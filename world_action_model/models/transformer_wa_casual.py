# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

try:
    from torch.nn.attention.flex_attention import create_block_mask as _create_flex_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
except ImportError:
    _create_flex_block_mask = None
    _flex_attention = None


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _build_teacher_forcing_mask(
    seq_len: int,
    num_state_tokens: int,
    num_ref_tokens: int,
    num_pred_action_tokens: int,
    num_gt_action_tokens: int,
    num_future_targets_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Self-attention mask for the teacher-forcing token layout
    [state | ref | pred_action | gt_action | future_state | value | future_image].

    Semantics (Q ↓ / K →):
        P  (state+ref):       only P
        A  (pred_action):     P + A                      (A is a sink — nothing else sees A)
        G  (gt_action):       P + G                      (cannot see A, otherwise A leaks via G's hidden state)
        SV (future_state/V):  P + G + SV                 (no A, no future_image)
        I  (future_image):    P + G + SV + I             (everything except A)
    """
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    neg_inf = float("-inf")
    p_end = num_state_tokens + num_ref_tokens
    a_end = p_end + num_pred_action_tokens
    g_end = a_end + num_gt_action_tokens
    sv_end = g_end + num_future_targets_tokens
    L = seq_len

    mask[0:p_end,      p_end:L]      = neg_inf  # P → rest
    mask[p_end:a_end,  a_end:L]      = neg_inf  # A → G, SV, I
    mask[a_end:g_end,  p_end:a_end]  = neg_inf  # G → A
    mask[a_end:g_end,  g_end:L]      = neg_inf  # G → SV, I
    mask[g_end:sv_end, p_end:a_end]  = neg_inf  # SV → A
    mask[g_end:sv_end, sv_end:L]     = neg_inf  # SV → I
    mask[sv_end:L,     p_end:a_end]  = neg_inf  # I → A
    return mask


def _dispatch_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    kernel_options: Optional[dict] = None,
) -> torch.Tensor:
    if _flex_attention is None:
        raise ImportError(
            "Flex attention is unavailable in the current PyTorch build. "
            "Upgrade PyTorch to a version that exposes torch.nn.attention.flex_attention."
        )

    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()
    hidden_states = _flex_attention(query, key, value, block_mask=block_mask, kernel_options=kernel_options)
    return hidden_states.permute(0, 2, 1, 3)


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def _project_attention_qkv(
    attn: "WanAttention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    if rotary_emb is not None:
        query = _apply_rotary_emb(query, *rotary_emb)
        key = _apply_rotary_emb(key, *rotary_emb)

    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAttnProcessor:
    _attention_backend = None
    _parallel_config = None
    _flex_block_mask_cache: dict[tuple, object] = {}

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    @classmethod
    def _get_flex_block_mask(
        cls,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        p_end: int,
        a_end: int,
        g_end: int,
        sv_end: int,
        device: torch.device,
        block_size: Union[int, tuple[int, int]],
    ):
        """Flex-attention block mask mirroring `_build_teacher_forcing_mask`.

        Segment boundaries (cumulative):
            0 .. p_end   -> prefix = state + ref
            p_end .. a_end   -> A = pred_action
            a_end .. g_end   -> G = gt_action
            g_end .. sv_end  -> SV = future_state + value
            sv_end .. seq_len-> I = future_image
        """
        if _create_flex_block_mask is None:
            raise ImportError(
                "Flex attention block masks are unavailable in the current PyTorch build. "
                "Upgrade PyTorch to a version that exposes torch.nn.attention.flex_attention.create_block_mask."
            )

        cache_key = (
            str(device),
            int(batch_size),
            int(num_heads),
            int(seq_len),
            int(p_end),
            int(a_end),
            int(g_end),
            int(sv_end),
            block_size if isinstance(block_size, tuple) else int(block_size),
        )
        block_mask = cls._flex_block_mask_cache.get(cache_key)
        if block_mask is not None:
            return block_mask

        def teacher_forcing_mask_mod(batch, head, q_idx, kv_idx):
            k_in_pred = (kv_idx >= p_end) & (kv_idx < a_end)
            k_not_pred = ~k_in_pred

            is_prefix_q = q_idx < p_end
            is_pred_q = (q_idx >= p_end) & (q_idx < a_end)
            is_gt_q = (q_idx >= a_end) & (q_idx < g_end)
            is_sv_q = (q_idx >= g_end) & (q_idx < sv_end)
            # else: is_img_q (q_idx >= sv_end)

            prefix_allowed = kv_idx < p_end                                           # P → P
            pred_allowed = kv_idx < a_end                                             # A → P + A
            gt_allowed = (kv_idx < p_end) | ((kv_idx >= a_end) & (kv_idx < g_end))    # G → P + G
            sv_allowed = k_not_pred & (kv_idx < sv_end)                               # SV → P + G + SV
            img_allowed = k_not_pred                                                  # I → everything except A

            return torch.where(
                is_prefix_q,
                prefix_allowed,
                torch.where(
                    is_pred_q,
                    pred_allowed,
                    torch.where(
                        is_gt_q,
                        gt_allowed,
                        torch.where(is_sv_q, sv_allowed, img_allowed),
                    ),
                ),
            )

        block_mask = _create_flex_block_mask(
            teacher_forcing_mask_mod,
            B=int(batch_size),
            H=int(num_heads),
            Q_LEN=int(seq_len),
            KV_LEN=int(seq_len),
            device=str(device),
            BLOCK_SIZE=block_size,
        )
        cls._flex_block_mask_cache[cache_key] = block_mask
        return block_mask

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_layout: Optional[Tuple[int, int, int]] = None,
        attention_implementation: str = "sdpa",
        flex_block_size: Union[int, tuple[int, int]] = 128,
        flex_kernel_options: Optional[dict] = None,
    ) -> torch.Tensor:
        is_self_attention = encoder_hidden_states is None and attn.cross_attention_dim_head is None
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _project_attention_qkv(attn, hidden_states, encoder_hidden_states, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        use_flex_attention = (
            attention_implementation == "flex"
            and is_self_attention
            and attention_layout is not None
            and attention_mask is None
        )
        if use_flex_attention:
            (
                num_state_tokens,
                num_ref_tokens,
                num_pred_action_tokens,
                num_gt_action_tokens,
                num_future_targets_tokens,
            ) = attention_layout
            p_end = int(num_state_tokens) + int(num_ref_tokens)
            a_end = p_end + int(num_pred_action_tokens)
            g_end = a_end + int(num_gt_action_tokens)
            sv_end = g_end + int(num_future_targets_tokens)
            block_mask = self._get_flex_block_mask(
                batch_size=query.shape[0],
                num_heads=attn.heads,
                seq_len=query.shape[1],
                p_end=p_end,
                a_end=a_end,
                g_end=g_end,
                sv_end=sv_end,
                device=query.device,
                block_size=flex_block_size,
            )
            hidden_states = _dispatch_flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                kernel_options=flex_kernel_options,
            )
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


class WanRotaryPosEmbed1D(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freq_cos, freq_sin = get_1d_rotary_pos_embed(
            attention_head_dim,
            max_seq_len,
            theta,
            use_real=True,
            repeat_interleave_real=True,
            freqs_dtype=freqs_dtype,
        )
        self.register_buffer("freqs_cos", freq_cos, persistent=False)
        self.register_buffer("freqs_sin", freq_sin, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, num_heads, head_dim)
        seq_len = hidden_states.shape[1]
        freqs_cos = self.freqs_cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
        freqs_sin = self.freqs_sin[:seq_len].unsqueeze(0).unsqueeze(2)
        return freqs_cos, freqs_sin


class RobotTokenAdapter(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down_proj = nn.Linear(dim, rank)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(rank, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.up_proj(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        robot_adapter_rank: int = 0,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.robot_ffn_adapter = RobotTokenAdapter(dim, int(robot_adapter_rank)) if int(robot_adapter_rank) > 0 else None

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        self_attention_layout: Optional[Tuple[int, int, int, int, int]] = None,
        self_attention_implementation: str = "sdpa",
        flex_block_size: Union[int, tuple[int, int]] = 128,
        flex_kernel_options: Optional[dict] = None,
        num_state_tokens: int = 0,
        num_ref_tokens: int = 0,
        num_robot_tokens: int = 0,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        # FP32LayerNorm already upcasts internally and returns origin dtype; the outer
        # .float() merely duplicates the input as fp32 (doubling saved activation size).
        norm_hidden_states = (self.norm1(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            None,
            self_attention_mask,
            rotary_emb,
            attention_layout=self_attention_layout,
            attention_implementation=self_attention_implementation,
            flex_block_size=flex_block_size,
            flex_kernel_options=flex_kernel_options,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        if self.robot_ffn_adapter is not None and (num_state_tokens > 0 or num_robot_tokens > 0):
            robot_start = num_state_tokens + num_ref_tokens
            robot_end = robot_start + num_robot_tokens
            adapter_input = torch.cat(
                [
                    norm_hidden_states[:, :num_state_tokens],
                    norm_hidden_states[:, robot_start:robot_end],
                ],
                dim=1,
            )
            adapter_output = self.robot_ffn_adapter(adapter_input)
            state_adapter = adapter_output[:, :num_state_tokens]
            robot_adapter = adapter_output[:, num_state_tokens:]
            ff_output = torch.cat(
                [
                    ff_output[:, :num_state_tokens] + state_adapter,
                    ff_output[:, num_state_tokens:robot_start],
                    ff_output[:, robot_start:robot_end] + robot_adapter,
                    ff_output[:, robot_end:],
                ],
                dim=1,
            )
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

    def forward_prefix_cache(
        self,
        prefix_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        prefix_temb: torch.Tensor,
        prefix_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        self_attention_mask: Optional[torch.Tensor] = None,
        num_state_tokens: int = 0,
        num_ref_tokens: int = 0,
        num_robot_tokens: int = 0,
    ) -> dict[str, torch.Tensor]:
        if prefix_temb.ndim != 4:
            raise ValueError("prefix cache requires per-token timestep embeddings")

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + prefix_temb.float()
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)

        norm_prefix = (self.norm1(prefix_hidden_states) * (1 + scale_msa) + shift_msa).type_as(prefix_hidden_states)
        prefix_query, prefix_key, prefix_value = _project_attention_qkv(self.attn1, norm_prefix, None, prefix_rotary_emb)
        prefix_attn = dispatch_attention_fn(
            prefix_query,
            prefix_key,
            prefix_value,
            attn_mask=self_attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=WanAttnProcessor._attention_backend,
            parallel_config=WanAttnProcessor._parallel_config,
        )
        prefix_attn = prefix_attn.flatten(2, 3).type_as(prefix_key)
        prefix_attn = self.attn1.to_out[0](prefix_attn)
        prefix_attn = self.attn1.to_out[1](prefix_attn)
        prefix_hidden_states = (prefix_hidden_states.float() + prefix_attn * gate_msa).type_as(prefix_hidden_states)

        norm_prefix = self.norm2(prefix_hidden_states)
        prefix_attn = self.attn2(norm_prefix, encoder_hidden_states, None, None)
        prefix_hidden_states = prefix_hidden_states + prefix_attn

        norm_prefix = (self.norm3(prefix_hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(prefix_hidden_states)
        prefix_ff = self.ffn(norm_prefix)
        if self.robot_ffn_adapter is not None and (num_state_tokens > 0 or num_robot_tokens > 0):
            robot_start = num_state_tokens + num_ref_tokens
            robot_end = robot_start + num_robot_tokens
            adapter_input = torch.cat(
                [
                    norm_prefix[:, :num_state_tokens],
                    norm_prefix[:, robot_start:robot_end],
                ],
                dim=1,
            )
            adapter_output = self.robot_ffn_adapter(adapter_input)
            state_adapter = adapter_output[:, :num_state_tokens]
            robot_adapter = adapter_output[:, num_state_tokens:]
            prefix_ff = torch.cat(
                [
                    prefix_ff[:, :num_state_tokens] + state_adapter,
                    prefix_ff[:, num_state_tokens:robot_start],
                    prefix_ff[:, robot_start:robot_end] + robot_adapter,
                    prefix_ff[:, robot_end:],
                ],
                dim=1,
            )
        prefix_hidden_states = (prefix_hidden_states.float() + prefix_ff.float() * c_gate_msa).type_as(prefix_hidden_states)

        return {
            "hidden_states": prefix_hidden_states,
            "self_key": prefix_key.detach(),
            "self_value": prefix_value.detach(),
        }

    def forward_with_prefix_cache(
        self,
        suffix_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        suffix_temb: torch.Tensor,
        suffix_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        prefix_cache: dict[str, torch.Tensor],
        num_state_tokens: int = 0,
        num_ref_tokens: int = 0,
        num_robot_tokens: int = 0,
    ) -> torch.Tensor:
        if suffix_temb.ndim != 4:
            raise ValueError("prefix cache requires per-token timestep embeddings")

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + suffix_temb.float()
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)

        norm_suffix = (self.norm1(suffix_hidden_states) * (1 + scale_msa) + shift_msa).type_as(suffix_hidden_states)
        suffix_query, suffix_key, suffix_value = _project_attention_qkv(self.attn1, norm_suffix, None, suffix_rotary_emb)
        key = torch.cat([prefix_cache["self_key"], suffix_key], dim=1)
        value = torch.cat([prefix_cache["self_value"], suffix_value], dim=1)
        attn_output = dispatch_attention_fn(
            suffix_query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=WanAttnProcessor._attention_backend,
            parallel_config=WanAttnProcessor._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3).type_as(suffix_query)
        attn_output = self.attn1.to_out[0](attn_output)
        attn_output = self.attn1.to_out[1](attn_output)
        suffix_hidden_states = (suffix_hidden_states.float() + attn_output * gate_msa).type_as(suffix_hidden_states)

        norm_suffix = self.norm2(suffix_hidden_states)
        attn_output = self.attn2(norm_suffix, encoder_hidden_states, None, None)
        suffix_hidden_states = suffix_hidden_states + attn_output

        norm_suffix = (self.norm3(suffix_hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(suffix_hidden_states)
        ff_output = self.ffn(norm_suffix)
        if self.robot_ffn_adapter is not None and num_robot_tokens > 0:
            robot_adapter = self.robot_ffn_adapter(norm_suffix)
            ff_output = ff_output + robot_adapter
        suffix_hidden_states = (suffix_hidden_states.float() + ff_output.float() * c_gate_msa).type_as(suffix_hidden_states)

        return suffix_hidden_states


class CasualWorldActionTransformer(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*": {
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        robot_adapter_rank: int = 0,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    robot_adapter_rank=robot_adapter_rank,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # Robot-token encoders/decoders: current state and future state share a dedicated state encoder
        # because they live in joint-position space (state_mean/std), while `action` lives in delta space.
        self.action_rope = WanRotaryPosEmbed1D(attention_head_dim, rope_max_seq_len)
        self.state_encoder = self._make_vector_encoder(14, inner_dim)
        self.action_encoder = self._make_vector_encoder(14, inner_dim)
        self.value_encoder = self._make_vector_encoder(1, inner_dim)
        self.action_decoder = self._make_vector_decoder(inner_dim, 14)
        self.future_state_decoder = self._make_vector_decoder(inner_dim, 14)
        self.value_decoder = self._make_vector_decoder(inner_dim, 1)
        self.self_attention_implementation = "sdpa"
        self.flex_attention_block_size = 128
        self.flex_attention_kernel_options = None

    def set_robot_adapter_rank(self, rank: int):
        rank = int(rank)
        inner_dim = int(self.config.num_attention_heads) * int(self.config.attention_head_dim)
        for block in self.blocks:
            block.robot_ffn_adapter = RobotTokenAdapter(inner_dim, rank) if rank > 0 else None
        if hasattr(self, "config"):
            self.config.robot_adapter_rank = rank
        return self

    @staticmethod
    def _make_vector_encoder(input_dim: int, inner_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(int(input_dim), 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, inner_dim),
        )

    @staticmethod
    def _make_vector_decoder(inner_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(inner_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, int(output_dim)),
        )

    def reset_action_heads(self, action_dim: int = 14):
        inner_dim = int(self.config.num_attention_heads) * int(self.config.attention_head_dim)
        rope_max_seq_len = int(getattr(self.action_rope, "max_seq_len", getattr(self.config, "rope_max_seq_len", 1024)))
        attention_head_dim = int(self.config.attention_head_dim)
        self.state_encoder = self._make_vector_encoder(int(action_dim), inner_dim)
        self.action_encoder = self._make_vector_encoder(int(action_dim), inner_dim)
        self.value_encoder = self._make_vector_encoder(1, inner_dim)
        self.action_decoder = self._make_vector_decoder(inner_dim, int(action_dim))
        self.future_state_decoder = self._make_vector_decoder(inner_dim, int(action_dim))
        self.value_decoder = self._make_vector_decoder(inner_dim, 1)
        self.action_rope = WanRotaryPosEmbed1D(attention_head_dim, rope_max_seq_len)
        return self

    def set_self_attention_implementation(self, implementation: str):
        implementation = str(implementation).strip().lower()
        if implementation not in {"sdpa", "flex"}:
            raise ValueError(f"Unsupported self attention implementation: {implementation}")
        self.self_attention_implementation = implementation
        if hasattr(self, "config"):
            self.config.self_attention_implementation = implementation
        return self

    def set_flex_attention_block_size(self, block_size: Union[int, tuple[int, int]]):
        self.flex_attention_block_size = block_size
        if hasattr(self, "config"):
            self.config.flex_attention_block_size = block_size
        return self

    def set_flex_attention_kernel_options(self, kernel_options: Optional[dict]):
        self.flex_attention_kernel_options = None if kernel_options is None else dict(kernel_options)
        if hasattr(self, "config"):
            self.config.flex_attention_kernel_options = self.flex_attention_kernel_options
        return self

    def _use_flex_attention(self) -> bool:
        if self.self_attention_implementation != "flex":
            return False
        if _flex_attention is None or _create_flex_block_mask is None:
            raise ImportError(
                "self_attention_implementation='flex' requires a newer PyTorch build with "
                "torch.nn.attention.flex_attention."
            )
        return True

    def _compute_timestep_embeddings(
        self,
        timestep: torch.Tensor,
        dtype: torch.dtype,
        timestep_seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = self.condition_embedder.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.condition_embedder.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.condition_embedder.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.condition_embedder.time_proj(self.condition_embedder.act_fn(temb))
        return temb, timestep_proj

    def _encode_robot_targets(
        self,
        *,
        state: Optional[torch.Tensor] = None,
        pred_action: Optional[torch.Tensor] = None,
        gt_action: Optional[torch.Tensor] = None,
        future_state: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], dict[str, int]]:
        """Encode the robot-side tokens for the teacher-forcing layout.

        Order of concatenation matches the full-training sequence:
            [state | pred_action | gt_action | future_state | value]
        Any argument may be None — its segment is skipped (0 tokens). Used this
        way by the two-stage inference forward paths (Stage 1: only state + pred_action;
        Stage 2: state + dummy pred_action + gt_action + future_state + value,
        optionally followed by future_image tokens in the video branch).
        """
        chunks: list[torch.Tensor] = []
        counts = {
            "num_state_tokens": 0,
            "num_pred_action_tokens": 0,
            "num_gt_action_tokens": 0,
            "num_future_state_tokens": 0,
            "num_value_tokens": 0,
        }
        if state is not None:
            s = self.state_encoder(state)
            chunks.append(s)
            counts["num_state_tokens"] = int(s.shape[1])
        if pred_action is not None:
            a = self.action_encoder(pred_action)
            chunks.append(a)
            counts["num_pred_action_tokens"] = int(a.shape[1])
        if gt_action is not None:
            g = self.action_encoder(gt_action)  # shared encoder with pred_action
            chunks.append(g)
            counts["num_gt_action_tokens"] = int(g.shape[1])
        if future_state is not None:
            fs = self.state_encoder(future_state)
            chunks.append(fs)
            counts["num_future_state_tokens"] = int(fs.shape[1])
        if value is not None:
            v = self.value_encoder(value)
            chunks.append(v)
            counts["num_value_tokens"] = int(v.shape[1])

        counts["num_robot_tokens"] = (
            counts["num_pred_action_tokens"]
            + counts["num_gt_action_tokens"]
            + counts["num_future_state_tokens"]
            + counts["num_value_tokens"]
        )
        counts["num_future_targets_tokens"] = (
            counts["num_future_state_tokens"] + counts["num_value_tokens"]
        )

        extra_states = torch.cat(chunks, dim=1) if chunks else None
        if extra_states is not None:
            rope_max_seq_len = int(
                getattr(self.action_rope, "max_seq_len", getattr(self.config, "rope_max_seq_len", 1024))
            )
            if int(extra_states.shape[1]) > rope_max_seq_len:
                raise ValueError(
                    f"Robot token length {int(extra_states.shape[1])} exceeds action_rope.max_seq_len="
                    f"{rope_max_seq_len}. Increase rope_max_seq_len or reduce action_chunk/state_repeats."
                )
        return extra_states, counts

    def _reorder_with_robot_tokens(
        self,
        hidden_states: torch.Tensor,
        extra_states: torch.Tensor,
        num_state_tokens: int,
        num_ref_tokens: int,
        extra_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        include_noisy_latents: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Interleave the robot-token extras with the video tokens to produce
        [state | ref | pred_action | gt_action | future_state | value | (future_image)].
        The caller's `extra_states` are already ordered as
        [state | pred_action | gt_action | future_state | value] (any subset allowed);
        we pull `state` out of the front and drop the remainder after `ref`.
        """
        state_part = extra_states[:, :num_state_tokens]
        robot_part = extra_states[:, num_state_tokens:]
        state_rope_cos = extra_rotary_emb[0][:, :num_state_tokens]
        state_rope_sin = extra_rotary_emb[1][:, :num_state_tokens]
        robot_rope_cos = extra_rotary_emb[0][:, num_state_tokens:]
        robot_rope_sin = extra_rotary_emb[1][:, num_state_tokens:]

        ref_part = hidden_states[:, :num_ref_tokens]
        ref_rope_cos = rotary_emb[0][:, :num_ref_tokens]
        ref_rope_sin = rotary_emb[1][:, :num_ref_tokens]

        if include_noisy_latents:
            video_part = hidden_states[:, num_ref_tokens:]
            video_rope_cos = rotary_emb[0][:, num_ref_tokens:]
            video_rope_sin = rotary_emb[1][:, num_ref_tokens:]
            hidden_states = torch.cat([state_part, ref_part, robot_part, video_part], dim=1)
            rotary_emb = (
                torch.cat([state_rope_cos, ref_rope_cos, robot_rope_cos, video_rope_cos], dim=1),
                torch.cat([state_rope_sin, ref_rope_sin, robot_rope_sin, video_rope_sin], dim=1),
            )
        else:
            hidden_states = torch.cat([state_part, ref_part, robot_part], dim=1)
            rotary_emb = (
                torch.cat([state_rope_cos, ref_rope_cos, robot_rope_cos], dim=1),
                torch.cat([state_rope_sin, ref_rope_sin, robot_rope_sin], dim=1),
            )
        return hidden_states, rotary_emb

    def _split_robot_outputs(
        self,
        hidden_states: torch.Tensor,
        num_state_tokens: int,
        num_ref_tokens: int,
        num_pred_action_tokens: int,
        num_gt_action_tokens: int,
        num_future_state_tokens: int,
        num_value_tokens: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Decode the heads that were present in this forward's sequence.
        Layout: [state | ref | pred_action | gt_action | future_state | value | ...].
        The `gt_action` slot is a K/V-only condition — no decoder head runs on it.
        Any absent segment (token count 0) produces None.
        """
        robot_start = int(num_state_tokens) + int(num_ref_tokens)
        pred_end = robot_start + int(num_pred_action_tokens)
        gt_end = pred_end + int(num_gt_action_tokens)
        future_state_end = gt_end + int(num_future_state_tokens)
        value_end = future_state_end + int(num_value_tokens)

        action_pred = None
        if num_pred_action_tokens > 0:
            action_pred = self.action_decoder(hidden_states[:, robot_start:pred_end])
        future_state_pred = None
        if num_future_state_tokens > 0:
            future_state_pred = self.future_state_decoder(hidden_states[:, gt_end:future_state_end])
        value_pred = None
        if num_value_tokens > 0:
            value_pred = self.value_decoder(hidden_states[:, future_state_end:value_end])
        return action_pred, future_state_pred, value_pred

    def _validate_prefix_cache(
        self,
        prefix_cache: dict[str, Any],
        *,
        expected_prefix_len: int,
        batch_size: int,
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
        blocks = prefix_cache.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != len(self.blocks):
            raise ValueError(
                f"prefix cache block count mismatch: expected {len(self.blocks)}, got "
                f"{0 if blocks is None else len(blocks)}"
            )
        if int(prefix_cache.get("prefix_len", -1)) != int(expected_prefix_len):
            raise ValueError(
                f"prefix cache length mismatch: expected {int(expected_prefix_len)}, "
                f"got {prefix_cache.get('prefix_len')}"
            )
        encoder_hidden_states = prefix_cache.get("encoder_hidden_states")
        if encoder_hidden_states is None or int(encoder_hidden_states.shape[0]) != int(batch_size):
            got = None if encoder_hidden_states is None else int(encoder_hidden_states.shape[0])
            raise ValueError(f"prefix cache batch mismatch: expected {int(batch_size)}, got {got}")
        if encoder_hidden_states.device != device:
            raise ValueError(f"prefix cache device mismatch: expected {device}, got {encoder_hidden_states.device}")

        for idx, block_cache in enumerate(blocks):
            if not isinstance(block_cache, dict) or "self_key" not in block_cache or "self_value" not in block_cache:
                raise ValueError(f"prefix cache block {idx} is missing self_key/self_value")
            self_key = block_cache["self_key"]
            self_value = block_cache["self_value"]
            if self_key.shape != self_value.shape:
                raise ValueError(f"prefix cache block {idx} key/value shape mismatch")
            if self_key.ndim != 4:
                raise ValueError(f"prefix cache block {idx} must be rank-4, got shape {tuple(self_key.shape)}")
            if int(self_key.shape[0]) != int(batch_size) or int(self_key.shape[1]) != int(expected_prefix_len):
                raise ValueError(
                    f"prefix cache block {idx} shape mismatch: expected batch/prefix "
                    f"{int(batch_size)}/{int(expected_prefix_len)}, got {tuple(self_key.shape[:2])}"
                )
            if self_key.device != device or self_value.device != device:
                raise ValueError(f"prefix cache block {idx} device mismatch: expected {device}")
        return blocks

    def _check_prefix_cache_supported(self) -> None:
        if self.self_attention_implementation != "sdpa":
            raise ValueError("prefix cache supports only self_attention_implementation='sdpa'.")
        if (
            getattr(self.config, "image_dim", None) is not None
            or getattr(self.config, "added_kv_proj_dim", None) is not None
            or self.condition_embedder.image_embedder is not None
        ):
            raise ValueError("prefix cache does not support image-conditioned transformers.")

    def forward(self, *args, **kwargs):
        if self.training:
            return self._forward_train(*args, **kwargs)
        else:
            action_only = kwargs.pop("action_only", False)
            if not action_only:
                return self._forward_inference_future(*args, **kwargs)

            if any(kwargs.get(name) is not None for name in ("gt_action", "future_state", "value")):
                return self._forward_inference_future_action_only(*args, **kwargs)

            return self._forward_inference_action_only(*args, **kwargs)

    def _forward_inference_future(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        pred_action: Optional[torch.Tensor] = None,
        gt_action: Optional[torch.Tensor] = None,
        future_state: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Stage-2 inference forward. Sequence:
        [state | ref | pred_action | gt_action | future_state | value | future_image].
        `pred_action` is a dummy noisy slot kept only to preserve the training-time
        token layout. `gt_action` is the clean converged output of Stage 1.
        `future_state`, `value`, `noisy_latents` (future_image) are noisy and
        denoised here. The decoded `action_pred` from the dummy slot should be ignored."""
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states = torch.cat([ref_latents, noisy_latents], dim=2)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        extra_states, layout = self._encode_robot_targets(
            state=state,
            pred_action=pred_action,
            gt_action=gt_action,
            future_state=future_state,
            value=value,
        )
        num_state_tokens = layout["num_state_tokens"]
        num_pred_action_tokens = layout["num_pred_action_tokens"]
        num_gt_action_tokens = layout["num_gt_action_tokens"]
        num_future_state_tokens = layout["num_future_state_tokens"]
        num_value_tokens = layout["num_value_tokens"]
        num_future_targets_tokens = layout["num_future_targets_tokens"]
        num_robot_tokens = layout["num_robot_tokens"]
        num_ref_tokens = post_patch_width * post_patch_height

        rotary_emb = self.rope(hidden_states)
        extra_rotary_emb = self.action_rope(extra_states)

        hidden_states = self.patch_embedding(hidden_states.to(dtype=self.patch_embedding.weight.dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        hidden_states, rotary_emb = self._reorder_with_robot_tokens(
            hidden_states=hidden_states,
            extra_states=extra_states,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            extra_rotary_emb=extra_rotary_emb,
            rotary_emb=rotary_emb,
            include_noisy_latents=True,
        )

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        self_attention_layout = (
            num_state_tokens,
            num_ref_tokens,
            num_pred_action_tokens,
            num_gt_action_tokens,
            num_future_targets_tokens,
        )
        if self._use_flex_attention():
            self_attention_mask = None
        else:
            self_attention_mask = _build_teacher_forcing_mask(
                seq_len=hidden_states.shape[1],
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_pred_action_tokens=num_pred_action_tokens,
                num_gt_action_tokens=num_gt_action_tokens,
                num_future_targets_tokens=num_future_targets_tokens,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                self_attention_mask,
                self_attention_layout=self_attention_layout,
                self_attention_implementation=self.self_attention_implementation,
                flex_block_size=self.flex_attention_block_size,
                flex_kernel_options=self.flex_attention_kernel_options,
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_robot_tokens=num_robot_tokens,
            )

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift, scale = shift.to(hidden_states.device), scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)

        action_pred, future_state_pred, value_pred = self._split_robot_outputs(
            hidden_states=hidden_states,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            num_pred_action_tokens=num_pred_action_tokens,
            num_gt_action_tokens=num_gt_action_tokens,
            num_future_state_tokens=num_future_state_tokens,
            num_value_tokens=num_value_tokens,
        )

        p_end = num_state_tokens + num_ref_tokens
        video_start = p_end + num_robot_tokens
        video_hidden = torch.cat(
            [hidden_states[:, num_state_tokens:p_end], hidden_states[:, video_start:]],
            dim=1,
        )
        video_output = self.proj_out(video_hidden)
        video_output = video_output.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        video_output = video_output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = video_output.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output, action_pred, future_state_pred, value_pred

        return Transformer2DModelOutput(sample=output)

    def _forward_inference_future_action_only(
        self,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        pred_action: Optional[torch.Tensor] = None,
        gt_action: Optional[torch.Tensor] = None,
        future_state: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        noisy_latents: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        prefix_cache: Optional[dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Stage-2 inference without future-image denoising. Sequence:
        [state | ref | pred_action | gt_action | future_state | value].
        `pred_action` is a dummy noisy slot kept only to match the training-time
        layout; `gt_action` is the clean Stage-1 action. Only `future_state` and
        `value` are decoded and denoised here."""
        del noisy_latents, action  # accepted for signature compatibility
        if prefix_cache is not None:
            return self._forward_inference_future_action_only_cached(
                ref_latents=ref_latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                state=state,
                pred_action=pred_action,
                gt_action=gt_action,
                future_state=future_state,
                value=value,
                prefix_cache=prefix_cache,
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states = ref_latents
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        extra_states, layout = self._encode_robot_targets(
            state=state,
            pred_action=pred_action,
            gt_action=gt_action,
            future_state=future_state,
            value=value,
        )
        num_state_tokens = layout["num_state_tokens"]
        num_pred_action_tokens = layout["num_pred_action_tokens"]
        num_gt_action_tokens = layout["num_gt_action_tokens"]
        num_future_state_tokens = layout["num_future_state_tokens"]
        num_value_tokens = layout["num_value_tokens"]
        num_future_targets_tokens = layout["num_future_targets_tokens"]
        num_robot_tokens = layout["num_robot_tokens"]
        num_ref_tokens = post_patch_width * post_patch_height

        rotary_emb = self.rope(hidden_states)
        extra_rotary_emb = self.action_rope(extra_states)

        hidden_states = self.patch_embedding(hidden_states.to(dtype=self.patch_embedding.weight.dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        hidden_states, rotary_emb = self._reorder_with_robot_tokens(
            hidden_states=hidden_states,
            extra_states=extra_states,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            extra_rotary_emb=extra_rotary_emb,
            rotary_emb=rotary_emb,
            include_noisy_latents=False,
        )

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        self_attention_layout = (
            num_state_tokens,
            num_ref_tokens,
            num_pred_action_tokens,
            num_gt_action_tokens,
            num_future_targets_tokens,
        )
        if self._use_flex_attention():
            self_attention_mask = None
        else:
            self_attention_mask = _build_teacher_forcing_mask(
                seq_len=hidden_states.shape[1],
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_pred_action_tokens=num_pred_action_tokens,
                num_gt_action_tokens=num_gt_action_tokens,
                num_future_targets_tokens=num_future_targets_tokens,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                self_attention_mask,
                self_attention_layout=self_attention_layout,
                self_attention_implementation=self.self_attention_implementation,
                flex_block_size=self.flex_attention_block_size,
                flex_kernel_options=self.flex_attention_kernel_options,
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_robot_tokens=num_robot_tokens,
            )

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift, scale = shift.to(hidden_states.device), scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)

        _, future_state_pred, value_pred = self._split_robot_outputs(
            hidden_states=hidden_states,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            num_pred_action_tokens=num_pred_action_tokens,
            num_gt_action_tokens=num_gt_action_tokens,
            num_future_state_tokens=num_future_state_tokens,
            num_value_tokens=num_value_tokens,
        )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return None, None, future_state_pred, value_pred

        return Transformer2DModelOutput(sample=None)

    def _forward_inference_action_only(
        self,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        pred_action: Optional[torch.Tensor] = None,
        # accepted for signature compatibility with the full inference path — ignored in Stage 1
        noisy_latents: Optional[torch.Tensor] = None,
        gt_action: Optional[torch.Tensor] = None,
        future_state: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        prefix_cache: Optional[dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Stage-1 inference forward. Sequence: [state | ref | pred_action].
        Only `action` (noisy pred_action) is denoised; no gt_action, no future_state,
        no value, no future_image. Returns `(None, action_pred, None, None)`."""
        del noisy_latents, gt_action, future_state, value, pred_action  # not used in Stage 1
        if prefix_cache is not None:
            return self._forward_inference_action_only_cached(
                ref_latents=ref_latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                state=state,
                action=action,
                prefix_cache=prefix_cache,
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # Stage 1 uses only the ref frame(s) — no future video latents in the sequence.
        hidden_states = ref_latents
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        extra_states, layout = self._encode_robot_targets(state=state, pred_action=action)
        num_state_tokens = layout["num_state_tokens"]
        num_pred_action_tokens = layout["num_pred_action_tokens"]
        num_gt_action_tokens = layout["num_gt_action_tokens"]  # 0
        num_future_targets_tokens = layout["num_future_targets_tokens"]  # 0
        num_robot_tokens = layout["num_robot_tokens"]
        num_ref_tokens = post_patch_width * post_patch_height

        rotary_emb = self.rope(hidden_states)
        extra_rotary_emb = self.action_rope(extra_states)

        hidden_states = self.patch_embedding(hidden_states.to(dtype=self.patch_embedding.weight.dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        hidden_states, rotary_emb = self._reorder_with_robot_tokens(
            hidden_states=hidden_states,
            extra_states=extra_states,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            extra_rotary_emb=extra_rotary_emb,
            rotary_emb=rotary_emb,
            include_noisy_latents=False,
        )

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        self_attention_layout = (
            num_state_tokens,
            num_ref_tokens,
            num_pred_action_tokens,
            num_gt_action_tokens,
            num_future_targets_tokens,
        )
        if self._use_flex_attention():
            self_attention_mask = None
        else:
            self_attention_mask = _build_teacher_forcing_mask(
                seq_len=hidden_states.shape[1],
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_pred_action_tokens=num_pred_action_tokens,
                num_gt_action_tokens=num_gt_action_tokens,
                num_future_targets_tokens=num_future_targets_tokens,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                self_attention_mask,
                self_attention_layout=self_attention_layout,
                self_attention_implementation=self.self_attention_implementation,
                flex_block_size=self.flex_attention_block_size,
                flex_kernel_options=self.flex_attention_kernel_options,
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_robot_tokens=num_robot_tokens,
            )

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift, scale = shift.to(hidden_states.device), scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)

        action_pred, _, _ = self._split_robot_outputs(
            hidden_states=hidden_states,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            num_pred_action_tokens=num_pred_action_tokens,
            num_gt_action_tokens=num_gt_action_tokens,
            num_future_state_tokens=0,
            num_value_tokens=0,
        )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return None, action_pred, None, None

        return Transformer2DModelOutput(sample=None)

    def _build_inference_action_prefix_cache(
        self,
        ref_latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        state: torch.Tensor,
    ) -> dict[str, Any]:
        self._check_prefix_cache_supported()
        hidden_states = ref_latents
        _batch_size, _num_channels, _num_frames, height, width = hidden_states.shape
        _p_t, p_h, p_w = self.config.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        state_states, state_layout = self._encode_robot_targets(state=state)
        num_state_tokens = state_layout["num_state_tokens"]
        num_ref_tokens = post_patch_width * post_patch_height

        rotary_emb = self.rope(hidden_states)
        state_rotary_emb = self.action_rope(state_states)
        hidden_states = self.patch_embedding(hidden_states.to(dtype=self.patch_embedding.weight.dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        prefix_hidden_states = torch.cat([state_states, hidden_states[:, :num_ref_tokens]], dim=1)
        prefix_rotary_emb = (
            torch.cat([state_rotary_emb[0][:, :num_state_tokens], rotary_emb[0][:, :num_ref_tokens]], dim=1),
            torch.cat([state_rotary_emb[1][:, :num_state_tokens], rotary_emb[1][:, :num_ref_tokens]], dim=1),
        )

        timestep = torch.zeros(
            prefix_hidden_states.shape[:2],
            device=prefix_hidden_states.device,
            dtype=prefix_hidden_states.dtype,
        )
        ts_seq_len = timestep.shape[1]
        _temb, timestep_proj = self._compute_timestep_embeddings(
            timestep.flatten(),
            dtype=encoder_hidden_states.dtype,
            timestep_seq_len=ts_seq_len,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        encoder_hidden_states = self.condition_embedder.text_embedder(encoder_hidden_states)

        blocks = []
        for block in self.blocks:
            block_cache = block.forward_prefix_cache(
                prefix_hidden_states,
                encoder_hidden_states,
                timestep_proj,
                prefix_rotary_emb,
                self_attention_mask=None,
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_robot_tokens=0,
            )
            prefix_hidden_states = block_cache["hidden_states"]
            blocks.append(
                {
                    "self_key": block_cache["self_key"],
                    "self_value": block_cache["self_value"],
                }
            )

        return {
            "blocks": blocks,
            "encoder_hidden_states": encoder_hidden_states,
            "num_state_tokens": num_state_tokens,
            "num_ref_tokens": num_ref_tokens,
            "prefix_len": num_state_tokens + num_ref_tokens,
        }

    def build_inference_action_prefix_cache(
        self,
        ref_latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        state: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        prefix_cache = self._build_inference_action_prefix_cache(
            ref_latents=ref_latents,
            encoder_hidden_states=encoder_hidden_states,
            state=state,
        )
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        return prefix_cache

    def _forward_inference_action_only_cached(
        self,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        prefix_cache: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if encoder_hidden_states_image is not None:
            raise ValueError("prefix cache does not support encoder_hidden_states_image/image-conditioned inputs.")
        del kwargs
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        if prefix_cache is None:
            prefix_cache = self._build_inference_action_prefix_cache(
                ref_latents=ref_latents,
                encoder_hidden_states=encoder_hidden_states,
                state=state,
            )
        num_state_tokens = int(prefix_cache["num_state_tokens"])
        num_ref_tokens = int(prefix_cache["num_ref_tokens"])
        num_pred_action_tokens = int(action.shape[1])
        num_gt_action_tokens = 0
        num_future_targets_tokens = 0
        num_robot_tokens = num_pred_action_tokens

        action_states, action_layout = self._encode_robot_targets(pred_action=action)
        if int(action_layout["num_pred_action_tokens"]) != num_pred_action_tokens:
            raise ValueError("cached action token count mismatch")
        prefix_blocks = self._validate_prefix_cache(
            prefix_cache,
            expected_prefix_len=num_state_tokens + num_ref_tokens,
            batch_size=int(action_states.shape[0]),
            device=action_states.device,
        )
        action_rope_states = action_states.new_empty(
            action_states.shape[0],
            num_state_tokens + num_pred_action_tokens,
            action_states.shape[-1],
        )
        action_rotary_emb = self.action_rope(action_rope_states)
        suffix_rotary_emb = (
            action_rotary_emb[0][:, num_state_tokens : num_state_tokens + num_pred_action_tokens],
            action_rotary_emb[1][:, num_state_tokens : num_state_tokens + num_pred_action_tokens],
        )

        if timestep.ndim != 2:
            raise ValueError("cached action-only inference requires per-token timestep")
        suffix_timestep = timestep[:, num_state_tokens + num_ref_tokens :]
        if int(suffix_timestep.shape[1]) != num_pred_action_tokens:
            raise ValueError("cached action-only timestep token count mismatch")
        ts_seq_len = suffix_timestep.shape[1]
        temb, timestep_proj = self._compute_timestep_embeddings(
            suffix_timestep.flatten(),
            dtype=prefix_cache["encoder_hidden_states"].dtype,
            timestep_seq_len=ts_seq_len,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        hidden_states = action_states
        for block, block_cache in zip(self.blocks, prefix_blocks):
            hidden_states = block.forward_with_prefix_cache(
                hidden_states,
                prefix_cache["encoder_hidden_states"],
                timestep_proj,
                suffix_rotary_emb,
                block_cache,
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_robot_tokens=num_robot_tokens,
            )

        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
        shift, scale = shift.to(hidden_states.device), scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        action_pred = self.action_decoder(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return None, action_pred, None, None

        return Transformer2DModelOutput(sample=None)

    def _build_inference_future_action_prefix_cache(
        self,
        ref_latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        state: torch.Tensor,
        gt_action: torch.Tensor,
    ) -> dict[str, Any]:
        self._check_prefix_cache_supported()
        hidden_states = ref_latents
        _batch_size, _num_channels, _num_frames, height, width = hidden_states.shape
        _p_t, p_h, p_w = self.config.patch_size
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        prefix_robot_states, prefix_layout = self._encode_robot_targets(state=state, gt_action=gt_action)
        num_state_tokens = prefix_layout["num_state_tokens"]
        num_gt_action_tokens = prefix_layout["num_gt_action_tokens"]
        num_ref_tokens = post_patch_width * post_patch_height
        num_robot_tokens = num_gt_action_tokens

        rotary_emb = self.rope(hidden_states)
        full_robot_len = num_state_tokens + int(gt_action.shape[1]) + num_gt_action_tokens
        robot_rope_states = prefix_robot_states.new_empty(
            prefix_robot_states.shape[0],
            full_robot_len,
            prefix_robot_states.shape[-1],
        )
        full_robot_rotary_emb = self.action_rope(robot_rope_states)
        state_rotary_emb = (
            full_robot_rotary_emb[0][:, :num_state_tokens],
            full_robot_rotary_emb[1][:, :num_state_tokens],
        )
        gt_start = num_state_tokens + int(gt_action.shape[1])
        gt_end = gt_start + num_gt_action_tokens
        gt_rotary_emb = (
            full_robot_rotary_emb[0][:, gt_start:gt_end],
            full_robot_rotary_emb[1][:, gt_start:gt_end],
        )

        hidden_states = self.patch_embedding(hidden_states.to(dtype=self.patch_embedding.weight.dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        prefix_hidden_states = torch.cat(
            [
                prefix_robot_states[:, :num_state_tokens],
                hidden_states[:, :num_ref_tokens],
                prefix_robot_states[:, num_state_tokens:],
            ],
            dim=1,
        )
        prefix_rotary_emb = (
            torch.cat([state_rotary_emb[0], rotary_emb[0][:, :num_ref_tokens], gt_rotary_emb[0]], dim=1),
            torch.cat([state_rotary_emb[1], rotary_emb[1][:, :num_ref_tokens], gt_rotary_emb[1]], dim=1),
        )

        timestep = torch.zeros(
            prefix_hidden_states.shape[:2],
            device=prefix_hidden_states.device,
            dtype=prefix_hidden_states.dtype,
        )
        ts_seq_len = timestep.shape[1]
        _temb, timestep_proj = self._compute_timestep_embeddings(
            timestep.flatten(),
            dtype=encoder_hidden_states.dtype,
            timestep_seq_len=ts_seq_len,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        encoder_hidden_states = self.condition_embedder.text_embedder(encoder_hidden_states)

        p_end = num_state_tokens + num_ref_tokens
        seq_len = prefix_hidden_states.shape[1]
        self_attention_mask = torch.zeros(
            (seq_len, seq_len),
            device=prefix_hidden_states.device,
            dtype=prefix_hidden_states.dtype,
        )
        self_attention_mask[:p_end, p_end:] = float("-inf")

        blocks = []
        for block in self.blocks:
            block_cache = block.forward_prefix_cache(
                prefix_hidden_states,
                encoder_hidden_states,
                timestep_proj,
                prefix_rotary_emb,
                self_attention_mask=self_attention_mask,
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_robot_tokens=num_robot_tokens,
            )
            prefix_hidden_states = block_cache["hidden_states"]
            blocks.append(
                {
                    "self_key": block_cache["self_key"],
                    "self_value": block_cache["self_value"],
                }
            )

        return {
            "blocks": blocks,
            "encoder_hidden_states": encoder_hidden_states,
            "num_state_tokens": num_state_tokens,
            "num_ref_tokens": num_ref_tokens,
            "num_gt_action_tokens": num_gt_action_tokens,
            "num_prefix_robot_tokens": num_robot_tokens,
            "prefix_len": num_state_tokens + num_ref_tokens + num_robot_tokens,
            "dummy_action_tokens": int(gt_action.shape[1]),
        }

    def build_inference_future_action_prefix_cache(
        self,
        ref_latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        state: torch.Tensor,
        gt_action: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        prefix_cache = self._build_inference_future_action_prefix_cache(
            ref_latents=ref_latents,
            encoder_hidden_states=encoder_hidden_states,
            state=state,
            gt_action=gt_action,
        )
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        return prefix_cache

    def _forward_inference_future_action_only_cached(
        self,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        pred_action: Optional[torch.Tensor] = None,
        gt_action: Optional[torch.Tensor] = None,
        future_state: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        prefix_cache: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if encoder_hidden_states_image is not None:
            raise ValueError("prefix cache does not support encoder_hidden_states_image/image-conditioned inputs.")
        del pred_action, kwargs
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        if prefix_cache is None:
            prefix_cache = self._build_inference_future_action_prefix_cache(
                ref_latents=ref_latents,
                encoder_hidden_states=encoder_hidden_states,
                state=state,
                gt_action=gt_action,
            )
        num_state_tokens = int(prefix_cache["num_state_tokens"])
        num_ref_tokens = int(prefix_cache["num_ref_tokens"])
        num_gt_action_tokens = int(prefix_cache["num_gt_action_tokens"])
        if gt_action is not None and int(gt_action.shape[1]) != num_gt_action_tokens:
            raise ValueError("cached gt_action token count mismatch")
        num_future_state_tokens = int(future_state.shape[1])
        num_value_tokens = int(value.shape[1])
        num_future_targets_tokens = num_future_state_tokens + num_value_tokens

        suffix_states, suffix_layout = self._encode_robot_targets(future_state=future_state, value=value)
        if (
            int(suffix_layout["num_future_state_tokens"]) != num_future_state_tokens
            or int(suffix_layout["num_value_tokens"]) != num_value_tokens
        ):
            raise ValueError("cached future/value token count mismatch")
        prefix_blocks = self._validate_prefix_cache(
            prefix_cache,
            expected_prefix_len=num_state_tokens + num_ref_tokens + num_gt_action_tokens,
            batch_size=int(suffix_states.shape[0]),
            device=suffix_states.device,
        )

        suffix_offset = (
            num_state_tokens
            + int(prefix_cache["dummy_action_tokens"])
            + num_gt_action_tokens
        )
        suffix_rope_states = suffix_states.new_empty(
            suffix_states.shape[0],
            suffix_offset + num_future_targets_tokens,
            suffix_states.shape[-1],
        )
        suffix_rotary_full = self.action_rope(suffix_rope_states)
        suffix_rotary_emb = (
            suffix_rotary_full[0][:, suffix_offset : suffix_offset + num_future_targets_tokens],
            suffix_rotary_full[1][:, suffix_offset : suffix_offset + num_future_targets_tokens],
        )

        if timestep.ndim != 2:
            raise ValueError("cached future action-only inference requires per-token timestep")
        p_end_original = num_state_tokens + num_ref_tokens
        dummy_tokens = int(prefix_cache["dummy_action_tokens"])
        gt_end_original = p_end_original + dummy_tokens + num_gt_action_tokens
        suffix_timestep = timestep[:, gt_end_original : gt_end_original + num_future_targets_tokens]
        if int(suffix_timestep.shape[1]) != num_future_targets_tokens:
            raise ValueError("cached future/value timestep token count mismatch")
        ts_seq_len = suffix_timestep.shape[1]
        temb, timestep_proj = self._compute_timestep_embeddings(
            suffix_timestep.flatten(),
            dtype=prefix_cache["encoder_hidden_states"].dtype,
            timestep_seq_len=ts_seq_len,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        hidden_states = suffix_states
        for block, block_cache in zip(self.blocks, prefix_blocks):
            hidden_states = block.forward_with_prefix_cache(
                hidden_states,
                prefix_cache["encoder_hidden_states"],
                timestep_proj,
                suffix_rotary_emb,
                block_cache,
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_robot_tokens=num_future_targets_tokens,
            )

        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
        shift, scale = shift.to(hidden_states.device), scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)

        future_state_pred = self.future_state_decoder(hidden_states[:, :num_future_state_tokens])
        value_pred = self.value_decoder(hidden_states[:, num_future_state_tokens:])

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return None, None, future_state_pred, value_pred

        return Transformer2DModelOutput(sample=None)

    def _forward_train(
        self,
        noisy_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        gt_action: Optional[torch.Tensor] = None,
        future_state: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training forward with the full teacher-forcing sequence
        [state | ref | pred_action | gt_action | future_state | value | future_image].
        `action` is the noisy pred_action; `gt_action` is the clean condition. All four
        heads (action_pred, future_state_pred, value_pred, video) are decoded."""
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = torch.cat([ref_latents, noisy_latents], dim=2)
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        extra_states, layout = self._encode_robot_targets(
            state=state,
            pred_action=action,
            gt_action=gt_action,
            future_state=future_state,
            value=value,
        )
        num_state_tokens = layout["num_state_tokens"]
        num_pred_action_tokens = layout["num_pred_action_tokens"]
        num_gt_action_tokens = layout["num_gt_action_tokens"]
        num_future_state_tokens = layout["num_future_state_tokens"]
        num_value_tokens = layout["num_value_tokens"]
        num_future_targets_tokens = layout["num_future_targets_tokens"]
        num_robot_tokens = layout["num_robot_tokens"]
        num_ref_tokens = post_patch_width * post_patch_height

        rotary_emb = self.rope(hidden_states)
        extra_rotary_emb = self.action_rope(extra_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        hidden_states, rotary_emb = self._reorder_with_robot_tokens(
            hidden_states=hidden_states,
            extra_states=extra_states,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            extra_rotary_emb=extra_rotary_emb,
            rotary_emb=rotary_emb,
            include_noisy_latents=True,
        )

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        self_attention_layout = (
            num_state_tokens,
            num_ref_tokens,
            num_pred_action_tokens,
            num_gt_action_tokens,
            num_future_targets_tokens,
        )
        if self._use_flex_attention():
            self_attention_mask = None
        else:
            self_attention_mask = _build_teacher_forcing_mask(
                seq_len=hidden_states.shape[1],
                num_state_tokens=num_state_tokens,
                num_ref_tokens=num_ref_tokens,
                num_pred_action_tokens=num_pred_action_tokens,
                num_gt_action_tokens=num_gt_action_tokens,
                num_future_targets_tokens=num_future_targets_tokens,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    self_attention_mask,
                    self_attention_layout,
                    self.self_attention_implementation,
                    self.flex_attention_block_size,
                    self.flex_attention_kernel_options,
                    num_state_tokens,
                    num_ref_tokens,
                    num_robot_tokens,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    self_attention_mask,
                    self_attention_layout=self_attention_layout,
                    self_attention_implementation=self.self_attention_implementation,
                    flex_block_size=self.flex_attention_block_size,
                    flex_kernel_options=self.flex_attention_kernel_options,
                    num_state_tokens=num_state_tokens,
                    num_ref_tokens=num_ref_tokens,
                    num_robot_tokens=num_robot_tokens,
                )

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)

        action_pred, future_state_pred, value_pred = self._split_robot_outputs(
            hidden_states=hidden_states,
            num_state_tokens=num_state_tokens,
            num_ref_tokens=num_ref_tokens,
            num_pred_action_tokens=num_pred_action_tokens,
            num_gt_action_tokens=num_gt_action_tokens,
            num_future_state_tokens=num_future_state_tokens,
            num_value_tokens=num_value_tokens,
        )

        p_end = num_state_tokens + num_ref_tokens
        video_start = p_end + num_robot_tokens
        video_hidden = torch.cat(
            [hidden_states[:, num_state_tokens:p_end], hidden_states[:, video_start:]],
            dim=1,
        )
        video_hidden = self.proj_out(video_hidden)

        video_hidden = video_hidden.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        video_hidden = video_hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = video_hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output, action_pred, future_state_pred, value_pred

        return Transformer2DModelOutput(sample=output)
