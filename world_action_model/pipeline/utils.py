import json
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch


NormMode = Literal["minmax", "zscore"]


def pad_t5_embedding(t5_embedding: torch.Tensor, target_len: int) -> torch.Tensor:
    if t5_embedding.ndim != 2:
        raise ValueError(f"t5_embedding must be 2D [seq, dim], got {t5_embedding.shape}")
    if t5_embedding.shape[0] >= target_len:
        return t5_embedding[:target_len]
    return torch.nn.functional.pad(t5_embedding, (0, 0, 0, target_len - t5_embedding.shape[0]), value=0)


def load_t5_embedding_from_pkl(pkl_path: str, target_len: int = 32) -> torch.Tensor:
    with open(pkl_path, "rb") as f:
        t5_embedding = torch.load(f, weights_only=True)
    if not isinstance(t5_embedding, torch.Tensor):
        t5_embedding = torch.as_tensor(t5_embedding)
    return pad_t5_embedding(t5_embedding, target_len=target_len)


def load_stats(stats_dict_path: str) -> Dict:
    with open(stats_dict_path, "r") as f:
        return json.load(f)


@dataclass(frozen=True)
class NormalizationTensors:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    state_min: torch.Tensor
    state_max: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    action_min: torch.Tensor
    action_max: torch.Tensor
    value_min: torch.Tensor
    value_max: torch.Tensor


def extract_normalization_tensors(stats: Dict, device: torch.device, state_dim: int, action_dim: int) -> NormalizationTensors:
    state_mean = torch.tensor(stats["norm_stats"]["observation.state"]["mean"])[..., :state_dim].to(device=device)
    state_std = torch.tensor(stats["norm_stats"]["observation.state"]["std"])[..., :state_dim].to(device=device)
    state_min = torch.tensor(stats["norm_stats"]["observation.state"]["min"])[..., :state_dim].to(device=device)
    state_max = torch.tensor(stats["norm_stats"]["observation.state"]["max"])[..., :state_dim].to(device=device)

    action_mean = torch.tensor(stats["norm_stats"]["action"]["mean"][:action_dim])[..., :action_dim].to(device=device)
    action_std = torch.tensor(stats["norm_stats"]["action"]["std"][:action_dim])[..., :action_dim].to(device=device)
    action_min = torch.tensor(stats["norm_stats"]["action"]["min"][:action_dim])[..., :action_dim].to(device=device)
    action_max = torch.tensor(stats["norm_stats"]["action"]["max"][:action_dim])[..., :action_dim].to(device=device)
    value_stats = stats["norm_stats"]["value"]
    value_min = torch.tensor(value_stats["min"])[..., :1].to(device=device)
    value_max = torch.tensor(value_stats["max"])[..., :1].to(device=device)

    return NormalizationTensors(
        state_mean=state_mean,
        state_std=state_std,
        state_min=state_min,
        state_max=state_max,
        action_mean=action_mean,
        action_std=action_std,
        action_min=action_min,
        action_max=action_max,
        value_min=value_min,
        value_max=value_max,
    )


def normalize_state(state: torch.Tensor, stats: NormalizationTensors, mode: NormMode) -> torch.Tensor:
    eps = 1e-8
    state = state[..., : stats.state_mean.shape[-1]]
    if mode == "minmax":
        state_range = stats.state_max - stats.state_min + eps
        norm = ((state - stats.state_min) / state_range) * 2 - 1
        return norm.clamp(-1, 1)
    if mode == "zscore":
        return (state - stats.state_mean) / stats.state_std.clamp_min(eps)
    raise ValueError(f"unknown mode: {mode}")


def denormalize_action(action: torch.Tensor, stats: NormalizationTensors, mode: NormMode) -> torch.Tensor:
    eps = 1e-8
    action = action[..., : stats.action_mean.shape[-1]]
    if mode == "minmax":
        action_range = stats.action_max - stats.action_min + eps
        return ((action + 1) / 2) * action_range + stats.action_min
    if mode == "zscore":
        return action * stats.action_std.clamp_min(eps) + stats.action_mean
    raise ValueError(f"unknown mode: {mode}")


def denormalize_state(state: torch.Tensor, stats: NormalizationTensors, mode: NormMode) -> torch.Tensor:
    eps = 1e-8
    state = state[..., : stats.state_mean.shape[-1]]
    if mode == "minmax":
        state_range = stats.state_max - stats.state_min + eps
        return ((state + 1) / 2) * state_range + stats.state_min
    if mode == "zscore":
        return state * stats.state_std.clamp_min(eps) + stats.state_mean
    raise ValueError(f"unknown mode: {mode}")


def denormalize_value(value: torch.Tensor, stats: NormalizationTensors) -> torch.Tensor:
    eps = 1e-8
    value = value[..., :1]
    value_range = stats.value_max - stats.value_min + eps
    return ((value + 1) / 2) * value_range + stats.value_min


ROBOT_TOKEN_PARAM_PREFIXES: Tuple[str, ...] = (
    "state_encoder.",
    "action_encoder.",
    "action_decoder.",
    "value_encoder.",
    "future_state_decoder.",
    "value_decoder.",
)
ROBOT_TOKEN_PARAM_SUBSTRINGS: Tuple[str, ...] = (".robot_ffn_adapter.",)


def is_robot_token_param(name: str, module_prefix: str = "") -> bool:
    for pat in ROBOT_TOKEN_PARAM_PREFIXES:
        if name.startswith(f"{module_prefix}{pat}"):
            return True
    return any(pat in name for pat in ROBOT_TOKEN_PARAM_SUBSTRINGS)


def infer_robot_adapter_rank(state_dict: Dict[str, torch.Tensor]) -> int:
    for name, tensor in state_dict.items():
        if name.endswith("robot_ffn_adapter.down_proj.weight") and getattr(tensor, "ndim", 0) == 2:
            return int(tensor.shape[0])
    return 0


def build_teacher_forcing_per_token_timestep(
    *,
    batch_size: int,
    num_state_tokens: int,
    num_ref_tokens: int,
    num_pred_action_tokens: int,
    num_gt_action_tokens: int,
    num_future_state_tokens: int,
    num_value_tokens: int,
    num_noisy_latent_tokens: int,
    noise_t: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-token timestep for the teacher-forcing layout
    [state | ref | pred_action | gt_action | future_state | value | future_image].
    Clean segments (state, ref, gt_action) stay at 0; noisy segments
    (pred_action, future_state, value, future_image) get noise_t.
    Any segment count may be 0 for stage-specific inference sequences.
    """
    total_len = (
        num_state_tokens
        + num_ref_tokens
        + num_pred_action_tokens
        + num_gt_action_tokens
        + num_future_state_tokens
        + num_value_tokens
        + num_noisy_latent_tokens
    )
    timestep = torch.zeros(batch_size, total_len, device=device, dtype=dtype)
    pred_start = num_state_tokens + num_ref_tokens
    pred_end = pred_start + num_pred_action_tokens
    gt_end = pred_end + num_gt_action_tokens
    fut_start = gt_end
    timestep[:, pred_start:pred_end] = noise_t
    timestep[:, fut_start:] = noise_t
    return timestep


def add_state_to_action(action: torch.Tensor, state: torch.Tensor, action_chunk: int, mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=action.device)
    if mask.numel() != action.shape[-1]:
        raise ValueError(f"mask length ({mask.numel()}) must match action_dim ({action.shape[-1]})")
    state = state[..., : action.shape[-1]]
    state_rep = state.repeat(action_chunk, 1)
    return action + state_rep * mask
