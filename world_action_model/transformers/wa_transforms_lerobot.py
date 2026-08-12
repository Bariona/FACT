import copy
import io
import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as torch_F
from PIL import Image as _PILImage
from torchvision.transforms import functional as _tv_F

from fact_train import TRANSFORMS

from world_action_model.image_layouts import build_robotwin_three_view_tensor, get_robotwin_side_view_size

from .wa_transforms import WATransforms


_DELTA_MASK_TEMPLATES: Dict[int, np.ndarray] = {
    0: np.array(
        [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
        dtype=bool,
    ),
    1: np.array(
        [True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False],
        dtype=bool,
    ),
}


@TRANSFORMS.register
class WATransformsLerobot(WATransforms):
    def __init__(
        self,
        is_train=False,
        dst_size=None,
        num_frames=1,
        fps=16,
        norm_path=None,
        robotype_to_embed_id=None,
        robotype_default_embed_id=0,
        model_action_dim=None,
        image_cfg=None,
        num_views=1,
        view_keys=None,
        state_key="observation.state",
        action_key="action",
        task_key="task",
        t5_len=64,
        value_penalty_scale=1.0,
        delta_mask_templates=None,
        image_aug_cfg=None,
    ):
        if norm_path is None:
            raise ValueError("norm_path is None")
        if isinstance(norm_path, (str, os.PathLike)):
            norm_paths = [str(norm_path)]
        else:
            norm_paths = [str(p) for p in norm_path]
            if len(norm_paths) == 0:
                raise ValueError("norm_path list is empty")
 
        super().__init__(
            is_train=is_train,
            dst_size=dst_size,
            num_frames=num_frames,
            fps=fps,
            norm_path=norm_paths[0],
            image_cfg=image_cfg,
            num_views=num_views,
        )
 
        self.robotype_default_embed_id = int(robotype_default_embed_id)
        self.robotype_to_embed_id = dict(robotype_to_embed_id or {})
        self.model_action_dim = None if model_action_dim is None else int(model_action_dim)
        self.delta_mask_templates = {
            int(k): np.asarray(v, dtype=bool)
            for k, v in dict(_DELTA_MASK_TEMPLATES).items()
        }
        if delta_mask_templates is not None:
            for k, v in dict(delta_mask_templates).items():
                self.delta_mask_templates[int(k)] = np.asarray(v, dtype=bool)
 
        self.norm_paths = norm_paths
        self.stats_dicts = []
        for json_path in self.norm_paths:
            with open(json_path, "r", encoding="utf-8") as f:
                self.stats_dicts.append(json.load(f))
            print("Loading stats dict from:", json_path)
 
        if view_keys is None:
            view_keys = [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ]
        self.view_keys = list(view_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.task_key = task_key
        self.t5_len = int(t5_len)
        self.value_penalty_scale = float(value_penalty_scale)
        self.image_aug_cfg = dict(image_aug_cfg) if image_aug_cfg else None
        self._warned_unknown_robotype = False
        assert self.model_action_dim is not None, "model_action_dim must be provided"
        self._stats_arrays_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._delta_template_cache: Dict[int, Dict[str, np.ndarray]] = {}
        for embed_id in self.delta_mask_templates:
            self._delta_template_cache[embed_id] = self._build_delta_template_arrays(embed_id)

    def _build_delta_template_arrays(self, embed_id: int) -> Dict[str, np.ndarray]:
        raw = self.delta_mask_templates.get(embed_id, None)
        assert raw is not None, f"robotype_embed_id {embed_id} not found in delta_mask_templates"
        d = int(self.model_action_dim)
        raw_len = int(raw.shape[0])
        if d > raw_len:
            base = np.pad(raw, (0, d - raw_len), constant_values=False)
        else:
            base = raw[:d].copy()
        effective_dim = min(raw_len, d)
        action_dim_mask = np.arange(d) < effective_dim
        return {
            "mask": base,
            "idx": np.nonzero(base)[0].astype(np.int64),
            "action_dim_mask": action_dim_mask,
        }

    def _build_stats_arrays(self, stats_dict: dict) -> Dict[str, np.ndarray]:
        d = int(self.model_action_dim)

        def _pad(x, pad_value: float) -> np.ndarray:
            arr = np.asarray(x, dtype=np.float32).flatten()
            if arr.size >= d:
                return arr[:d].copy()
            out = np.full((d,), float(pad_value), dtype=np.float32)
            if arr.size > 0:
                out[: arr.size] = arr
            return out

        ns = stats_dict["norm_stats"]
        value_stats = ns.get("value", None)
        if value_stats is None:
            raise KeyError("Missing norm_stats['value'] in stats file")
        return {
            "state_mean": _pad(ns["observation.state"]["mean"], 0.0),
            "state_std": _pad(ns["observation.state"]["std"], 1.0),
            "delta_mean": _pad(ns["action"]["mean"], 0.0),
            "delta_std": _pad(ns["action"]["std"], 1.0),
            "value_min": np.asarray(value_stats["min"], dtype=np.float32).flatten()[:1].copy(),
            "value_max": np.asarray(value_stats["max"], dtype=np.float32).flatten()[:1].copy(),
        }

    def _get_stats_arrays(self, embed_id: int) -> Dict[str, np.ndarray]:
        if embed_id not in self._stats_arrays_cache:
            self._stats_arrays_cache[embed_id] = self._build_stats_arrays(self._get_stats_dict(embed_id))
        return self._stats_arrays_cache[embed_id]

    def _get_delta_template(self, embed_id: int) -> Dict[str, np.ndarray]:
        if embed_id not in self._delta_template_cache:
            self._delta_template_cache[embed_id] = self._build_delta_template_arrays(embed_id)
        return self._delta_template_cache[embed_id]

    def _parse_robotype(self, robotype):
        if robotype is None:
            return None
        if isinstance(robotype, bytes):
            robotype = robotype.decode("utf-8", errors="ignore")
        if hasattr(robotype, "item"):
            try:
                robotype = robotype.item()
            except Exception as e:
                print(f"Error parsing robotype={robotype!r}: {e}")
        if isinstance(robotype, str):
            robotype = robotype.strip()
        return robotype
 
    def _get_robotype_embed_id(self, data_dict) -> int:
        robotype = self._parse_robotype(data_dict.get("robotype", None))
        if robotype in self.robotype_to_embed_id:
            return int(self.robotype_to_embed_id[robotype])
        if isinstance(robotype, str):
            robotype_l = robotype.lower()
            if "agibot" in robotype_l and "agibot" in self.robotype_to_embed_id:
                return int(self.robotype_to_embed_id["agibot"])
            if "aloha" in robotype_l and "aloha" in self.robotype_to_embed_id:
                return int(self.robotype_to_embed_id["aloha"])
            if "agilex" in robotype_l and "agilex" in self.robotype_to_embed_id:
                return int(self.robotype_to_embed_id["agilex"])
        if not self._warned_unknown_robotype:
            print(f"Unknown robotype={robotype!r}, fallback to {self.robotype_default_embed_id}")
            self._warned_unknown_robotype = True
        return self.robotype_default_embed_id
 
    def _get_stats_dict(self, embed_id: int):
        if not self.stats_dicts:
            return self.stats_dict
        if 0 <= embed_id < len(self.stats_dicts):
            return self.stats_dicts[embed_id]
        if not self._warned_unknown_robotype:
            print(f"robotype_embed_id={embed_id} out of range for norm_paths (len={len(self.stats_dicts)}), fallback to 0")
            self._warned_unknown_robotype = True
        return self.stats_dicts[0]
 
    def _to_nchw_uint8(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[0] in (1, 3):
                x = x[None, ...]
            elif x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)[None, ...]
            else:
                x = x[None, ...]
        if x.dim() != 4:
            raise ValueError(f"Unexpected image tensor shape: {tuple(x.shape)}")
        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dtype != torch.uint8:
            x_f = x.to(dtype=torch.float32)
            x_max = float(x_f.max().item()) if x_f.numel() > 0 else 0.0
            if x_max <= 1.0:
                x_f = x_f * 255.0
            x = x_f.clamp(0.0, 255.0).to(dtype=torch.uint8)
        return x
 
    def _sample_frames(self, frames: torch.Tensor, num: int) -> torch.Tensor:
        t = int(frames.shape[0])
        if t <= 0:
            raise ValueError("Empty video frames")
        if t == 1:
            return frames.repeat(num, 1, 1, 1)
        idx = np.linspace(0, t - 1, num=num, dtype=int)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=frames.device)
        return torch.index_select(frames, 0, idx_t)
 
    def _process_images(
        self,
        input_images: torch.Tensor,
        dst_width: int,
        dst_height: int,
        aug_params: dict | None = None,
    ) -> torch.Tensor:
        input_images = input_images.to(dtype=torch.float32) / 255.0
        input_images = torch.nn.functional.interpolate(
            input_images,
            size=(int(dst_height), int(dst_width)),
            mode="bilinear",
            align_corners=False,
        )
        if aug_params is not None:
            input_images = self._apply_image_aug_float(input_images, aug_params)
        input_images = self.normalize(input_images)
        return input_images

    def _process_view_tensor(
        self,
        input_images: torch.Tensor,
        dst_size: tuple[int, int],
        aug_params: dict | None = None,
    ) -> torch.Tensor:
        dst_width, dst_height = dst_size
        return self._process_images(
            input_images,
            dst_width=dst_width,
            dst_height=dst_height,
            aug_params=aug_params,
        )

    def _sample_image_aug_params(self) -> dict | None:
        """Sample augmentation params shared across all views and frames in this clip.

        Returns None when augmentation is disabled or this is not a training pass.
        Sampling once per `__call__` enforces temporal consistency: the model never
        sees a brightness or JPEG-quality jump between ref-frame and future frames,
        which matches the live-camera behavior at deploy time.
        """
        cfg = self.image_aug_cfg
        if cfg is None or not self.is_train:
            return None
        apply_prob = float(cfg.get("apply_prob", 1.0))
        if apply_prob < 1.0 and float(np.random.random()) >= apply_prob:
            return None
        params: dict = {}

        def _u(key: str, default: float) -> float:
            return float(cfg.get(key, default))

        b = _u("brightness", 0.0)
        if b > 0.0:
            params["brightness"] = float(np.random.uniform(max(1.0 - b, 0.0), 1.0 + b))
        c = _u("contrast", 0.0)
        if c > 0.0:
            params["contrast"] = float(np.random.uniform(max(1.0 - c, 0.0), 1.0 + c))
        s = _u("saturation", 0.0)
        if s > 0.0:
            params["saturation"] = float(np.random.uniform(max(1.0 - s, 0.0), 1.0 + s))
        h = _u("hue", 0.0)
        if h > 0.0:
            params["hue"] = float(np.random.uniform(-h, h))
        gs = _u("gauss_std", 0.0)
        if gs > 0.0:
            # Per-clip scaled std; per-frame independent realization sampled at apply time.
            scale = float(np.random.uniform(0.5, 1.0))
            params["gauss_std"] = gs * scale
        jq = cfg.get("jpeg_quality", None)
        if jq is not None:
            if isinstance(jq, (list, tuple)) and len(jq) == 2:
                lo, hi = int(jq[0]), int(jq[1])
                params["jpeg_quality"] = int(np.random.randint(min(lo, hi), max(lo, hi) + 1))
            else:
                params["jpeg_quality"] = int(jq)
        if not params:
            return None
        return params

    def _apply_image_aug_float(self, view: torch.Tensor, params: dict) -> torch.Tensor:
        """view: float [T, 3, h, w] in [0, 1]. Returns same shape, augmented.

        Color params are deterministic per clip (already sampled in `params`).
        Gaussian noise has fixed std but a fresh per-pixel realization here, which
        mirrors real per-frame sensor shot noise.
        """
        if not params:
            return view
        v = view
        if "brightness" in params:
            v = _tv_F.adjust_brightness(v, params["brightness"])
        if "contrast" in params:
            v = _tv_F.adjust_contrast(v, params["contrast"])
        if "saturation" in params:
            v = _tv_F.adjust_saturation(v, params["saturation"])
        if "hue" in params:
            v = _tv_F.adjust_hue(v, params["hue"])
        if "gauss_std" in params:
            std = float(params["gauss_std"])
            if std > 0.0:
                v = v + torch.randn_like(v) * std
        v = v.clamp(0.0, 1.0)
        if "jpeg_quality" in params:
            v = self._apply_jpeg_per_frame(v, int(params["jpeg_quality"]))
        return v

    @staticmethod
    def _apply_jpeg_per_frame(view: torch.Tensor, quality: int) -> torch.Tensor:
        """JPEG-roundtrip every frame at fixed quality. view: [T, 3, h, w] float in [0,1]."""
        out_frames = []
        for t in range(int(view.shape[0])):
            arr = (view[t].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            buf = io.BytesIO()
            _PILImage.fromarray(arr).save(buf, format="JPEG", quality=int(quality), optimize=False)
            buf.seek(0)
            decoded = np.asarray(_PILImage.open(buf).convert("RGB"), dtype=np.float32) / 255.0
            out_frames.append(torch.from_numpy(decoded).permute(2, 0, 1))
        out = torch.stack(out_frames, dim=0)
        if view.device.type != "cpu":
            out = out.to(device=view.device)
        return out

    def _build_robotwin_layout(self, views: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.dst_size is None:
            raise ValueError("dst_size is required")
        return build_robotwin_three_view_tensor(views, main_dst_size=tuple(int(v) for v in self.dst_size))
 
    def __call__(self, data_dict):
        if self.dst_size is None:
            raise ValueError("dst_size is required")
        dst_width, dst_height = self.dst_size

        if "robotype" not in data_dict:
            raise KeyError("Missing robotype key")
        robotype_embed_id = self._get_robotype_embed_id(data_dict)
        stats_arrays = self._get_stats_arrays(robotype_embed_id)
        delta_template = self._get_delta_template(robotype_embed_id)

        # If the dataset provided a cached visual_latents tensor (pre-computed VAE
        # encode), we skip video loading + resize + normalize + layout compose.
        # The trainer consumes visual_latents directly and skips forward_vae.
        cached_visual_latents = data_dict.get("visual_latents", None)
        if cached_visual_latents is None:
            # Sample augmentation params ONCE per clip so all views/frames use the same
            # color/JPEG settings (Gaussian noise still gets a fresh realization per frame).
            aug_params = self._sample_image_aug_params()
            views_clean: dict = {}
            views_aug: dict = {}
            for idx, k in enumerate(self.view_keys[: self.num_views]):
                if k not in data_dict:
                    raise KeyError(f"Missing view key: {k}")
                v = data_dict[k]
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                if not isinstance(v, torch.Tensor):
                    raise TypeError(f"Unsupported image type for {k}: {type(v)}")
                v = self._to_nchw_uint8(v)
                if self.num_views == 1:
                    view_dst_size = (dst_width, dst_height)
                elif self.num_views == 3:
                    view_dst_size = (dst_width, dst_height) if idx == 0 else get_robotwin_side_view_size((dst_width, dst_height))
                else:
                    raise ValueError(f"Unsupported num_views={self.num_views}; expected 1 or 3")
                # Clean version: used as the diffusion target (input_images). Always built.
                views_clean[k] = self._process_view_tensor(v, dst_size=view_dst_size, aug_params=None)
                # Augmented version: used only for the ref slot of input_ref_images, so the
                # model sees noisy input but is asked to predict clean future frames. Skip
                # the extra work entirely when augmentation is off.
                if aug_params is not None:
                    views_aug[k] = self._process_view_tensor(v, dst_size=view_dst_size, aug_params=aug_params)
                else:
                    views_aug[k] = views_clean[k]

            if self.num_views == 1:
                input_images = views_clean[self.view_keys[0]]
                input_images_aug = views_aug[self.view_keys[0]]
            else:
                input_images = self._build_robotwin_layout({k: views_clean[k] for k in self.view_keys[: self.num_views]})
                input_images_aug = self._build_robotwin_layout({k: views_aug[k] for k in self.view_keys[: self.num_views]})

            data_dict["input_images"] = input_images  # CLEAN — diffusion targets

            if self.image_cfg is not None:
                ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict["input_images"].shape[0])
                ref_masks = ref_masks[:, None, None, None]
                ref_latent_masks = ref_latent_masks[None, :, None, None]
                # ref input takes the AUGMENTED frames (only the ref slot survives the mask).
                ref_images = input_images_aug.clone() * ref_masks
                data_dict["input_ref_images"] = ref_images
                data_dict["input_ref_masks"] = ref_latent_masks
 
        action = data_dict[self.action_key]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        state = data_dict[self.state_key]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        future_state = data_dict.get("future_state", None)
        if future_state is not None and isinstance(future_state, np.ndarray):
            future_state = torch.from_numpy(future_state)

        action = action.to(dtype=torch.float32)
        state = state.to(dtype=torch.float32)
        if future_state is not None:
            future_state = future_state.to(dtype=torch.float32)

        if action.dim() == 1:
            action = action[None, :]
        if state.dim() == 1:
            state = state[None, :]
        if state.dim() == 2 and state.shape[0] > 1:
            state = state[:1]
        if future_state is None:
            raise KeyError("Missing future_state key")
        if future_state.dim() == 1:
            future_state = future_state[None, :]
        if future_state.dim() == 2 and future_state.shape[0] > 1:
            future_state = future_state[:1]

        if action.shape[0] != self.num_frames:
            t = int(self.num_frames)
            cur_t = int(action.shape[0])
            if cur_t >= t:
                action = action[:t]
            else:
                pad = torch.zeros((t - cur_t, action.shape[1]), dtype=action.dtype, device=action.device)
                action = torch.cat([action, pad], dim=0)

        d = int(self.model_action_dim)
        if state.shape[-1] > d:
            state = state[..., :d]
        if state.shape[-1] < d:
            state = torch_F.pad(state, (0, d - int(state.shape[-1])), value=0.0)
        if future_state.shape[-1] > d:
            future_state = future_state[..., :d]
        if future_state.shape[-1] < d:
            future_state = torch_F.pad(future_state, (0, d - int(future_state.shape[-1])), value=0.0)
        if action.shape[-1] > d:
            action = action[..., :d]
        if action.shape[-1] < d:
            action = torch_F.pad(action, (0, d - int(action.shape[-1])), value=0.0)

        idx = torch.from_numpy(delta_template["idx"]).to(action.device).clone()
        delta = action.clone()
        if idx.numel() > 0:
            delta[:, idx] = action[:, idx] - state[:, idx]

        state_mean = torch.from_numpy(stats_arrays["state_mean"]).to(state.device).clone()
        state_std = torch.from_numpy(stats_arrays["state_std"]).to(state.device).clone()
        delta_mean = torch.from_numpy(stats_arrays["delta_mean"]).to(action.device).clone()
        delta_std = torch.from_numpy(stats_arrays["delta_std"]).to(action.device).clone()

        eps = 1e-8
        norm_state = (state - state_mean) / state_std.clamp_min(eps)
        norm_future_state = (future_state - state_mean) / state_std.clamp_min(eps)
        norm_delta = (delta - delta_mean) / delta_std.clamp_min(eps)
        value_min = torch.from_numpy(stats_arrays["value_min"]).to(state.device).clone()
        value_max = torch.from_numpy(stats_arrays["value_max"]).to(state.device).clone()
        frame_index = int(data_dict.get("frame_index", data_dict.get("index", 0)))
        episode_length = int(data_dict.get("episode_length", 0))
        future_state_index = int(data_dict.get("future_state_index", min(frame_index + self.num_frames, max(0, episode_length - 1))))
        failure_episode = bool(data_dict.get("failure_episode", False))
        failure_active = bool(data_dict.get("failure_active", False))
        if episode_length <= 1:
            value = torch.zeros((1,), dtype=torch.float32, device=state.device)
        else:
            value = torch.tensor(
                [(float(episode_length - future_state_index - 1) / float(episode_length - 1))],
                dtype=torch.float32,
                device=state.device,
            )
        value_range = (value_max - value_min).clamp_min(eps)
        if failure_active:
            value = value + float(self.value_penalty_scale)
        value_normalized = ((value - value_min) / value_range) * 2.0 - 1.0
        value_normalized = value_normalized.clamp(-1.0, 1.0)

        # Gates the action loss on real failure episodes.
        action_loss_mask = torch.tensor(0.0 if failure_episode else 1.0, dtype=torch.float32, device=norm_delta.device)

        prompt = data_dict.get("t5_embedding", None)
        assert prompt is not None, "t5_embedding must be provided"
        if isinstance(prompt, np.ndarray):
            prompt = torch.from_numpy(prompt)
        prompt = prompt.to(dtype=torch.float32)
        t5_len = int(self.t5_len)
        prompt = prompt[:t5_len]
        prompt_embeds = torch_F.pad(prompt, (0, 0, 0, t5_len - prompt.shape[0]), value=0)
 
        out = {}
        out["fps"] = torch.tensor(self.fps, dtype=torch.float32)
        if cached_visual_latents is not None:
            # Pre-computed VAE latents path: no images/ref_images needed.
            # Trainer reads visual_latents and skips forward_vae().
            out["visual_latents"] = cached_visual_latents
        else:
            out["images"] = data_dict["input_images"]
            out["ref_images"] = data_dict.get("input_ref_images", None)
            out["ref_masks"] = data_dict.get("input_ref_masks", None)
        out["prompt_embeds"] = prompt_embeds
        out["action"] = norm_delta
        out["state"] = norm_state
        out["future_state"] = norm_future_state
        out["value_normalized"] = value_normalized
        out["frame_index"] = torch.tensor(frame_index, dtype=torch.long)
        out["episode_length"] = torch.tensor(episode_length, dtype=torch.long)
        out["future_state_index"] = torch.tensor(future_state_index, dtype=torch.long)
        out["robotype_embed_id"] = torch.tensor(int(robotype_embed_id), dtype=torch.long)
        out["failure_episode_mask"] = torch.tensor(float(failure_episode), dtype=torch.float32)
        out["failure_active_mask"] = torch.tensor(float(failure_active), dtype=torch.float32)

        keys = list(out.keys())
        for k in keys:
            if out[k] is None:
                out.pop(k)

        out["action_dim_mask"] = torch.from_numpy(delta_template["action_dim_mask"]).to(out["action"].device).clone()
        out["action_loss_mask"] = action_loss_mask

        return out
