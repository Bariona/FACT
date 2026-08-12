import argparse
from io import BytesIO
import json
import os
import time
import types
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from PIL import Image

from world_action_model import apply_runtime_compat
from world_action_model.image_layouts import build_robotwin_ref_tensor, infer_robotwin_main_view_size, tensor_chw01_to_pil_rgb
from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
from world_action_model.pipeline.utils import (
    add_state_to_action,
    denormalize_action,
    denormalize_state,
    denormalize_value,
    extract_normalization_tensors,
    infer_robot_adapter_rank,
    load_stats,
    load_t5_embedding_from_pkl,
    pad_t5_embedding,
    normalize_state,
)
from world_action_model.pipeline.wa_pipeline import WAPipeline
from world_action_model.sockets import RobotInferenceServer
from diffusers.models import AutoencoderKLWan


def _parse_bool_list(value: str) -> list[bool]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if part in ("1", "true", "True"):
            out.append(True)
            continue
        if part in ("0", "false", "False"):
            out.append(False)
            continue
        raise ValueError(f"invalid mask element: {part!r} (expected 0/1/true/false)")
    return out


def _coerce_prompt_embeds(prompt_embeds, target_len: int) -> torch.Tensor:
    if isinstance(prompt_embeds, bytes):
        buffer = BytesIO(prompt_embeds)
        prompt_embeds = torch.load(buffer, map_location="cpu", weights_only=False)
    if not isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = torch.as_tensor(prompt_embeds)
    if prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
        prompt_embeds = prompt_embeds.squeeze(0)
    if prompt_embeds.ndim != 2:
        raise ValueError(f"prompt_embeds must be 2D [seq, dim], got {tuple(prompt_embeds.shape)}")
    return pad_t5_embedding(prompt_embeds.detach().cpu().to(torch.float32), target_len=int(target_len))


def _coerce_image_tensor(image) -> torch.Tensor:
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image)).convert("RGB")
        image = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).contiguous()
        return image
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    return image


class InstructionEmbeddingProvider:
    def __init__(self, model_id: str, text_len: int, target_len: int, device: str):
        self.model_id = model_id
        self.text_len = int(text_len)
        self.target_len = int(target_len)
        self.device = str(device)
        self._tokenizer = None
        self._text_encoder = None
        self._cache: dict[str, torch.Tensor] = {}

    def _lazy_init(self) -> None:
        if self._tokenizer is not None and self._text_encoder is not None:
            return
        tokenizer_path = os.path.join(self.model_id, "tokenizer")
        text_encoder_path = os.path.join(self.model_id, "text_encoder")
        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer dir not found under model_id: {tokenizer_path}")
        if not os.path.isdir(text_encoder_path):
            raise FileNotFoundError(f"Text encoder dir not found under model_id: {text_encoder_path}")
        encoder_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._text_encoder = UMT5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=encoder_dtype)
        self._text_encoder.eval().requires_grad_(False)
        self._text_encoder.to(device=self.device, dtype=encoder_dtype)
        print(f"Loaded text encoder from {text_encoder_path} on {self.device}")

    @torch.no_grad()
    def encode(self, instruction: str) -> torch.Tensor:
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("instruction is empty")
        cached = self._cache.get(instruction)
        if cached is not None:
            return cached

        self._lazy_init()
        text_inputs = self._tokenizer(
            [instruction],
            padding="max_length",
            max_length=self.text_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        last_hidden_state = self._text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        seq_len = int(attention_mask.gt(0).sum(dim=1)[0].item())
        prompt_embeds = pad_t5_embedding(last_hidden_state[0, :seq_len].detach().cpu().to(torch.float32), target_len=self.target_len)
        self._cache[instruction] = prompt_embeds
        return prompt_embeds


def _clamp_like(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(x, upper), lower)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _json_default(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _load_transformer_checkpoint(transformer_path: str, dtype: torch.dtype) -> CasualWorldActionTransformer:
    transformer_path_obj = Path(transformer_path).expanduser().resolve()
    checkpoint_config = CasualWorldActionTransformer.load_config(transformer_path_obj)

    weights_path = transformer_path_obj / "diffusion_pytorch_model.bin"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing transformer weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    patched_config = dict(checkpoint_config)
    inferred_rank = infer_robot_adapter_rank(state_dict)
    patched_config["robot_adapter_rank"] = max(int(patched_config.get("robot_adapter_rank", 0)), inferred_rank)

    transformer = CasualWorldActionTransformer.from_config(patched_config, torch_dtype=dtype)
    load_result = transformer.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys:
        print(f"Warning: missing transformer keys during server load ({len(missing_keys)}): {missing_keys[:20]}")
    if unexpected_keys:
        print(f"Warning: unexpected transformer keys during server load ({len(unexpected_keys)}): {unexpected_keys[:20]}")
    return transformer


def _set_scheduler_flow_shift(pipe, flow_shift: float) -> None:
    for name in ("scheduler", "action_scheduler", "future_state_scheduler", "value_scheduler"):
        scheduler = getattr(pipe, name, None)
        if scheduler is None or not hasattr(scheduler, "register_to_config"):
            continue
        updates = {}
        if "flow_shift" in scheduler.config:
            updates["flow_shift"] = float(flow_shift)
        if "shift" in scheduler.config:
            updates["shift"] = float(flow_shift)
        if updates:
            scheduler.register_to_config(**updates)


def get_policy(args: argparse.Namespace):
    device = torch.device(args.device)
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"unknown dtype: {args.dtype}")

    default_t5_embedding = None
    if args.t5_embedding_pkl:
        default_t5_embedding = load_t5_embedding_from_pkl(args.t5_embedding_pkl, target_len=int(args.t5_len)).to(
            device=device, dtype=torch.float32
        )
    stats = load_stats(args.stats_path)
    norm = extract_normalization_tensors(stats, device=device, state_dim=args.state_dim, action_dim=args.action_dim)
    delta_mask = torch.tensor(_parse_bool_list(args.delta_mask), device=device, dtype=torch.bool)
    if delta_mask.numel() != args.action_dim:
        raise ValueError(
            f"--delta_mask length ({delta_mask.numel()}) must match --action_dim ({args.action_dim})"
        )
    prompt_provider = InstructionEmbeddingProvider(
        model_id=args.model_id,
        text_len=args.text_len,
        target_len=args.t5_len,
        device=args.text_encoder_device,
    )

    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    transformer = _load_transformer_checkpoint(args.transformer_path, dtype=dtype).to(dtype).eval()
    pipe = WAPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        transformer=transformer,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe.text_encoder = None
    pipe.tokenizer = None
    _set_scheduler_flow_shift(pipe, float(args.scheduler_flow_shift))
    pipe.to(device)
    verbose_dir = Path(args.verbose_dir).expanduser().resolve()
    request_counter = 0

    def _sync_timing_device(timing_device) -> None:
        timing_device = torch.device(timing_device)
        if timing_device.type == "cuda":
            torch.cuda.synchronize(timing_device)

    @torch.no_grad()
    def inference(self, observation):
        nonlocal request_counter
        timing: dict[str, float] = {}
        total_t0 = time.perf_counter()
        request_counter += 1
        request_id = request_counter

        t0 = time.perf_counter()
        state = observation["observation.state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(device=device, dtype=torch.float32)
        timing["state_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        images = {
            "observation.images.cam_high": observation["observation.images.cam_high"],
            "observation.images.cam_left_wrist": observation["observation.images.cam_left_wrist"],
            "observation.images.cam_right_wrist": observation["observation.images.cam_right_wrist"],
        }
        for k, v in list(images.items()):
            images[k] = _coerce_image_tensor(v)
        timing["image_decode_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ref_image = build_robotwin_ref_tensor(
            images=images,
            main_dst_size=infer_robotwin_main_view_size((args.dst_width, args.dst_height)),
        )
        timing["image_layout_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        prompt_embeds = observation.get("prompt_embeds", None)
        if prompt_embeds is None:
            prompt_embeds = observation.get("prompt_embed", None)
        instruction = observation.get("instruction", None)
        if instruction is None:
            instruction = observation.get("prompt", None)
        if instruction is None:
            instruction = observation.get("lang", None)
        if prompt_embeds is not None:
            prompt_embeds = _coerce_prompt_embeds(prompt_embeds, target_len=int(args.t5_len))
        elif instruction is not None:
            _sync_timing_device(args.text_encoder_device)
            prompt_embeds = prompt_provider.encode(str(instruction))
            _sync_timing_device(args.text_encoder_device)
        elif default_t5_embedding is not None:
            prompt_embeds = default_t5_embedding.detach().cpu()
        else:
            raise ValueError(
                "No language input provided. Supply observation['instruction'] or observation['prompt_embeds'], "
                "or start the server with --t5_embedding_pkl."
            )
        timing["prompt_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        norm_state = normalize_state(state, norm, mode=args.norm_mode).to(device=device, dtype=dtype)
        timing["state_normalize_ms"] = (time.perf_counter() - t0) * 1000.0

        best_of_N = max(1, int(observation.get("best_of_N", 1)))
        skip_future_state_value = bool(args.skip_future_state_value)
        if skip_future_state_value and args.return_images:
            raise ValueError("--skip_future_state_value is only valid when --return_images is not set.")
        if skip_future_state_value and best_of_N > 1:
            raise ValueError("--skip_future_state_value is incompatible with best_of_N > 1 because BoN ranks by value.")

        def _run_pipe(gen, sample_count: int = 1):
            sample_count = int(sample_count)
            if sample_count < 1:
                raise ValueError(f"sample_count must be >= 1, got {sample_count}")
            if sample_count == 1:
                pipe_state = norm_state
                pipe_prompt_embeds = prompt_embeds.unsqueeze(0)
            else:
                pipe_state = norm_state.repeat(sample_count, 1)
                pipe_prompt_embeds = prompt_embeds.unsqueeze(0).repeat(sample_count, 1, 1)
            return pipe(
                height=args.dst_height,
                width=args.dst_width,
                action_chunk=args.action_chunk,
                action_dim=args.action_dim,
                state=pipe_state,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                image=ref_image,
                action_only=not args.return_images,
                return_dict=False,
                prompt_embeds=pipe_prompt_embeds.to(device=device, dtype=torch.float32),
                generator=gen,
                enable_prefix_cache=bool(args.enable_prefix_cache),
                skip_future_state_value=skip_future_state_value,
            )

        def _postprocess_action(action_raw, index: int = 0):
            a = denormalize_action(action_raw[index].float(), norm, mode=args.norm_mode)
            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            a = _clamp_like(a, norm.action_min, norm.action_max)
            ac = int(a.shape[0])
            a = add_state_to_action(a, state[0].float(), action_chunk=ac, mask=delta_mask)
            sf = state[0].float().unsqueeze(0).repeat(ac, 1)[..., : a.shape[-1]]
            a = torch.where(torch.isfinite(a), a, sf)
            return _clamp_like(a, norm.state_min, norm.state_max)

        def _postprocess(action_raw, future_state_raw, value_raw, index: int = 0):
            a = _postprocess_action(action_raw, index=index)
            fs = denormalize_state(future_state_raw[index].float(), norm, mode=args.norm_mode)
            fs = torch.nan_to_num(fs, nan=0.0, posinf=0.0, neginf=0.0)
            fs = _clamp_like(fs, norm.state_min, norm.state_max)
            v_denorm = denormalize_value(value_raw[index].float(), norm)
            # Keep raw scalar (may be NaN/inf) for ranking; return sanitized tensor as `value`.
            v_scalar_raw = v_denorm.view(-1)[0].detach().clone()
            v = torch.nan_to_num(v_denorm, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
            return a, fs, v, v_scalar_raw

        def _action_diversity(action_tensors):
            # Mean pairwise L2 distance between flattened action chunks.
            if len(action_tensors) < 2:
                return 0.0
            stacked = torch.stack(action_tensors).float()  # [N, T, D]
            flat = stacked.reshape(stacked.shape[0], -1)
            diffs = flat.unsqueeze(0) - flat.unsqueeze(1)
            dists = torch.linalg.norm(diffs, dim=-1)
            n = dists.shape[0]
            iu = torch.triu_indices(n, n, offset=1, device=dists.device)
            pair_dists = dists[iu[0], iu[1]]
            return float(pair_dists.mean().item())

        def _expand_keyframes_to_rollout(imgs: torch.Tensor, action_chunk: int, num_frames: int) -> torch.Tensor:
            """Hold each predicted keyframe over its segment to produce a rollout-length video.

            imgs: [B, C, num_frames, H, W] — frame 0 is the reference, frames 1..n-1 are predictions.
            returns: [B, C, action_chunk, H, W]
            """
            B, C, _T, H, W = imgs.shape
            offsets = [round(action_chunk * i / (num_frames - 1)) for i in range(num_frames)]
            out = torch.empty(B, C, action_chunk, H, W, dtype=imgs.dtype, device=imgs.device)
            for i in range(num_frames - 1):
                start, end = offsets[i], offsets[i + 1]
                out[:, :, start:end] = imgs[:, :, i + 1 : i + 2]
            return out

        if best_of_N == 1:
            _sync_timing_device(device)
            t0 = time.perf_counter()
            imgs, action_raw, future_state_raw, value_raw = _run_pipe(None)
            _sync_timing_device(device)
            timing["model_forward_ms"] = (time.perf_counter() - t0) * 1000.0
            _sync_timing_device(device)
            t0 = time.perf_counter()
            if skip_future_state_value:
                action = _postprocess_action(action_raw)
                future_state = None
                value = None
                v_scalar_raw = None
            else:
                action, future_state, value, v_scalar_raw = _postprocess(
                    action_raw, future_state_raw, value_raw
                )
            _sync_timing_device(device)
            timing["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0
            # Report as a length-1 vector so the client can log/plot uniformly.
            values_per_sample = None if v_scalar_raw is None else v_scalar_raw.unsqueeze(0)
            selected_index = 0
            action_diversity = 0.0
        else:
            base_seed = int(time.time_ns() & 0xFFFFFFFF)
            samples = []
            raw_scalars = []
            generators = [torch.Generator(device=device).manual_seed(base_seed + i) for i in range(best_of_N)]
            _sync_timing_device(device)
            t0 = time.perf_counter()
            imgs_batch, action_raw, future_state_raw, value_raw = _run_pipe(generators, sample_count=best_of_N)
            _sync_timing_device(device)
            timing["model_forward_ms"] = (time.perf_counter() - t0) * 1000.0
            _sync_timing_device(device)
            t0 = time.perf_counter()
            for i in range(best_of_N):
                a, fs, v, v_raw = _postprocess(action_raw, future_state_raw, value_raw, index=i)
                samples.append((a, fs, v))
                raw_scalars.append(v_raw)
            _sync_timing_device(device)
            timing["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0
            raw_stack = torch.stack(raw_scalars)
            # Rank with +inf for non-finite so blown-up samples can't win argmin.
            rank_vals = torch.where(
                torch.isfinite(raw_stack),
                raw_stack,
                torch.full_like(raw_stack, float("inf")),
            )
            selected_index = int(torch.argmin(rank_vals).item())
            action, future_state, value = samples[selected_index]
            imgs = None if imgs_batch is None else imgs_batch[selected_index : selected_index + 1]
            # Report raw values (NaN/inf preserved) so the client can see the blow-ups.
            values_per_sample = raw_stack
            n_bad = int((~torch.isfinite(raw_stack)).sum().item())
            action_diversity = _action_diversity([s[0] for s in samples])
            finite_vals = raw_stack[torch.isfinite(raw_stack)]
            v_spread = float((finite_vals.max() - finite_vals.min()).item()) if finite_vals.numel() > 1 else 0.0
            v_std = float(finite_vals.std().item()) if finite_vals.numel() > 1 else 0.0
            print(
                f"[BoN] selected={selected_index} "
                f"values={[float(x) for x in raw_stack.cpu().tolist()]} "
                f"v_spread={v_spread:.4f} v_std={v_std:.4f} "
                f"action_diversity={action_diversity:.4f} "
                f"non_finite={n_bad}"
            )
        if imgs is not None and args.num_frames > 1:
            t0 = time.perf_counter()
            imgs = _expand_keyframes_to_rollout(imgs, args.action_chunk, args.num_frames)
            timing["image_expand_ms"] = (time.perf_counter() - t0) * 1000.0
        _sync_timing_device(device)
        t0 = time.perf_counter()
        result = {"action": action.cpu()}
        if future_state is not None:
            result["future_state"] = future_state.cpu()
        if value is not None:
            result["value"] = value.cpu()
        _sync_timing_device(device)
        timing["result_cpu_ms"] = (time.perf_counter() - t0) * 1000.0
        timing["total_policy_ms"] = (time.perf_counter() - total_t0) * 1000.0
        result["_policy_timing_ms"] = timing
        if values_per_sample is not None:
            result["values_per_sample"] = values_per_sample.cpu()
            result["selected_index"] = int(selected_index)
            result["action_diversity"] = float(action_diversity)
        if imgs is not None:
            # Wan VAE decodes in bf16; numpy/pickle can't serialize bf16, so cast for the socket payload.
            result["images"] = imgs.detach().float().cpu()
        if args.verbose:
            try:
                dump_dir = verbose_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{request_id:06d}"
                dump_dir.mkdir(parents=True, exist_ok=True)
                tensor_chw01_to_pil_rgb(ref_image).save(dump_dir / "input_t_image.png")
                np.save(dump_dir / "input_state.npy", _to_numpy(state[0].float()))
                np.save(dump_dir / "output_action_abs.npy", _to_numpy(result["action"]))
                if "future_state" in result:
                    np.save(dump_dir / "output_future_state.npy", _to_numpy(result["future_state"]))
                if "value" in result:
                    np.save(dump_dir / "output_value.npy", _to_numpy(result["value"]))
                if "values_per_sample" in result:
                    np.save(dump_dir / "output_values_per_sample.npy", _to_numpy(result["values_per_sample"]))
                if "images" in result:
                    # imgs is [1, C, action_chunk, H, W] in [-1, 1] after _expand_keyframes_to_rollout.
                    # The expanded rollout is just 4 unique future keyframes held over 12-step segments
                    # (the reference frame is dropped by _expand_keyframes_to_rollout).
                    imgs_t = result["images"]
                    if imgs_t.ndim == 5:
                        imgs_t = imgs_t[0]
                    imgs_uint8 = ((imgs_t.float().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
                    imgs_uint8 = imgs_uint8.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
                    key_indices = [
                        round(args.action_chunk * i / (args.num_frames - 1))
                        for i in range(args.num_frames - 1)
                    ]
                    for k, idx in enumerate(key_indices):
                        Image.fromarray(imgs_uint8[idx]).save(dump_dir / f"pred_future_keyframe_{k+1}.png")
                    try:
                        import imageio.v3 as iio
                        iio.imwrite(
                            dump_dir / "pred_future.mp4",
                            imgs_uint8,
                            fps=max(1, int(args.action_chunk / 4)),
                            codec="libx264",
                        )
                    except Exception as exc:
                        print(f"Warning: mp4 write failed: {exc}")
                metadata = {
                    "request_id": int(request_id),
                    "instruction": instruction,
                    "selected_index": result.get("selected_index", None),
                    "action_diversity": result.get("action_diversity", None),
                    "delta_mask": _parse_bool_list(args.delta_mask),
                    "action_is_absolute": True,
                    "action_note": "output_action_abs.npy is after denormalization and add_state_to_action; this is what is returned to the client.",
                    "state_dim": int(args.state_dim),
                    "action_dim": int(args.action_dim),
                    "action_chunk": int(args.action_chunk),
                    "timing_ms": result.get("_policy_timing_ms", {}),
                }
                (dump_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default))
                print(f"[verbose] dumped inference input/output to {dump_dir}")
            except Exception as exc:
                print(f"Warning: failed to dump verbose inference artifacts: {exc}")
        return result

    pipe.inference = types.MethodType(inference, pipe)
    return pipe


def main():
    apply_runtime_compat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--transformer_path", type=str, required=True)
    parser.add_argument("--stats_path", type=str, required=True)
    parser.add_argument("--t5_embedding_pkl", type=str, default=None)
    parser.add_argument("--t5_len", type=int, default=64)
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument("--text_encoder_device", type=str, default="cpu")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--dst_width", type=int, default=384)
    parser.add_argument("--dst_height", type=int, default=192)
    parser.add_argument("--action_chunk", type=int, default=48)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--scheduler_flow_shift", type=float, default=3.0)
    parser.add_argument("--norm_mode", type=str, default="zscore", choices=["minmax", "zscore"])
    parser.add_argument("--return_images", action="store_true")
    parser.add_argument(
        "--enable_prefix_cache",
        action="store_true",
        help="Enable prefix KV cache for the supported action-only SDPA path.",
    )
    parser.add_argument(
        "--skip_future_state_value",
        action="store_true",
        help="Return only actions in action-only inference by skipping the future_state/value denoise stage.",
    )
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--verbose", action="store_true", help="Dump each inference input T image and output action/value on the server.")
    parser.add_argument("--verbose_dir", type=str, default="./tmp/inference_server_verbose", help="Directory used when --verbose is set.")
    parser.add_argument(
        "--delta_mask",
        type=str,
        default="1,1,1,1,1,1,0,1,1,1,1,1,1,0",
    )
    args = parser.parse_args()

    policy = get_policy(args)
    server = RobotInferenceServer(policy, host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
