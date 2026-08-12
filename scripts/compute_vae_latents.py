"""
Pre-compute VAE latents for RoboTwin dataset subtasks.

For each (episode, start_t) training sample, we encode the 5-frame composite
(3 views stacked into a 384x192 layout) with the frozen Wan VAE encoder and
store the resulting latents per-episode. At training time, the dataset can
read the cached latent and skip video decode + VAE encode entirely.

Storage schema:
    {task_dir}/vae_latents/
        _manifest.json                     — schema/dtype gate (see dataset side)
        episode_{ep_idx:06d}.pt            — torch.Tensor,
            shape=[N_samples, z=48, T_lat=2, H_lat=12, W_lat=24], dtype=bfloat16,
            where N_samples = ep_length (out-of-range offsets clip to the last
            frame, matching LeRobotDataset._get_query_indices).

Each latent is whitened by `(raw - vae.config.latents_mean) * (1 / vae.config.latents_std)`
to match `CasualWATrainer.forward_vae` exactly.

Launch on 8 GPUs:
    python -m scripts.compute_vae_latents --batch_size 32

  `--batch_size` should match training's `BATCH_SIZE_PER_GPU` for bit-exact
  parity with the live forward_vae (bf16 conv kernel selection varies with
  batch size; the cache is still numerically valid at any batch size).

Launch on free GPUs only (training using 0-3):
    python -m scripts.compute_vae_latents --devices cuda:4,cuda:5,cuda:6,cuda:7
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_action_model.image_layouts import (
    ROBOTWIN_VIEW_KEYS,
    build_robotwin_three_view_tensor,
    get_robotwin_side_view_size,
)
from world_action_model.robotwin_utils import select_robotwin_task_specs

VAE_FOLDER_NAME = "vae_latents"
MANIFEST_NAME = "_manifest.json"
# Bumped whenever the preprocessing semantics change in a way that invalidates
# existing cache files. Consumers (dataset) must compare this to the manifest's
# value and refuse to use the cache if they differ.
#   v1: un-normalized vae.encode().mode() (INVALID; broken by omission of whitening)
#   v2: added whitening (latents - mean) * (1/std); fp16 storage
#   v3: storage dtype switched to bf16 to match trainer's vae.dtype
#       (fp16→bf16 round-trip at load time loses ~3 bits of mantissa)
CACHE_SCHEMA_VERSION = 3
# Mirror the trainer's vae.dtype. bf16 also keeps storage at 2 bytes/element.
CACHE_TENSOR_DTYPE = torch.bfloat16


def _parse_csv(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        parts = [p.strip() for p in str(value).split(",")]
        out.extend(p for p in parts if p)
    return out


def _parse_devices(raw: str | None) -> list[str]:
    if raw is not None and raw.strip():
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        # Normalize bare indices ("0", "3") to torch's expected "cuda:N" form.
        # torch._C._nn._parse_to rejects naked integer strings.
        return [f"cuda:{p}" if p.isdigit() else p for p in parts]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def _load_episodes_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _decode_video_all_frames(
    video_path: Path,
    pyav_thread_count: int = 1,
    assert_uniform_pts: bool = True,
    pts_jitter_tolerance: float = 0.25,
) -> np.ndarray:
    """Decode all frames from an mp4 file via PyAV. Returns uint8 array [T, H, W, 3].

    When `assert_uniform_pts=True`, verifies that frame timestamps are spaced
    at (approximately) 1/fps. The live path in LeRobotDataset decodes by
    timestamp, so our position-indexed preprocessing only matches it when PTS
    are uniform — the tolerance catches the dataset's expected spacing within
    25% of the mean gap.
    """
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        if pyav_thread_count and pyav_thread_count > 1:
            stream.thread_count = int(pyav_thread_count)
        time_base = float(stream.time_base) if stream.time_base is not None else 0.0
        frames: list[np.ndarray] = []
        ptss: list[float] = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if assert_uniform_pts and frame.pts is not None and time_base > 0:
                ptss.append(float(frame.pts) * time_base)
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    if assert_uniform_pts and len(ptss) >= 3:
        gaps = np.diff(np.asarray(ptss, dtype=np.float64))
        mean_gap = float(gaps.mean())
        if mean_gap > 0:
            max_dev = float(np.abs(gaps - mean_gap).max()) / mean_gap
            if max_dev > pts_jitter_tolerance:
                raise RuntimeError(
                    f"Non-uniform PTS in {video_path}: max_dev={max_dev:.3f} > tolerance "
                    f"{pts_jitter_tolerance}. Position-indexed preprocessing will diverge "
                    f"from timestamp-indexed decode at train time; refusing to produce cache."
                )
    return np.stack(frames, axis=0)  # [T, H, W, 3] uint8


def _resolve_video_path(task_dir: Path, view_key: str, ep_idx: int) -> Path:
    """Find the mp4 for (task, view, episode). Handle chunk-* subdirs."""
    videos_root = task_dir / "videos"
    # Try common case first
    p = videos_root / "chunk-000" / view_key / f"episode_{ep_idx:06d}.mp4"
    if p.exists():
        return p
    # Fall back to glob
    matches = list(videos_root.glob(f"*/{view_key}/episode_{ep_idx:06d}.mp4"))
    if not matches:
        raise FileNotFoundError(f"Video not found under {videos_root} for view={view_key} ep={ep_idx}")
    return matches[0]


def _per_view_transform(frames_thwc_u8: np.ndarray, view_wh: Tuple[int, int]) -> torch.Tensor:
    """Replicate WATransformsLerobot._process_images:
        uint8[T,H,W,3] -> float32/255 -> bilinear resize to (dst_h, dst_w) -> normalize((x-0.5)/0.5).
    Returns [T, 3, dst_h, dst_w] float32 in [-1, 1]."""
    # [T, H, W, 3] -> [T, 3, H, W]
    frames = torch.from_numpy(frames_thwc_u8).permute(0, 3, 1, 2).contiguous()
    frames = frames.to(dtype=torch.float32) / 255.0
    dst_w, dst_h = int(view_wh[0]), int(view_wh[1])
    frames = F.interpolate(frames, size=(dst_h, dst_w), mode="bilinear", align_corners=False)
    frames = (frames - 0.5) / 0.5  # mirrors torchvision.transforms.Normalize([0.5],[0.5])
    return frames


def _build_composite(
    views_raw: dict[str, np.ndarray], main_wh: Tuple[int, int], view_keys: list[str]
) -> torch.Tensor:
    """Apply per-view transform, then the RoboTwin 3-view layout. Returns [T, 3, 192, 384] float32."""
    side_wh = get_robotwin_side_view_size(main_wh)
    views_processed: dict[str, torch.Tensor] = {}
    for idx, view_key in enumerate(view_keys):
        view_wh = main_wh if idx == 0 else side_wh
        views_processed[view_key] = _per_view_transform(views_raw[view_key], view_wh)
    if tuple(view_keys) != tuple(ROBOTWIN_VIEW_KEYS):
        views_processed = {
            robotwin_key: views_processed[source_key]
            for robotwin_key, source_key in zip(ROBOTWIN_VIEW_KEYS, view_keys, strict=True)
        }
    # build_robotwin_three_view_tensor does an additional same-size resize (no-op)
    # and stacks into 384x192 composite.
    return build_robotwin_three_view_tensor(views_processed, main_wh)


def _encode_samples(
    vae,
    composite: torch.Tensor,
    frame_offsets: list[int],
    valid_start_ts: list[int],
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    ep_length: int,
    latents_mean: torch.Tensor,
    latents_std_inv: torch.Tensor,
) -> torch.Tensor:
    """For each start_t in valid_start_ts, build the 5-frame clip [composite[t+o] for o in offsets],
    with frame indices clipped to [0, ep_length-1] (matching LeRobotDataset._get_query_indices).
    Batch encode through VAE and apply the trainer's per-channel whitening
    `(latents - latents_mean) * latents_std_inv` to match forward_vae exactly.
    Returns a CPU tensor [N, z, T_lat, H_lat, W_lat] in `CACHE_TENSOR_DTYPE`
    (bf16 — matches the trainer's vae.dtype to avoid a lossy round-trip)."""
    n_samples = len(valid_start_ts)
    n_frames = len(frame_offsets)
    out_parts: list[torch.Tensor] = []
    # Pre-move composite to GPU once to avoid repeated H2D
    composite_gpu = composite.to(device=device, dtype=dtype, non_blocking=True)
    H, W = composite_gpu.shape[-2], composite_gpu.shape[-1]
    max_idx = ep_length - 1

    # Precompute all [t + offset] indices once per episode (clamped to last
    # frame, matching LeRobotDataset._get_query_indices). Flat gather + reshape
    # in the loop avoids a per-sample Python list comprehension and per-sample
    # torch.tensor() allocation on GPU — both measurable at bs=32 and tens of
    # batches per episode.
    ts_t = torch.as_tensor(valid_start_ts, device=device, dtype=torch.long)  # [N]
    offsets_t = torch.as_tensor(frame_offsets, device=device, dtype=torch.long)  # [F]
    all_idx = (ts_t.unsqueeze(1) + offsets_t.unsqueeze(0)).clamp_(max=max_idx)  # [N, F]

    # inference_mode is strictly faster than no_grad (skips view tracking);
    # the produced tensors are .cpu()-copied before saving so the "tainted"
    # inference flag never leaks to the training path.
    with torch.inference_mode():
        for batch_start in range(0, n_samples, batch_size):
            batch_idx = all_idx[batch_start : batch_start + batch_size]  # [B, F]
            bsz = batch_idx.shape[0]
            gathered = composite_gpu.index_select(0, batch_idx.reshape(-1))  # [B*F, 3, H, W]
            # rearrange to [B, 3, T=F, H, W] expected by Wan VAE
            gathered = gathered.view(bsz, n_frames, 3, H, W).permute(0, 2, 1, 3, 4).contiguous()
            lat = vae.encode(gathered).latent_dist.mode()  # [B, z, T_lat, H_lat, W_lat]
            # Apply the same per-channel whitening that CasualWATrainer.forward_vae does,
            # so the cached tensor is drop-in replacement for forward_vae output.
            # latents_std_inv == 1.0 / vae.config.latents_std (pre-inverted).
            lat = (lat - latents_mean) * latents_std_inv
            # Store in the trainer's native precision (bf16) so the load-time
            # .to(vae.dtype) is a no-op, not a round-trip through fp16.
            out_parts.append(lat.to(dtype=CACHE_TENSOR_DTYPE).cpu())
    return torch.cat(out_parts, dim=0)


def _assert_frame_index_contiguous(task_dir: Path, ep_idx: int, ep_length: int) -> None:
    """Verify the episode's parquet has frame_index == 0..ep_length-1 contiguously.

    The dataset indexes our cache by `data_dict["frame_index"]`, which is the
    parquet value; the cache itself is position-indexed (0..ep_length-1). They
    agree only when the parquet's frame_index column is contiguous 0..N-1.
    """
    import pandas as _pd
    parquet_path = task_dir / "data" / f"chunk-{ep_idx // 1000:03d}" / f"episode_{ep_idx:06d}.parquet"
    if not parquet_path.is_file():
        # Try glob fallback for non-standard chunk layouts
        matches = list((task_dir / "data").glob(f"*/episode_{ep_idx:06d}.parquet"))
        if not matches:
            raise FileNotFoundError(f"Parquet not found for episode {ep_idx} under {task_dir}/data")
        parquet_path = matches[0]
    df = _pd.read_parquet(parquet_path, columns=["frame_index"])
    fidx = df["frame_index"].to_numpy()
    if fidx.size != ep_length:
        raise RuntimeError(
            f"Episode {ep_idx}: parquet has {fidx.size} rows but meta/episodes.jsonl says length={ep_length}"
        )
    expected = np.arange(ep_length, dtype=fidx.dtype)
    if not np.array_equal(np.sort(fidx), expected):
        raise RuntimeError(
            f"Episode {ep_idx}: frame_index column is not 0..{ep_length-1} contiguous. "
            "Cache indexing would silently map the wrong row; refusing to produce cache."
        )


def _decode_view_frames(
    task_dir: Path,
    view_key: str,
    ep_idx: int,
    pyav_thread_count: int,
    camera_by_view_key: dict[str, str] | None,
) -> np.ndarray:
    del camera_by_view_key
    vp = _resolve_video_path(task_dir, view_key, ep_idx)
    return _decode_video_all_frames(vp, pyav_thread_count=pyav_thread_count)


def _prepare_episode_cpu(
    task_dir: Path,
    cache_task_dir: Path,
    ep_idx: int,
    ep_length: int,
    dst_wh: Tuple[int, int],
    view_keys: list[str],
    camera_by_view_key: dict[str, str] | None,
    overwrite: bool,
    pyav_thread_count: int,
) -> Tuple[torch.Tensor | None, dict[str, Any]]:
    """CPU-only stage: resume check → frame_index assertion → parallel video
    decode → composite build.

    Returns (composite, meta). When meta has `skipped=True` the GPU stage
    should be bypassed; the caller publishes `meta` as-is. Otherwise composite
    is a CPU tensor [ep_length, 3, 192, 384] and meta is empty metadata.
    Any assertion failure (bad parquet, jittery PTS) raises — the worker's
    outer try/except reports it as an error.
    """
    out_path = cache_task_dir / VAE_FOLDER_NAME / f"episode_{ep_idx:06d}.pt"
    # Note: main() already filters out existing-and-valid caches. If we reached
    # the worker, the file either doesn't exist yet OR has a stale shape. Only
    # skip when overwrite is False AND the file is fully valid.
    if out_path.exists() and not overwrite:
        try:
            cached = torch.load(out_path, map_location="cpu", weights_only=True)
            if int(cached.shape[0]) >= ep_length:
                return None, {"skipped": True, "episode_idx": ep_idx, "reason": "exists"}
        except Exception:
            pass  # fall through and regenerate

    if ep_length <= 0:
        return None, {"skipped": True, "episode_idx": ep_idx, "reason": "too_short", "length": ep_length}

    # Verify the dataset-side assumption that frame_index is a contiguous 0..N-1
    # column. Without this the cache would be indexed by a value that doesn't
    # match its row positions at train time.
    if not (task_dir / "low_dim_obs.pkl").is_file():
        _assert_frame_index_contiguous(task_dir, ep_idx, ep_length)

    # Decode 3 views in parallel. PyAV releases the GIL during decode, so
    # Python threads deliver real wall-time speedup (not just concurrency
    # bookkeeping). Single-view decode is still throttled by pyav_thread_count.
    views_raw: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=len(view_keys)) as ex:
        futures = {
            view_key: ex.submit(_decode_view_frames, task_dir, view_key, ep_idx, pyav_thread_count, camera_by_view_key)
            for view_key in view_keys
        }
        for view_key in view_keys:
            frames = futures[view_key].result()
            if frames.shape[0] < ep_length:
                return None, {
                    "skipped": True,
                    "episode_idx": ep_idx,
                    "reason": f"video_short:{view_key}:{frames.shape[0]}<{ep_length}",
                }
            views_raw[view_key] = frames[:ep_length]

    composite = _build_composite(views_raw, dst_wh, view_keys)  # [T, 3, 192, 384]
    return composite, {}


def _encode_and_save_episode(
    task_dir: Path,
    cache_task_dir: Path,
    ep_idx: int,
    ep_length: int,
    composite: torch.Tensor,
    vae,
    device: str,
    dtype: torch.dtype,
    frame_offsets: list[int],
    batch_size: int,
    latents_mean: torch.Tensor,
    latents_std_inv: torch.Tensor,
) -> dict[str, Any]:
    """GPU stage: encode the CPU composite into whitened latents and atomically
    write the per-episode .pt file."""
    n_samples = ep_length
    valid_start_ts = list(range(n_samples))
    latents = _encode_samples(
        vae, composite, frame_offsets, valid_start_ts, batch_size, device, dtype,
        ep_length=ep_length,
        latents_mean=latents_mean,
        latents_std_inv=latents_std_inv,
    )
    if latents.shape[0] != n_samples:
        raise RuntimeError(f"latents shape {tuple(latents.shape)} n_samples {n_samples} mismatch")

    out_path = cache_task_dir / VAE_FOLDER_NAME / f"episode_{ep_idx:06d}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(latents, tmp_path)
    tmp_path.replace(out_path)

    return {
        "skipped": False,
        "episode_idx": ep_idx,
        "length": ep_length,
        "n_samples": n_samples,
        "latent_shape": list(latents.shape),
        "size_mb": out_path.stat().st_size / 1024 / 1024,
    }


def _worker_main(
    worker_id: int,
    device: str,
    task_queue: "mp.Queue",
    result_queue: "mp.Queue",
    vae_path: str,
    dtype_str: str,
    dst_wh: Tuple[int, int],
    frame_offsets: list[int],
    view_keys: list[str],
    camera_by_view_key: dict[str, str] | None,
    batch_size: int,
    overwrite: bool,
    pyav_thread_count: int,
    cudnn_benchmark: bool,
) -> None:
    # Lazy import inside worker to avoid CUDA init in parent
    from diffusers.models import AutoencoderKLWan

    from world_action_model import apply_runtime_compat

    apply_runtime_compat()

    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":", 1)[1]))
        # Only worth it when few distinct batch shapes occur: every new B triggers
        # a cuDNN kernel search (~30 conv layers x 100-500ms) per worker, which can
        # cost more than the steady-state speedup buys.
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]

    vae = AutoencoderKLWan.from_pretrained(vae_path)
    vae.requires_grad_(False)
    vae.to(device, dtype=dtype)
    vae.eval()

    # Per-channel whitening constants (shape: [1, z_dim, 1, 1, 1]). We pre-invert
    # latents_std (as the trainer does) and ship both to the encoder helper.
    z_dim = int(vae.config.z_dim)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device, dtype=dtype)
    latents_std_inv = (1.0 / torch.tensor(vae.config.latents_std)).view(1, z_dim, 1, 1, 1).to(device, dtype=dtype)

    # Single-slot prefetch: decodes episode N+1 on CPU while the GPU encodes N.
    # max_workers=1 caps peak memory at two episodes' composites; the 3-view
    # decode inside _prepare_episode_cpu has its own pool.
    cpu_pool = ThreadPoolExecutor(max_workers=1)

    def _kick_next():
        """Pull the next work item and submit its CPU stage. Returns
        (future, item, t0) or (None, None, None) on the end-of-work sentinel."""
        item = task_queue.get()
        if item is None:
            return None, None, None
        task_dir_str, cache_task_dir_str, ep_idx, ep_len, force_regen = item
        t0 = time.perf_counter()
        fut = cpu_pool.submit(
            _prepare_episode_cpu,
            Path(task_dir_str),
            Path(cache_task_dir_str),
            ep_idx,
            ep_len,
            dst_wh,
            view_keys,
            camera_by_view_key,
            overwrite or force_regen,
            pyav_thread_count,
        )
        return fut, item, t0

    pending_future, pending_item, pending_t0 = _kick_next()
    while pending_future is not None:
        fut, item, t0 = pending_future, pending_item, pending_t0
        # Prefetch the next episode BEFORE we start GPU work, so its CPU
        # stage overlaps with our current episode's encode/save.
        pending_future, pending_item, pending_t0 = _kick_next()

        task_dir_str, cache_task_dir_str, ep_idx, ep_len, _force_regen = item
        task_dir = Path(task_dir_str)
        cache_task_dir = Path(cache_task_dir_str)
        try:
            composite, meta = fut.result()
            if meta.get("skipped"):
                meta.update({
                    "worker_id": worker_id,
                    "device": device,
                    "task_dir": task_dir_str,
                    "seconds": time.perf_counter() - t0,
                    "ok": True,
                })
                result_queue.put(meta)
                continue
            res = _encode_and_save_episode(
                task_dir=task_dir,
                cache_task_dir=cache_task_dir,
                ep_idx=ep_idx,
                ep_length=ep_len,
                composite=composite,
                vae=vae,
                device=device,
                dtype=dtype,
                frame_offsets=frame_offsets,
                batch_size=batch_size,
                latents_mean=latents_mean,
                latents_std_inv=latents_std_inv,
            )
            res.update({
                "worker_id": worker_id,
                "device": device,
                "task_dir": task_dir_str,
                "seconds": time.perf_counter() - t0,
                "ok": True,
            })
            result_queue.put(res)
        except Exception as exc:
            result_queue.put({
                "worker_id": worker_id,
                "device": device,
                "task_dir": task_dir_str,
                "episode_idx": ep_idx,
                "ok": False,
                "error": f"{exc!r}\n{traceback.format_exc()}",
            })

    cpu_pool.shutdown(wait=True)
    result_queue.put({"worker_id": worker_id, "device": device, "done": True})


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Wan VAE latents for RoboTwin data.")
    parser.add_argument("--config_module", default="world_action_model.configs.robotwin", help="Training config module.")
    parser.add_argument("--robotwin_root", default=None, help="Override ROBOTWIN_ROOT.")
    parser.add_argument("--vae_path", default=None, help="Override Wan VAE dir (default <WAN_MODEL_DIR>/vae).")
    parser.add_argument("--dataset_id", action="append", default=[], help="Override dataset_ids (repeatable or comma-separated).")
    parser.add_argument("--dataset_glob", action="append", default=[], help="Override dataset_globs.")
    parser.add_argument("--devices", default=None, help="Comma-separated CUDA devices, e.g. cuda:4,cuda:5. Default: all CUDA GPUs.")
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="VAE encode batch size per worker. The default matches "
             "BATCH_SIZE_PER_GPU in world_action_model/configs/robotwin.py, "
             "which gives bit-exact parity with the live forward_vae. Another "
             "value only adds ~1e-2 bf16 kernel roundoff; the cache stays valid.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--pyav_thread_count", type=int, default=2)
    parser.add_argument("--max_episodes_per_task", type=int, default=0, help="0 = all; set >0 for quick tests.")
    parser.add_argument(
        "--cudnn_benchmark", action="store_true",
        help="Enable torch.backends.cudnn.benchmark. Worth it only if the "
             "dataset has a narrow set of unique ep_length tail-batch sizes; "
             "otherwise the per-shape kernel search (~100-500ms/conv × ~30 "
             "layers) dominates. Default: off.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # The config module resolves the dataset at import time, so CLI overrides
    # must reach the environment before it is imported.
    if args.robotwin_root:
        os.environ["ROBOTWIN_ROOT"] = args.robotwin_root
    if args.dataset_id:
        os.environ["ROBOTWIN_DATASET_IDS"] = ",".join(_parse_csv(args.dataset_id))
    if args.dataset_glob:
        os.environ["ROBOTWIN_DATASET_GLOBS"] = ",".join(_parse_csv(args.dataset_glob))

    cfg = importlib.import_module(args.config_module)
    robotwin_root = args.robotwin_root or getattr(cfg, "ROBOTWIN_ROOT", None)
    vae_path = args.vae_path or str(Path(cfg.WAN_MODEL_DIR) / "vae")
    dst_wh = tuple(int(v) for v in cfg.DST_SIZE)  # (256, 192)
    num_frames = int(cfg.NUM_FRAMES)
    frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]

    dataset_ids = _parse_csv(args.dataset_id) or list(getattr(cfg, "ROBOTWIN_DATASET_IDS", []))
    dataset_globs = _parse_csv(args.dataset_glob) or list(getattr(cfg, "ROBOTWIN_DATASET_GLOBS", []))
    specs = select_robotwin_task_specs(
        robotwin_root=robotwin_root,
        dataset_ids=dataset_ids,
        dataset_globs=dataset_globs,
    )
    view_keys = list(ROBOTWIN_VIEW_KEYS)
    camera_by_view_key = None
    spec_rows = [(spec.task_dir, int(ep["episode_index"]), int(ep["length"])) for spec in specs for ep in _load_episodes_jsonl(spec.task_dir / "meta" / "episodes.jsonl")]

    devices = _parse_devices(args.devices)
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    def _build_manifest() -> dict[str, Any]:
        # Consumers check `normalized` and `schema_version`; the rest is for
        # config-drift diagnostics.
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "num_frames": num_frames,
            "frame_offsets": list(frame_offsets),
            "dst_wh": list(dst_wh),
            "view_keys": list(view_keys),
            "backend": "robotwin",
            "vae_path": str(vae_path),
            "normalized": True,
            "dtype": str(CACHE_TENSOR_DTYPE).replace("torch.", ""),
            "notes": (
                "Latents are whitened by (latents - vae.config.latents_mean) * "
                "(1 / vae.config.latents_std), matching CasualWATrainer.forward_vae. "
                "Stored in bf16 to match trainer's vae.dtype; the load-time cast is a no-op."
            ),
        }

    def _read_manifest(task_dir: Path) -> dict[str, Any] | None:
        p = task_dir / VAE_FOLDER_NAME / MANIFEST_NAME
        if not p.is_file():
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    # (source_task_dir, cache_task_dir, ep_idx, ep_length, force_regen); force_regen
    # regenerates even when a correctly-shaped .pt exists, as its schema may be stale.
    work_items: list[Tuple[str, str, int, int, bool]] = []
    skipped_exists = 0
    total_eps = 0
    total_samples = 0
    tasks_needing_full_regen: list[Path] = []
    rows_by_task: dict[Path, list[tuple[Path, int, int]]] = {}
    for task_dir, ep_idx, ep_len in spec_rows:
        task_dir_path = Path(task_dir)
        rows_by_task.setdefault(task_dir_path, []).append((task_dir_path, int(ep_idx), int(ep_len)))

    for cache_task_dir, ep_rows in rows_by_task.items():
        existing_manifest = _read_manifest(cache_task_dir)
        # A mismatched or missing manifest means every episode in this task
        # must be regenerated: the schema bump captures things like missing
        # normalization that a shape check wouldn't notice.
        task_is_stale = (
            existing_manifest is None
            or int(existing_manifest.get("schema_version", 0)) != CACHE_SCHEMA_VERSION
        )
        if task_is_stale and (cache_task_dir / VAE_FOLDER_NAME).is_dir():
            tasks_needing_full_regen.append(cache_task_dir)

        if args.max_episodes_per_task and args.max_episodes_per_task > 0:
            ep_rows = ep_rows[: args.max_episodes_per_task]
        for source_task_dir, ep_idx, ep_len in ep_rows:
            total_eps += 1
            total_samples += max(0, ep_len)
            out_path = cache_task_dir / VAE_FOLDER_NAME / f"episode_{ep_idx:06d}.pt"
            if task_is_stale:
                # Schema mismatch: ignore any existing per-episode file, regen.
                # force_regen=True so the worker's skip-check doesn't rescue it
                # by shape alone when the on-disk bytes are stale (e.g. fp16
                # tensor under a v3 manifest that promised bf16).
                work_items.append((str(source_task_dir), str(cache_task_dir), ep_idx, ep_len, True))
                continue
            if out_path.exists() and not args.overwrite:
                # Verify the cached file covers the full ep_length; older caches
                # only had ep_length - 48 samples and must be regenerated.
                try:
                    cached = torch.load(out_path, map_location="cpu", weights_only=True)
                    if int(cached.shape[0]) >= ep_len:
                        skipped_exists += 1
                        continue
                except Exception:
                    pass
                # Fall through: regenerate this episode's cache (it's stale or broken).
            work_items.append((str(source_task_dir), str(cache_task_dir), ep_idx, ep_len, args.overwrite))

    print("Backend: robotwin")
    print(f"Tasks selected: {len(rows_by_task)}")
    print(f"Total episodes: {total_eps}  |  total samples: {total_samples}")
    print(f"Already cached: {skipped_exists}  |  to process: {len(work_items)}")
    print(f"Frame offsets: {frame_offsets}  (NUM_FRAMES={num_frames})")
    print(f"dst_wh (main view): {dst_wh}  |  composite canvas: {dst_wh[0] + dst_wh[0]//2}x{dst_wh[1]}")
    print(f"Devices: {devices}  |  batch_size/worker: {args.batch_size}  |  dtype: {args.dtype}")
    print(f"VAE: {vae_path}")
    print(f"Cache schema version: {CACHE_SCHEMA_VERSION}")
    if tasks_needing_full_regen:
        print(f"Schema mismatch / missing manifest → full regen for {len(tasks_needing_full_regen)} task(s).")

    if args.dry_run:
        print("Dry run; exiting.")
        return
    if not work_items:
        print("Nothing to do.")
        return

    # Write the manifest per task *before* workers start. If the run is
    # interrupted, a valid-but-partial cache still presents a manifest with the
    # correct schema version, and the dataset will detect missing episode files
    # at load time (it raises FileNotFoundError on a cache miss).
    manifest = _build_manifest()
    for cache_task_dir in rows_by_task:
        (cache_task_dir / VAE_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
        tmp_path = cache_task_dir / VAE_FOLDER_NAME / (MANIFEST_NAME + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        tmp_path.replace(cache_task_dir / VAE_FOLDER_NAME / MANIFEST_NAME)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    worker_count = min(len(devices), len(work_items))
    workers: list[mp.Process] = []

    for wid, device in enumerate(devices[:worker_count]):
        p = ctx.Process(
            target=_worker_main,
            args=(
                wid,
                device,
                task_queue,
                result_queue,
                vae_path,
                args.dtype,
                dst_wh,
                frame_offsets,
                view_keys,
                camera_by_view_key,
                args.batch_size,
                args.overwrite,
                args.pyav_thread_count,
                args.cudnn_benchmark,
            ),
        )
        p.start()
        workers.append(p)

    for item in work_items:
        task_queue.put(item)
    for _ in workers:
        task_queue.put(None)

    t_start = time.perf_counter()
    completed = 0
    done_workers = 0
    errors: list[dict[str, Any]] = []
    encoded_samples = 0
    total_size_mb = 0.0

    while done_workers < len(workers):
        try:
            res = result_queue.get(timeout=60.0)
        except queue.Empty:
            alive = sum(p.is_alive() for p in workers)
            print(f"[wait] completed={completed}/{len(work_items)} alive_workers={alive}")
            continue

        if res.get("done"):
            done_workers += 1
            continue

        if not res.get("ok", False):
            errors.append(res)
            print(f"[ERR] wid={res.get('worker_id')} task={res.get('task_dir')} ep={res.get('episode_idx')} -> {res.get('error')}")
            continue

        completed += 1
        if not res.get("skipped"):
            encoded_samples += int(res.get("n_samples", 0))
            total_size_mb += float(res.get("size_mb", 0.0))

        if completed % 20 == 0 or completed == len(work_items):
            elapsed = time.perf_counter() - t_start
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (len(work_items) - completed) / rate if rate > 0 else 0.0
            print(
                f"[{completed}/{len(work_items)}] ep={res.get('episode_idx')} dev={res.get('device')} "
                f"t={res.get('seconds', 0):.2f}s samples_total={encoded_samples} "
                f"size={total_size_mb:.0f}MB rate={rate:.2f}ep/s eta={eta:.0f}s"
            )

    for p in workers:
        p.join()

    elapsed = time.perf_counter() - t_start
    print(
        f"\nDone in {elapsed:.0f}s. completed={completed}/{len(work_items)} "
        f"encoded_samples={encoded_samples} total_size={total_size_mb:.1f}MB errors={len(errors)}"
    )
    if errors:
        for e in errors[:5]:
            print("  - " + str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
