"""
Standalone dataloader smoke-test for FACT (World-Action Model).
Builds the dataloader from any FACT training config and iterates N batches,
printing per-key shapes / value ranges / timing — and the per-sample token
count for the Transformer (no GPU or model weights required).

Run (basic):
    python -m scripts.test_dataloader

Run with per-component profiling:
    python -m scripts.test_dataloader --profile --profile_samples 20

Run and print every batch:
    python -m scripts.test_dataloader --num_batches 40 --batch_size 10 --verbose
"""
from __future__ import annotations

import argparse
import copy
import importlib
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import BatchSampler, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

# Wan2.2 architecture constants used for token estimation
_VAE_SPATIAL = 8
_VAE_TEMPORAL = 4
_PATCH_H = 2
_PATCH_W = 2


# ── Dataloader builder ────────────────────────────────────────────────────────

def _build_dataloader(
    data_config: dict,
    batch_size: int,
    num_workers: int,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> DataLoader:
    from fact_datasets import DefaultCollator, load_dataset
    from fact_train import build_sampler, build_transform
    import world_action_model.transformers  # noqa: F401 — triggers @TRANSFORMS.register decorators

    dataset = load_dataset(data_config["data_or_config"])
    filter_cfg = data_config.get("filter", None)
    if filter_cfg is not None:
        dataset.filter(**filter_cfg)
    transform = build_transform(data_config["transform"])
    dataset.set_transform(transform)

    if "batch_sampler" in data_config:
        batch_sampler_cfg = data_config["batch_sampler"]
        batch_sampler = build_sampler(
            batch_sampler_cfg,
            dataset=dataset,
            batch_size_per_gpu=batch_size,
            batch_size=batch_size,
        )
    else:
        sampler_cfg = data_config.get("sampler", {"type": "DefaultSampler"})
        sampler = build_sampler(sampler_cfg, dataset=dataset, batch_size=batch_size)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    collator_cfg = data_config.get("collator", {})
    collator = DefaultCollator(**collator_cfg)

    if pin_memory is None:
        pin_memory = bool(data_config.get("pin_memory", False))
    if persistent_workers is None:
        persistent_workers = bool(data_config.get("persistent_workers", False))
    if prefetch_factor is None:
        prefetch_factor = data_config.get("prefetch_factor", None)

    dataloader_kwargs = dict(
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) and num_workers > 0,
    )
    if prefetch_factor is not None and num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(dataset, **dataloader_kwargs)


def _override_pyav_thread_count(data_config: dict, pyav_thread_count: int) -> None:
    data_or_config = data_config.get("data_or_config", [])
    if isinstance(data_or_config, dict):
        items = [data_or_config]
    else:
        items = list(data_or_config)
    for item in items:
        if not isinstance(item, dict):
            continue
        video_kwargs = dict(item.get("video_backend_kwargs", {}))
        video_kwargs["pyav_thread_count"] = int(pyav_thread_count)
        item["video_backend_kwargs"] = video_kwargs


def _get_pyav_thread_counts(data_config: dict) -> list[int]:
    data_or_config = data_config.get("data_or_config", [])
    if isinstance(data_or_config, dict):
        items = [data_or_config]
    else:
        items = list(data_or_config)
    counts = []
    for item in items:
        if isinstance(item, dict):
            video_kwargs = item.get("video_backend_kwargs", {})
            if "pyav_thread_count" in video_kwargs:
                counts.append(int(video_kwargs["pyav_thread_count"]))
    return sorted(set(counts))


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _print_batch_stats(batch: dict, batch_idx: int) -> None:
    print(f"\n--- Batch {batch_idx} ---")
    for key in sorted(batch.keys()):
        val = batch[key]
        if val is None:
            print(f"  {key}: None")
        elif isinstance(val, torch.Tensor):
            vf = val.float()
            print(
                f"  {key:45s}  shape={str(tuple(val.shape)):30s}"
                f"  min={vf.min().item():+.4f}  max={vf.max().item():+.4f}"
            )
        else:
            print(f"  {key:45s}  type={type(val).__name__}")


def _print_token_count(batch: dict) -> None:
    required = {"action", "state", "prompt_embeds"}
    if not required.issubset(batch.keys()):
        missing = required - batch.keys()
        print(f"\n[token count] skipping — missing keys: {missing}")
        return
    if "images" not in batch and "visual_latents" not in batch:
        print("\n[token count] skipping — missing either 'images' or 'visual_latents'")
        return

    action = batch["action"]         # [B, T_action, D]
    state  = batch["state"]          # [B, T_state, D]
    prompt = batch["prompt_embeds"]  # [B, T_text, D]
    if "visual_latents" in batch:
        visual_latents = batch["visual_latents"]  # [B, z, T_lat, H_lat, W_lat]
        T_latent = int(visual_latents.shape[2])
        H_lat = int(visual_latents.shape[-2])
        W_lat = int(visual_latents.shape[-1])
        T_img = "cached"
        tokens_per_frame = (H_lat // _PATCH_H) * (W_lat // _PATCH_W)
    else:
        images = batch["images"]  # [B, T_img, C, H, W]
        _, T_img, _, H, W = images.shape
        T_latent = (T_img - 1) // _VAE_TEMPORAL + 1
        H_lat = H // _VAE_SPATIAL
        W_lat = W // _VAE_SPATIAL
        tokens_per_frame = (H_lat // _PATCH_H) * (W_lat // _PATCH_W)

    n_state  = state.shape[1]
    n_ref    = tokens_per_frame
    n_action = action.shape[1]
    n_future_state = int(batch["future_state"].shape[1]) if "future_state" in batch else 0
    n_value = int(batch["value_normalized"].shape[1]) if "value_normalized" in batch else 0
    n_noisy  = (T_latent - 1) * tokens_per_frame
    n_self   = n_state + n_ref + n_action + n_future_state + n_value + n_noisy
    n_text   = prompt.shape[1]

    print("\n=== Token count (per sample) ===")
    print(f"  FACT constants: VAE {_VAE_SPATIAL}x spatial / {_VAE_TEMPORAL}x temporal, "
          f"patch (1,{_PATCH_H},{_PATCH_W})")
    print(f"  Image frames  T_img={T_img}  →  T_latent={T_latent}")
    print(f"  Latent spatial: H_lat={H_lat}  W_lat={W_lat}  "
          f"(={H_lat // _PATCH_H} × {W_lat // _PATCH_W} patches)")
    print(f"  Tokens per latent frame: {tokens_per_frame}")
    print()
    print(f"  {'state':22s}: {n_state:6d}  (robot state)")
    print(f"  {'ref visual':22s}: {n_ref:6d}  (1 clean ref frame × {tokens_per_frame})")
    print(f"  {'action':22s}: {n_action:6d}  (action chunk)")
    if n_future_state:
        print(f"  {'future_state':22s}: {n_future_state:6d}  (future-state token)")
    if n_value:
        print(f"  {'value':22s}: {n_value:6d}  (value token)")
    print(f"  {'noisy visual':22s}: {n_noisy:6d}  ({T_latent - 1} noisy frame(s) × {tokens_per_frame})")
    print(f"  {'─' * 38}")
    print(f"  {'TOTAL self-attn':22s}: {n_self:6d}")
    print(f"  {'cross-attn text (T5)':22s}: {n_text:6d}  (separate cross-attention, not in self-attn)")
    if "frame_index" in batch and "episode_length" in batch and "future_state_index" in batch:
        frame_index = batch["frame_index"].view(-1)
        episode_length = batch["episode_length"].view(-1)
        future_state_index = batch["future_state_index"].view(-1)
        expected = torch.minimum(frame_index + n_action, episode_length - 1)
        valid = torch.all((frame_index >= 0) & (frame_index < episode_length))
        aligned = torch.all(future_state_index == expected)
        print(f"  metadata valid: frame/episode={bool(valid)}  future_state_index={bool(aligned)}")


# ── Per-component profiler ────────────────────────────────────────────────────

def _time_image_processing(transform, raw_sample: dict) -> float:
    """Return seconds spent doing image resize/normalize and layout composition."""
    import numpy as np
    if "visual_latents" in raw_sample:
        return 0.0
    dst_width, dst_height = transform.dst_size
    t0 = time.perf_counter()
    views = {}
    for idx, k in enumerate(transform.view_keys[:transform.num_views]):
        v = raw_sample[k]
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        v = transform._to_nchw_uint8(v)
        if transform.num_views == 1:
            view_dst_size = (dst_width, dst_height)
        else:
            view_dst_size = (dst_width, dst_height) if idx == 0 else (dst_width // 2, dst_height // 2)
        views[k] = transform._process_view_tensor(v, dst_size=view_dst_size)
    if transform.num_views == 1:
        _ = views[transform.view_keys[0]]
    else:
        _ = transform._build_robotwin_layout({k: views[k] for k in transform.view_keys[:transform.num_views]})
    return time.perf_counter() - t0


def _profile(data_config: dict, n_samples: int) -> None:
    """Measure time spent in each pipeline stage for `n_samples` samples."""
    from fact_datasets import DefaultCollator, load_dataset
    from fact_train import build_transform
    import world_action_model.transformers  # noqa: F401

    print(f"\n=== Per-component profiling  (n_samples={n_samples}, single-process) ===\n")

    # ── Stage 1: raw dataset fetch (cache load or video decode + metadata) ───
    dataset_raw = load_dataset(data_config["data_or_config"])
    _ = dataset_raw[0]   # warm up (open file handles etc.)

    fetch_times = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        _ = dataset_raw[i]
        fetch_times.append(time.perf_counter() - t0)

    # ── Stage 2: transform (total) ────────────────────────────────────────────
    transform = build_transform(data_config["transform"])
    raw_samples = [dataset_raw[i] for i in range(n_samples)]   # pre-fetched

    transform_times = []
    for raw in raw_samples:
        t0 = time.perf_counter()
        _ = transform(dict(raw))      # copy so we don't mutate cached sample
        transform_times.append(time.perf_counter() - t0)

    # ── Stage 2a: image sub-stage only ───────────────────────────────────────
    cached_latent_samples = sum(1 for raw in raw_samples if "visual_latents" in raw)
    img_times = [_time_image_processing(transform, dict(raw)) for raw in raw_samples]

    # ── Stage 3: collation ────────────────────────────────────────────────────
    collator_cfg = data_config.get("collator", {})
    collator = DefaultCollator(**collator_cfg)
    transformed = [transform(dict(raw)) for raw in raw_samples]

    collate_times = []
    for item in transformed:
        t0 = time.perf_counter()
        _ = collator([item])
        collate_times.append(time.perf_counter() - t0)

    # ── Report ────────────────────────────────────────────────────────────────
    def _ms(ts):
        return (statistics.mean(ts) * 1000,
                statistics.median(ts) * 1000,
                max(ts) * 1000)

    f_mean, f_med, f_max = _ms(fetch_times)
    t_mean, t_med, t_max = _ms(transform_times)
    c_mean, c_med, c_max = _ms(collate_times)
    i_mean = statistics.mean(img_times) * 1000
    r_mean = max(t_mean - i_mean, 0.0)   # rest = total transform - image proc
    total = f_mean + t_mean + c_mean

    def _row(label, mean, med, mx, pct=None, indent=""):
        pct_str = f"  {pct:5.1f}%" if pct is not None else ""
        print(f"  {indent}{label:38s}  {mean:7.1f}ms  {med:7.1f}ms  {mx:7.1f}ms{pct_str}")

    print(f"  {'Stage':38s}  {'mean':>7s}    {'median':>7s}    {'max':>7s}    {'%total':>7s}")
    print(f"  {'─' * 72}")
    _row("1. raw fetch  (cache/video)",    f_mean, f_med, f_max, f_mean / total * 100)
    _row("2. transform  (total)",          t_mean, t_med, t_max, t_mean / total * 100)
    _row("   2a. image resize/norm/layout", i_mean, i_mean, i_mean,
         i_mean / total * 100, indent="")
    _row("   2b. action/state/T5/mask",    r_mean, r_mean, r_mean,
         r_mean / total * 100, indent="")
    _row("3. collate    (batch_size=1)",   c_mean, c_med, c_max, c_mean / total * 100)
    print(f"  {'─' * 72}")
    print(f"  {'TOTAL (sequential, 1 worker)':38s}  {total:7.1f}ms")
    print()
    print(f"  Theoretical max throughput (1 worker) : {1000/total:6.1f} samples/s")
    if cached_latent_samples:
        print(f"  Cached VAE samples                  : {cached_latent_samples}/{len(raw_samples)}")
    top = "raw fetch (cache/video)" if f_mean >= t_mean else "transform"
    print(f"  Bottleneck                            : {top}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def _measure_throughput(
    dataloader: DataLoader,
    batch_size: int,
    num_batches: int,
    warmup_batches: int = 0,
) -> tuple[int, int, float]:
    it = iter(dataloader)
    for _ in range(max(0, warmup_batches)):
        next(it)

    t0 = time.perf_counter()
    n_iters = 0
    for _ in range(num_batches):
        next(it)
        n_iters += 1
    elapsed = time.perf_counter() - t0
    total_samples = n_iters * batch_size
    return n_iters, total_samples, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone dataloader test for FACT (World-Action Model).")
    parser.add_argument(
        "--config", type=str, default="world_action_model.configs.robotwin",
        help="Python module path exporting a `config` dict.",
    )
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Number of batches to iterate.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size. Defaults to config['dataloaders']['train']['batch_size_per_gpu'].")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (0 = main process, safest for debugging).")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true",
                        help="Enable DataLoader pin_memory.")
    parser.add_argument("--no-pin_memory", dest="pin_memory", action="store_false",
                        help="Disable DataLoader pin_memory.")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true",
                        help="Enable DataLoader persistent_workers.")
    parser.add_argument("--no-persistent_workers", dest="persistent_workers", action="store_false",
                        help="Disable DataLoader persistent_workers.")
    parser.add_argument("--prefetch_factor", type=int, default=None,
                        help="Override DataLoader prefetch_factor (only used when num_workers > 0).")
    parser.add_argument("--pyav_thread_count", type=int, default=None,
                        help="Override PyAV decoder thread count in LeRobotDataset video_backend_kwargs.")
    parser.add_argument("--quiet", action="store_true",
                        help="Skip per-batch tensor printing and only report throughput.")
    parser.add_argument("--warmup_batches", type=int, default=1,
                        help="Number of warmup batches before timing.")
    parser.add_argument("--profile", action="store_true",
                        help="Run per-component timing breakdown before the main loop.")
    parser.add_argument("--profile_samples", type=int, default=20,
                        help="Number of samples to use for --profile (default: 20).")
    parser.set_defaults(pin_memory=None, persistent_workers=None)
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    mod = importlib.import_module(args.config)
    config = mod.config

    data_config = copy.deepcopy(config["dataloaders"]["train"])
    if args.pyav_thread_count is not None:
        _override_pyav_thread_count(data_config, args.pyav_thread_count)
    batch_size = int(args.batch_size if args.batch_size is not None else data_config.get("batch_size_per_gpu", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else data_config.get("num_workers", 0))

    # ── Optional per-component profiling ──────────────────────────────────────
    if args.profile:
        _profile(copy.deepcopy(data_config), n_samples=args.profile_samples)

    # ── Main dataloader loop ──────────────────────────────────────────────────
    print(
        "Building dataloader"
        f"  (batch_size={batch_size}, num_workers={num_workers},"
        f" pyav_thread_count={_get_pyav_thread_counts(data_config) or 'default'},"
        f" pin_memory={args.pin_memory if args.pin_memory is not None else data_config.get('pin_memory', False)},"
        f" persistent_workers={args.persistent_workers if args.persistent_workers is not None else data_config.get('persistent_workers', False)},"
        f" prefetch_factor={args.prefetch_factor if args.prefetch_factor is not None else data_config.get('prefetch_factor', None)}) ..."
    )
    dataloader = _build_dataloader(
        data_config,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print(f"Dataset size: {len(dataloader.dataset)}")

    if args.quiet:
        n_iters, total_samples, elapsed = _measure_throughput(
            dataloader,
            batch_size=batch_size,
            num_batches=args.num_batches,
            warmup_batches=args.warmup_batches,
        )
    else:
        token_printed = False
        t0 = time.perf_counter()
        n_iters = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
            n_iters += 1

            _print_batch_stats(batch, batch_idx)

            if not token_printed:
                _print_token_count(batch)
                token_printed = True
        elapsed = time.perf_counter() - t0
        total_samples = n_iters * batch_size

    print(f"\n=== Dataloader throughput ({num_workers} workers, batch_size={batch_size}) ===")
    print(f"  {n_iters} batches / {total_samples} samples in {elapsed:.2f}s")
    if elapsed > 0 and n_iters > 0:
        print(f"  {n_iters / elapsed:.2f} batches/s  |  {total_samples / elapsed:.2f} samples/s")
        print(f"  {elapsed / n_iters * 1000:.1f} ms/batch  |  {elapsed / total_samples * 1000:.1f} ms/sample")


if __name__ == "__main__":
    main()
