import logging
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as _LeRobotDatasetMetadata

from .base_dataset import BaseDataset
from .dataset import register_dataset
from .. import utils


def _open_pyav_container(video_path: str, video_backend_kwargs: dict[str, Any] | None = None):
    kwargs = dict(video_backend_kwargs or {})
    pyav_thread_count = int(kwargs.pop("pyav_thread_count", 1))
    pyav_thread_type = str(kwargs.pop("pyav_thread_type", "SLICE"))
    container = av.open(video_path, **kwargs)
    if len(container.streams.video) == 0:
        container.close()
        raise ValueError(f"No video stream found in {video_path}")
    stream = container.streams.video[0]
    try:
        stream.codec_context.thread_count = max(1, pyav_thread_count)
    except Exception:
        pass
    try:
        stream.codec_context.thread_type = pyav_thread_type
    except Exception:
        pass
    return container, stream


def _decode_frames_by_timestamps_pyav(
    video_path: str,
    timestamps: np.ndarray,
    video_backend_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    container = None
    try:
        container, stream = _open_pyav_container(video_path, video_backend_kwargs)
        time_base = float(stream.time_base) if stream.time_base is not None else 0.0
        if not np.isfinite(time_base) or time_base <= 0:
            raise ValueError(f"Invalid time_base for {video_path}: {stream.time_base}")

        order = np.argsort(timestamps)
        sorted_timestamps = timestamps[order]

        if stream.duration is not None and stream.duration > 0:
            duration = float(stream.duration * time_base)
            sorted_timestamps = np.clip(sorted_timestamps, 0.0, max(0.0, duration))
        else:
            sorted_timestamps = np.maximum(sorted_timestamps, 0.0)

        seek_start_ts = max(0.0, float(sorted_timestamps[0]) - 0.5)
        try:
            container.seek(int(seek_start_ts / time_base), stream=stream, backward=True, any_frame=False)
        except Exception:
            container.seek(0)

        frames_sorted: list[np.ndarray | None] = [None] * len(sorted_timestamps)
        target_idx = 0
        prev_ts = None
        prev_frame = None

        for frame in container.decode(video=stream.index):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * time_base)
            frame_data = frame.to_ndarray(format="rgb24")

            if prev_ts is None:
                prev_ts = current_ts
                prev_frame = frame_data
                continue

            while target_idx < len(sorted_timestamps) and sorted_timestamps[target_idx] <= current_ts:
                diff_prev = abs(float(sorted_timestamps[target_idx]) - prev_ts)
                diff_curr = abs(float(sorted_timestamps[target_idx]) - current_ts)
                frames_sorted[target_idx] = frame_data if diff_curr < diff_prev else prev_frame
                target_idx += 1

            prev_ts = current_ts
            prev_frame = frame_data
            if target_idx >= len(sorted_timestamps):
                break

        if prev_frame is None:
            raise ValueError(f"Unable to decode any frame from {video_path}")

        while target_idx < len(sorted_timestamps):
            frames_sorted[target_idx] = prev_frame
            target_idx += 1

        frames_out: list[np.ndarray] = [None] * len(sorted_timestamps)  # type: ignore[list-item]
        for sorted_idx, original_idx in enumerate(order):
            frame_value = frames_sorted[sorted_idx]
            if frame_value is None:
                raise ValueError(f"Failed to map frame for ts={float(sorted_timestamps[sorted_idx]):.6f}")
            frames_out[original_idx] = frame_value
        return np.asarray(frames_out)
    finally:
        if container is not None:
            container.close()


def _torch_load_weights_only(path: Path | str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


@register_dataset
class LeRobotDataset(BaseDataset):
    def __init__(
        self,
        data_path,
        repo_id=None,
        data_size=None,
        delta_info=None,
        delta_frames=None,
        delta_timestamps=None,
        t5_embedding_dir=None,
        t5_embedding_pattern="episode_{episode_index:06d}.pt",
        t5_embedding_key="t5_embedding",
        t5_cache_size=256,
        vae_latent_dir=None,
        vae_latent_pattern="episode_{episode_index:06d}.pt",
        vae_latent_key="visual_latents",
        vae_latent_cache_size=4,
        skip_video_decode=None,
        meta_name=None,
        **kwargs,
    ):
        super(LeRobotDataset, self).__init__(data_path=data_path)
        self.repo_id = str(repo_id).strip() if repo_id is not None else None
        self.data_size = data_size
        self.delta_info = delta_info
        self.delta_frames = delta_frames
        self.delta_timestamps = delta_timestamps
        self.t5_embedding_dir = t5_embedding_dir
        self.t5_embedding_pattern = str(t5_embedding_pattern)
        self.t5_embedding_key = str(t5_embedding_key)
        self.t5_cache_size = int(t5_cache_size) if t5_cache_size is not None else 0
        # True LRU: OrderedDict; on hit, move_to_end to mark as most-recent.
        self._t5_cache: OrderedDict[int, Any] = OrderedDict()
        # VAE latent cache (per-episode dense tensor
        # [N=ep_length, z, T_lat, H_lat, W_lat] in bf16). When set, avoids video
        # decode + live VAE encode at train time.
        self.vae_latent_dir = vae_latent_dir
        self.vae_latent_pattern = str(vae_latent_pattern)
        self.vae_latent_key = str(vae_latent_key)
        self.vae_latent_cache_size = int(vae_latent_cache_size) if vae_latent_cache_size is not None else 0
        # True LRU: OrderedDict; on hit, move_to_end to mark as most-recent.
        self._vae_latent_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        # If None, auto: skip video decode iff vae_latent_dir is set.
        self.skip_video_decode = skip_video_decode
        self.meta_name = meta_name
        self.kwargs = kwargs
        self._is_open = False
        self.dataset = None
        self.robotype = None
        self.info = None
        self.codebase_version = ""
        self.fps = None
        self.video_backend = None
        self.video_backend_kwargs = {}
        self.data_path_pattern = None
        self.video_path_pattern = None
        self.chunk_size = None
        self.tasks_by_index: dict[int, str] = {}
        self.episode_rows: list[dict[str, Any]] = []
        self.failure_rows_by_episode: dict[int, dict[str, Any]] = {}
        self.episode_starts = None
        self.episode_stops = None
        self._episode_frame_cache: dict[int, pd.DataFrame] = {}
        self._episode_cache_order: list[int] = []

    @classmethod
    def load(cls, data_or_config):
        from .dataset import load_config

        config = load_config(data_or_config)
        keys = list(config.keys())
        for key in keys:
            if key.startswith('_'):
                config.pop(key)
        return cls(**config)

    def open(self):
        if not self._is_open:
            info = None
            for name in ("info.json", "info.yaml", "info.yml"):
                p = os.path.join(self.data_path, "meta", name)
                if os.path.isfile(p):
                    info = utils.load_file(p)
                    break
            if info is None:
                raise FileNotFoundError(f"missing meta/info.(json|yaml|yml) under {self.data_path}")

            robotype = None
            for k in ("robotype", "robot_type", "robot", "robot_name", "robot_model"):
                if k in info:
                    robotype = info[k]
                    break
            if robotype is None:
                raise KeyError(f"missing robotype in meta/info.* under {self.data_path}, keys={list(info.keys())}")
            if isinstance(robotype, str):
                robotype = robotype.strip()

            self.info = info
            self.robotype = robotype
            self.codebase_version = str(info.get("codebase_version", "")).strip()
            self.fps = int(info.get("fps", 1))
            self.video_backend = self.kwargs.get("video_backend", None) or "pyav"
            self.video_backend_kwargs = dict(self.kwargs.get("video_backend_kwargs", {}))
            self.data_path_pattern = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
            self.video_path_pattern = str(info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"))
            self.chunk_size = int(info.get("chunks_size", 1000))

            if self.codebase_version.startswith("v2."):
                self._open_local_v2()
            else:
                repo_id = self.repo_id or os.path.basename(self.data_path)
                dataset_meta = _LeRobotDatasetMetadata(repo_id, root=self.data_path)
                delta_timestamps = self._build_delta_timestamps(dataset_meta.fps)
                self.dataset = _LeRobotDataset(
                    repo_id,
                    root=self.data_path,
                    delta_timestamps=delta_timestamps,
                    **self.kwargs,
                )
                if self.data_size is not None:
                    assert self.data_size == len(self.dataset)
                else:
                    self.data_size = len(self.dataset)
            if self.t5_embedding_dir is None:
                d = os.path.join(self.data_path, "t5_embedding")
                if os.path.isdir(d):
                    self.t5_embedding_dir = d
            if self.t5_embedding_dir is not None and not os.path.isdir(self.t5_embedding_dir):
                raise FileNotFoundError(
                    f"Configured t5_embedding_dir does not exist for dataset {self.data_path}: {self.t5_embedding_dir}"
                )
            # vae_latent_dir semantics:
            #   None  -> auto-discover {data_path}/vae_latents/ if it exists
            #   ""    -> explicit disable (do not use cache even if directory exists)
            #   False -> explicit disable (same as empty string)
            #   path  -> use this path (must exist)
            if self.vae_latent_dir is False or self.vae_latent_dir == "":
                self.vae_latent_dir = None
            elif self.vae_latent_dir is None:
                d = os.path.join(self.data_path, "vae_latents")
                if os.path.isdir(d):
                    self.vae_latent_dir = d
            if self.vae_latent_dir is not None and not os.path.isdir(self.vae_latent_dir):
                raise FileNotFoundError(
                    f"Configured vae_latent_dir does not exist for dataset {self.data_path}: {self.vae_latent_dir}"
                )
            # Validate cache manifest. We don't hardcode the exact
            # schema_version a dataset accepts (so a preprocessor bump doesn't
            # silently reject a still-compatible cache); instead we require a
            # minimum version + the `normalized=True` flag that marks the fix
            # from v2 onwards. Callers can override the minimum with
            # `vae_latent_min_schema_version` kwarg.
            if self.vae_latent_dir is not None:
                import json as _json
                manifest_path = os.path.join(self.vae_latent_dir, "_manifest.json")
                if not os.path.isfile(manifest_path):
                    raise FileNotFoundError(
                        f"VAE latent cache at {self.vae_latent_dir} has no _manifest.json. "
                        "Regenerate with `python -m scripts.compute_vae_latents`."
                    )
                with open(manifest_path) as _f:
                    manifest = _json.load(_f)
                schema_version = int(manifest.get("schema_version", 0))
                min_schema_version = int(self.kwargs.get("vae_latent_min_schema_version", 3))
                if schema_version < min_schema_version:
                    raise ValueError(
                        f"VAE latent cache at {self.vae_latent_dir} has schema_version={schema_version}, "
                        f"minimum supported is {min_schema_version}. The older format is missing either "
                        "the whitening normalization or the bf16 storage. "
                        "Delete the directory and regenerate with `python -m scripts.compute_vae_latents`."
                    )
                if not bool(manifest.get("normalized", False)):
                    raise ValueError(
                        f"VAE latent cache at {self.vae_latent_dir} claims normalized=False. "
                        "Regenerate with a post-fix compute_vae_latents."
                    )
                # Explicit dtype guard: the trainer loads the cache as vae.dtype (bf16).
                # If the cache were stored in fp16, the load-time cast is a lossy
                # round-trip through fp16 mantissa. Refuse anything that isn't bf16
                # so "normalized but wrong dtype" caches don't sneak through a future
                # schema_version check.
                cache_dtype = str(manifest.get("dtype", "")).lower()
                accepted_dtypes = {str(d).lower() for d in self.kwargs.get("vae_latent_accepted_dtypes", ["bfloat16"])}
                if cache_dtype not in accepted_dtypes:
                    raise ValueError(
                        f"VAE latent cache at {self.vae_latent_dir} has dtype={cache_dtype!r}, "
                        f"expected one of {sorted(accepted_dtypes)}. The trainer dispatches through "
                        "vae.dtype (bf16); other storage dtypes force a lossy round-trip. "
                        "Regenerate with `python -m scripts.compute_vae_latents`."
                    )
                self._vae_latent_manifest = manifest
                # One-shot log so operators can see which path is active.
                logging.getLogger(__name__).info(
                    "LeRobotDataset %s using cached VAE latents from %s "
                    "(schema_version=%d dtype=%s)",
                    self.data_path,
                    self.vae_latent_dir,
                    schema_version,
                    manifest.get("dtype", "unknown"),
                )
            self._is_open = True

    def close(self):
        self._is_open = False
        if self.dataset is not None:
            self.dataset = None
        self.info = None
        self.tasks_by_index.clear()
        self.episode_rows = []
        self.failure_rows_by_episode.clear()
        self.episode_starts = None
        self.episode_stops = None
        self._episode_frame_cache.clear()
        self._episode_cache_order.clear()
        self._t5_cache.clear()
        self._vae_latent_cache.clear()
        super(LeRobotDataset, self).close()

    def _effective_skip_video_decode(self) -> bool:
        if self.skip_video_decode is None:
            return self.vae_latent_dir is not None
        return bool(self.skip_video_decode)

    def __len__(self):
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index):
        if self.codebase_version.startswith("v2."):
            data_dict = self._get_local_v2_data(int(index))
        else:
            data_dict = self.dataset[index]
        data_dict["robotype"] = self.robotype
        if self.meta_name is not None:
            assert self.meta_name not in data_dict
            data_dict[self.meta_name] = self.info if self.codebase_version.startswith("v2.") else self.dataset.meta
        if self.t5_embedding_dir is not None:
            episode_index = data_dict.get("episode_index", None)
            if hasattr(episode_index, "item"):
                try:
                    episode_index = episode_index.item()
                except Exception as e:
                    print(f"error in episode_index.item(): {e}")
            try:
                cache_key = int(episode_index)
            except Exception as e:
                print(f"error in int(episode_index): {e}")
                cache_key = None
            if cache_key is not None:
                if cache_key in self._t5_cache:
                    # LRU: mark as most-recently-used on hit.
                    self._t5_cache.move_to_end(cache_key)
                    data_dict[self.t5_embedding_key] = self._t5_cache[cache_key]
                else:
                    p = self._resolve_t5_embedding_path(cache_key, data_dict)
                    if p is None:
                        default_path = Path(self.t5_embedding_dir) / self.t5_embedding_pattern.format(episode_index=cache_key)
                        raise FileNotFoundError(
                            f"Missing T5 embedding cache for dataset {self.data_path}, episode {cache_key}: {default_path}. "
                            "Generate it with `python -m scripts.compute_t5_embedding --root <task_dir> --wan_path <wan_model_path>` "
                            "or `python -m scripts.prepare_robotwin --robotwin_root <root> --wan_model_path <wan_model_path>`."
                        )
                    obj = _torch_load_weights_only(p)
                    if isinstance(obj, dict):
                        if self.t5_embedding_key in obj:
                            emb = obj[self.t5_embedding_key]
                        elif "prompt_embeds" in obj:
                            emb = obj["prompt_embeds"]
                        elif "condition_dict" in obj and isinstance(obj["condition_dict"], dict) and "prompt_embeds" in obj["condition_dict"]:
                            emb = obj["condition_dict"]["prompt_embeds"]
                        else:
                            emb = next(iter(obj.values()))
                    else:
                        emb = obj
                    data_dict[self.t5_embedding_key] = emb
                    if self.t5_cache_size > 0:
                        self._t5_cache[cache_key] = emb
                        self._t5_cache.move_to_end(cache_key)
                        while len(self._t5_cache) > self.t5_cache_size:
                            self._t5_cache.popitem(last=False)
        if self.vae_latent_dir is not None:
            episode_index = data_dict.get("episode_index", None)
            # Prefer local_frame_index (always contiguous 0..ep_length-1, matches
            # cache row order). Fall back to parquet frame_index for non-v2 path
            # where local_frame_index isn't populated — in that case the dataset
            # is trusting the upstream lerobot to keep frame_index contiguous.
            local_fi = data_dict.get("local_frame_index", None)
            frame_index = local_fi if local_fi is not None else data_dict.get("frame_index", None)
            if hasattr(episode_index, "item"):
                try:
                    episode_index = episode_index.item()
                except Exception:
                    pass
            if hasattr(frame_index, "item"):
                try:
                    frame_index = frame_index.item()
                except Exception:
                    pass
            try:
                ep_key = int(episode_index)
                frame_idx = int(frame_index)
            except Exception as e:
                raise ValueError(
                    f"vae_latent_dir set but episode_index/local_frame_index missing/invalid in sample: {e}"
                )
            if ep_key in self._vae_latent_cache:
                # LRU: mark as most-recently-used on hit.
                self._vae_latent_cache.move_to_end(ep_key)
                ep_tensor = self._vae_latent_cache[ep_key]
            else:
                latent_path = Path(self.vae_latent_dir) / self.vae_latent_pattern.format(episode_index=ep_key)
                if not latent_path.is_file():
                    raise FileNotFoundError(
                        f"Missing VAE latent cache for dataset {self.data_path}, episode {ep_key}: {latent_path}. "
                        "Generate it with `python -m scripts.compute_vae_latents`."
                    )
                ep_tensor = _torch_load_weights_only(latent_path)
                if not isinstance(ep_tensor, torch.Tensor):
                    raise TypeError(
                        f"Expected torch.Tensor in {latent_path}, got {type(ep_tensor)}"
                    )
                if self.vae_latent_cache_size > 0:
                    self._vae_latent_cache[ep_key] = ep_tensor
                    self._vae_latent_cache.move_to_end(ep_key)
                    while len(self._vae_latent_cache) > self.vae_latent_cache_size:
                        self._vae_latent_cache.popitem(last=False)
            n = int(ep_tensor.shape[0])
            if frame_idx < 0 or frame_idx >= n:
                raise IndexError(
                    f"frame_index={frame_idx} out of range [0, {n}) for episode {ep_key} at {self.vae_latent_dir}"
                )
            # Clone so downstream code can't mutate the cached tensor.
            data_dict[self.vae_latent_key] = ep_tensor[frame_idx].clone()
        return data_dict

    def _resolve_t5_embedding_path(self, episode_index: int, data_dict: dict[str, Any]) -> Path | None:
        candidates: list[Path] = []
        metadata_path = data_dict.get("t5_embedding_path", None)
        if metadata_path:
            p = Path(str(metadata_path))
            if not p.is_absolute():
                p = Path(self.data_path) / p
            candidates.append(p)

        if self.t5_embedding_dir is not None:
            candidates.append(Path(self.t5_embedding_dir) / self.t5_embedding_pattern.format(episode_index=episode_index))

        for p in candidates:
            if p.is_file():
                return p
        return None

    def _build_delta_timestamps(self, fps: int) -> dict[str, list[float]]:
        delta_timestamps = {}
        if self.delta_info is not None:
            for delta_name, delta_size in self.delta_info.items():
                delta_timestamps[delta_name] = [i / fps for i in range(int(delta_size))]
        if self.delta_frames is not None:
            for delta_name, frames in dict(self.delta_frames).items():
                delta_timestamps[delta_name] = [float(i) / fps for i in frames]
        if self.delta_timestamps is not None:
            for delta_name, stamps in dict(self.delta_timestamps).items():
                delta_timestamps[delta_name] = [float(t) for t in stamps]
        if not delta_timestamps:
            delta_timestamps = {"action": [0.0]}
        return delta_timestamps

    def _open_local_v2(self) -> None:
        self.tasks_by_index.clear()
        self.episode_rows = []
        self.failure_rows_by_episode.clear()
        tasks_path = Path(self.data_path) / "meta" / "tasks.jsonl"
        if tasks_path.is_file():
            with open(tasks_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.tasks_by_index[int(row["task_index"])] = str(row.get("task", row.get("task_name", "")))

        episodes_path = Path(self.data_path) / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            raise FileNotFoundError(f"missing v2 episode metadata: {episodes_path}")
        starts: list[int] = []
        stops: list[int] = []
        running = 0
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                length = int(row.get("length", 0))
                start = running
                stop = start + max(0, length)
                self.episode_rows.append(row)
                starts.append(start)
                stops.append(stop)
                running = stop

        failure_path = Path(self.data_path) / "meta" / "failure_rollouts.jsonl"
        if failure_path.is_file():
            with open(failure_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.failure_rows_by_episode[int(row["episode_index"])] = row

        self.episode_starts = np.asarray(starts, dtype=np.int64)
        self.episode_stops = np.asarray(stops, dtype=np.int64)
        if self.data_size is not None:
            assert int(self.data_size) == int(running), f"Expected data_size={self.data_size}, got {running}"
        else:
            self.data_size = int(running)

    def _get_episode_row(self, index: int) -> tuple[int, dict[str, Any], int]:
        assert self.episode_stops is not None and self.episode_starts is not None
        if index < 0 or index >= int(self.data_size):
            raise IndexError(index)
        episode_pos = int(np.searchsorted(self.episode_stops, index, side="right"))
        episode_row = self.episode_rows[episode_pos]
        episode_start = int(self.episode_starts[episode_pos])
        local_frame_index = int(index - episode_start)
        return episode_pos, episode_row, local_frame_index

    def _load_episode_frame_df(self, episode_index: int) -> pd.DataFrame:
        if episode_index in self._episode_frame_cache:
            return self._episode_frame_cache[episode_index]

        parquet_path = Path(self.data_path) / self.data_path_pattern.format(
            episode_chunk=int(episode_index) // int(self.chunk_size),
            episode_index=int(episode_index),
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing episode parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df = df.sort_values("frame_index", kind="stable").reset_index(drop=True)
        self._episode_frame_cache[episode_index] = df
        self._episode_cache_order.append(episode_index)
        while len(self._episode_cache_order) > 4:
            old = self._episode_cache_order.pop(0)
            self._episode_frame_cache.pop(old, None)
        return df

    def _get_query_indices(self, base_index: int, episode_length: int, key: str) -> np.ndarray:
        if self.delta_info is not None and key in self.delta_info:
            offsets = np.arange(int(self.delta_info[key]), dtype=np.int64)
        elif self.delta_frames is not None and key in self.delta_frames:
            offsets = np.asarray(list(self.delta_frames[key]), dtype=np.int64)
        elif self.delta_timestamps is not None and key in self.delta_timestamps:
            offsets = np.rint(np.asarray(list(self.delta_timestamps[key]), dtype=np.float64) * float(self.fps)).astype(np.int64)
        else:
            return np.asarray([base_index], dtype=np.int64)
        query_indices = base_index + offsets
        return np.clip(query_indices, 0, max(0, episode_length - 1))

    def _load_video_frames(self, episode_index: int, video_key: str, timestamps: np.ndarray) -> np.ndarray:
        video_path = Path(self.data_path) / self.video_path_pattern.format(
            episode_chunk=int(episode_index) // int(self.chunk_size),
            episode_index=int(episode_index),
            video_key=video_key,
        )
        if not video_path.is_file():
            raise FileNotFoundError(f"missing episode video: {video_path}")
        if self.video_backend != "pyav":
            raise ValueError(f"Unsupported local v2 video_backend={self.video_backend!r}; only 'pyav' is supported.")
        return _decode_frames_by_timestamps_pyav(
            video_path=str(video_path),
            timestamps=timestamps,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def _get_local_v2_data(self, index: int) -> dict[str, Any]:
        _episode_pos, episode_row, local_frame_index = self._get_episode_row(index)
        episode_index = int(episode_row["episode_index"])
        episode_length = int(episode_row["length"])
        episode_df = self._load_episode_frame_df(episode_index)
        if local_frame_index >= len(episode_df):
            raise IndexError(
                f"Frame index {local_frame_index} exceeds episode parquet length {len(episode_df)} "
                f"for dataset {self.data_path}, episode {episode_index}"
            )
        episode_length = min(episode_length, len(episode_df))
        row = episode_df.iloc[local_frame_index]

        data_dict: dict[str, Any] = {
            "observation.state": np.array(row["observation.state"], dtype=np.float32, copy=True),
            "episode_index": np.int64(episode_index),
            "episode_length": np.int64(episode_length),
            "frame_index": np.int64(row["frame_index"]),
            # local_frame_index is the POSITION within this episode's sorted
            # parquet (0..ep_length-1). Always contiguous; safe to use as a
            # direct cache index (the preprocessor also uses positions).
            "local_frame_index": np.int64(local_frame_index),
            "index": np.int64(row.get("index", index)),
            "task_index": np.int64(row["task_index"]),
            "timestamp": np.float32(row["timestamp"]),
        }
        if "t5_embedding_path" in episode_row:
            data_dict["t5_embedding_path"] = episode_row["t5_embedding_path"]

        action_indices = self._get_query_indices(local_frame_index, episode_length, "action")
        action_rows = episode_df.iloc[action_indices]["action"].to_numpy()
        data_dict["action"] = np.stack([np.array(value, dtype=np.float32, copy=True) for value in action_rows], axis=0)
        future_state_index = min(local_frame_index + max(1, len(action_indices)), max(0, episode_length - 1))
        future_state_row = episode_df.iloc[future_state_index]
        data_dict["future_state"] = np.array(future_state_row["observation.state"], dtype=np.float32, copy=True)
        data_dict["future_state_index"] = np.int64(future_state_index)

        if int(data_dict["task_index"]) in self.tasks_by_index:
            data_dict["task"] = self.tasks_by_index[int(data_dict["task_index"])]

        failure_row = self.failure_rows_by_episode.get(int(episode_index))
        if failure_row is not None:
            failure_episode = bool(failure_row.get("failure_episode", True))
            active_from = failure_row.get("failure_active_from_frame", 0)
            active_from = 0 if active_from is None else int(active_from)
            data_dict["failure_episode"] = np.bool_(failure_episode)
            data_dict["failure_active"] = np.bool_(failure_episode and local_frame_index >= active_from)

        if not self._effective_skip_video_decode():
            image_keys = set()
            if self.delta_frames is not None:
                image_keys.update(key for key in self.delta_frames.keys() if key != "action")
            if self.delta_timestamps is not None:
                image_keys.update(key for key in self.delta_timestamps.keys() if key != "action")

            if not image_keys:
                image_keys.update(
                    key for key, feature in (self.info or {}).get("features", {}).items() if feature.get("dtype") == "video"
                )

            for image_key in sorted(image_keys):
                query_indices = self._get_query_indices(local_frame_index, episode_length, image_key)
                timestamps = episode_df.iloc[query_indices]["timestamp"].to_numpy(dtype=np.float64)
                frames = self._load_video_frames(episode_index=episode_index, video_key=image_key, timestamps=timestamps)
                data_dict[image_key] = frames

        return data_dict

