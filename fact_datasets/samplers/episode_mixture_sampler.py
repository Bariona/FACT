import math
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from ..datasets import ConcatDataset


class EpisodeMixtureSampler(torch.utils.data.Sampler):
    """Sample a ``ConcatDataset`` by dataset weight, then episode, then frame.

    Each child dataset is expected to be a ``LeRobotDataset``-like task dataset
    with episode metadata under ``meta/episodes/**/*.parquet``. Sampling is:
    1. choose child dataset with probability proportional to
       ``dataset_multiplier * episode_count``
    2. choose an episode uniformly inside that dataset
    3. choose a frame/sample uniformly inside the chosen episode

    ``dataset_weights`` are therefore per-dataset multipliers, not direct
    group-level probabilities. Equal multipliers still produce sampling
    proportional to episode counts.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int | None = None,
        shuffle: bool = True,
        infinite: bool = True,
        seed: int = 6666,
        dataset_weights: list[float] | None = None,
        sample_epoch_size: int | None = None,
    ) -> None:
        assert isinstance(dataset, ConcatDataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = int(seed)
        self.epoch = 0
        self._dataset_weights = np.asarray(dataset_weights if dataset_weights is not None else [1.0] * len(dataset.datasets), dtype=np.float64)
        if self._dataset_weights.shape[0] != len(dataset.datasets):
            raise ValueError("dataset_weights length must match number of sub-datasets")
        self._sample_epoch_size = int(sample_epoch_size) if sample_epoch_size is not None else None

        self._episode_ranges_per_dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._dataset_sizes: list[int] = []
        self._offsets: list[int] = []
        self._episode_counts: list[int] = []

        start_idx = 0
        for child in dataset.datasets:
            episode_starts, episode_stops, dataset_size = self._load_episode_ranges(child)
            if episode_starts.size == 0:
                raise ValueError(f"Dataset at {getattr(child, 'data_path', None)} has no episodes")
            self._episode_ranges_per_dataset.append((episode_starts, episode_stops, episode_stops - episode_starts))
            self._dataset_sizes.append(dataset_size)
            self._offsets.append(start_idx)
            self._episode_counts.append(int(episode_starts.size))
            start_idx += dataset_size

        self._dataset_sizes_arr = np.asarray(self._dataset_sizes, dtype=np.int64)
        self._episode_counts_arr = np.asarray(self._episode_counts, dtype=np.int64)
        self._effective_weights = self._dataset_weights * self._episode_counts_arr
        self._effective_weights = np.maximum(self._effective_weights, 1e-8)
        self._effective_weights = self._effective_weights / self._effective_weights.sum()

        base_epoch_size = self._sample_epoch_size or int(self._dataset_sizes_arr.sum())
        if batch_size is not None:
            self.total_size = int(math.ceil(base_epoch_size / batch_size)) * batch_size
        else:
            self.total_size = base_epoch_size

    @staticmethod
    def _load_episode_ranges(dataset) -> tuple[np.ndarray, np.ndarray, int]:
        data_path = getattr(dataset, "data_path", None)
        if data_path is None:
            raise ValueError("EpisodeMixtureSampler requires datasets with a data_path")

        episode_starts: list[int] = []
        episode_stops: list[int] = []
        meta_dir = Path(data_path) / "meta"
        v2_episodes_path = meta_dir / "episodes.jsonl"
        v3_meta_root = meta_dir / "episodes"

        if v2_episodes_path.is_file():
            start = 0
            with open(v2_episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    length = int(row.get("length", 0))
                    stop = start + length
                    if stop > start:
                        episode_starts.append(start)
                        episode_stops.append(stop)
                    start = stop
        else:
            parquet_paths = sorted(v3_meta_root.rglob("*.parquet"))
            if not parquet_paths:
                raise FileNotFoundError(
                    f"Could not find LeRobot episode metadata under {v2_episodes_path} or {v3_meta_root}"
                )
            for parquet_path in parquet_paths:
                df = pd.read_parquet(parquet_path, columns=["dataset_from_index", "dataset_to_index"])
                for row in df.itertuples(index=False):
                    start = int(row.dataset_from_index)
                    stop = int(row.dataset_to_index)
                    if stop <= start:
                        continue
                    episode_starts.append(start)
                    episode_stops.append(stop)

        starts_arr = np.asarray(episode_starts, dtype=np.int64)
        stops_arr = np.asarray(episode_stops, dtype=np.int64)
        inferred_size = int(stops_arr.max(initial=0))
        explicit_size = getattr(dataset, "data_size", None)
        if explicit_size is not None:
            dataset_size = int(explicit_size)
            if dataset_size > inferred_size:
                raise ValueError(
                    f"Dataset at {data_path} reports data_size={dataset_size}, larger than episode metadata size={inferred_size}"
                )
            if dataset_size < inferred_size:
                valid_mask = starts_arr < dataset_size
                starts_arr = starts_arr[valid_mask]
                stops_arr = np.minimum(stops_arr[valid_mask], dataset_size)
                valid_mask = stops_arr > starts_arr
                starts_arr = starts_arr[valid_mask]
                stops_arr = stops_arr[valid_mask]
        else:
            dataset_size = inferred_size
        return starts_arr, stops_arr, dataset_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.total_size

    def __iter__(self) -> Iterator[int]:
        while True:
            rng = np.random.default_rng(self.seed + self.epoch)
            self.epoch += 1
            dataset_indices = rng.choice(len(self._episode_ranges_per_dataset), size=self.total_size, p=self._effective_weights)
            indices = np.empty((self.total_size,), dtype=np.int64)
            for dataset_idx in range(len(self._episode_ranges_per_dataset)):
                sample_mask = dataset_indices == dataset_idx
                sample_count = int(sample_mask.sum())
                if sample_count == 0:
                    continue
                episode_starts, _episode_stops, episode_lengths = self._episode_ranges_per_dataset[dataset_idx]
                chosen_episode_indices = rng.integers(episode_starts.size, size=sample_count)
                chosen_episode_starts = episode_starts[chosen_episode_indices]
                chosen_episode_lengths = episode_lengths[chosen_episode_indices]
                frame_offsets = rng.integers(0, chosen_episode_lengths)
                indices[sample_mask] = self._offsets[dataset_idx] + chosen_episode_starts + frame_offsets
            if self.shuffle:
                rng.shuffle(indices)
            yield from indices.tolist()
            if not self.infinite:
                break
