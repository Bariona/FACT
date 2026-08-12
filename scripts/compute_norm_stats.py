import pathlib
from typing import Any

import numpy as np
import numpydantic
import pandas as pd
import pydantic
import tyro
from tqdm import tqdm


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    min: numpydantic.NDArray | None = None
    max: numpydantic.NDArray | None = None
    q01: numpydantic.NDArray | None = None
    q99: numpydantic.NDArray | None = None


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000

    def update(self, batch: np.ndarray) -> None:
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1) for i in range(vector_length)]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(
            mean=self._mean,
            std=stddev,
            min=self._min,
            max=self._max,
            q01=q01,
            q99=q99,
        )

    def _adjust_histograms(self):
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def _build_default_value_stats() -> NormStats:
    # max is 2.0 rather than the raw time-to-go upper bound of 1.0, leaving
    # headroom for failure episodes, whose V target is `gt_raw +
    # value_penalty_scale`. The normalized range [-1, 1] then covers both expert
    # values (gt_raw in [0, 1]) and failure values (raw in (1, 2]).
    value_vec = np.asarray([0.0], dtype=np.float64)
    return NormStats(
        mean=value_vec.copy(),
        std=np.asarray([1.0], dtype=np.float64),
        min=np.asarray([-1.0], dtype=np.float64),
        max=np.asarray([2.0], dtype=np.float64),
        q01=np.asarray([-1.0], dtype=np.float64),
        q99=np.asarray([2.0], dtype=np.float64),
    )


def _pad_to_dim(values: np.ndarray, target_dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] >= target_dim:
        return values
    padded = np.zeros((*values.shape[:-1], target_dim), dtype=np.float64)
    padded[..., : values.shape[-1]] = values
    return padded


def _sample_indices(num_frames: int, sample_rate: float, rng: np.random.Generator) -> np.ndarray:
    if num_frames <= 0:
        return np.empty((0,), dtype=np.int64)
    if sample_rate >= 1.0:
        return np.arange(num_frames, dtype=np.int64)
    keep_mask = rng.random(num_frames) < sample_rate
    indices = np.flatnonzero(keep_mask)
    if indices.size == 0:
        indices = np.array([0], dtype=np.int64)
    return indices


def compute_norm_stats(
    data_paths: list[str],
    output_path: str | pathlib.Path,
    embodiment_id: int,
    delta_mask: list[bool],
    sample_rate: float = 1.0,
    action_chunk: int = 50,
    action_dim: int = 32,
) -> None:
    """Compute normalization statistics from LeRobot parquet files and write them to JSON.

    The scan is sequential (RunningStats is stateful); use ``sample_rate`` for speed.
    """

    if not data_paths:
        raise ValueError("data_paths must not be empty")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be > 0")

    keys = ["observation.state", "action"]
    stats = {key: RunningStats() for key in keys}
    rng = np.random.default_rng(0)
    delta_mask_arr = np.asarray(delta_mask, dtype=bool)

    for data_path in tqdm(data_paths, desc="Datasets"):
        root = pathlib.Path(data_path).expanduser().resolve()
        parquet_paths = sorted((root / "data").rglob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

        for parquet_path in tqdm(parquet_paths, desc=f"Parquet {root.name}", leave=False):
            frame_df = pd.read_parquet(
                parquet_path,
                columns=["observation.state", "action", "episode_index", "frame_index"],
            )
            if frame_df.empty:
                continue

            frame_df = frame_df.sort_values(["episode_index", "frame_index"], kind="stable")
            for _, episode_df in frame_df.groupby("episode_index", sort=False):
                states = np.stack(episode_df["observation.state"].to_numpy()).astype(np.float64, copy=False)
                actions = np.stack(episode_df["action"].to_numpy()).astype(np.float64, copy=False)
                indices = _sample_indices(num_frames=len(states), sample_rate=sample_rate, rng=rng)
                if indices.size == 0:
                    continue

                sampled_states = states[indices]
                stats["observation.state"].update(_pad_to_dim(sampled_states, action_dim))

                padded_actions = np.concatenate(
                    [actions, np.repeat(actions[-1:, :], repeats=max(0, action_chunk - 1), axis=0)],
                    axis=0,
                )
                action_chunks = np.stack([padded_actions[idx : idx + action_chunk] for idx in indices], axis=0)
                dims = min(delta_mask_arr.size, action_chunks.shape[-1], sampled_states.shape[-1])
                if dims > 0:
                    delta_base = np.where(delta_mask_arr[:dims][None, None, :], sampled_states[:, None, :dims], 0.0)
                    action_chunks[..., :dims] -= delta_base
                stats["action"].update(_pad_to_dim(action_chunks, action_dim).reshape(-1, max(action_dim, action_chunks.shape[-1])))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    # Keep value normalization configurable from stats instead of hard-coding [0, 1] -> [-1, 1] in model code.
    norm_stats["value"] = _build_default_value_stats()

    print(f"Writing stats to: {output_path}")
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_json(norm_stats))


if __name__ == "__main__":
    tyro.cli(compute_norm_stats)
