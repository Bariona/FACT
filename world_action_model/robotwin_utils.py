from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_EXCLUDED_TOP_LEVEL_GROUPS = ("EvalRollouts", "Fail")
SUPPORTED_ROBOTWIN_CODEBASE_VERSION = "v2.1"


@dataclass(frozen=True)
class RobotWinTaskSpec:
    dataset_id: str
    task_dir: Path
    task_name: str
    group_name: str | None
    repo_id: str


def _info_path(task_dir: Path) -> Path:
    return task_dir / "meta" / "info.json"


def is_robotwin_task_dir(path: str | Path) -> bool:
    return _info_path(Path(path).expanduser().resolve()).is_file()


def get_codebase_version(task_dir: str | Path) -> str:
    info_path = _info_path(Path(task_dir).expanduser().resolve())
    if not info_path.is_file():
        return ""
    return str(json.loads(info_path.read_text()).get("codebase_version", "")).strip()


def _should_skip_task(rel_parts: tuple[str, ...]) -> bool:
    if not rel_parts:
        return False
    return any(part.endswith("_old") for part in rel_parts)


def _has_glob_wildcards(pattern: str) -> bool:
    return any(token in pattern for token in ("*", "?", "["))


def _pattern_specificity(pattern: str) -> tuple[int, int, int, int, int]:
    normalized = pattern.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    literal_segments = sum(1 for part in parts if not _has_glob_wildcards(part))
    wildcard_chars = sum(1 for ch in normalized if ch in "*?[")
    literal_chars = sum(1 for ch in normalized if ch not in "*?[]")
    return (
        int(not _has_glob_wildcards(normalized)),
        literal_segments,
        len(parts),
        literal_chars,
        -wildcard_chars,
    )


def _build_task_spec(task_dir: Path, root_path: Path) -> RobotWinTaskSpec:
    task_dir = task_dir.expanduser().resolve()
    root_path = root_path.expanduser().resolve()
    if task_dir == root_path:
        dataset_id = task_dir.name
        rel_parts = (task_dir.name,)
    else:
        try:
            rel_path = task_dir.relative_to(root_path)
            dataset_id = rel_path.as_posix()
            rel_parts = rel_path.parts
        except ValueError:
            dataset_id = task_dir.name
            rel_parts = (task_dir.name,)
    group_name = rel_parts[0] if len(rel_parts) >= 2 else None
    return RobotWinTaskSpec(
        dataset_id=dataset_id,
        task_dir=task_dir,
        task_name=task_dir.name,
        group_name=group_name,
        repo_id=task_dir.name,
    )


def discover_robotwin_task_specs(robotwin_root: str | Path) -> list[RobotWinTaskSpec]:
    root_path = Path(robotwin_root).expanduser().resolve()
    if is_robotwin_task_dir(root_path):
        return [_build_task_spec(root_path, root_path)]

    task_dirs = sorted(
        {
            info_path.parent.parent.resolve()
            for info_path in root_path.rglob("meta/info.json")
            if not _should_skip_task(info_path.parent.parent.resolve().relative_to(root_path).parts)
        }
    )
    return [_build_task_spec(task_dir, root_path) for task_dir in task_dirs]


def validate_robotwin_codebase_versions(
    specs: Sequence[RobotWinTaskSpec],
    supported_version: str = SUPPORTED_ROBOTWIN_CODEBASE_VERSION,
) -> None:
    mismatched: list[tuple[str, str]] = []
    for spec in specs:
        version = get_codebase_version(spec.task_dir)
        if version != supported_version:
            mismatched.append((spec.dataset_id, version))

    if mismatched:
        preview = ", ".join(f"{dataset_id} ({version or 'unknown'})" for dataset_id, version in mismatched[:5])
        raise RuntimeError(
            f"RoboTwin loader only supports codebase_version={supported_version}. "
            f"Found incompatible tasks under the selected root: {preview}"
        )


def select_robotwin_task_specs(
    robotwin_root: str | Path,
    dataset_ids: Sequence[str] | None = None,
    dataset_globs: Sequence[str] | None = None,
    excluded_top_level_groups: Sequence[str] | None = None,
    supported_version: str | None = SUPPORTED_ROBOTWIN_CODEBASE_VERSION,
) -> list[RobotWinTaskSpec]:
    specs = discover_robotwin_task_specs(robotwin_root)
    if not specs:
        root_path = Path(robotwin_root).expanduser().resolve()
        raise FileNotFoundError(f"No LeRobot task directories found under {root_path}")

    dataset_ids = [s for s in (dataset_ids or []) if str(s).strip()]
    dataset_globs = [s for s in (dataset_globs or []) if str(s).strip()]

    if dataset_ids or dataset_globs:
        selected = [
            spec
            for spec in specs
            if spec.dataset_id in dataset_ids or any(fnmatch(spec.dataset_id, pattern) for pattern in dataset_globs)
        ]
    else:
        excluded = set(excluded_top_level_groups or DEFAULT_EXCLUDED_TOP_LEVEL_GROUPS)
        grouped_specs = [spec for spec in specs if spec.group_name is not None]
        if grouped_specs:
            selected = [spec for spec in specs if spec.group_name not in excluded]
        else:
            selected = specs

    if not selected:
        available = ", ".join(spec.dataset_id for spec in specs[:20])
        raise FileNotFoundError(
            "No LeRobot task directories matched the selection under "
            f"{Path(robotwin_root).expanduser().resolve()}. Available examples: {available}"
        )
    selected = sorted(selected, key=lambda spec: spec.dataset_id)
    if supported_version is not None:
        validate_robotwin_codebase_versions(selected, supported_version=supported_version)
    return selected


def resolve_dataset_multipliers(
    specs: Sequence[RobotWinTaskSpec],
    multiplier_overrides: Mapping[str, float] | None = None,
) -> list[float]:
    overrides = multiplier_overrides or {}
    multipliers: list[float] = []
    for spec in specs:
        multiplier = 1.0
        matched_score: tuple[int, int, int, int, int] | None = None
        matched_order = -1
        for order, (pattern, value) in enumerate(overrides.items()):
            if fnmatch(spec.dataset_id, pattern):
                score = _pattern_specificity(pattern)
                if matched_score is None or score > matched_score or (score == matched_score and order > matched_order):
                    multiplier = float(value)
                    matched_score = score
                    matched_order = order
        multipliers.append(float(multiplier))
    return multipliers


def describe_robotwin_selection(
    specs: Sequence[RobotWinTaskSpec],
    multipliers: Sequence[float] | None = None,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for idx, spec in enumerate(specs):
        row: dict[str, str | float] = {
            "dataset_id": spec.dataset_id,
            "path": str(spec.task_dir),
            "repo_id": spec.repo_id,
        }
        if multipliers is not None:
            row["multiplier"] = float(multipliers[idx])
        rows.append(row)
    return rows


