from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_action_model.robotwin_utils import (
    describe_robotwin_selection,
    resolve_dataset_multipliers,
    select_robotwin_task_specs,
)

ALOHA_DELTA_MASK = [
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
]


def _parse_csv_or_repeatable(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        parts = [part.strip() for part in str(value).split(",")]
        out.extend(part for part in parts if part)
    return out


def _parse_multiplier_overrides(values: list[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in _parse_csv_or_repeatable(values):
        if "=" not in value:
            raise ValueError(f"Invalid multiplier override {value!r}; expected pattern=value")
        pattern, raw_weight = value.split("=", 1)
        overrides[pattern.strip()] = float(raw_weight.strip())
    return overrides


def main() -> None:
    from world_action_model import apply_runtime_compat

    apply_runtime_compat()

    from scripts.compute_norm_stats import compute_norm_stats
    from scripts.compute_t5_embedding import compute_t5_embeddings_for_dataset

    parser = argparse.ArgumentParser(description="Prepare grouped RoboTwin LeRobot tasks for FACT RoboTwin training.")
    parser.add_argument(
        "--robotwin_root",
        type=str,
        required=True,
        help="RoboTwin root directory or a single LeRobot task directory.",
    )
    parser.add_argument(
        "--wan_model_path",
        type=str,
        required=True,
        help="Path to Wan2.2-TI2V-5B-Diffusers.",
    )
    parser.add_argument("--output_dir", type=str, default="./artifacts/robotwin", help="Directory for generated manifests and stats.")
    parser.add_argument(
        "--dataset_id",
        type=str,
        action="append",
        default=[],
        help="Explicit dataset id(s) such as Clean/place_can_basket. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--dataset_glob",
        type=str,
        action="append",
        default=[],
        help="Shell-style matcher(s) over dataset ids such as Clean/* . Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--dataset_multiplier",
        type=str,
        action="append",
        default=[],
        help="Optional pattern=value overrides for sampling multipliers, e.g. Clean/*=1.0 Randomized/*=1.0.",
    )
    parser.add_argument("--max_tasks", type=int, default=0, help="Use at most N task directories (0 = all).")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Fraction of samples to use for norm stats.")
    parser.add_argument("--action_chunk", type=int, default=48, help="Action horizon used by training.")
    parser.add_argument("--action_dim", type=int, default=14, help="Target action/state dimension for padding.")
    parser.add_argument("--embodiment_id", type=int, default=0, help="Embodiment id used for delta-mask selection.")
    parser.add_argument("--text_len", type=int, default=512, help="Text encoder max sequence length.")
    parser.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu for T5 embedding generation.")
    parser.add_argument("--t5_folder_name", type=str, default="t5_embedding", help="Per-task cache directory name.")
    parser.add_argument("--overwrite_t5", action="store_true", help="Regenerate existing per-episode T5 embeddings.")
    parser.add_argument("--max_episodes", type=int, default=0, help="Maximum episodes per task for T5 generation (0 = all).")
    parser.add_argument("--skip_norm_stats", action="store_true", help="Skip norm_stats_delta.json generation.")
    parser.add_argument("--skip_t5", action="store_true", help="Skip per-task T5 embedding generation.")
    parser.add_argument("--norm_output_path", type=str, default=None, help="Override output path for norm_stats_delta.json.")
    args = parser.parse_args()

    selected_specs = select_robotwin_task_specs(
        robotwin_root=args.robotwin_root,
        dataset_ids=_parse_csv_or_repeatable(args.dataset_id),
        dataset_globs=_parse_csv_or_repeatable(args.dataset_glob),
    )
    if args.max_tasks > 0:
        selected_specs = selected_specs[: args.max_tasks]

    multipliers = resolve_dataset_multipliers(selected_specs, _parse_multiplier_overrides(args.dataset_multiplier))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "robotwin_task_dirs.json"
    manifest_path.write_text(json.dumps(describe_robotwin_selection(selected_specs, multipliers), indent=2))
    print(f"Discovered {len(selected_specs)} task directories. Manifest: {manifest_path}")

    norm_output_path = Path(args.norm_output_path).expanduser().resolve() if args.norm_output_path else (output_dir / "norm_stats_delta.json")
    if not args.skip_norm_stats:
        print(f"Computing norm stats for {len(selected_specs)} task directories ...")
        compute_norm_stats(
            data_paths=[str(spec.task_dir) for spec in selected_specs],
            output_path=norm_output_path,
            embodiment_id=args.embodiment_id,
            delta_mask=ALOHA_DELTA_MASK,
            sample_rate=args.sample_rate,
            action_chunk=args.action_chunk,
            action_dim=args.action_dim,
        )
    else:
        print("Skipping norm stats generation.")

    if not args.skip_t5:
        # Carried across tasks so the text encoder is loaded once.
        t5_encoder = None
        for spec in selected_specs:
            print(f"Generating T5 embeddings for {spec.dataset_id} ...")
            result = compute_t5_embeddings_for_dataset(
                repo_id=spec.repo_id,
                root=str(spec.task_dir),
                wan_path=args.wan_model_path,
                device=args.device,
                text_len=args.text_len,
                t5_folder_name=args.t5_folder_name,
                overwrite=args.overwrite_t5,
                max_episodes=args.max_episodes,
                encoder=t5_encoder,
            )
            t5_encoder = result["encoder"]
            print(
                f"Done {spec.dataset_id}: updated={result['updated']} skipped={result['skipped']} root={result['dataset_root']}"
            )
    else:
        print("Skipping T5 embedding generation.")

    print("\nPreparation complete.")
    print(f"Train config can use ROBOTWIN_ROOT={Path(args.robotwin_root).expanduser().resolve()}")
    print(f"Train config can use WAN_MODEL_DIR={Path(args.wan_model_path).expanduser().resolve()}")
    print(f"Train config can use ROBOTWIN_NORM_PATH={norm_output_path}")


if __name__ == "__main__":
    main()
