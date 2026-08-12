"""
Add episode-level T5 embedding cache to an existing LeRobot dataset.

What it does
------------
- For LeRobot v2 datasets, ensure each row in `meta/episodes.jsonl` has a `t5_embedding_path`
  pointing to `{dataset_root}/{t5_folder_name}/episode_XXXXXX.pt`.
- For LeRobot v3 datasets, read episode metadata from `meta/episodes/**/*.parquet`
  and write the same `{dataset_root}/{t5_folder_name}/episode_XXXXXX.pt` cache files.
- If the cache is missing, encode the episode instruction using WAN's text encoder,
  write the `.pt` file, and update v2 `meta/episodes.jsonl` in-place when that metadata exists.

The cache lets training and inference skip on-the-fly text encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, UMT5EncoderModel


def _load_jsonlines(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_v3_episode_rows(meta_dir: Path) -> List[Dict[str, Any]]:
    episode_rows: List[Dict[str, Any]] = []
    episode_paths = sorted((meta_dir / "episodes").rglob("*.parquet"))
    for episode_path in episode_paths:
        df = pd.read_parquet(episode_path, columns=["episode_index", "tasks"])
        episode_rows.extend(df.to_dict("records"))
    episode_rows.sort(key=lambda row: int(row["episode_index"]))
    return episode_rows


def _write_jsonlines_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _resolve_dataset_root(repo_id: str, root: Optional[str]) -> Path:
    if root is not None:
        root_path = Path(root).expanduser().resolve()
        if root_path.is_dir():
            return root_path
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    return Path(meta.root)


def _infer_repo_id(repo_id: Optional[str], root: Optional[str]) -> str:
    if repo_id is not None and str(repo_id).strip():
        return str(repo_id).strip()
    if root is None:
        raise ValueError("repo_id is required when root is not provided")
    return Path(root).expanduser().resolve().name


def _init_wan_t5_encoder_diffusers(
    wan_path: str,
    device: str,
    text_len: int = 512,
) -> Dict[str, Any]:
    text_encoder_path = os.path.join(wan_path, "text_encoder")
    tokenizer_path = os.path.join(wan_path, "tokenizer")
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(f"Diffusers text_encoder dir not found: {text_encoder_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Diffusers tokenizer dir not found: {tokenizer_path}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    text_encoder = UMT5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=dtype)
    text_encoder.eval().requires_grad_(False)
    text_encoder.to(device=device, dtype=dtype)
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "text_len": int(text_len),
    }


def _init_wan_t5_encoder(
    wan_path: str,
    device: str,
    text_len: int = 512,
) -> Tuple[str, Any]:
    diffusers_text_encoder = os.path.join(wan_path, "text_encoder")
    diffusers_tokenizer = os.path.join(wan_path, "tokenizer")

    if os.path.exists(diffusers_text_encoder) and os.path.exists(diffusers_tokenizer):
        return "diffusers", _init_wan_t5_encoder_diffusers(wan_path=wan_path, device=device, text_len=text_len)

    raise FileNotFoundError(
        "Unsupported Wan model layout. Expected the diffusers layout "
        f"({diffusers_text_encoder}, {diffusers_tokenizer})."
    )


def _encode_t5(encoder: Tuple[str, Any], instruction: str, device: str) -> torch.Tensor:
    encoder_type, encoder_obj = encoder
    if encoder_type == "diffusers":
        tokenizer = encoder_obj["tokenizer"]
        text_encoder = encoder_obj["text_encoder"]
        text_len = int(encoder_obj["text_len"])
        text_inputs = tokenizer(
            [instruction],
            padding="max_length",
            max_length=text_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        with torch.no_grad():
            last_hidden_state = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        seq_len = int(attention_mask.gt(0).sum(dim=1)[0].item())
        return last_hidden_state[0, :seq_len].detach().cpu()

    raise ValueError(f"Unknown encoder_type={encoder_type!r}")


def _episode_instruction_from_meta(ep_row: Dict[str, Any]) -> str:
    tasks = ep_row.get("tasks", None)
    if tasks is not None and not isinstance(tasks, str):
        try:
            if len(tasks) > 0 and isinstance(tasks[0], str):
                return tasks[0]
        except TypeError:
            pass
    task = ep_row.get("task", "")
    if isinstance(task, str):
        return task
    return str(task)


def compute_t5_embeddings_for_dataset(
    repo_id: Optional[str],
    root: Optional[str],
    wan_path: str,
    device: Optional[str] = None,
    text_len: int = 512,
    t5_folder_name: str = "t5_embedding",
    overwrite: bool = False,
    max_episodes: int = 0,
    encoder: Optional[Tuple[str, Any]] = None,
) -> Dict[str, Any]:
    """Cache one T5 embedding per episode instruction.

    Pass the ``encoder`` from a previous call's result dict to avoid reloading
    the text encoder for every dataset.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    repo_id = _infer_repo_id(repo_id=repo_id, root=root)
    dataset_root = _resolve_dataset_root(repo_id, root)
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    v3_meta_dir = dataset_root / "meta"
    v3_episode_paths = sorted((v3_meta_dir / "episodes").rglob("*.parquet"))
    if not episodes_path.exists() and not v3_episode_paths:
        raise FileNotFoundError(f"Could not find LeRobot episode metadata under {dataset_root / 'meta'}")

    out_dir = dataset_root / t5_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    is_v3 = not episodes_path.exists()
    episodes = _load_v3_episode_rows(v3_meta_dir) if is_v3 else _load_jsonlines(episodes_path)
    if max_episodes and max_episodes > 0:
        episodes = episodes[:max_episodes]

    updated = 0
    skipped = 0

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        rel = f"{t5_folder_name}/episode_{ep_idx:06d}.pt"
        abs_pt = dataset_root / rel

        has_ptr = ("t5_embedding_path" in ep) and isinstance(ep.get("t5_embedding_path"), str)
        ptr_ok = has_ptr and (Path(dataset_root) / str(ep["t5_embedding_path"])).exists()

        if not overwrite:
            if ptr_ok or abs_pt.exists():
                if not has_ptr and abs_pt.exists():
                    ep["t5_embedding_path"] = rel
                    updated += 1
                else:
                    skipped += 1
                continue

        instr = _episode_instruction_from_meta(ep)
        if encoder is None:
            print(f"Loading WAN T5 encoder from {wan_path} on {device} ...")
            encoder = _init_wan_t5_encoder(wan_path=wan_path, device=device, text_len=int(text_len))

        emb = _encode_t5(encoder, instr, device=device)
        torch.save(emb, abs_pt)
        ep["t5_embedding_path"] = rel
        updated += 1

        if updated % 10 == 0:
            print(f"Processed {updated} episodes (latest: {ep_idx})")

    if not is_v3:
        if max_episodes and max_episodes > 0:
            full = _load_jsonlines(episodes_path)
            patch_map = {int(ep["episode_index"]): ep for ep in episodes}
            for i, ep in enumerate(full):
                ep_idx = int(ep["episode_index"])
                if ep_idx in patch_map:
                    full[i] = patch_map[ep_idx]
            _write_jsonlines_atomic(episodes_path, full)
        else:
            _write_jsonlines_atomic(episodes_path, episodes)

    return dict(
        updated=updated,
        skipped=skipped,
        dataset_root=str(dataset_root),
        repo_id=repo_id,
        t5_folder_name=t5_folder_name,
        metadata_format="v3" if is_v3 else "v2",
        encoder=encoder,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Add WAN T5 embedding cache to a LeRobot dataset (episode-level pt + episodes.jsonl pointer)"
    )
    parser.add_argument("--repo_id", type=str, default=None, help="LeRobot dataset repo_id (defaults to basename(root)).")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset root (contains meta/data/videos). If omitted, LeRobot default cache is used.",
    )
    parser.add_argument(
        "--wan_path",
        type=str,
        default=None,
        help="Path to Wan2.2-TI2V-5B-Diffusers or the original Wan2.2-TI2V-5B directory.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument("--text_len", type=int, default=512, help="T5 text_len")
    parser.add_argument("--t5_folder_name", type=str, default="t5_embedding", help="Cache folder name under dataset root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files and meta pointers")
    parser.add_argument("--max_episodes", type=int, default=0, help="Process at most N episodes (0 = all)")
    args = parser.parse_args()

    if not args.wan_path:
        raise ValueError("WAN path not provided. Pass --wan_path <Wan2.2-TI2V-5B-Diffusers dir>.")

    result = compute_t5_embeddings_for_dataset(
        repo_id=args.repo_id,
        root=args.root,
        wan_path=args.wan_path,
        device=args.device,
        text_len=args.text_len,
        t5_folder_name=args.t5_folder_name,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
    )

    print(f"Done. updated={result['updated']}, skipped={result['skipped']}.")
    print(f"Dataset root: {result['dataset_root']}")


if __name__ == "__main__":
    main()
