<div align="center">

# FACT: Failure-Aware Causal Training<br>for World-Action Models

**One causal diffusion transformer — act, then imagine.**

[Quanquan Peng](https://pengqq.com)<sup>\*</sup> · [Yutong Liang](https://lyt0112.com)<sup>\*</sup> · [Rui Yan](https://jerryyan24.github.io) · [Nicklas Hansen](https://nicklashansen.com) · [Xiaolong Wang](https://xiaolonw.github.io)


[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-fact--wam.github.io-4b8bbe)](https://fact-wam.github.io/)
[![Paper](https://img.shields.io/badge/%F0%9F%93%84%20Paper-PDF-b31b1b)](https://arxiv.org/abs/2608.10232)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-fact--wam-ffd21e)](https://huggingface.co/Bariona/fact-wam)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Spaces-ff7c00)](https://huggingface.co/spaces/Bariona/fact-world-action-model)
[![License](https://img.shields.io/badge/License-Apache%202.0-3da639)](LICENSE)

</div>

## 📖 Overview

**FACT** is a causal world-action model: one causal diffusion transformer jointly denoises **robot actions**, a **task-progress value**, and **future video**, all conditioned on the executed action.

<p align="center">
  <img src="https://fact-wam.github.io/static/images/method/frame_14.png" width="92%" alt="FACT architecture: a shared causal diffusion transformer denoises action, value, and future-video tokens; value and future video condition on the clean action slot, not the noisy one.">
</p>

- **Act, then imagine.** Future video and value condition on the clean action, never the reverse — future prediction sharpens actions without leaking targets, and deployment decodes actions without generating video.
- **Failures teach consequences.** Failure rollouts skip the action-imitation loss but still supervise the observed failed future and a lowered value.
- **Optional best-of-N scoring.** The value head ranks sampled action candidates at inference.

This repository is the official implementation, containing the end-to-end RoboTwin pipeline: **data prep → training → inference → closed-loop simulator evaluation**.

> 🎮 **Try it live** — run the released checkpoint in your browser on Hugging Face Spaces: [Bariona/fact-world-action-model](https://huggingface.co/spaces/Bariona/fact-world-action-model)

| Path | What it is |
| --- | --- |
| `world_action_model/` | Model, trainer, inference pipeline, transforms, config (`configs/robotwin.py`) |
| `fact_train/`, `fact_datasets/` | Training harness and dataset library |
| `scripts/` | CLI entrypoints, run as `python -m scripts.<name>` from the repo root |
| `evaluation/robotwin/` | Closed-loop simulator evaluation |

## 🛠️ Installation

```bash
bash setup_env.sh        # conda env `fact`, pinned deps, Wan2.2 + RoboTwin download
conda activate fact
```

Already have the checkpoint or dataset: `SKIP_MODEL_DOWNLOAD=1 SKIP_DATA_DOWNLOAD=1 bash setup_env.sh`

## 📦 Model & Data Download

Equivalent of what `setup_env.sh` does:

```bash
# Wan2.2 base model
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir ./models/Wan2.2-TI2V-5B-Diffusers

# RoboTwin demonstrations
huggingface-cli download Bariona/robotwin-v2 robotwin-v2.tar \
  --repo-type dataset --local-dir ./datasets
tar -xf ./datasets/robotwin-v2.tar -C ./datasets   # -> datasets/RoboTwin/{Clean,Randomized}/<task>/
```

Trained FACT checkpoint ([`Bariona/fact-wam`](https://huggingface.co/Bariona/fact-wam)):

```bash
huggingface-cli download Bariona/fact-wam --local-dir ./models/fact-wam
```

## 📊 Data Preprocessing

**1. Norm stats + per-episode T5 embedding caches** (required; subset via `--dataset_glob 'Clean/*'`):

```bash
python -m scripts.prepare_robotwin \
  --robotwin_root ./datasets/RoboTwin \
  --wan_model_path ./models/Wan2.2-TI2V-5B-Diffusers \
  --output_dir ./artifacts/robotwin
```

**2. VAE latent cache** — training reads it by default; export `FACT_USE_CACHED_VAE_LATENTS=0` to train from raw video instead:

```bash
python -m scripts.compute_vae_latents --batch_size 32   # match training BATCH_SIZE_PER_GPU
```

**3. Dataloader check** (optional):

```bash
python -m scripts.test_dataloader --config world_action_model.configs.robotwin \
  --num_workers 0 --batch_size 2 --num_batches 2
```

## 🚀 Training

Edit the USER SETTINGS block at the top of `world_action_model/configs/robotwin.py`, then:

```bash
python -m scripts.train --config world_action_model.configs.robotwin.config
```

## ⚡ Inference

Serve a trained checkpoint (to use the released one instead, pass `--transformer_path ./models/fact-wam/transformer --stats_path ./models/fact-wam/norm_stats_delta.json`):

```bash
python -m scripts.inference_server \
  --model_id ./models/Wan2.2-TI2V-5B-Diffusers \
  --transformer_path ./experiments/robotwin/models/<checkpoint>/transformer \
  --stats_path ./artifacts/robotwin/norm_stats_delta.json \
  --port 8093
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--verbose --verbose_dir ./tmp/verbose` | Dump each request's input image and output action/value |
| `--return_images` | Decode and return predicted frames (needed for the client-side `VIS_DIR` video dump) |
| `--enable_prefix_cache` | KV-cache the fixed prefix |
| `--skip_future_state_value` | Action-only decoding |

## 🤖 RoboTwin Evaluation

Set up the simulator in its own conda env following [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/2eeec322d95799f537cbfe5f291a8220d965ccb8), then apply `git -C <robotwin> apply <fact>/evaluation/robotwin/robotwin_test_num.patch` so `TEST_NUM` takes effect (upstream hardcodes 100 episodes).

```bash
# settings live in evaluation/robotwin/launch_config.yml; point ROBOTWIN_PATH at your checkout
bash evaluation/robotwin/launch_server.sh                                # terminal 1
bash evaluation/robotwin/launch_client.sh beat_block_hammer demo_clean   # terminal 2
bash evaluation/robotwin/eval_all_tasks.sh demo_clean 50                 # ... or sweep all 50 tasks
```

Per-episode results land in RoboTwin's `eval_result/`; the sweep also writes per-task and average success rates to `./eval_runs/<config>_<timestamp>/`.

## 📝 Citation

If you find FACT useful, please consider citing:

```bibtex
@article{peng2026fact,
  title   = {FACT: Failure-Aware Causal Training for World-Action Models},
  author  = {Peng, Quanquan and Liang, Yutong and Yan, Rui and Hansen, Nicklas and Wang, Xiaolong},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🙏 Acknowledgements

FACT builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2), [giga-world-policy](https://github.com/open-gigaai/giga-world-policy), [GigaTrain](https://github.com/open-gigaai/giga-train), and [GigaDatasets](https://github.com/open-gigaai/giga-datasets); simulation experiments use the [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) benchmark. Thanks to the authors for open-sourcing their work.
