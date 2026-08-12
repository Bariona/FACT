import copy
import functools
import json
import math
import os
import shutil
import time

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from accelerate import DistributedType
from ..models.transformer_wa_casual import CasualWorldActionTransformer
from ..pipeline.utils import (
    add_state_to_action,
    build_teacher_forcing_per_token_timestep,
    denormalize_action,
    extract_normalization_tensors,
    is_robot_token_param,
    load_stats,
    normalize_state,
    pad_t5_embedding,
)
from einops import rearrange
from fact_datasets import load_dataset
from fact_train import Trainer, ModuleDict, build_optimizer
from fact_train.utils import as_list
from PIL import Image
import imageio
import numpy as np
import matplotlib.pyplot as plt
from diffusers.video_processor import VideoProcessor

import world_action_model.transformers  # noqa: F401 - registers project transforms for saved JSON configs
from .. import apply_numa_affinity, apply_runtime_compat
from ..image_layouts import build_robotwin_ref_tensor, get_robotwin_canvas_size
from ..pipeline.wa_pipeline import WAPipeline


def _resolve_torch_dtype(dtype_value, default_dtype: torch.dtype) -> torch.dtype:
    if dtype_value is None:
        return default_dtype
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    if isinstance(dtype_value, str):
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "float": torch.float32,
        }
        key = dtype_value.strip().lower()
        if key in dtype_map:
            return dtype_map[key]
    raise ValueError(f"Unsupported vae_dtype: {dtype_value!r}")


def _is_action_param_name(name: str) -> bool:
    return is_robot_token_param(name, module_prefix="transformer.")


def _is_no_decay_param(name: str, param: torch.Tensor) -> bool:
    # 1D tensors (LN/RMSNorm gain, bias, gates) and embedding-like / scale-shift tables
    # should not be regularised — applying weight decay to them hurts convergence.
    if param.ndim <= 1:
        return True
    if name.endswith(".bias"):
        return True
    lowered = name.lower()
    if "norm" in lowered:
        return True
    if "scale_shift_table" in name:
        return True
    if "pos_embed" in name:
        return True
    return False


def build_optimizer_param_groups(model: torch.nn.Module, base_lr: float, action_lr: float | None) -> list[dict]:
    if action_lr is None:
        return []

    buckets: dict[tuple[str, bool], list] = {
        ("action", False): [],
        ("action", True): [],
        ("base", False): [],
        ("base", True): [],
    }
    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        lr_key = "action" if _is_action_param_name(name) else "base"
        no_decay = _is_no_decay_param(name, param)
        buckets[(lr_key, no_decay)].append(param)

    param_groups = []
    for (lr_key, no_decay), params in buckets.items():
        if not params:
            continue
        lr = float(action_lr) if lr_key == "action" else float(base_lr)
        suffix = "_no_decay" if no_decay else ""
        group = {"params": params, "lr": lr, "name": f"{lr_key}{suffix}"}
        if no_decay:
            group["weight_decay"] = 0.0
        param_groups.append(group)
    return param_groups


class CasualWATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Before prepare() forks the dataloader workers and before any forward.
        apply_numa_affinity()
        apply_runtime_compat()
        super().__init__(*args, **kwargs)
        self._validation_dataset = None
        self._validation_pipe = None
        self._validation_stats = None
        self._validation_norm = None
        self._best_validation_action_mse = float("inf")
        self._best_validation_step = -1
        self._best_validation_checkpoint = ""

    def _get_loss_weights(self) -> dict[str, float]:
        loss_weights = self.kwargs.get("loss_weights", {})
        if loss_weights is None:
            return {}
        return {str(key): float(value) for key, value in dict(loss_weights).items()}

    def prepare(self, dataloaders, models, optimizers, schedulers) -> None:
        super().prepare(dataloaders, models, optimizers, schedulers)
        if bool(self.kwargs.get("enable_validation", False)):
            self._validation_dataset = self._build_validation_dataset()

    def _build_validation_dataset(self):
        data_config = self.kwargs.get("validation_data", None)
        if data_config is None:
            raise ValueError("enable_validation=True requires train.validation_data")
        data_config = copy.deepcopy(data_config)
        dataset = load_dataset(data_config["data_or_config"])
        if self.is_main_process:
            self.logger.info("Validation dataset size: %d", len(dataset))
        return dataset

    def _get_validation_pipe(self):
        if self._validation_pipe is not None:
            return self._validation_pipe
        unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        transformer = unwrapped["transformer"]
        pretrained = self.kwargs.get("validation_pretrained", None) or getattr(self, "pretrained", None) or self._validation_wan_model_path()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained,
            subfolder="scheduler",
            shift=float(self.kwargs.get("validation_flow_shift", self.flow_shift)),
        )
        pipe = WAPipeline(
            tokenizer=None,
            text_encoder=None,
            vae=self.vae,
            scheduler=scheduler,
            transformer=transformer,
            expand_timesteps=self.expand_timesteps,
        )
        flow_shift = float(self.kwargs.get("validation_flow_shift", self.flow_shift))
        self._set_pipeline_flow_shift(pipe, flow_shift)
        pipe = pipe.to(self.device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        pipe.text_encoder = None
        pipe.tokenizer = None
        self._validation_pipe = pipe
        return pipe

    def _validation_wan_model_path(self) -> str:
        vae_name_or_path = getattr(getattr(self.vae, "config", None), "_name_or_path", "")
        if vae_name_or_path:
            return os.path.dirname(str(vae_name_or_path))
        raise ValueError("validation_pretrained must be set when the VAE config has no _name_or_path")

    @staticmethod
    def _set_pipeline_flow_shift(pipe, flow_shift: float) -> None:
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

    def _get_validation_norm(self):
        if self._validation_norm is not None:
            return self._validation_norm
        stats_path = self.kwargs.get("validation_stats_path", None)
        if stats_path is None:
            raise ValueError("validation_stats_path is required for validation")
        action_dim = int(self.kwargs.get("validation_action_dim", self.action_dim))
        self._validation_stats = load_stats(str(stats_path))
        self._validation_norm = extract_normalization_tensors(
            self._validation_stats,
            device=self.device,
            state_dim=action_dim,
            action_dim=action_dim,
        )
        return self._validation_norm

    def _checkpoint_due(self) -> bool:
        if self._by_epoch and self.checkpoint_interval < self.max_epochs:
            checkpoint_interval = int(self.checkpoint_interval * self.epoch_size)
        else:
            checkpoint_interval = int(self.checkpoint_interval)
        return self.cur_step % checkpoint_interval == 0 or self.cur_step == self.max_steps

    def _current_checkpoint_dir(self) -> str:
        output_name = "checkpoint_epoch_{}_step_{}".format(self.cur_epoch, self.cur_step)
        return os.path.join(self.model_dir, output_name)

    def _best_validation_checkpoint_dir(self) -> str:
        return os.path.join(self.model_dir, "best_validation_action_mse")

    def _sync_best_validation_from_metadata(self) -> None:
        metadata_path = os.path.join(self._best_validation_checkpoint_dir(), "best_validation_metrics.json")
        if not os.path.isfile(metadata_path):
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            action_mse = float(metadata["value"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if math.isfinite(action_mse) and action_mse < self._best_validation_action_mse:
            self._best_validation_action_mse = action_mse
            self._best_validation_step = int(metadata.get("step", -1))
            self._best_validation_checkpoint = str(metadata.get("source_checkpoint", ""))

    def _strip_optimizer_from_checkpoint_dir(self, checkpoint_dir: str) -> None:
        if self.distributed_type == DistributedType.DEEPSPEED:
            optimizer_dir = os.path.join(checkpoint_dir, "pytorch_model")
            if os.path.exists(optimizer_dir):
                shutil.rmtree(optimizer_dir)
            return
        if self.distributed_type == DistributedType.FSDP:
            for i in range(len(self.optimizers)):
                optimizer_name = "optimizer.bin" if i == 0 else f"optimizer_{i}.bin"
                optimizer_path = os.path.join(checkpoint_dir, optimizer_name)
                if os.path.exists(optimizer_path):
                    os.remove(optimizer_path)
                optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{i}")
                if os.path.exists(optimizer_path):
                    shutil.rmtree(optimizer_path)
            return
        for i in range(len(self.optimizers)):
            optimizer_name = "optimizer.bin" if i == 0 else f"optimizer_{i}.bin"
            optimizer_path = os.path.join(checkpoint_dir, optimizer_name)
            if os.path.exists(optimizer_path):
                os.remove(optimizer_path)

    def _maybe_save_best_validation_checkpoint(self, metrics: dict[str, float]) -> None:
        self._sync_best_validation_from_metadata()
        action_mse = float(metrics["val/action_mse"])
        if not math.isfinite(action_mse) or action_mse >= self._best_validation_action_mse:
            return

        source_dir = self._current_checkpoint_dir()
        if not os.path.isdir(source_dir):
            self.logger.info("Skip best validation checkpoint: source checkpoint is missing: %s", source_dir)
            return

        best_dir = self._best_validation_checkpoint_dir()
        tmp_dir = best_dir + ".tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        shutil.copytree(source_dir, tmp_dir)
        if not self.checkpoint_save_optimizer:
            self._strip_optimizer_from_checkpoint_dir(tmp_dir)

        metadata = {
            "metric": "val/action_mse",
            "value": action_mse,
            "step": int(self.cur_step),
            "epoch": int(self.cur_epoch),
            "source_checkpoint": os.path.basename(source_dir),
            "metrics": {key: float(value) for key, value in metrics.items()},
        }
        with open(os.path.join(tmp_dir, "best_validation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")

        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.replace(tmp_dir, best_dir)

        self._best_validation_action_mse = action_mse
        self._best_validation_step = int(self.cur_step)
        self._best_validation_checkpoint = os.path.basename(source_dir)
        self.logger.info(
            "New best validation checkpoint: action_mse=%.6f step=%d path=%s",
            action_mse,
            int(self.cur_step),
            best_dir,
        )

    def save_checkpoint_step(self) -> None:
        should_eval = bool(self.kwargs.get("enable_validation", False)) and self._checkpoint_due()
        super().save_checkpoint_step()
        if should_eval:
            self.accelerator.wait_for_everyone()
            self.run_validation_step()
            self.accelerator.wait_for_everyone()

    def run_validation_step(self) -> None:
        if not self.is_main_process:
            return
        if self._validation_dataset is None:
            return

        model_was_training = self.model.training
        vae_was_training = self.vae.training
        self.model.eval()
        self.vae.eval()
        try:
            metrics = self._run_validation()
            self.accelerator.log(metrics, self.cur_step)
            self._maybe_save_best_validation_checkpoint(metrics)
            self.logger.info(
                "Validation step=%d action_mse=%.6f action_norm_mse=%.6f samples=%d forward_ms=%.2f",
                self.cur_step,
                metrics["val/action_mse"],
                metrics["val/action_norm_mse"],
                int(metrics["val/num_samples"]),
                metrics["val/model_forward_ms_mean"],
            )
        finally:
            if model_was_training:
                self.model.train()
            if vae_was_training:
                self.vae.train()

    def _run_validation(self) -> dict[str, float]:
        pipe = self._get_validation_pipe()
        norm = self._get_validation_norm()
        action_dim = int(self.kwargs.get("validation_action_dim", self.action_dim))
        action_chunk = int(self.kwargs.get("validation_action_chunk", 48))
        dst_size = tuple(int(v) for v in self.kwargs.get("validation_dst_size", (256, 192)))
        dst_width, dst_height = get_robotwin_canvas_size(dst_size)
        view_keys = [str(v) for v in self.kwargs.get("validation_view_keys", [])]
        t5_len = int(self.kwargs.get("validation_t5_len", 64))
        image_num_frames = int(self.kwargs.get("validation_image_num_frames", 5))
        max_samples = int(self.kwargs.get("validation_max_samples", 256))
        num_inference_steps = int(self.kwargs.get("validation_num_inference_steps", 10))
        guidance_scale = float(self.kwargs.get("validation_guidance_scale", 1.0))
        skip_future_state_value = bool(self.kwargs.get("validation_skip_future_state_value", True))
        enable_prefix_cache = bool(self.kwargs.get("validation_enable_prefix_cache", False))
        delta_mask = torch.as_tensor(
            self.kwargs.get("validation_delta_mask", [False] * action_dim),
            dtype=torch.float32,
            device=self.device,
        )[:action_dim]
        if delta_mask.numel() < action_dim:
            delta_mask = torch.nn.functional.pad(delta_mask, (0, action_dim - int(delta_mask.numel())), value=0.0)

        action_mses = []
        action_norm_mses = []
        forward_ms_values = []
        rng = np.random.default_rng(int(self.kwargs.get("validation_seed", self.seed)) + int(self.cur_step))
        val_len = len(self._validation_dataset)
        if max_samples > 0 and val_len > max_samples:
            selected_indices = rng.choice(val_len, size=max_samples, replace=False).tolist()
        else:
            selected_indices = list(range(val_len))
        sample_count = len(selected_indices)

        with torch.no_grad():
            for sample_index in selected_indices:
                sample = self._validation_dataset[int(sample_index)]
                state_raw = torch.as_tensor(sample["observation.state"], dtype=torch.float32)
                if state_raw.ndim == 1:
                    state_raw = state_raw.unsqueeze(0)
                state_raw = state_raw[:1, :action_dim]
                norm_state = normalize_state(state_raw.to(self.device), stats=norm, mode="zscore").to(dtype=self.dtype)

                ref_dict = {key: torch.as_tensor(sample[key])[0] for key in view_keys}
                ref_image = build_robotwin_ref_tensor(ref_dict, main_dst_size=dst_size)
                prompt_embeds = (
                    pad_t5_embedding(torch.as_tensor(sample["t5_embedding"]), target_len=t5_len)
                    .unsqueeze(0)
                    .to(device=self.device, dtype=torch.float32)
                )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                forward_t0 = time.perf_counter()
                _imgs, pred_action_norm, _pred_future_state_norm, _pred_value_norm = pipe(
                    image=ref_image,
                    height=dst_height,
                    width=dst_width,
                    action_chunk=action_chunk,
                    state=norm_state,
                    num_frames=image_num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    action_only=True,
                    return_dict=False,
                    prompt_embeds=prompt_embeds,
                    enable_prefix_cache=enable_prefix_cache,
                    skip_future_state_value=skip_future_state_value,
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                forward_ms_values.append((time.perf_counter() - forward_t0) * 1000.0)

                pred_action_raw = denormalize_action(pred_action_norm.squeeze(0).float(), norm, mode="zscore").cpu()
                pred_abs = add_state_to_action(pred_action_raw, state_raw, action_chunk=action_chunk, mask=delta_mask.cpu())
                pred_abs_np = pred_abs.numpy()
                gt_action = np.asarray(sample["action"][:action_chunk], dtype=np.float32)

                gt_delta = torch.as_tensor(gt_action[:action_chunk, :action_dim], dtype=torch.float32).clone()
                delta_mask_bool = delta_mask.cpu().to(dtype=torch.bool)
                if delta_mask_bool.any():
                    gt_delta[:, delta_mask_bool] = gt_delta[:, delta_mask_bool] - state_raw[:, delta_mask_bool]
                gt_action_norm = ((gt_delta.to(self.device) - norm.action_mean) / norm.action_std.clamp_min(1e-8)).cpu()
                pred_action_norm_cpu = pred_action_norm.squeeze(0).float().cpu()[
                    : gt_action_norm.shape[0],
                    : gt_action_norm.shape[1],
                ]

                t_common = min(gt_action.shape[0], pred_abs_np.shape[0])
                action_mses.append(float(((gt_action[:t_common] - pred_abs_np[:t_common]) ** 2).mean()))
                action_norm_mses.append(float(((gt_action_norm[:t_common] - pred_action_norm_cpu[:t_common]) ** 2).mean()))

        if sample_count == 0:
            return {
                "val/action_mse": float("nan"),
                "val/action_norm_mse": float("nan"),
                "val/num_samples": 0.0,
                "val/model_forward_ms_mean": float("nan"),
                "val/model_forward_ms_median": float("nan"),
            }
        return {
            "val/action_mse": float(np.mean(action_mses)),
            "val/action_norm_mse": float(np.mean(action_norm_mses)),
            "val/num_samples": float(sample_count),
            "val/model_forward_ms_mean": float(np.mean(forward_ms_values)),
            "val/model_forward_ms_median": float(np.median(forward_ms_values)),
        }

    def state_dict(self):
        state = super().state_dict()
        state.update(
            best_validation_action_mse=float(self._best_validation_action_mse),
            best_validation_step=int(self._best_validation_step),
            best_validation_checkpoint=str(self._best_validation_checkpoint),
        )
        return state

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        self._best_validation_action_mse = float(state_dict.get("best_validation_action_mse", float("inf")))
        self._best_validation_step = int(state_dict.get("best_validation_step", -1))
        self._best_validation_checkpoint = str(state_dict.get("best_validation_checkpoint", ""))

    def parse_losses(self, losses):
        if not isinstance(losses, dict):
            return super().parse_losses(losses)

        assert "total_loss" not in losses
        # Keys starting with "_aux_" are diagnostics (e.g. action_loss_mask_rate). They
        # flow through the same gather / log-interval path as losses but are
        # NOT multiplied by any weight and NOT summed into the training loss or
        # total_loss. Keeps monitoring cheap without changing training dynamics.
        aux_keys = [k for k in losses.keys() if k.startswith("_aux_")]
        loss_keys = [k for k in losses.keys() if not k.startswith("_aux_")]

        loss_weights = self._get_loss_weights()
        reduced = [losses[k].mean() for k in loss_keys]
        device = reduced[0].device
        dtype = reduced[0].dtype
        reduced_aux = [losses[k].mean().to(device=device, dtype=dtype) for k in aux_keys]
        weights = torch.tensor(
            [loss_weights.get(k, 1.0) for k in loss_keys], device=device, dtype=dtype
        )

        loss = (torch.stack(reduced) * weights).sum()

        # Single gather across losses + aux metrics (N+2 individual gathers → 1).
        # On single-GPU accelerate.gather is an identity; the reshape still holds.
        all_keys = loss_keys + aux_keys
        stacked = torch.stack(reduced + reduced_aux)
        gathered = self.accelerator.gather(stacked).view(-1, len(all_keys)).mean(dim=0)
        total_loss = (gathered[: len(loss_keys)] * weights).sum()

        # On-device accumulator — avoids per-step .item()/.any() sync points. Only
        # syncs once per log_interval when we need to hand values to print_step.
        if not getattr(self, "_loss_accum", None):
            self._loss_accum = {
                k: {"sum": torch.zeros((), device=device, dtype=dtype), "num": 0}
                for k in all_keys + ["total_loss"]
            }
        # Lazily initialize entries for keys that appear for the first time
        # after _loss_accum was created (keeps the stack invariant at log-sync).
        for k in all_keys:
            if k not in self._loss_accum:
                self._loss_accum[k] = {
                    "sum": torch.zeros((), device=device, dtype=dtype),
                    "num": 0,
                }
        for idx, k in enumerate(all_keys):
            self._loss_accum[k]["sum"] = self._loss_accum[k]["sum"] + gathered[idx]
            self._loss_accum[k]["num"] += 1
        self._loss_accum["total_loss"]["sum"] = (
            self._loss_accum["total_loss"]["sum"] + total_loss
        )
        self._loss_accum["total_loss"]["num"] += 1

        if self.cur_step % self.log_interval == 0:
            accum_keys = list(self._loss_accum.keys())
            summed = torch.stack([self._loss_accum[k]["sum"] for k in accum_keys])
            vals = summed.detach().cpu().tolist()  # single GPU->CPU sync per interval
            total_val = None
            for k, v in zip(accum_keys, vals):
                entry = self._loss_accum[k]
                if k not in self._outputs:
                    self._outputs[k] = {"sum": 0.0, "num": 0}
                self._outputs[k]["sum"] += v
                self._outputs[k]["num"] += entry["num"]
                entry["sum"] = torch.zeros_like(entry["sum"])
                entry["num"] = 0
                if k == "total_loss":
                    total_val = v / max(self._outputs[k]["num"], 1)

            loss_nan_total_limit = self.kwargs.get("loss_nan_total_limit", 100)
            if total_val is not None and math.isnan(total_val):
                if loss_nan_total_limit > 0:
                    self._loss_nan_count += 1
                    if self._loss_nan_count > loss_nan_total_limit:
                        exit(-1)
            else:
                self._loss_nan_count = 0

        return loss

    def get_models(self, model_config):
        pretrained = get_model_path(model_config.pretrained)
        self.pretrained = pretrained
        self.flow_shift = model_config.flow_shift
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.action_repeats = model_config.get("action_repeats", 1)
        self.state_repeats = model_config.get("state_repeats", 1)
        self.action_dim = int(model_config.get("action_dim", 14))
        self.enable_train_vis = bool(model_config.get("enable_train_vis", True))
        self.view_interval = int(model_config.get("view_interval", 50))
        self.view_dir = model_config.view_dir
        model = dict()
        # vae
        vae_pretrained = model_config.get('vae_pretrained', os.path.join(pretrained, 'vae'))
        vae_dtype = _resolve_torch_dtype(model_config.get('vae_dtype', None), self.dtype)
        vae = AutoencoderKLWan.from_pretrained(vae_pretrained)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype)
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        # transformer
        transformer_pretrained = model_config.get('transformer_pretrained', os.path.join(pretrained, 'transformer'))
        if model_config.get("unpretrain", False):
            print("Load unet from config only.")
            transformer = CasualWorldActionTransformer.from_config(transformer_pretrained, torch_dtype=self.dtype)
        else:
            transformer = CasualWorldActionTransformer.from_pretrained(transformer_pretrained, torch_dtype=self.dtype)

        transformer.reset_action_heads(self.action_dim)
        transformer_cfg = model_config.get('transformer', dict())
        transformer = process_transformer(transformer, transformer_cfg)
        attention_backend = getattr(transformer, "self_attention_implementation", "unknown")
        flex_block_size = getattr(transformer, "flex_attention_block_size", None)
        self.logger.info(
            "Transformer self-attention backend: %s%s",
            attention_backend,
            f" (block_size={flex_block_size})" if attention_backend == "flex" and flex_block_size is not None else "",
        )
        robot_adapter_rank = int(model_config.get("robot_adapter_rank", transformer_cfg.get("robot_adapter_rank", 0)))
        transformer.set_robot_adapter_rank(robot_adapter_rank)
        transformer.to(self.device, dtype=self.dtype)
        model.update(transformer=transformer)
        # model
        checkpoint = model_config.get('checkpoint', None)
        strict = model_config.get('strict', True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.train()
        return model

    def get_optimizers(self, optimizers):
        optimizers = as_list(optimizers)
        for i in range(len(optimizers)):
            if not isinstance(optimizers[i], dict):
                continue
            if len(optimizers) != 1 or len(self.models) != 1:
                params = []
                if len(optimizers) == 1 and len(self.models) > 1:
                    for model in self.models:
                        params += list(model.parameters())
                elif len(optimizers) == len(self.models):
                    params = self.models[i].parameters()
                else:
                    assert False
                params = list(filter(lambda p: p.requires_grad, params))
                optimizers[i] = build_optimizer(optimizers[i], params=params)
                continue

            optimizer_cfg = copy.deepcopy(optimizers[i])
            action_lr = optimizer_cfg.pop("action_lr", None)
            if action_lr is None:
                params = list(filter(lambda p: p.requires_grad, self.models[0].parameters()))
                optimizers[i] = build_optimizer(optimizer_cfg, params=params)
                continue

            base_lr = float(optimizer_cfg["lr"])
            param_groups = build_optimizer_param_groups(
                model=self.models[0],
                base_lr=base_lr,
                action_lr=float(action_lr),
            )
            if not param_groups:
                params = list(filter(lambda p: p.requires_grad, self.models[0].parameters()))
                optimizers[i] = build_optimizer(optimizer_cfg, params=params)
                continue

            optimizers[i] = build_optimizer(optimizer_cfg, params=param_groups)

            if self.is_main_process:
                for group in optimizers[i].param_groups:
                    group_name = str(group.get("name", "unnamed"))
                    tensors = len(group["params"])
                    num_params = sum(int(param.numel()) for param in group["params"])
                    self.logger.info(
                        "Optimizer group %s: lr=%.2e tensors=%d params=%d",
                        group_name,
                        float(group["lr"]),
                        tensors,
                        num_params,
                    )
        return optimizers

    def forward_step(self, batch_dict):
        transformer = functools.partial(self.model, 'transformer')
        # Support two input paths:
        #   (1) legacy: batch_dict['images'] -> forward_vae(images) on the fly
        #   (2) cached: batch_dict['visual_latents'] produced by compute_vae_latents.py
        cached_latents = 'visual_latents' in batch_dict and batch_dict.get('visual_latents', None) is not None
        if cached_latents:
            # The cached path reuses visual_latents[:, :, :1] as ref_latents and
            # mask-blends, which only matches the expand_timesteps branch below.
            # The legacy (channel-concat) branch would feed a different in_channels
            # to the transformer, so refuse to train silently on a mismatched config.
            assert self.expand_timesteps, (
                "cached VAE latents require expand_timesteps=True; the legacy "
                "channel-concat ref path is not supported with cached latents."
            )
            visual_latents = batch_dict['visual_latents']
            # self.vae is always loaded in get_models (we keep the decoder for
            # visualization), so dispatch to its dtype directly.
            visual_latents = visual_latents.to(self.device, dtype=self.vae.dtype)
            bs = visual_latents.shape[0]
            # In the latent space the rank is B,z,T_lat,H_lat,W_lat = 5 dims.
            sample_ndim = visual_latents.ndim
        else:
            images = batch_dict['images']
            bs = images.shape[0]
            sample_ndim = images.ndim
        prompt_embeds = batch_dict['prompt_embeds']
        timestep, sigma = self.get_timestep_and_sigma(bs, sample_ndim)
        # Teacher forcing uses one action track: the gt_action condition token that
        # future_state / value / future_image attend to, and the denoising target
        # for pred_action.
        action = batch_dict['action']
        state = batch_dict['state']
        future_state = batch_dict['future_state']
        value = batch_dict['value_normalized']
        self.vae_decode(action=action, sign='input_action')
        if self.state_repeats > 1:
            state = state.repeat(1, self.state_repeats, 1)
        if self.action_repeats > 1:
            action = action.repeat(1, self.action_repeats, 1)
        # inputs
        if not cached_latents:
            visual_latents = self.forward_vae(images)
        self.vae_decode(latents=visual_latents, sign='input_visual')
        visual_noise = visual_latents[:, :, :1] + torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * sigma + visual_latents * (1 - sigma)
        action_sigma = sigma.squeeze(-1).squeeze(-1)
        action_noise = torch.randn_like(action)
        action_target = action_noise - action
        noisy_action = action_noise * action_sigma + action * (1 - action_sigma)
        future_state_noise = torch.randn_like(future_state)
        future_state_target = future_state_noise - future_state
        noisy_future_state = future_state_noise * action_sigma + future_state * (1 - action_sigma)
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        if value.dim() == 2:
            value = value.unsqueeze(1)
        value_noise = torch.randn_like(value)
        value_target = value_noise - value
        noisy_value = value_noise * action_sigma + value * (1 - action_sigma)
        # loss
        prompt_embeds = prompt_embeds.to(self.dtype)
        # ref_latents path: either we have live ref_images from the image-decoding
        # dataloader, or we're using cached visual_latents (in which case we always
        # take the expand_timesteps branch that reuses visual_latents[:, :, :1]).
        has_ref_path = cached_latents or ('ref_images' in batch_dict)
        if has_ref_path:
            if not cached_latents and not self.expand_timesteps:
                ref_images = batch_dict['ref_images']
                ref_latents = self.forward_vae(ref_images)
                num_frames = images.shape[1]
                batch_size = ref_latents.shape[0]
                latent_height = ref_latents.shape[-2]
                latent_width = ref_latents.shape[-1]
                mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
                mask_lat_size[:, :, list(range(1, num_frames))] = 0
                first_frame_mask = mask_lat_size[:, :, 0:1]
                first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
                mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
                mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height,
                                                   latent_width)
                mask_lat_size = mask_lat_size.transpose(1, 2)
                mask_lat_size = mask_lat_size.to(ref_latents.device)
                condition = torch.concat([mask_lat_size, ref_latents], dim=1)
                noisy_latents = torch.concat([noisy_latents, condition], dim=1)
            else:
                num_latent_frames = visual_latents.shape[2]
                latent_height = visual_latents.shape[-2]
                latent_width = visual_latents.shape[-1]
                if not cached_latents:
                    ref_images = batch_dict['ref_images']
                    ref_first_frame = ref_images[:, :1]
                    if torch.equal(ref_first_frame, batch_dict['images'][:, :1]):
                        # Clean ref path: Wan VAE is causal in time, so encoding only the
                        # first frame is bit-exact equal to visual_latents[:, :, :1]. Reuse
                        # that latent to skip an extra VAE encode when augmentation is off.
                        if not getattr(self, "_ref_latent_reuse_verified", False):
                            with torch.no_grad():
                                reencoded = self.forward_vae(ref_first_frame)
                            assert torch.equal(reencoded, visual_latents[:, :, :1]), (
                                "ref_latents reuse is unsafe: forward_vae(ref_images[:, :1]) "
                                "!= visual_latents[:, :, :1]. Check MaskGenerator "
                                "(max_ref_frames=1, start=1) and Wan VAE causality."
                            )
                            self._ref_latent_reuse_verified = True
                        ref_latents = visual_latents[:, :, :1]
                    else:
                        # Augmented ref path: condition on the augmented first frame while
                        # keeping the future frames/targets clean.
                        ref_latents = self.forward_vae(ref_first_frame)
                else:
                    ref_latents = visual_latents[:, :, :1]
                first_frame_mask = torch.ones(
                    bs, 1, num_latent_frames, latent_height, latent_width, dtype=visual_latents.dtype, device=visual_latents.device
                )
                first_frame_mask[:, :, 0] = 0
                insert_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * noisy_latents
                # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                temp_ts = (first_frame_mask[:, :, :, ::2, ::2] * timestep[:, None, None, None, None]).reshape(bs, -1)
                # batch_size, seq_len
                timestep = temp_ts
        insert_noisy_latents = insert_noisy_latents.to(self.dtype)
        noise_t = timestep[:, -2:-1]
        noisy_action = noisy_action.to(self.dtype)
        gt_action_cond = action.to(self.dtype)
        state = state.to(self.dtype)
        noisy_future_state = noisy_future_state.to(self.dtype)
        noisy_value = noisy_value.to(self.dtype)
        ref_latents = insert_noisy_latents[:, :, :1]
        noisy_latents = insert_noisy_latents[:, :, 1:]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        timestep = build_teacher_forcing_per_token_timestep(
            batch_size=bs,
            num_state_tokens=int(state.shape[1]),
            num_ref_tokens=frame_per_tokens,
            num_pred_action_tokens=int(noisy_action.shape[1]),
            num_gt_action_tokens=int(gt_action_cond.shape[1]),
            num_future_state_tokens=int(noisy_future_state.shape[1]),
            num_value_tokens=int(noisy_value.shape[1]),
            num_noisy_latent_tokens=frame_per_tokens * (int(first_frame_mask.shape[2]) - 1),
            noise_t=noise_t,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        visual_pred, action_pred, future_state_pred, value_pred = transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action,
            gt_action=gt_action_cond,
            state=state,
            future_state=noisy_future_state,
            value=noisy_value,
        )
        if self.if_visualize():
            with torch.no_grad():
                pred_x0 = noisy_latents - visual_pred * sigma
                if self.expand_timesteps:
                    pred_x0 = (1 - first_frame_mask) * ref_latents + first_frame_mask * pred_x0
                self.vae_decode(latents=pred_x0, sign='pred_visual')
                pred_action = noisy_action - action_pred * action_sigma
                pred_future_state = noisy_future_state - future_state_pred * action_sigma
                pred_value = noisy_value - value_pred * action_sigma
                if self.action_repeats > 1:
                    pred_action = pred_action.reshape(bs, self.action_repeats, -1, 14)
                    pred_action = pred_action.mean(1)
                self.vae_decode(action=pred_action, sign='action_visual')
                if self.process_index == 0:
                    self.logger.info(
                        "train vis value_norm gt=%.4f pred=%.4f future_state_mse=%.6f",
                        float(value[0, 0, 0].detach().float().cpu().item()),
                        float(pred_value[0, 0, 0].detach().float().cpu().item()),
                        float(torch.mean((pred_future_state[0].float() - future_state[0].float()) ** 2).detach().cpu().item()),
                    )
        # action_loss_mask gates the action loss for real failure episodes, whose
        # actions should not be imitated. Value trains on all samples so failures
        # teach higher cost (via value_penalty_scale).
        action_loss_mask = batch_dict.get('action_loss_mask', None)

        def _masked_mean(per_sample_loss, mask):
            if mask is None:
                return per_sample_loss.mean()
            m = mask.to(per_sample_loss.device, dtype=per_sample_loss.dtype).reshape(-1)
            return (per_sample_loss * m).sum() / m.sum().clamp_min(1e-8)

        visual_loss_raw = ((visual_pred.float() - visual_target.float()) * first_frame_mask) ** 2
        visual_loss = visual_loss_raw.flatten(1).mean(dim=1).mean()
        action_loss_raw = (action_pred.float() - action_target.float()) ** 2
        action_loss = _masked_mean(action_loss_raw.flatten(1).mean(dim=1), action_loss_mask)
        future_state_loss_raw = (future_state_pred.float() - future_state_target.float()) ** 2
        future_state_loss = future_state_loss_raw.flatten(1).mean(dim=1).mean()
        value_loss = (value_pred.float() - value_target.float()) ** 2
        value_loss = value_loss.mean()
        loss = {
            'visual_loss': visual_loss,
            'action_loss': action_loss,
            'future_state_loss': future_state_loss,
            'value_loss': value_loss,
        }
        # Diagnostic: fraction of this batch coming from failure episodes. The
        # "_aux_" prefix makes parse_losses log it without training on it.
        if action_loss_mask is not None:
            with torch.no_grad():
                m = action_loss_mask.to(device=value_loss.device, dtype=torch.float32).reshape(-1)
                loss['_aux_action_loss_mask_rate'] = (1.0 - m.mean()).detach()
        return loss

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, 'b t c h w -> b c t h w')
            latents = self.vae.encode(images).latent_dist.mode()
        latents = (latents - self.latents_mean) * self.latents_std
        return latents

    def get_timestep_and_sigma(self, batch_size, ndim):
        sigma = torch.rand(batch_size).to(self.device)
        # flow_shift: 5.0 for 720P, 3.0 for 480P
        sigma = self.flow_shift * sigma / (1 + (self.flow_shift - 1) * sigma)
        timestep = torch.round(sigma * 1000).long()
        sigma = timestep.float() / 1000
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return timestep, sigma

    def if_visualize(self):
        if not self.enable_train_vis:
            return False
        return self.process_index == 0 and (self.cur_step == 1 or self.cur_step % self.view_interval == 0)

    def vae_decode(self, latents=None, action=None, images=None, sign=None, return_tensor=False):
        if self.if_visualize():
            save_dir = os.path.join(self.view_dir, "images", "{}".format(self.cur_step))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "{}.mp4".format(sign))
            if latents is not None:
                latents = latents.to(self.vae.dtype)
                latents = latents / self.latents_std + self.latents_mean
                with torch.no_grad():
                    tensor_video = self.vae.decode(latents, return_dict=False)[0].detach()
                video = self.video_processor.postprocess_video(tensor_video, output_type='pil')[0]
                vis_images = video
                imageio.mimsave(save_path, vis_images, fps=16)
                if return_tensor:
                    return tensor_video
                return vis_images
            if images is not None:
                image_tensor = images
                # [T, 3, H, W] to video
                images = (images + 1.0) / 2.0 * 255
                images = images.astype(np.uint8)
                images = [Image.fromarray(images[i]) for i in range(images.shape[0])]
                imageio.mimsave(save_path, images, fps=16)
                return image_tensor
            if action is not None:
                # action: [B, T, D]
                action = action.float().detach().cpu().numpy()
                T = action.shape[1]
                plot_dims = min(int(action.shape[2]), 32)
                cols = 4
                rows = (plot_dims + cols - 1) // cols
                fig = plt.figure(figsize=(cols * 3, max(1, rows) * 2.5))
                # plot sub plots with [D, T]
                for i in range(plot_dims):
                    plt.subplot(rows, cols, i + 1)
                    plt.plot(range(T), action[0, :, i])
                    plt.title("Dim {}".format(i))
                plt.tight_layout()
                save_path = os.path.join(save_dir, "{}.png".format(sign))
                plt.savefig(save_path)
                plt.close(fig)
                return action


def process_transformer(transformer, transformer_cfg):
    in_channels = transformer_cfg.get('in_channels', transformer.config.in_channels)
    if transformer.config.in_channels != in_channels:
        assert False
    num_checkpointing = transformer_cfg.get('num_checkpointing', None)
    if num_checkpointing is not None:
        transformer.enable_gradient_checkpointing()
        transformer.num_checkpointing = num_checkpointing
    self_attention_implementation = transformer_cfg.get("self_attention_implementation", None)
    if self_attention_implementation is not None and hasattr(transformer, "set_self_attention_implementation"):
        transformer.set_self_attention_implementation(self_attention_implementation)

    flex_attention_block_size = transformer_cfg.get("flex_attention_block_size", None)
    if flex_attention_block_size is not None and hasattr(transformer, "set_flex_attention_block_size"):
        transformer.set_flex_attention_block_size(flex_attention_block_size)

    flex_attention_kernel_options = transformer_cfg.get("flex_attention_kernel_options", None)
    if hasattr(transformer, "set_flex_attention_kernel_options"):
        transformer.set_flex_attention_kernel_options(flex_attention_kernel_options)
    return transformer

def get_model_path(model_name_or_path):
    if model_name_or_path is None or os.path.exists(model_name_or_path):
        return model_name_or_path
    raise ValueError(f'{model_name_or_path} does not exist')
