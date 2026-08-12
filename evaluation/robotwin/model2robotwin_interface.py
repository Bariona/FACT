from __future__ import annotations

import atexit
import os
from copy import deepcopy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from world_action_model.sockets import RobotInferenceClient


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _PlainLogStream:
    """Make RoboTwin's terminal-oriented output readable once redirected to a file:
    drop the per-step progress line, strip ANSI colors, drop carriage returns."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, data: str) -> int:
        if data.startswith("step: "):
            return len(data)
        cleaned = _ANSI_RE.sub("", data).replace("\r", "")
        if cleaned:
            self._stream.write(cleaned)
        return len(data)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_plain_log_stream() -> None:
    # Only when redirected; an interactive terminal keeps colors and the progress bar.
    if sys.stdout.isatty() or isinstance(sys.stdout, _PlainLogStream):
        return
    sys.stdout = _PlainLogStream(sys.stdout)


_install_plain_log_stream()


def _optional_env_bool(name: str) -> bool | None:
    if name not in os.environ:
        return None
    value = os.environ[name].strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    return None


def _patch_robotwin_task_config_overrides() -> None:
    eval_video_log = _optional_env_bool("FACT_ROBOTWIN_EVAL_VIDEO_LOG")
    if eval_video_log is None:
        return
    try:
        import yaml
    except Exception:
        return
    if getattr(yaml, "_fact_task_config_patch", False):
        return
    original_load = yaml.load

    def _load_with_overrides(*args, **kwargs):
        data = original_load(*args, **kwargs)
        if isinstance(data, dict) and "eval_video_log" in data:
            data = dict(data)
            data["eval_video_log"] = bool(eval_video_log)
        return data

    yaml.load = _load_with_overrides
    yaml._fact_task_config_patch = True


_patch_robotwin_task_config_overrides()


def _patch_robotwin_seed_tracking() -> None:
    try:
        from envs._base_task import Base_Task
    except Exception:
        return
    if getattr(Base_Task, "_fact_seed_tracking_patch", False):
        return
    original_init_task_env = Base_Task._init_task_env_

    def _init_task_env_with_seed(self, *args, **kwargs):
        if "seed" in kwargs:
            seed = int(kwargs["seed"])
            self.current_seed = seed
            self.episode_seed = seed
        try:
            return original_init_task_env(self, *args, **kwargs)
        finally:
            if "seed" in kwargs:
                seed = int(kwargs["seed"])
                self.current_seed = seed
                self.episode_seed = seed

    Base_Task._init_task_env_ = _init_task_env_with_seed
    Base_Task._fact_seed_tracking_patch = True


def _patch_robotwin_low_frequency_rgb() -> None:
    try:
        from envs._base_task import Base_Task
    except Exception:
        return
    if getattr(Base_Task, "_fact_low_frequency_rgb_patch", False):
        return
    original_get_obs = Base_Task.get_obs
    original_take_action = Base_Task.take_action

    def _light_obs(self):
        left_jointstate = self.robot.get_left_arm_jointState()
        right_jointstate = self.robot.get_right_arm_jointState()
        obs = {
            "observation": {},
            "pointcloud": [],
            "joint_action": {
                "left_arm": left_jointstate[:-1],
                "left_gripper": left_jointstate[-1],
                "right_arm": right_jointstate[:-1],
                "right_gripper": right_jointstate[-1],
                "vector": np.array(left_jointstate + right_jointstate),
            },
            "endpose": {},
            "_fact_light_obs": True,
        }
        self.now_obs = deepcopy(obs)
        return obs

    def _need_full_obs(self):
        if not getattr(Base_Task, "_fact_low_frequency_rgb_enabled", False):
            return True
        if getattr(self, "eval_video_path", None) is not None:
            return True
        if getattr(self, "save_data", False):
            return True
        data_type = getattr(self, "data_type", {}) or {}
        if any(data_type.get(key, False) for key in ("third_view", "mesh_segmentation", "actor_segmentation", "depth", "pointcloud")):
            return True
        interval = max(1, int(getattr(Base_Task, "_fact_execute_actions_per_plan", 1)))
        return int(getattr(self, "take_action_cnt", 0)) % interval == 0

    def _get_obs_low_frequency_rgb(self):
        if _need_full_obs(self):
            real_update_render = getattr(self, "_fact_real_update_render", None)
            if real_update_render is None:
                return original_get_obs(self)
            current_update_render = self._update_render
            try:
                self._update_render = real_update_render
                return original_get_obs(self)
            finally:
                self._update_render = current_update_render
        return _light_obs(self)

    def _force_full_obs(self):
        return original_get_obs(self)

    def _take_action_low_frequency_rgb(self, action, action_type="qpos"):
        if getattr(Base_Task, "_fact_skip_action_render_sync", False) and getattr(self, "eval_video_path", None) is None and not getattr(
            self, "render_freq", 0
        ):
            original_update_render = self._update_render
            self._fact_real_update_render = original_update_render

            def _skip_update_render():
                self.cameras.update_wrist_camera(self.robot.left_camera.get_pose(), self.robot.right_camera.get_pose())

            try:
                self._update_render = _skip_update_render
                return original_take_action(self, action, action_type=action_type)
            finally:
                self._update_render = original_update_render
                self._fact_real_update_render = original_update_render
        return original_take_action(self, action, action_type=action_type)

    Base_Task.get_obs = _get_obs_low_frequency_rgb
    Base_Task.take_action = _take_action_low_frequency_rgb
    Base_Task._fact_force_full_obs = _force_full_obs
    Base_Task._fact_low_frequency_rgb_patch = True


def _to_chw01(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={arr.shape}")
    if arr.shape[-1] != 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape={arr.shape}")
    is_uint8 = arr.dtype == np.uint8
    arr = arr.astype(np.float32, copy=False)
    if is_uint8 or arr.max() > 1.0:
        arr /= 255.0
    return np.transpose(arr, (2, 0, 1)).copy()


class ModelClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8093,
        timeout_ms: int = 30000,
        wait_seconds: float = 120.0,
        execute_actions_per_plan: int = 1,
        action_dim: int = 14,
        trace_root: str | None = None,
        print_action_stats: bool = False,
        enable_sample: bool = False,
        best_of_N: int = 1,
        enable_value_vis: bool = False,
        trace_value_only: bool = False,
        vis_dir: str | None = None,
        vis_fps: int = 10,
        low_frequency_rgb: bool = True,
        skip_action_render_sync: bool = True,
    ) -> None:
        # The TCP connect happens inside the client constructor, so retry the
        # whole construction until the server is listening and answers a ping.
        deadline = time.time() + float(wait_seconds)
        self.client = None
        while time.time() < deadline:
            try:
                client = RobotInferenceClient(host=host, port=port, timeout_ms=timeout_ms)
            except OSError:
                time.sleep(1.0)
                continue
            if client.ping():
                self.client = client
                break
            client.close()
            time.sleep(1.0)
        if self.client is None:
            raise TimeoutError(f"Timed out waiting for inference server at tcp://{host}:{port}")
        self.execute_actions_per_plan = max(1, int(execute_actions_per_plan))
        self.action_dim = int(action_dim)
        self.trace_root = None if not trace_root else Path(trace_root).expanduser().resolve()
        self.vis_dir = None if not vis_dir else Path(vis_dir).expanduser().resolve()
        self.print_action_stats = bool(print_action_stats)
        self.enable_sample = bool(enable_sample)
        self.best_of_N = max(1, int(best_of_N))
        self.enable_value_vis = bool(enable_value_vis)
        self.trace_value_only = bool(trace_value_only)
        self.vis_fps = int(vis_fps)
        self.low_frequency_rgb = bool(low_frequency_rgb)
        self.skip_action_render_sync = bool(skip_action_render_sync)
        self.planned_actions: np.ndarray | None = None
        self._last_images: list[np.ndarray] | None = None
        self._trace_episode_index: int | None = None
        self._trace_task_name: str | None = None
        self._trace_instruction: str | None = None
        self._trace_seed: int | None = None
        self._trace_success = False
        self._trace_steps: list[int] = []
        self._trace_states: list[np.ndarray] = []
        self._trace_actions: list[np.ndarray] = []
        self._trace_value_plans: list[list[float]] = []
        self._trace_selected_idx: list[int] = []
        self._trace_action_diversity: list[float] = []
        self._trace_images: list[torch.Tensor] = []
        self._configure_robotwin_render()
        atexit.register(self.close)

    def _configure_robotwin_render(self) -> None:
        try:
            from envs._base_task import Base_Task
        except Exception:
            return
        Base_Task._fact_low_frequency_rgb_enabled = bool(self.low_frequency_rgb)
        Base_Task._fact_execute_actions_per_plan = int(self.execute_actions_per_plan)
        Base_Task._fact_skip_action_render_sync = bool(self.low_frequency_rgb and self.skip_action_render_sync)

    def _trace_enabled(self) -> bool:
        return self.trace_root is not None or self.vis_dir is not None or self.enable_value_vis

    def _coerce_action_plan(self, response: Any) -> np.ndarray:
        if isinstance(response, dict) and "action" in response:
            response = response["action"]
        if isinstance(response, torch.Tensor):
            response = response.detach().cpu().numpy()
        actions = np.asarray(response, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"Expected action plan with shape [T, D], got {actions.shape}")
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {actions.shape[1]}")
        return actions.copy()

    def _extract_values(self, response: Any) -> tuple[np.ndarray | None, int | None, float | None]:
        if not isinstance(response, dict):
            return None, None, None
        vps = response.get("values_per_sample", None)
        sel = response.get("selected_index", None)
        if vps is None or sel is None:
            return None, None, None
        if isinstance(vps, torch.Tensor):
            vps = vps.detach().cpu().numpy()
        vps = np.asarray(vps, dtype=np.float32).reshape(-1)
        diversity = response.get("action_diversity", None)
        if diversity is not None:
            diversity = float(diversity)
        return vps, int(sel), diversity

    def _build_request(self, example: dict[str, Any]) -> dict[str, Any]:
        images = example["image"]
        self._last_images = [np.asarray(image).copy() for image in images]
        if len(images) != 3:
            raise ValueError(f"Expected 3 images [head, left, right], got {len(images)}")
        state = np.asarray(example["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != self.action_dim:
            raise ValueError(f"Expected state dim {self.action_dim}, got {state.shape[0]}")
        request: dict[str, Any] = {
            "observation.state": state,
            "observation.images.cam_high": _to_chw01(images[0]),
            "observation.images.cam_left_wrist": _to_chw01(images[1]),
            "observation.images.cam_right_wrist": _to_chw01(images[2]),
            "instruction": str(example["lang"]),
        }
        if self.enable_sample and self.best_of_N > 1:
            request["best_of_N"] = int(self.best_of_N)
        return request

    def needs_new_plan(self, step: int) -> bool:
        plan_offset = int(step % self.execute_actions_per_plan)
        return self.planned_actions is None or plan_offset == 0 or plan_offset >= len(self.planned_actions)

    def _print_plan_stats(self, actions: np.ndarray, state: np.ndarray, step: int) -> None:
        first = actions[0]
        delta = first - state
        print(
            "[FACT RoboTwin] "
            f"step={step} plan_len={actions.shape[0]} "
            f"first_action[min={first.min():.4f}, max={first.max():.4f}, mean={first.mean():.4f}] "
            f"delta_l1={np.abs(delta).mean():.4f}"
        )

    def _reset_trace_buffers(self) -> None:
        self._trace_episode_index = None
        self._trace_task_name = None
        self._trace_instruction = None
        self._trace_seed = None
        self._trace_success = False
        self._trace_steps = []
        self._trace_states = []
        self._trace_actions = []
        self._trace_value_plans = []
        self._trace_selected_idx = []
        self._trace_action_diversity = []
        self._trace_images = []

    def _render_value_plot(self, png_path: Path, values_per_plan: np.ndarray, selected_idx: np.ndarray) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[FACT RoboTwin] value-vis skipped (matplotlib unavailable: {exc})")
            return
        # Non-finite values come from server-side blow-ups (server already excludes them
        # from argmin). For plotting, turn them into NaN so matplotlib just leaves a gap
        # instead of stretching the axis to +/- inf.
        vals = np.where(np.isfinite(values_per_plan), values_per_plan, np.nan)
        num_plans, n_samples = vals.shape
        x = np.arange(num_plans)
        fig, ax = plt.subplots(figsize=(8, 4))
        if n_samples > 1:
            for i in range(n_samples):
                ax.plot(x, vals[:, i], color="tab:gray", alpha=0.35, linewidth=1.0)
        chosen = vals[x, selected_idx]
        label = "predicted value" if n_samples == 1 else f"selected (argmin, N={n_samples})"
        ax.plot(x, chosen, color="tab:red", linewidth=2.0, label=label)
        ax.set_xlabel("plan index")
        ax.set_ylabel("predicted value (time-to-go)")
        ax.set_title(f"{self._trace_task_name or ''} seed={self._trace_seed} ep={self._trace_episode_index}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)

    def _flush_trace(self, reason: str) -> None:
        have_trace_step_data = bool(self._trace_steps) and not self.trace_value_only
        have_value_data = bool(self._trace_value_plans) and bool(self._trace_selected_idx)
        can_save_trace = self.trace_root is not None and (have_trace_step_data or have_value_data)
        can_save_video = bool(self._trace_images)  # vis_dir is non-None whenever _trace_images is populated
        if self._trace_episode_index is None or not (can_save_trace or can_save_video):
            self._reset_trace_buffers()
            return
        task_name = self._trace_task_name or "unknown_task"
        seed_str = "unknown" if self._trace_seed is None else str(int(self._trace_seed))
        stem = f"seed_{seed_str}_episode_{int(self._trace_episode_index):06d}"
        if can_save_trace:
            task_dir = self.trace_root / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            npz_path = task_dir / f"{stem}.npz"
            json_path = task_dir / f"{stem}.json"
            npz_arrays: dict[str, np.ndarray] = {}
            if have_trace_step_data:
                npz_arrays["steps"] = np.asarray(self._trace_steps, dtype=np.int32)
                npz_arrays["states"] = np.asarray(self._trace_states, dtype=np.float32)
                npz_arrays["actions"] = np.asarray(self._trace_actions, dtype=np.float32)
            values_per_plan: np.ndarray | None = None
            selected_idx_arr: np.ndarray | None = None
            if have_value_data:
                n_samples = max(len(row) for row in self._trace_value_plans)
                values_per_plan = np.full((len(self._trace_value_plans), n_samples), np.nan, dtype=np.float32)
                for i, row in enumerate(self._trace_value_plans):
                    values_per_plan[i, : len(row)] = row
                selected_idx_arr = np.asarray(self._trace_selected_idx, dtype=np.int32)
                npz_arrays["values_per_plan"] = values_per_plan
                npz_arrays["selected_index_per_plan"] = selected_idx_arr
                if self._trace_action_diversity:
                    npz_arrays["action_diversity_per_plan"] = np.asarray(
                        self._trace_action_diversity, dtype=np.float32
                    )
            np.savez_compressed(npz_path, **npz_arrays)
            meta = {
                "task_name": task_name,
                "episode_index": int(self._trace_episode_index),
                "seed": None if self._trace_seed is None else int(self._trace_seed),
                "instruction": self._trace_instruction,
                "num_steps": len(self._trace_steps),
                "num_plans": len(self._trace_value_plans),
                "success": bool(self._trace_success),
                "finalize_reason": str(reason),
                "trace_value_only": bool(self.trace_value_only),
            }
            json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            if self.enable_value_vis and values_per_plan is not None and selected_idx_arr is not None:
                self._render_value_plot(task_dir / f"{stem}_values.png", values_per_plan, selected_idx_arr)
        if can_save_video:
            try:
                import torchvision
                all_frames = torch.cat(self._trace_images, dim=0)  # [T_total, H, W, C]
                video_dir = self.vis_dir / task_name
                video_dir.mkdir(parents=True, exist_ok=True)
                torchvision.io.write_video(str(video_dir / f"{stem}_pred_video.mp4"), all_frames, fps=self.vis_fps)
            except Exception as exc:
                print(f"[FACT RoboTwin] pred-video save failed: {exc}")
        self._reset_trace_buffers()

    def reset(self) -> None:
        self.planned_actions = None
        self._last_images = None
        self._flush_trace(reason="reset")

    def maybe_start_episode(self, task_env, instruction: str) -> None:
        if not self._trace_enabled():
            return
        episode_index = int(getattr(task_env, "test_num", 0))
        if self._trace_episode_index is not None and episode_index != self._trace_episode_index:
            self._flush_trace(reason="episode_switch")
        if self._trace_episode_index is None:
            self._trace_episode_index = episode_index
            self._trace_task_name = str(getattr(task_env, "task_name", "unknown_task"))
            self._trace_instruction = str(instruction)
            seed = None
            for seed_attr in ("current_seed", "episode_seed", "seed", "now_seed"):
                if hasattr(task_env, seed_attr):
                    seed = getattr(task_env, seed_attr)
                    break
            self._trace_seed = None if seed is None else int(seed)

    def record_step(
        self,
        task_env,
        instruction: str,
        state: np.ndarray,
        action: np.ndarray,
    ) -> None:
        if not self._trace_enabled():
            return
        # Always update episode metadata so the value-only trace can still flush
        # (episode_index is set here, not in step()).
        self.maybe_start_episode(task_env, instruction)
        if self.trace_root is None or self.trace_value_only:
            return
        self._trace_steps.append(int(getattr(task_env, "take_action_cnt", len(self._trace_steps))))
        self._trace_states.append(np.asarray(state, dtype=np.float32).reshape(-1).copy())
        self._trace_actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())

    def mark_episode(self, success: bool) -> None:
        if not self._trace_enabled():
            return
        self._trace_success = bool(success)

    def step(self, example: dict[str, Any], step: int = 0) -> np.ndarray:
        plan_offset = int(step % self.execute_actions_per_plan)
        need_new_plan = self.needs_new_plan(step)
        if need_new_plan:
            if "image" not in example and self._last_images is not None:
                example["image"] = self._last_images
            request = self._build_request(example)
            response = self.client.inference(request)
            self.planned_actions = self._coerce_action_plan(response)
            if self.vis_dir is not None and isinstance(response, dict):
                imgs_tensor = response.get("images", None)
                if isinstance(imgs_tensor, torch.Tensor) and imgs_tensor.ndim == 5:
                    # [B, C, T, H, W] float in [-1, 1] → [T, H, W, C] uint8
                    vid = imgs_tensor[0].permute(1, 2, 3, 0).float()
                    vid = ((vid + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
                    self._trace_images.append(vid.cpu())
            if self.enable_value_vis and self.trace_root is not None:
                vps, sel, diversity = self._extract_values(response)
                if vps is not None and sel is not None:
                    self._trace_value_plans.append([float(v) for v in vps.tolist()])
                    self._trace_selected_idx.append(int(sel))
                    if diversity is not None:
                        self._trace_action_diversity.append(diversity)
            plan_offset = 0
            if self.print_action_stats:
                self._print_plan_stats(self.planned_actions, np.asarray(example["state"], dtype=np.float32), int(step))
        return np.asarray(self.planned_actions[plan_offset], dtype=np.float32).copy()

    def close(self) -> None:
        self._flush_trace(reason="close")
        try:
            self.client.close()
        except Exception:
            pass


def _as_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default


def get_model(usr_args: dict[str, Any]):
    _patch_robotwin_seed_tracking()
    _patch_robotwin_low_frequency_rgb()
    return ModelClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 8093)),
        timeout_ms=int(usr_args.get("server_timeout_ms", 30000)),
        wait_seconds=float(usr_args.get("server_wait_seconds", 120.0)),
        execute_actions_per_plan=int(usr_args.get("execute_actions_per_plan", 1)),
        action_dim=int(usr_args.get("action_dim", 14)),
        trace_root=usr_args.get("trace_root", None),
        print_action_stats=_as_bool(usr_args.get("print_action_stats", False)),
        enable_sample=_as_bool(usr_args.get("enable_sample", False)),
        best_of_N=int(usr_args.get("best_of_n", usr_args.get("best_of_N", 1))),
        enable_value_vis=_as_bool(usr_args.get("enable_value_vis", False)),
        trace_value_only=_as_bool(usr_args.get("trace_value_only", False)),
        vis_dir=usr_args.get("vis_dir", None),
        vis_fps=int(usr_args.get("vis_fps", 10)),
        low_frequency_rgb=_as_bool(usr_args.get("low_frequency_rgb", True), True),
        skip_action_render_sync=_as_bool(usr_args.get("skip_action_render_sync", True), True),
    )


def reset_model(model) -> None:
    model.reset()


def eval(TASK_ENV, model, observation):  # noqa: A001
    instruction = str(TASK_ENV.get_instruction())
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    step = int(getattr(TASK_ENV, "take_action_cnt", 0))
    need_new_plan = model.needs_new_plan(step)
    if need_new_plan and observation.get("_fact_light_obs", False):
        observation = TASK_ENV._fact_force_full_obs()
    images = None
    if need_new_plan or not observation.get("_fact_light_obs", False):
        images = [
            observation["observation"]["head_camera"]["rgb"],
            observation["observation"]["left_camera"]["rgb"],
            observation["observation"]["right_camera"]["rgb"],
        ]
    example = {
        "lang": instruction,
        "state": state,
    }
    if images is not None:
        example["image"] = images
    action = model.step(example, step=step)
    model.record_step(TASK_ENV, instruction=instruction, state=state, action=action)
    TASK_ENV.take_action(action)
    if getattr(TASK_ENV, "eval_success", False):
        model.mark_episode(success=True)
