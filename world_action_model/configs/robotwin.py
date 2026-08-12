"""
FACT (World-Action Model) — Multi-GPU RoboTwin Training Config
=============================================
Edit the USER SETTINGS sections below, then launch with:

    python -m scripts.train --config world_action_model.configs.robotwin.config

All important hyperparameters are grouped at the top.
The config dict assembly at the bottom should rarely need editing.
"""

import os as _os

from world_action_model.robotwin_utils import resolve_dataset_multipliers, select_robotwin_task_specs

try:
    from torch.nn.attention.flex_attention import create_block_mask as _robotwin_flex_probe
except ImportError:
    _robotwin_flex_probe = None

# ══════════════════════════════════════════════════════════════════════════════
# § 1  PATHS
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_DIR = _os.environ.get("FACT_PROJECT_DIR", "./experiments/robotwin")
WAN_MODEL_DIR = _os.environ.get("WAN_MODEL_DIR", "./models/Wan2.2-TI2V-5B-Diffusers")
ROBOTWIN_ROOT = _os.environ.get("ROBOTWIN_ROOT", "./datasets/RoboTwin")
NORM_STATS_PATH = _os.environ.get("ROBOTWIN_NORM_PATH", "./artifacts/robotwin/norm_stats_delta.json")
CHECKPOINT = None

# ══════════════════════════════════════════════════════════════════════════════
# § 2  DATASET SELECTION / MIXTURE
# ══════════════════════════════════════════════════════════════════════════════

# Explicit dataset ids such as ["Clean/place_can_basket"].
ROBOTWIN_DATASET_IDS = []

# Glob-style selectors such as ["Clean/*", "Randomized/*"].
# Leave empty to use all discoverable training tasks under ROBOTWIN_ROOT.
ROBOTWIN_DATASET_GLOBS = []

# Optional multiplier overrides. Keys can be exact ids or globs.
# More-specific matches win over broad fallbacks such as "*".
# These are per-dataset multipliers before episode-count weighting, so with
# equal multipliers the final sampling mass is still proportional to the
# number of episodes in each selected dataset.
ROBOTWIN_DATASET_MULTIPLIERS = {
    "*": 1.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# § 3  HARDWARE
# ══════════════════════════════════════════════════════════════════════════════

GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
DEEPSPEED_ZERO_STAGE = 2
DEEPSPEED_OFFLOAD_OPTIMIZER = False

# ══════════════════════════════════════════════════════════════════════════════
# § 4  BATCH & STEPS
# ══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE_PER_GPU = 32
PYAV_THREAD_COUNT = 2
GRADIENT_ACCUMULATION_STEPS = 1
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
MAX_STEPS = 150_000
# When True, the dataloader reads pre-computed VAE latents from
# {task_dir}/vae_latents/episode_{ep:06d}.pt instead of decoding videos and
# running VAE encode at training time. Run `python -m scripts.compute_vae_latents`
# first to populate the cache. Set to False (or export
# FACT_USE_CACHED_VAE_LATENTS=0) to force the live video+VAE path.
USE_CACHED_VAE_LATENTS = _os.environ.get("FACT_USE_CACHED_VAE_LATENTS", "1").strip().lower() not in (
    "0", "false", "no", "off", ""
)

# ══════════════════════════════════════════════════════════════════════════════
# § 5  OPTIMIZER & SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

LR = 2e-5
ACTION_LR = 2e-4
WEIGHT_DECAY = 1e-4
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
SCHEDULER_TYPE = "WarmupCosineScheduler"
SCHEDULER_KWARGS = dict(
    warmup_steps=500,
    decay_steps=MAX_STEPS,
)

# ══════════════════════════════════════════════════════════════════════════════
# § 6  MODEL
# ══════════════════════════════════════════════════════════════════════════════

ACTION_DIM = 14
NUM_FRAMES = 48
FPS = 15
FLOW_SHIFT = 3.0
EXPAND_TIMESTEPS = True
STATE_REPEATS = 1
ACTION_REPEATS = 1
ROBOT_ADAPTER_RANK = 1024 # ffn adapter rank (for action), set to 0 to disable
ACTIVATION_CHECKPOINTING = True
ACTIVATION_CLASS_NAMES = ["WanTransformerBlock"]
SELF_ATTENTION_IMPLEMENTATION = "sdpa" # "flex" or "sdpa"
FLEX_ATTENTION_BLOCK_SIZE = 128
FLEX_ATTENTION_KERNEL_OPTIONS = None
VISUAL_LOSS_WEIGHT = 1.0
ACTION_LOSS_WEIGHT = 10.0
FUTURE_STATE_LOSS_WEIGHT = 0.4 # predict future state
VALUE_LOSS_WEIGHT = 0.4  # predict future value

# Raw-unit penalty added to the V target on failure episodes (failure_active).
# Requires norm_stats value.max >= 1 + VALUE_PENALTY_SCALE; the default is 2.0
# (see scripts/compute_norm_stats.py::_build_default_value_stats).
VALUE_PENALTY_SCALE = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# § 7  LOGGING & CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════════

WANDB_PROJECT = "fact-robotwin"
WANDB_ENTITY = None  # set to your wandb team/user, or leave None for the default entity
LOG_INTERVAL = 10
ENABLE_TRAIN_VIS = False
VIEW_INTERVAL = 500
CHECKPOINT_INTERVAL = 1_000
CHECKPOINT_TOTAL_LIMIT = 1
CHECKPOINT_SAVE_OPTIMIZER = False
IF_RESUME = True
# ══════════════════════════════════════════════════════════════════════════════
# § 8  DATASET  (camera views & data layout)
# ══════════════════════════════════════════════════════════════════════════════

# Main-view size. RoboTwin 3-view inputs are laid out as:
# - cam_high          -> 256x192
# - cam_left_wrist    -> 128x96
# - cam_right_wrist   -> 128x96
# Final composite canvas: 384x192.
DST_SIZE = (256, 192)
VIEW_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
T5_LEN = 64


def _normalize_csv_env(var_name: str) -> list[str]:
    raw = _os.environ.get(var_name, "")
    if not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _to_wandb_config(value):
    if isinstance(value, dict):
        return {str(k): _to_wandb_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wandb_config(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


_env_dataset_ids = _normalize_csv_env("ROBOTWIN_DATASET_IDS")
_env_dataset_globs = _normalize_csv_env("ROBOTWIN_DATASET_GLOBS")
if _env_dataset_ids:
    ROBOTWIN_DATASET_IDS = _env_dataset_ids
if _env_dataset_globs:
    ROBOTWIN_DATASET_GLOBS = _env_dataset_globs

_selected_specs = select_robotwin_task_specs(
    robotwin_root=ROBOTWIN_ROOT,
    dataset_ids=ROBOTWIN_DATASET_IDS,
    dataset_globs=ROBOTWIN_DATASET_GLOBS,
)
_dataset_multipliers = resolve_dataset_multipliers(_selected_specs, ROBOTWIN_DATASET_MULTIPLIERS)
_image_frame_offsets = [0, NUM_FRAMES // 4, NUM_FRAMES // 2, (3 * NUM_FRAMES) // 4, NUM_FRAMES]

_DS_CONFIG_MAP = {
    (1, False): "accelerate_configs/zero1.json",
    (2, False): "accelerate_configs/zero2.json",
    (2, True): "accelerate_configs/zero2_offload.json",
    (3, False): "accelerate_configs/zero3.json",
    (3, True): "accelerate_configs/zero3_offload.json",
}
_deepspeed_config_file = _DS_CONFIG_MAP[DEEPSPEED_ZERO_STAGE, DEEPSPEED_OFFLOAD_OPTIMIZER]
_scheduler_dict = dict(type=SCHEDULER_TYPE, **SCHEDULER_KWARGS)
_tracker_init_kwargs = dict(wandb=dict())
if WANDB_ENTITY is not None:
    _tracker_init_kwargs["wandb"]["entity"] = WANDB_ENTITY

_launch_config = dict(
    gpu_ids=GPU_IDS,
    distributed_type="DEEPSPEED",
    deepspeed_config=dict(
        deepspeed_config_file=_deepspeed_config_file,
    ),
    until_completion=False,
)

_dataloaders_config = dict(
    train=dict(
        data_or_config=[
            dict(
                _class_name="LeRobotDataset",
                data_path=str(spec.task_dir),
                repo_id=spec.task_name,
                data_size=None,
                delta_info={"action": NUM_FRAMES},
                delta_frames={k: _image_frame_offsets for k in VIEW_KEYS},
                t5_embedding_dir=_os.path.join(str(spec.task_dir), "t5_embedding"),
                # vae_latent_dir: set to the cache path when enabled so the
                # dataloader skips video decode + VAE encode at train time.
                # Empty string explicitly disables the cache.
                vae_latent_dir=(
                    _os.path.join(str(spec.task_dir), "vae_latents")
                    if USE_CACHED_VAE_LATENTS
                    else ""
                ),
                video_backend_kwargs=dict(pyav_thread_count=PYAV_THREAD_COUNT),
            )
            for spec in _selected_specs
        ],
        batch_size_per_gpu=BATCH_SIZE_PER_GPU,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        transform=dict(
            type="WATransformsLerobot",
            robotype_to_embed_id={
                "aloha": 0,
                "agilex": 0,
                "agibot": 1,
            },
            dst_size=DST_SIZE,
            num_frames=NUM_FRAMES,
            fps=FPS,
            is_train=True,
            norm_path=[NORM_STATS_PATH],
            model_action_dim=ACTION_DIM,
            num_views=len(VIEW_KEYS),
            view_keys=VIEW_KEYS,
            t5_len=T5_LEN,
            value_penalty_scale=VALUE_PENALTY_SCALE,
            image_cfg=dict(
                mask_generator=dict(
                    max_ref_frames=1,
                    start=1,
                    factor=4,
                ),
            ),
        ),
        sampler=dict(
            type="EpisodeMixtureSampler",
            dataset_weights=_dataset_multipliers,
        ),
        collator=dict(is_equal=True),
    ),
    test=dict(),
)

_models_config = dict(
    pretrained=WAN_MODEL_DIR,
    checkpoint=CHECKPOINT,
    strict=False,
    action_dim=ACTION_DIM,
    flow_shift=FLOW_SHIFT,
    expand_timesteps=EXPAND_TIMESTEPS,
    state_repeats=STATE_REPEATS,
    action_repeats=ACTION_REPEATS,
    robot_adapter_rank=ROBOT_ADAPTER_RANK,
    view_dir=PROJECT_DIR,
    enable_train_vis=ENABLE_TRAIN_VIS,
    view_interval=VIEW_INTERVAL,
    transformer=dict(
        self_attention_implementation=SELF_ATTENTION_IMPLEMENTATION,
        flex_attention_block_size=FLEX_ATTENTION_BLOCK_SIZE,
        flex_attention_kernel_options=FLEX_ATTENTION_KERNEL_OPTIONS,
    ),
)

_optimizers_config = dict(
    type="AdamW",
    lr=LR,
    action_lr=ACTION_LR,
    betas=ADAMW_BETAS,
    eps=ADAMW_EPS,
    weight_decay=WEIGHT_DECAY,
    fused=True,
    foreach=False,
)

_train_config = dict(
    max_steps=MAX_STEPS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    mixed_precision="bf16",
    activation_checkpointing=ACTIVATION_CHECKPOINTING,
    activation_class_names=ACTIVATION_CLASS_NAMES,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    checkpoint_total_limit=CHECKPOINT_TOTAL_LIMIT,
    checkpoint_save_optimizer=CHECKPOINT_SAVE_OPTIMIZER,
    resume=IF_RESUME,
    log_with="wandb",
    tracker_project_name=WANDB_PROJECT,
    tracker_init_kwargs=_tracker_init_kwargs,
    log_interval=LOG_INTERVAL,
    loss_weights=dict(
        visual_loss=VISUAL_LOSS_WEIGHT,
        action_loss=ACTION_LOSS_WEIGHT,
        future_state_loss=FUTURE_STATE_LOSS_WEIGHT,
        value_loss=VALUE_LOSS_WEIGHT,
    ),
)

_train_config["tracker_config"] = _to_wandb_config(
    dict(
        config_module="world_action_model.configs.robotwin",
        user_settings=dict(
            paths=dict(
                project_dir=PROJECT_DIR,
                wan_model_dir=WAN_MODEL_DIR,
                robotwin_root=ROBOTWIN_ROOT,
                norm_stats_path=NORM_STATS_PATH,
                checkpoint=CHECKPOINT,
            ),
            dataset_selection=dict(
                dataset_ids=ROBOTWIN_DATASET_IDS,
                dataset_globs=ROBOTWIN_DATASET_GLOBS,
                dataset_multipliers=ROBOTWIN_DATASET_MULTIPLIERS,
            ),
            hardware=dict(
                gpu_ids=GPU_IDS,
                deepspeed_zero_stage=DEEPSPEED_ZERO_STAGE,
                deepspeed_offload_optimizer=DEEPSPEED_OFFLOAD_OPTIMIZER,
            ),
            batch_and_steps=dict(
                batch_size_per_gpu=BATCH_SIZE_PER_GPU,
                pyav_thread_count=PYAV_THREAD_COUNT,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS,
                prefetch_factor=PREFETCH_FACTOR,
                max_steps=MAX_STEPS,
            ),
            optimizer_and_scheduler=dict(
                lr=LR,
                action_lr=ACTION_LR,
                weight_decay=WEIGHT_DECAY,
                adamw_betas=ADAMW_BETAS,
                adamw_eps=ADAMW_EPS,
                scheduler_type=SCHEDULER_TYPE,
                scheduler_kwargs=SCHEDULER_KWARGS,
            ),
            model=dict(
                action_dim=ACTION_DIM,
                num_frames=NUM_FRAMES,
                fps=FPS,
                flow_shift=FLOW_SHIFT,
                expand_timesteps=EXPAND_TIMESTEPS,
                state_repeats=STATE_REPEATS,
                action_repeats=ACTION_REPEATS,
                robot_adapter_rank=ROBOT_ADAPTER_RANK,
                self_attention_implementation=SELF_ATTENTION_IMPLEMENTATION,
                flex_attention_block_size=FLEX_ATTENTION_BLOCK_SIZE,
                flex_attention_kernel_options=FLEX_ATTENTION_KERNEL_OPTIONS,
                activation_checkpointing=ACTIVATION_CHECKPOINTING,
                activation_class_names=ACTIVATION_CLASS_NAMES,
                visual_loss_weight=VISUAL_LOSS_WEIGHT,
                action_loss_weight=ACTION_LOSS_WEIGHT,
                future_state_loss_weight=FUTURE_STATE_LOSS_WEIGHT,
                value_loss_weight=VALUE_LOSS_WEIGHT,
                value_penalty_scale=VALUE_PENALTY_SCALE,
            ),
            logging_and_checkpointing=dict(
                wandb_project=WANDB_PROJECT,
                wandb_entity=WANDB_ENTITY,
                log_interval=LOG_INTERVAL,
                enable_train_vis=ENABLE_TRAIN_VIS,
                view_interval=VIEW_INTERVAL,
                checkpoint_interval=CHECKPOINT_INTERVAL,
                checkpoint_total_limit=CHECKPOINT_TOTAL_LIMIT,
            ),
            dataset_layout=dict(
                dst_size=DST_SIZE,
                view_keys=VIEW_KEYS,
                t5_len=T5_LEN,
            ),
        ),
        selected_datasets=dict(
            count=len(_selected_specs),
            dataset_ids=[spec.task_name for spec in _selected_specs],
            dataset_dirs=[str(spec.task_dir) for spec in _selected_specs],
            resolved_dataset_weights=_dataset_multipliers,
        ),
        project_dir=PROJECT_DIR,
        runners=["world_action_model.trainer.wa_casual_trainer.CasualWATrainer"],
        launch=_launch_config,
        dataloaders=_dataloaders_config,
        models=_models_config,
        optimizers=_optimizers_config,
        schedulers=_scheduler_dict,
        train={k: v for k, v in _train_config.items() if k != "tracker_config"},
    )
)


config = dict(
    project_dir=PROJECT_DIR,
    runners=["world_action_model.trainer.wa_casual_trainer.CasualWATrainer"],
    launch=_launch_config,
    dataloaders=_dataloaders_config,
    models=_models_config,
    optimizers=_optimizers_config,
    schedulers=_scheduler_dict,
    train=_train_config,
)
