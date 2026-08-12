import copy
import html
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import regex as re
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.models import AutoencoderKLWan

from .utils import build_teacher_forcing_per_token_timestep


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    _XLA_AVAILABLE = True
else:
    _XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

if is_ftfy_available():
    import ftfy


def _basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _prompt_clean(text: str) -> str:
    return _whitespace_clean(_basic_clean(text))


def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class WAPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2", "image_encoder", "image_processor"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer=None,
        transformer_2=None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor
        self.action_scheduler = copy.deepcopy(scheduler)
        self.future_state_scheduler = copy.deepcopy(scheduler)
        self.value_scheduler = copy.deepcopy(scheduler)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [_prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(self, image: PipelineImageInput, device: Optional[torch.device] = None):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}.")
        if image is None and image_embeds is None:
            raise ValueError("Provide either `image` or `prompt_embeds`.")
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}.")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}."
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when `boundary_ratio` is not None.")
        if self.config.boundary_ratio is not None and image_embeds is not None:
            raise ValueError("Cannot forward `image_embeds` when `boundary_ratio` is configured.")

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        action_chunk: Optional[int] = None,
        action_dim: Optional[int] = 14,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}."
            )

        # Encode reference image first so latent_condition is available for initialization.
        image = image.unsqueeze(2)

        if self.config.expand_timesteps:
            video_condition = image
        elif last_image is None:
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            device, dtype
        )

        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        if latent_condition.shape[0] == 1 and batch_size > 1:
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)
        elif latent_condition.shape[0] != batch_size:
            if batch_size % latent_condition.shape[0] != 0:
                raise ValueError(
                    f"Cannot broadcast latent condition batch {latent_condition.shape[0]} to requested batch {batch_size}."
                )
            latent_condition = latent_condition.repeat_interleave(batch_size // latent_condition.shape[0], dim=0)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        # latent_condition[:, :, :1] is the ref frame; broadcasts over T to initialize
        # future latents near the current observation instead of pure Gaussian noise.
        if latents is None:
            latents = latent_condition[:, :, :1] + randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            if action_chunk is None:
                raise ValueError("action_chunk is required when latents is None")
            action_shape = (batch_size, action_chunk, action_dim)
            action = randn_tensor(action_shape, generator=generator, device=device, dtype=dtype)
            future_state = randn_tensor((batch_size, 1, action_dim), generator=generator, device=device, dtype=dtype)
            value = randn_tensor((batch_size, 1, 1), generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
            if action_chunk is None:
                raise ValueError("action_chunk is required when latents is provided")
            action_shape = (batch_size, action_chunk, action_dim)
            action = randn_tensor(action_shape, generator=generator, device=device, dtype=dtype)
            future_state = randn_tensor((batch_size, 1, action_dim), generator=generator, device=device, dtype=dtype)
            value = randn_tensor((batch_size, 1, 1), generator=generator, device=device, dtype=dtype)

        if self.config.expand_timesteps:
            first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device)
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask, action, future_state, value

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1), future_state, value

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        action_chunk: int,
        state: Optional[torch.Tensor] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        action_only: bool = False,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        action_dim: int = 14,
        enable_prefix_cache: bool = False,
        skip_future_state_value: bool = False,
        gt_action_condition: Optional[torch.Tensor] = None,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if skip_future_state_value and not action_only:
            raise ValueError("skip_future_state_value requires action_only=True.")
        if enable_prefix_cache and image_embeds is not None:
            raise ValueError("enable_prefix_cache does not support image_embeds/image-conditioned transformer inputs.")

        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        uses_image_condition = False
        if self.transformer is not None and getattr(self.transformer.config, "image_dim", None) is not None:
            uses_image_condition = True
            if enable_prefix_cache:
                raise ValueError("enable_prefix_cache does not support image-conditioned transformers.")
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self.action_scheduler.set_timesteps(num_inference_steps, device=device)
        self.future_state_scheduler.set_timesteps(num_inference_steps, device=device)
        self.value_scheduler.set_timesteps(num_inference_steps, device=device)
        action_timesteps = self.action_scheduler.timesteps
        if not torch.all(timesteps == action_timesteps):
            raise ValueError("timesteps and action_timesteps mismatch")

        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(device, dtype=torch.float32)

        if state is None:
            raise ValueError("state is required")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(f"state must have shape [D], [B, D], or [B, 1, D], got {tuple(state.shape)}")
        effective_batch_size = batch_size * num_videos_per_prompt
        if state.shape[0] == batch_size and num_videos_per_prompt > 1:
            state = state.repeat_interleave(num_videos_per_prompt, dim=0)
        elif state.shape[0] == 1 and effective_batch_size > 1:
            state = state.repeat(effective_batch_size, 1, 1)
        elif state.shape[0] != effective_batch_size:
            raise ValueError(
                f"state batch {state.shape[0]} must match prompt batch {batch_size} or effective batch {effective_batch_size}"
            )
        state = state.to(device=device, dtype=self.dtype)

        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
            action_chunk,
            action_dim,
        )
        if self.config.expand_timesteps:
            latents, condition, first_frame_mask, action, future_state, value = latents_outputs
        else:
            latents, condition, future_state, value = latents_outputs
            first_frame_mask = torch.ones(1, 1, latents.shape[2], latents.shape[3], latents.shape[4], device=device, dtype=latents.dtype)
            first_frame_mask[:, :, 0] = 0
            action = randn_tensor((latents.shape[0], action_chunk, action_dim), generator=generator, device=device, dtype=torch.float32)
            future_state = future_state.to(device=device, dtype=torch.float32)
            value = value.to(device=device, dtype=torch.float32)

        action = action.to(dtype=transformer_dtype, device=device)
        future_state = future_state.to(dtype=transformer_dtype, device=device)
        value = value.to(dtype=transformer_dtype, device=device)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4

        # --------------------------------------------------------------
        # Stage 1: diffuse pred_action alone conditioned on {state, ref, text}.
        # Sequence: [state | ref | pred_action]. No gt_action, no future_state,
        # no value, no future_image in this stage.
        # --------------------------------------------------------------
        if self.config.expand_timesteps:
            stage1_ref_latents = condition[:, :, :1].to(transformer_dtype)
        else:
            stage1_ref_latents = torch.cat(
                [latents[:, :, :1], condition[:, :, :1]], dim=1
            ).to(transformer_dtype)

        stage1_prefix_cache = None
        use_stage1_prefix_cache = bool(enable_prefix_cache)
        if use_stage1_prefix_cache:
            unsupported_prefix_cache_reasons = []
            if self.transformer is None:
                unsupported_prefix_cache_reasons.append("missing primary transformer")
            if self.transformer_2 is not None:
                unsupported_prefix_cache_reasons.append("dual-transformer/boundary pipeline")
            if boundary_timestep is not None:
                unsupported_prefix_cache_reasons.append("boundary_ratio pipeline")
            if not action_only:
                unsupported_prefix_cache_reasons.append("return_images/full future-image inference")
            if self.do_classifier_free_guidance:
                unsupported_prefix_cache_reasons.append("classifier-free guidance")
            if uses_image_condition:
                unsupported_prefix_cache_reasons.append("image-conditioned transformer")
            if callback_on_step_end is not None and any(
                name in callback_on_step_end_tensor_inputs for name in ("prompt_embeds", "negative_prompt_embeds")
            ):
                unsupported_prefix_cache_reasons.append("callback can mutate prompt embeddings")
            if getattr(self.transformer, "self_attention_implementation", "sdpa") != "sdpa":
                unsupported_prefix_cache_reasons.append("non-SDPA self attention")
            if not (
                hasattr(self.transformer, "build_inference_action_prefix_cache")
                and hasattr(self.transformer, "build_inference_future_action_prefix_cache")
            ):
                unsupported_prefix_cache_reasons.append("transformer lacks prefix-cache methods")
            if unsupported_prefix_cache_reasons:
                raise ValueError(
                    "enable_prefix_cache is unsupported for this request: "
                    + ", ".join(unsupported_prefix_cache_reasons)
                )
        if use_stage1_prefix_cache:
            stage1_prefix_cache = self.transformer.build_inference_action_prefix_cache(
                ref_latents=stage1_ref_latents,
                encoder_hidden_states=prompt_embeds,
                state=state,
                attention_kwargs=attention_kwargs,
            )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(action_timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                else:
                    current_model = self.transformer_2

                noise_t_s1 = t.reshape(1, 1).to(
                    device=stage1_ref_latents.device, dtype=stage1_ref_latents.dtype
                ).expand(stage1_ref_latents.shape[0], 1)
                timestep_s1 = build_teacher_forcing_per_token_timestep(
                    batch_size=stage1_ref_latents.shape[0],
                    num_state_tokens=int(state.shape[1]),
                    num_ref_tokens=frame_per_tokens,
                    num_pred_action_tokens=int(action.shape[1]),
                    num_gt_action_tokens=0,
                    num_future_state_tokens=0,
                    num_value_tokens=0,
                    num_noisy_latent_tokens=0,
                    noise_t=noise_t_s1,
                    device=stage1_ref_latents.device,
                    dtype=stage1_ref_latents.dtype,
                )

                with current_model.cache_context("cond"):
                    _, action_pred, _, _ = current_model(
                        ref_latents=stage1_ref_latents,
                        timestep=timestep_s1,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                        action=action,
                        state=state,
                        action_only=True,
                        prefix_cache=stage1_prefix_cache,
                    )

                action = self.action_scheduler.step(action_pred, t, action, return_dict=False)[0]

                if callback_on_step_end is not None and action_only:
                    callback_kwargs = {"action": action}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    action = callback_outputs.pop("action", action)

                if i == len(action_timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.action_scheduler.order == 0
                ):
                    progress_bar.update()

                if _XLA_AVAILABLE:
                    xm.mark_step()

        if skip_future_state_value:
            if not return_dict:
                return None, action, None, None
            return {"images": None, "action": action, "future_state": None, "value": None}

        # The clean, fully-denoised action from Stage 1 becomes the teacher-forcing
        # gt_action condition for Stage 2 unless the caller provides an explicit
        # normalized gt_action_condition. We keep a dummy pred_action slot so the
        # Stage-2 token layout matches training exactly.
        if gt_action_condition is None:
            gt_action_clean = action.to(dtype=transformer_dtype, device=device)
        else:
            gt_action_clean = gt_action_condition
            if gt_action_clean.ndim == 2:
                gt_action_clean = gt_action_clean.unsqueeze(0)
            if gt_action_clean.ndim != 3:
                raise ValueError(
                    f"gt_action_condition must have shape [T, D] or [B, T, D], got {tuple(gt_action_clean.shape)}"
                )
            if int(gt_action_clean.shape[1]) != int(action_chunk):
                raise ValueError(
                    f"gt_action_condition length ({gt_action_clean.shape[1]}) must match action_chunk ({action_chunk})"
                )
            if int(gt_action_clean.shape[2]) != int(action_dim):
                raise ValueError(
                    f"gt_action_condition dim ({gt_action_clean.shape[2]}) must match action_dim ({action_dim})"
                )
            if gt_action_clean.shape[0] == batch_size and num_videos_per_prompt > 1:
                gt_action_clean = gt_action_clean.repeat_interleave(num_videos_per_prompt, dim=0)
            elif gt_action_clean.shape[0] == 1 and effective_batch_size > 1:
                gt_action_clean = gt_action_clean.repeat(effective_batch_size, 1, 1)
            elif gt_action_clean.shape[0] != effective_batch_size:
                raise ValueError(
                    f"gt_action_condition batch {gt_action_clean.shape[0]} must match prompt batch "
                    f"{batch_size} or effective batch {effective_batch_size}"
                )
            gt_action_clean = gt_action_clean.to(dtype=transformer_dtype, device=device)
        pred_action_dummy = torch.zeros_like(gt_action_clean)
        stage2_prefix_cache = None
        use_stage2_prefix_cache = use_stage1_prefix_cache
        if use_stage2_prefix_cache:
            stage2_prefix_cache = self.transformer.build_inference_future_action_prefix_cache(
                ref_latents=stage1_ref_latents,
                encoder_hidden_states=prompt_embeds,
                state=state,
                gt_action=gt_action_clean,
                attention_kwargs=attention_kwargs,
            )

        latent_model_input = None

        # --------------------------------------------------------------
        # Stage 2: diffuse future_state / value / future_image jointly, conditioned
        # on the clean gt_action from Stage 1. Sequence:
        # [state | ref | pred_action(dummy) | gt_action | future_state | value | future_image].
        # When action_only=True, this stage still runs for future_state/value and only
        # skips future_image denoising.
        # --------------------------------------------------------------
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                if action_only:
                    latent_model_input = stage1_ref_latents
                    noise_t = t.reshape(1, 1).to(
                        device=latent_model_input.device, dtype=latent_model_input.dtype
                    ).expand(latent_model_input.shape[0], 1)
                    timestep_s2 = build_teacher_forcing_per_token_timestep(
                        batch_size=latent_model_input.shape[0],
                        num_state_tokens=int(state.shape[1]),
                        num_ref_tokens=frame_per_tokens,
                        num_pred_action_tokens=int(pred_action_dummy.shape[1]),
                        num_gt_action_tokens=int(gt_action_clean.shape[1]),
                        num_future_state_tokens=int(future_state.shape[1]),
                        num_value_tokens=int(value.shape[1]),
                        num_noisy_latent_tokens=0,
                        noise_t=noise_t,
                        device=latent_model_input.device,
                        dtype=latent_model_input.dtype,
                    )

                    with current_model.cache_context("cond"):
                        _, _, future_state_pred, value_pred = current_model(
                            ref_latents=latent_model_input,
                            timestep=timestep_s2,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                            pred_action=pred_action_dummy,
                            gt_action=gt_action_clean,
                            state=state,
                            future_state=future_state,
                            value=value,
                            action_only=True,
                            prefix_cache=stage2_prefix_cache,
                        )
                else:
                    if self.config.expand_timesteps:
                        latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                        latent_model_input = latent_model_input.to(transformer_dtype)
                        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                    else:
                        latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                        timestep = t.expand(latents.shape[0])

                    noise_t = timestep[:, -2:-1]
                    timestep_s2 = build_teacher_forcing_per_token_timestep(
                        batch_size=latent_model_input.shape[0],
                        num_state_tokens=int(state.shape[1]),
                        num_ref_tokens=frame_per_tokens,
                        num_pred_action_tokens=int(pred_action_dummy.shape[1]),
                        num_gt_action_tokens=int(gt_action_clean.shape[1]),
                        num_future_state_tokens=int(future_state.shape[1]),
                        num_value_tokens=int(value.shape[1]),
                        num_noisy_latent_tokens=frame_per_tokens * (int(first_frame_mask.shape[2]) - 1),
                        noise_t=noise_t,
                        device=latent_model_input.device,
                        dtype=latent_model_input.dtype,
                    )

                    with current_model.cache_context("cond"):
                        model_out = current_model(
                            ref_latents=latent_model_input[:, :, :1],
                            noisy_latents=latent_model_input[:, :, 1:],
                            timestep=timestep_s2,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                            pred_action=pred_action_dummy,
                            gt_action=gt_action_clean,
                            state=state,
                            future_state=future_state,
                            value=value,
                            action_only=False,
                        )
                        noise_pred, _, future_state_pred, value_pred = model_out

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            uncond_out = current_model(
                                ref_latents=latent_model_input[:, :, :1],
                                noisy_latents=latent_model_input[:, :, 1:],
                                timestep=timestep_s2,
                                encoder_hidden_states=negative_prompt_embeds,
                                return_dict=False,
                                pred_action=pred_action_dummy,
                                gt_action=gt_action_clean,
                                state=state,
                                future_state=future_state,
                                value=value,
                                action_only=False,
                            )
                            noise_uncond, _, _, _ = uncond_out
                            noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                future_state = self.future_state_scheduler.step(
                    future_state_pred, t, future_state, return_dict=False
                )[0]
                value = self.value_scheduler.step(value_pred, t, value, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if _XLA_AVAILABLE:
                    xm.mark_step()

        imgs = None
        if not action_only:
            latents[:, :, :1] = latent_model_input[:, :, :1]
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )

            latents = latents / latents_std + latents_mean
            imgs = self.vae.decode(latents.bfloat16(), return_dict=False)[0]

        if not return_dict:
            return imgs, action, future_state, value

        return {"images": imgs, "action": action, "future_state": future_state, "value": value}
