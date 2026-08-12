from __future__ import annotations

from typing import Dict, Tuple

import PIL
import torch
import torch.nn.functional as torch_F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


ROBOTWIN_VIEW_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def get_robotwin_side_view_size(main_dst_size: Tuple[int, int]) -> Tuple[int, int]:
    main_width, main_height = main_dst_size
    return main_width // 2, main_height // 2


def get_robotwin_canvas_size(main_dst_size: Tuple[int, int]) -> Tuple[int, int]:
    main_width, main_height = main_dst_size
    side_width, _ = get_robotwin_side_view_size(main_dst_size)
    return main_width + side_width, main_height


def infer_robotwin_main_view_size(canvas_size: Tuple[int, int]) -> Tuple[int, int]:
    canvas_width, canvas_height = canvas_size
    main_width = (int(canvas_width) * 2) // 3
    if main_width <= 0 or int(canvas_height) <= 0:
        raise ValueError(f"Invalid canvas_size={canvas_size}")
    if main_width + main_width // 2 != int(canvas_width):
        raise ValueError(
            f"canvas_size={canvas_size} is incompatible with the RoboTwin side-stack layout"
        )
    return main_width, int(canvas_height)


def tensor_chw01_to_pil_rgb(image_chw: torch.Tensor) -> Image.Image:
    if image_chw.ndim != 3:
        raise ValueError(f"expected CHW tensor, got {image_chw.shape}")
    image = image_chw.detach().cpu()
    if image.dtype != torch.float32:
        image = image.float()
    if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
        image = image.permute(2, 0, 1)
    if image.shape[0] not in (1, 3):
        raise ValueError(f"expected CHW or HWC image tensor, got {image_chw.shape}")
    if image.max().item() > 1.0:
        image = image / 255.0
    image = image.clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return PIL.Image.fromarray(image)


def resize_tensor_image_or_video(image: torch.Tensor, dst_size: Tuple[int, int]) -> torch.Tensor:
    if image.ndim not in (3, 4):
        raise ValueError(f"expected CHW or TCHW tensor, got {tuple(image.shape)}")
    dst_width, dst_height = dst_size
    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    image = F.resize(image, (int(dst_height), int(dst_width)), InterpolationMode.BILINEAR)
    if squeeze:
        image = image.squeeze(0)
    return image


def _to_nchw_uint8_float01(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        if image.shape[0] in (1, 3):
            image = image.unsqueeze(0)
        elif image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"expected CHW or HWC image tensor, got {tuple(image.shape)}")
    elif image.ndim == 4:
        if image.shape[1] in (1, 3):
            pass
        elif image.shape[-1] in (1, 3):
            image = image.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"expected NCHW or NHWC image tensor, got {tuple(image.shape)}")
    else:
        raise ValueError(f"expected image tensor with 3 or 4 dims, got {tuple(image.shape)}")

    if image.dtype != torch.uint8:
        image_f = image.to(dtype=torch.float32)
        if image_f.numel() > 0 and float(image_f.max().item()) <= 1.0:
            image_f = image_f * 255.0
        image = image_f.clamp(0.0, 255.0).to(dtype=torch.uint8)
    return image.to(dtype=torch.float32) / 255.0


def _resize_float01_image_or_video(image: torch.Tensor, dst_size: Tuple[int, int]) -> torch.Tensor:
    dst_width, dst_height = dst_size
    if tuple(image.shape[-2:]) == (int(dst_height), int(dst_width)):
        return image
    return torch_F.interpolate(
        image,
        size=(int(dst_height), int(dst_width)),
        mode="bilinear",
        align_corners=False,
    )


def build_robotwin_ref_tensor(
    images: Dict[str, torch.Tensor],
    main_dst_size: Tuple[int, int],
) -> torch.Tensor:
    high = _resize_float01_image_or_video(_to_nchw_uint8_float01(images[ROBOTWIN_VIEW_KEYS[0]]), main_dst_size)
    side_dst_size = get_robotwin_side_view_size(main_dst_size)
    left = _resize_float01_image_or_video(_to_nchw_uint8_float01(images[ROBOTWIN_VIEW_KEYS[1]]), side_dst_size)
    right = _resize_float01_image_or_video(_to_nchw_uint8_float01(images[ROBOTWIN_VIEW_KEYS[2]]), side_dst_size)
    return build_robotwin_three_view_tensor(
        {
            ROBOTWIN_VIEW_KEYS[0]: high,
            ROBOTWIN_VIEW_KEYS[1]: left,
            ROBOTWIN_VIEW_KEYS[2]: right,
        },
        main_dst_size=main_dst_size,
    ).squeeze(0)


def resize_pil_image(image: Image.Image, dst_size: Tuple[int, int]) -> Image.Image:
    dst_width, dst_height = dst_size
    return image.resize((int(dst_width), int(dst_height)), resample=Image.BILINEAR)


def build_robotwin_three_view_tensor(
    images: Dict[str, torch.Tensor],
    main_dst_size: Tuple[int, int],
) -> torch.Tensor:
    high = resize_tensor_image_or_video(images[ROBOTWIN_VIEW_KEYS[0]], main_dst_size)
    side_dst_size = get_robotwin_side_view_size(main_dst_size)
    left = resize_tensor_image_or_video(images[ROBOTWIN_VIEW_KEYS[1]], side_dst_size)
    right = resize_tensor_image_or_video(images[ROBOTWIN_VIEW_KEYS[2]], side_dst_size)
    side_stack = torch.cat([left, right], dim=-2)
    return torch.cat([high, side_stack], dim=-1)


def build_robotwin_ref_image(
    images: Dict[str, torch.Tensor],
    main_dst_size: Tuple[int, int],
) -> Image.Image:
    high = resize_pil_image(tensor_chw01_to_pil_rgb(images[ROBOTWIN_VIEW_KEYS[0]]), main_dst_size)
    side_dst_size = get_robotwin_side_view_size(main_dst_size)
    left = resize_pil_image(tensor_chw01_to_pil_rgb(images[ROBOTWIN_VIEW_KEYS[1]]), side_dst_size)
    right = resize_pil_image(tensor_chw01_to_pil_rgb(images[ROBOTWIN_VIEW_KEYS[2]]), side_dst_size)

    _, side_height = side_dst_size
    canvas_width, canvas_height = get_robotwin_canvas_size(main_dst_size)
    out = Image.new("RGB", (canvas_width, canvas_height))
    out.paste(high, (0, 0))
    out.paste(left, (main_dst_size[0], 0))
    out.paste(right, (main_dst_size[0], side_height))
    return out
