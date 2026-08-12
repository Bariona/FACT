import json
import random

import torch
from torchvision import transforms


class WATransforms:
    """Shared base: image normalization, ref-frame mask generator, norm stats.

    Subclasses implement ``__call__``; see ``WATransformsLerobot``.
    """

    def __init__(
        self,
        is_train=False,
        dst_size=None,
        num_frames=1,
        fps=16,
        norm_path=None,
        image_cfg=None,
        num_views=1,
        t5_len=32,
    ):
        self.fps = fps
        self.is_train = is_train
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.dst_size = dst_size
        self.num_frames = num_frames
        self.image_cfg = image_cfg
        self.mask_generator = MaskGenerator(**image_cfg['mask_generator'])
        self.num_views = num_views
        self.t5_len = int(t5_len)

        with open(norm_path, "r", encoding="utf-8") as f:
            self.stats_dict = json.load(f)
        print("Loading stats dict from:", norm_path)
        self.use_delta = True
        print("Using delta mode")

    def __call__(self, data_dict):
        raise NotImplementedError("WATransforms is a base class; use WATransformsLerobot.")


class MaskGenerator:
    def __init__(self, max_ref_frames, factor=8, start=1):
        assert max_ref_frames > 0 and (max_ref_frames - 1) % factor == 0
        self.max_ref_frames = max_ref_frames
        self.factor = factor
        self.start = start
        self.max_ref_latents = 1 + (max_ref_frames - 1) // factor
        assert self.start <= self.max_ref_latents

    def get_mask(self, num_frames):
        assert num_frames > 0 and (num_frames - 1) % self.factor == 0 and num_frames >= self.max_ref_frames
        num_latents = 1 + (num_frames - 1) // self.factor
        num_ref_latents = random.randint(self.start, self.max_ref_latents)
        if num_ref_latents > 0:
            num_ref_frames = 1 + (num_ref_latents - 1) * self.factor
        else:
            num_ref_frames = 0
        ref_masks = torch.zeros((num_frames,), dtype=torch.float32)
        ref_masks[:num_ref_frames] = 1
        ref_latent_masks = torch.zeros((num_latents,), dtype=torch.float32)
        ref_latent_masks[:num_ref_latents] = 1
        return ref_masks, ref_latent_masks
