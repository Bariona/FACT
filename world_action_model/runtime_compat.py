import functools


def _patch_scaled_dot_product_attention() -> None:
    import torch.nn.functional as torch_F

    original_sdpa = getattr(torch_F, "scaled_dot_product_attention", None)
    if original_sdpa is None or getattr(original_sdpa, "_fact_enable_gqa_compat", False):
        return

    @functools.wraps(original_sdpa)
    def _scaled_dot_product_attention_compat(*args, **kwargs):
        try:
            return original_sdpa(*args, **kwargs)
        except TypeError as exc:
            if "enable_gqa" not in str(exc) or "enable_gqa" not in kwargs:
                raise
            kwargs = dict(kwargs)
            kwargs.pop("enable_gqa", None)
            return original_sdpa(*args, **kwargs)

    _scaled_dot_product_attention_compat._fact_enable_gqa_compat = True
    torch_F.scaled_dot_product_attention = _scaled_dot_product_attention_compat


def apply_runtime_compat() -> None:
    # diffusers==0.36.0 may pass `enable_gqa` to SDPA, but torch 2.4.x does not
    # expose that keyword yet. Patch once at package import time so training and
    # inference do not depend on a local site-packages modification.
    _patch_scaled_dot_product_attention()
