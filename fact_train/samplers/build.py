from typing import Any

from ..registry import Registry, build_module

SAMPLERS = Registry()


def build_sampler(params_or_type: dict | str | None, *args: Any, **kwargs: Any):
    """Build a sampler or batch sampler from registry using a config or type
    name.

    Args:
        params_or_type: Dict with key ``'type'`` or a type name string.
        *args, **kwargs: Forwarded to the sampler constructor.

    Returns:
        Any | None: Instantiated sampler or ``None`` if type missing.
    """
    return build_module(SAMPLERS, params_or_type, *args, **kwargs)


try:
    from fact_datasets import DefaultSampler, EpisodeMixtureSampler

    SAMPLERS.register_module(DefaultSampler)
    SAMPLERS.register_module(EpisodeMixtureSampler)

except ImportError:
    pass
