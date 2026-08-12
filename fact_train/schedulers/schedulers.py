import math

from .build import SCHEDULERS


class BaseScheduler(object):
    """Base scheduler that produces a value by interpolating start/end.

    Args:
        epoch_size: Number of steps in one epoch.
        max_epochs: Total number of epochs.
        max_steps: Total number of steps.
        start_value: Starting value for interpolation.
        end_value: Ending value for interpolation.
    """

    def __init__(self, epoch_size: int, max_epochs: int, max_steps: int, start_value: float = 1.0, end_value: float = 0.0) -> None:
        self.epoch_size = epoch_size
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.start_value = start_value
        self.end_value = end_value

    def get_value(self, num_update: int) -> float:
        assert 0 <= num_update <= self.max_steps
        start_factor, end_factor = self.get_factor(float(num_update))
        value = self.start_value * start_factor + self.end_value * end_factor
        return float(value)

    def get_factor(self, num_update: float) -> tuple[float, float]:
        raise NotImplementedError



@SCHEDULERS.register
class WarmupCosineScheduler(BaseScheduler):
    def __init__(self, warmup_steps: int, decay_steps: int, **kwargs) -> None:
        super(WarmupCosineScheduler, self).__init__(**kwargs)
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def get_factor(self, num_update: float) -> tuple[float, float]:
        if num_update == 0:
            factor = 1.0 / (self.warmup_steps + 1)
            return factor, 0
        elif num_update <= self.warmup_steps:
            factor = num_update / self.warmup_steps
            return factor, 0
        elif num_update <= self.decay_steps:
            alpha = (num_update - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
            factor = 0.5 * (1 + math.cos(math.pi * alpha))
            return factor, 1 - factor
        else:
            factor = 0
            return factor, 1 - factor

