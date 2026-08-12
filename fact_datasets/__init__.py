__version__ = '1.0.0'

from .collators import DefaultCollator
from .datasets import (
    BaseDataset,
    BaseProcessor,
    ConcatDataset,
    Dataset,
    load_config,
    load_dataset,
    register_dataset,
)
from .samplers import DefaultSampler, EpisodeMixtureSampler
from .utils import is_lerobot_available

# Mirror the guard in datasets/__init__.py: importing LeRobotDataset
# unconditionally breaks `import fact_datasets` on an install without lerobot.
if is_lerobot_available():
    from .datasets import LeRobotDataset  # noqa: F401
