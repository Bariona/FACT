import os


def get_root_dir() -> str:
    """Project root directory inferred from current file path."""
    return os.path.abspath(__file__).split('fact_datasets')[0][:-1]


def get_data_dir() -> str:
    """Data directory read from env var ``FACT_DATASETS_DIR`` with fallback."""
    return os.environ.get('FACT_DATASETS_DIR', './data/')
