from typing import Any

import numpy as np
import torch


def to_numpy(data: Any):
    """Recursively convert nested containers to NumPy arrays when possible."""
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = to_numpy(val)
    elif isinstance(data, list):
        data = [to_numpy(d) for d in data]
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cpu().numpy()
    elif hasattr(data, 'tensor'):
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)
    return data
