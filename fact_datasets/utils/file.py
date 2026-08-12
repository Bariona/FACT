import json
import os
import pickle
from typing import Any

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def load_file(file_path: str, **kwargs: Any) -> Any:
    """Load a structured file based on its extension.

    Args:
        file_path (str): Path to file ending with one of {'.pkl', '.pickle',
            '.json', '.yaml', '.yml'}.
        **kwargs (Any): Extra keyword arguments forwarded to the corresponding
            loader:
            - pickle.load(..., **kwargs)
            - json.load(..., **kwargs)
            - yaml.load(..., Loader=..., **kwargs)  (``Loader`` defaulted to CLoader if available)

    Returns:
        Any: The data structure loaded from the file.

    Raises:
        AssertionError: If the extension is unsupported.
    """
    if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        data = pickle.load(open(file_path, 'rb'), **kwargs)
    elif file_path.endswith('.json'):
        data = json.load(open(file_path, 'r'), **kwargs)
    elif file_path.endswith('.yaml') or file_path.endswith('yml'):
        kwargs.setdefault('Loader', Loader)
        data = yaml.load(open(file_path, 'r'), **kwargs)
    else:
        assert False
    return data


def save_file(file_path: str, data: Any, **kwargs: Any) -> None:
    """Save a Python object to disk based on file extension.

    Args:
        file_path (str): Destination path ending with one of {'.pkl', '.pickle',
            '.json', '.yaml', '.yml'}.
        data (Any): Object to serialize.
        **kwargs (Any): Extra keyword arguments forwarded to the corresponding
            dumper:
            - pickle.dump(obj, file, **kwargs)
            - json.dump(obj, file, indent=4, **kwargs)  (``indent`` defaulted to 4)
            - yaml.dump(obj, file, Dumper=..., **kwargs)  (``Dumper`` defaulted to CDumper if available)

    Raises:
        AssertionError: If the extension is unsupported.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        pickle.dump(data, open(file_path, 'wb'), **kwargs)
    elif file_path.endswith('.json'):
        kwargs.setdefault('indent', 4)
        json.dump(data, open(file_path, 'w'), **kwargs)
    elif file_path.endswith('.yaml') or file_path.endswith('yml'):
        kwargs.setdefault('Dumper', Dumper)
        yaml.dump(data, open(file_path, 'w'), **kwargs)
    else:
        assert False
