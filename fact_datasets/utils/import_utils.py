from diffusers.utils.import_utils import _is_package_available
from packaging.version import Version

_lerobot_available, _lerobot_version = _is_package_available('lerobot')


def is_lerobot_available() -> bool:
    """Check if the optional dependency 'lerobot' is available in the expected
    version.

    Returns:
        bool: True if the package 'lerobot' is installed and its version is compatible with the wrapper.
    """
    if not _lerobot_available:
        return False
    if Version(_lerobot_version) < Version('0.3.2'):
        return False
    return True
