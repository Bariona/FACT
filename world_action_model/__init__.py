"""FACT world-action model.

``apply_numa_affinity`` (pins the process to its GPU's NUMA node) and
``apply_runtime_compat`` (SDPA ``enable_gqa`` shim for Torch < 2.5) are opt-in:
importing a library should not reconfigure the host process. Entrypoints call
them directly; ``FACT_AUTO_RUNTIME_SETUP=1`` applies both at import instead.
"""

import os as _os

from .numa_affinity import apply_numa_affinity
from .runtime_compat import apply_runtime_compat

__all__ = ["apply_numa_affinity", "apply_runtime_compat"]

if _os.environ.get("FACT_AUTO_RUNTIME_SETUP", "0").strip().lower() in ("1", "true", "yes", "on"):
    apply_numa_affinity()
    apply_runtime_compat()
