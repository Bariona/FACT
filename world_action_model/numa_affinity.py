"""Per-rank NUMA CPU affinity for multi-GPU training.

On dual-socket servers (e.g. 8×A100 split across 2 NUMA nodes), each distributed
rank should pin itself to the NUMA node hosting its GPU so that dataloader
workers (forked after we set affinity) and pinned memory buffers are allocated
on the local node via Linux first-touch policy, avoiding cross-UPI traffic.

Activated by importing this module and calling apply_numa_affinity(). No-ops
when LOCAL_RANK is unset, when sysfs NUMA info is unavailable, or when the
machine only exposes one NUMA node.
"""
import os
import subprocess


def _parse_cpulist(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _node_cpus(numa_node: int) -> list[int]:
    with open(f"/sys/devices/system/node/node{numa_node}/cpulist") as f:
        return _parse_cpulist(f.read())


def _gpu_numa_node(physical_gpu_id: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader",
                "-i", str(physical_gpu_id),
            ],
            text=True, timeout=5,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    # "00000000:07:00.0" -> "0000:07:00.0"
    bus_id = out.replace("00000000:", "0000:").lower()
    numa_path = f"/sys/bus/pci/devices/{bus_id}/numa_node"
    try:
        with open(numa_path) as f:
            node = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None
    return node if node >= 0 else None


def apply_numa_affinity() -> None:
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        return
    try:
        local_rank = int(local_rank_str)
    except ValueError:
        return

    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_vis:
        return
    try:
        visible_gpus = [int(x) for x in cuda_vis.split(",") if x.strip()]
    except ValueError:
        return
    if local_rank >= len(visible_gpus):
        return
    physical_gpu = visible_gpus[local_rank]

    numa_node = _gpu_numa_node(physical_gpu)
    if numa_node is None:
        return

    try:
        cpus = _node_cpus(numa_node)
    except (FileNotFoundError, ValueError):
        return
    if not cpus:
        return

    try:
        os.sched_setaffinity(0, set(cpus))
    except (OSError, AttributeError):
        return

    print(
        f"[numa_affinity] rank={local_rank} gpu={physical_gpu} "
        f"-> node{numa_node} ({len(cpus)} cpus)",
        flush=True,
    )
