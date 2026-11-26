"""Hardware probing: discrete GPUs, CUDA toolkits, and device tiers.

`auto_hardware` uses this to pick the best backend and to report honest
telemetry. Probing is cached per process (WMI/nvidia-smi calls are not free).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache

_NVIDIA_SMI_CANDIDATES = (
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)


@lru_cache(maxsize=1)
def probe_gpu() -> dict:
    """Detect display/discrete GPUs and a usable CUDA toolchain."""
    names: list[str] = []
    try:
        output = subprocess.run(
            ["wmic", "path", "win32_videocontroller", "get", "name"],
            capture_output=True,
            text=True,
            timeout=10,
            errors="replace",
        ).stdout
        for line in output.splitlines()[1:]:
            line = line.strip()
            if line:
                names.append(line)
    except Exception:
        pass

    gpu_name = names[0] if names else None
    vendor = "unknown"
    lowered = " ".join(names).lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered:
        vendor = "nvidia"
    elif "radeon" in lowered or "amd" in lowered:
        vendor = "amd"
    elif "intel" in lowered or "arc" in lowered or "iris" in lowered:
        vendor = "intel"

    nvcc = shutil.which("nvcc")
    nvidia_smi = shutil.which("nvidia-smi") or next(
        (p for p in _NVIDIA_SMI_CANDIDATES if os.path.isfile(p)), None
    )
    cuda_gpu = False
    if nvidia_smi:
        try:
            listing = subprocess.run(
                [nvidia_smi, "-L"],
                capture_output=True,
                text=True,
                timeout=10,
                errors="replace",
            ).stdout
            cuda_gpu = listing.strip().lower().startswith("gpu")
            if cuda_gpu:
                vendor = "nvidia"
                gpu_name = listing.strip().splitlines()[0]
        except Exception:
            cuda_gpu = False

    if vendor == "nvidia" and nvcc and cuda_gpu:
        tier = "cuda"
    elif vendor in ("nvidia", "amd"):
        tier = "discrete-gpu"
    elif vendor == "intel":
        tier = "integrated-gpu"
    else:
        tier = "cpu"

    return {
        "vendor": vendor,
        "name": gpu_name,
        "cuda_gpu": cuda_gpu,
        "nvcc": nvcc,
        "tier": tier,
    }


def cuda_ready() -> bool:
    info = probe_gpu()
    return bool(info["cuda_gpu"] and info["nvcc"])


def dispatch_target(preferred: str = "auto") -> tuple[str, str | None]:
    """Resolve a requested target to an executable backend + note."""
    if preferred == "cuda":
        if cuda_ready():
            return "cuda", None
        info = probe_gpu()
        if not info["cuda_gpu"]:
            return (
                "c",
                "no CUDA-capable GPU detected; dispatching to the C99 backend",
            )
        return (
            "c",
            "CUDA GPU present but nvcc was not found; dispatching to the C99 backend",
        )
    if preferred == "auto":
        if cuda_ready():
            return "cuda", None
        return "c", None
    return preferred, None
