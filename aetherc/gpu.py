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


_OPENCL_LIBRARIES = (
    r"C:\Windows\System32\OpenCL.dll",
    r"C:\Windows\SysWOW64\OpenCL.dll",
)
_OPENCL_ICD_REGISTRY_KEYS = (
    r"SOFTWARE\Khronos\OpenCL\Vendors",
    r"SOFTWARE\WOW6432Node\Khronos\OpenCL\Vendors",
)


def _windows_opencl_icd_registered() -> bool:
    """A DLL alone is not enough: a vendor ICD must be registered, otherwise
    calling into the stub loader crashes. The Khronos key proves it."""
    try:
        import winreg

        for key_path in _OPENCL_ICD_REGISTRY_KEYS:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path):
                    return True
            except OSError:
                continue
    except ImportError:
        pass
    return False


@lru_cache(maxsize=1)
def probe_opencl() -> bool:
    """Detect a *working* OpenCL runtime (loader DLL + registered vendor ICD)."""
    if os.name == "nt":
        dll = any(os.path.isfile(path) for path in _OPENCL_LIBRARIES)
        return dll and _windows_opencl_icd_registered()
    for path in (
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
        "/usr/lib/libOpenCL.so.1",
        "/usr/lib/libOpenCL.so",
        "/opt/rocm/lib/libOpenCL.so",
    ):
        if os.path.isfile(path):
            return True
    return shutil.which("clinfo") is not None


@lru_cache(maxsize=1)
def probe_gpu() -> dict:
    """Detect display/discrete GPUs, CUDA toolchain, and OpenCL runtime."""
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

    opencl = probe_opencl()

    if vendor == "nvidia" and nvcc and cuda_gpu:
        tier = "cuda"
    elif opencl and vendor in ("nvidia", "amd", "intel"):
        tier = "opencl"
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
        "opencl": opencl,
        "tier": tier,
    }


def cuda_ready() -> bool:
    info = probe_gpu()
    return bool(info["cuda_gpu"] and info["nvcc"])


def opencl_ready() -> bool:
    return bool(probe_gpu().get("opencl"))


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
    if preferred == "opencl":
        # Emission + compilation need no OpenCL SDK (the generated binary loads
        # the ICD at runtime); only execution reports the honest fallback.
        return "opencl", None
    if preferred == "auto":
        if cuda_ready():
            return "cuda", None
        if opencl_ready():
            return "opencl", "auto-dispatched to the OpenCL tier"
        return "c", None
    return preferred, None
