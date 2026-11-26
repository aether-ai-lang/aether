"""Frozen-binary support (PyInstaller) and source-tree path helpers."""

from __future__ import annotations

import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))

# Source-tree root; inside a PyInstaller bundle this resolves to the
# extraction dir (sys._MEIPASS), where bundled data files live.
ROOT = Path(__file__).resolve().parent.parent


def entry_command(*args: str) -> list[str]:
    """Command vector that runs the aether CLI, frozen or from source."""
    if FROZEN:
        return [sys.executable, *args]
    return [sys.executable, str(ROOT / "aether.py"), *args]


def resource(name: str) -> Path:
    """Locate a bundled data file (works frozen and from source)."""
    return ROOT / name
