"""Build a standalone `aether` binary with PyInstaller (one command).

    python build_standalone.py

Produces `dist/aether.exe` (Windows) or `dist/aether` (Linux/macOS): a single
executable embedding the compiler, the Python backend (NumPy/OpenBLAS), and the
language spec — no system Python required to run it. The native C backend still
uses gcc/clang/MSVC at runtime (found via PATH / AETHER_CC), which is by design:
Aether *emits and compiles* C, it does not need Python for that.

`dist/` and `build/` are gitignored; attach release zips to GitHub Releases.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SEP = os.pathsep


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is required: python -m pip install pyinstaller")
        return 1

    for directory in (DIST, BUILD):
        shutil.rmtree(directory, ignore_errors=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "aether",
        "--clean",
        "--noconfirm",
        # Bundle the language spec so `aether mcp` works from the binary.
        "--add-data",
        f"{ROOT / 'AETHER_SPEC.md'}{SEP}.",
        # The examples make the binary useful out of the box.
        "--add-data",
        f"{ROOT / 'hello_aether.ae'}{SEP}.",
        "--add-data",
        f"{ROOT / 'examples' / 'control_flow.ae'}{SEP}examples",
        str(ROOT / "aether.py"),
    ]
    print("building:", " ".join(command))
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        print("PyInstaller failed")
        return result.returncode

    binary = DIST / ("aether.exe" if os.name == "nt" else "aether")
    if not binary.is_file():
        print(f"expected binary missing: {binary}")
        return 1
    size_mb = binary.stat().st_size / (1024 * 1024)
    print(f"\nbuilt {binary} ({size_mb:.1f} MB)")

    print("\nsmoke test (version + compile + run):")
    for args in (["version"], ["check", "hello_aether.ae", "--compact"]):
        probe = subprocess.run(
            [str(binary), *args], capture_output=True, text=True, cwd=str(ROOT)
        )
        print(f"  aether {' '.join(args)} -> rc={probe.returncode}")
        if probe.stdout.strip():
            print("   ", probe.stdout.strip().splitlines()[0])
        if probe.returncode != 0:
            return probe.returncode

    print(
        "\nDone. The binary is self-contained (Python runtime + NumPy + OpenBLAS\n"
        "bundled). The native C backend still invokes gcc/clang/MSVC at runtime —\n"
        "that is Aether compiling your intent to machine code, not a dependency."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
