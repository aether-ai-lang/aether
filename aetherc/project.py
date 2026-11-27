"""`aether init` and `aether install`: project scaffolding + package manager.

Projects are plain directories with an `aether.toml` manifest, `src/main.ae`,
and optionally `.aether_packages/` holding installed reusable `.ae` modules
(function libraries). `use pkg.<name>` resolves against `.aether_packages/`
found near the source file.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .diagnostics import AETHER_VERSION

MANIFEST_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
aether = "{version}"
entry = "src/main.ae"

[dependencies]
# Install more with: aether install <git-url-or-path>
"""

MAIN_TEMPLATE = """\
use ai.core
use agent.core

task Main:
    probe = tensor(8, 8) -> auto_hardware
    log sum(probe)
    log mean(probe)
    let goal = "Ship something great"
    let steps = plan(goal) -> max_steps(3)
    log steps
    log "Project {name} is live"
"""

README_TEMPLATE = """\
# {name}

An [Aether](https://github.com/aether-ai-lang/aether) project.

## Run

```powershell
aether run src/main.ae
```

## Layout

- `aether.toml` — project manifest (name, version, dependencies)
- `src/main.ae` — entry program
- `.aether_packages/` — installed modules (`aether install <git-url-or-path>`)
"""

GITIGNORE_TEMPLATE = """\
__pycache__/
.aether_build/
*.exe
*.bin
*.memory.json
dist/
"""


def init_project(name: str, parent: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name):
        raise ValueError(
            "project name must start with a letter/underscore and use only "
            "letters, digits, '_' or '-'"
        )
    root = parent / name
    if root.exists():
        raise FileExistsError(f"directory already exists: {root}")
    src = root / "src"
    src.mkdir(parents=True)
    (root / "aether.toml").write_text(
        MANIFEST_TEMPLATE.format(name=name, version=AETHER_VERSION),
        encoding="utf-8",
        newline="\n",
    )
    (src / "main.ae").write_text(
        MAIN_TEMPLATE.format(name=name), encoding="utf-8", newline="\n"
    )
    (root / "README.md").write_text(
        README_TEMPLATE.format(name=name), encoding="utf-8", newline="\n"
    )
    (root / ".gitignore").write_text(
        GITIGNORE_TEMPLATE, encoding="utf-8", newline="\n"
    )
    (root / ".aether_packages").mkdir()
    return root


def _slug_from_url(source: str) -> str:
    text = source.rstrip("/").rstrip("\\")
    if text.endswith(".git"):
        text = text[:-4]
    slug = re.split(r"[\\/]", text)[-1]
    return slug or "package"


def install_package(source: str, project_root: Path) -> tuple[str, str]:
    """Install a package into `<project_root>/.aether_packages/<name>`.

    Returns (package_name, how). Sources: a git URL, a local directory of
    `.ae` modules, or a local `.ae` file (copied as a single-file package).
    """
    packages = project_root / ".aether_packages"
    packages.mkdir(parents=True, exist_ok=True)

    if source.endswith(".git") or source.startswith(("http://", "https://", "git@")):
        name = _slug_from_url(source)
        target = packages / name
        if target.exists():
            raise FileExistsError(f"package already installed: {name}")
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", source, str(target)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git clone failed: {(proc.stderr or proc.stdout).strip()}")
        return name, "git"

    local = Path(source)
    if local.is_dir():
        name = _slug_from_url(source)
        ae_files = list(local.glob("*.ae"))
        if not ae_files:
            raise ValueError(f"no .ae modules found in {local}")
        target = packages / name
        if target.exists():
            raise FileExistsError(f"package already installed: {name}")
        target.mkdir(parents=True)
        for ae_file in ae_files:
            shutil.copy2(ae_file, target / ae_file.name)
        return name, "local directory"

    if local.is_file() and local.suffix == ".ae":
        name = local.stem
        target = packages / name
        if target.exists():
            raise FileExistsError(f"package already installed: {name}")
        target.mkdir(parents=True)
        shutil.copy2(local, target / local.name)
        return name, "single file"

    raise ValueError(
        f"cannot install '{source}': expected a git URL, a local directory "
        "with .ae modules, or a local .ae file"
    )


def find_packages_dir(start: Path) -> Path | None:
    """Locate `.aether_packages` near a source file (walks up a few levels)."""
    current = start if start.is_dir() else start.parent
    for _ in range(4):
        candidate = current / ".aether_packages"
        if candidate.is_dir():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def package_functions(package_dir: Path) -> list[Path]:
    """`.ae` files of an installed package that declare top-level functions."""
    return sorted(package_dir.glob("*.ae"))
