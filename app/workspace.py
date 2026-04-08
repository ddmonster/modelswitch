"""Workspace resolution for ModelSwitch.

The workspace is the root directory containing config.yaml, logs/, and data/.
Resolved in priority order:
  1. --workspace CLI flag
  2. MODELSWITCH_WORKSPACE environment variable
  3. ~/.modelswitch (default)
"""

from __future__ import annotations

import os
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Optional

_workspace: Optional[Path] = None


def resolve_workspace(explicit: Optional[str] = None) -> Path:
    """Resolve workspace directory from CLI flag, env var, or default."""
    global _workspace
    if _workspace is not None:
        return _workspace

    if explicit:
        _workspace = Path(explicit).resolve()
    elif os.environ.get("MODELSWITCH_WORKSPACE"):
        _workspace = Path(os.environ["MODELSWITCH_WORKSPACE"]).resolve()
    else:
        _workspace = Path.home() / ".modelswitch"

    _workspace.mkdir(parents=True, exist_ok=True)
    return _workspace


def get_workspace() -> Path:
    """Return the workspace directory. Call resolve_workspace() first."""
    if _workspace is None:
        resolve_workspace()
    return _workspace  # type: ignore[return-value]


def config_path() -> Path:
    """Return path to config.yaml inside the workspace."""
    return get_workspace() / "config.yaml"


def logs_dir() -> Path:
    """Return path to the logs directory inside the workspace."""
    return get_workspace() / "logs"


def data_dir() -> Path:
    """Return path to the data directory inside the workspace."""
    return get_workspace() / "data"


def web_dir() -> Path:
    """Return path to the web/ static files bundled with the package."""
    return Path(str(files("app") / "web"))


def config_example_path() -> Path:
    """Return path to config.yaml.example bundled with the package."""
    return Path(str(files("app") / "config.yaml.example"))


def ensure_config_exists() -> bool:
    """Copy config.yaml.example to workspace/config.yaml if it does not exist.

    Returns True if a new config was created, False if one already existed.
    """
    cfg = config_path()
    if not cfg.exists():
        example = config_example_path()
        if example.exists():
            shutil.copy2(str(example), str(cfg))
            return True
    return False
