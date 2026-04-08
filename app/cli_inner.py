"""Internal entry point for service invocation.

Called by systemd/launchd as: python -m app.cli_inner --workspace /path --start

This exists separately from app.cli so that service unit files can invoke
the installed Python module directly (which is always on the venv path)
without depending on the console_scripts wrapper being on PATH.
"""

from __future__ import annotations

import sys


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    if not args.start:
        print("Use: python -m app.cli_inner --workspace /path --start")
        sys.exit(1)

    from app.cli import _cmd_start
    from app.workspace import data_dir, logs_dir, resolve_workspace

    resolve_workspace(args.workspace)
    logs_dir().mkdir(parents=True, exist_ok=True)
    data_dir().mkdir(parents=True, exist_ok=True)
    _cmd_start(resolve_workspace())


if __name__ == "__main__":
    main()
