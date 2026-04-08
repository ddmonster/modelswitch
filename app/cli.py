"""ModelSwitch CLI entry point.

Usage:
    modelswitch --workspace /path/to/dir  # set workspace (default: ~/.modelswitch)
    modelswitch --install                 # install systemd/launchd service
    modelswitch --uninstall               # remove systemd/launchd service
    modelswitch --start                   # start the server (foreground)
    modelswitch --status                  # check if service is installed/running
    modelswitch --version                 # print version
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="modelswitch",
        description="ModelSwitch LLM Gateway Proxy",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        metavar="DIR",
        help="Set workspace directory (default: ~/.modelswitch). "
        "Also respected via MODELSWITCH_WORKSPACE env var.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install system service (systemd on Linux, launchd on macOS)",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove installed system service",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the ModelSwitch server in foreground",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show service installation and workspace status",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="store_true",
        help="Print version and exit",
    )

    args = parser.parse_args(argv)

    if args.version:
        from app import __version__

        print(f"modelswitch {__version__}")
        return

    # Resolve workspace and ensure config exists
    from app.workspace import config_path, ensure_config_exists, resolve_workspace

    workspace = resolve_workspace(args.workspace)
    created = ensure_config_exists()

    if created:
        print(f"Created default config at {config_path()}")
        print("Edit this file to configure your providers and API keys before starting.")

    if args.install:
        _cmd_install(workspace)
    elif args.uninstall:
        _cmd_uninstall()
    elif args.start:
        _cmd_start(workspace)
    elif args.status:
        _cmd_status(workspace)
    else:
        print(f"Workspace: {workspace}")
        print(f"Config:    {config_path()}")
        if created:
            print("\nNew config created. Edit it before starting the server.")
        else:
            print("\nConfig exists. Run 'modelswitch --start' to launch the server.")


# ── Install commands ──────────────────────────────────────────────────


def _cmd_install(workspace: Path) -> None:
    """Install system service."""
    system = platform.system()
    if system == "Linux":
        _install_systemd(workspace)
    elif system == "Darwin":
        _install_launchd(workspace)
    else:
        print(f"ERROR: Service installation not supported on {system}")
        sys.exit(1)


def _install_systemd(workspace: Path) -> None:
    """Generate and install systemd user unit file."""
    unit_content = f"""\
[Unit]
Description=ModelSwitch LLM Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={sys.executable} -m app.cli_inner --workspace {workspace} --start
Environment=MODELSWITCH_WORKSPACE={workspace}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / "modelswitch.service"
    unit_path.write_text(unit_content)

    # Enable lingering so the service runs without user login
    subprocess.run(
        ["loginctl", "enable-linger", os.environ.get("USER", "")],
        check=False,
    )

    print(f"Installed systemd user service: {unit_path}")
    print("To start:  systemctl --user start modelswitch")
    print("To enable: systemctl --user enable modelswitch")
    print("To view logs: journalctl --user -u modelswitch -f")


def _install_launchd(workspace: Path) -> None:
    """Generate and install macOS launchd plist."""
    plist_name = "com.modelswitch.gateway"
    plist_content = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{plist_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>-m</string>
        <string>app.cli_inner</string>
        <string>--workspace</string>
        <string>{workspace}</string>
        <string>--start</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MODELSWITCH_WORKSPACE</key>
        <string>{workspace}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{workspace}/logs/launchd-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{workspace}/logs/launchd-stderr.log</string>
</dict>
</plist>
"""
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / f"{plist_name}.plist"
    plist_path.write_text(plist_content)

    print(f"Installed launchd agent: {plist_path}")
    print(f"To load:   launchctl load ~/Library/LaunchAgents/{plist_name}.plist")
    print(f"Logs:      {workspace}/logs/launchd-stdout.log")


# ── Uninstall ─────────────────────────────────────────────────────────


def _cmd_uninstall() -> None:
    """Remove installed system service."""
    system = platform.system()
    if system == "Linux":
        unit_path = (
            Path.home() / ".config" / "systemd" / "user" / "modelswitch.service"
        )
        if unit_path.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "modelswitch"], check=False
            )
            subprocess.run(
                ["systemctl", "--user", "stop", "modelswitch"], check=False
            )
            unit_path.unlink()
            print(f"Removed systemd service: {unit_path}")
            print("Run 'systemctl --user daemon-reload' to clean up.")
        else:
            print("No systemd service found.")
    elif system == "Darwin":
        plist_name = "com.modelswitch.gateway"
        plist_path = (
            Path.home() / "Library" / "LaunchAgents" / f"{plist_name}.plist"
        )
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
            plist_path.unlink()
            print(f"Removed launchd agent: {plist_path}")
        else:
            print("No launchd agent found.")
    else:
        print(f"Service uninstall not supported on {platform.system()}")


# ── Start ─────────────────────────────────────────────────────────────


def _cmd_start(workspace: Path) -> None:
    """Start the uvicorn server in foreground."""
    from app.workspace import data_dir, logs_dir

    logs_dir().mkdir(parents=True, exist_ok=True)
    data_dir().mkdir(parents=True, exist_ok=True)

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",
    )


# ── Status ────────────────────────────────────────────────────────────


def _cmd_status(workspace: Path) -> None:
    """Show current status."""
    from app.workspace import config_path, data_dir, logs_dir

    print(f"Workspace: {workspace}")
    print(
        f"Config:    {config_path()} ({'exists' if config_path().exists() else 'MISSING'})"
    )
    print(f"Logs dir:  {logs_dir()}")
    print(f"Data dir:  {data_dir()}")

    system = platform.system()
    if system == "Linux":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "modelswitch"],
            capture_output=True,
            text=True,
        )
        status = result.stdout.strip() if result.returncode == 0 else "not installed"
        print(f"Service:   {status}")
    elif system == "Darwin":
        plist_name = "com.modelswitch.gateway"
        plist_path = (
            Path.home() / "Library" / "LaunchAgents" / f"{plist_name}.plist"
        )
        if plist_path.exists():
            print(f"Service:   installed ({plist_path})")
        else:
            print("Service:   not installed")
