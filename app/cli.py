"""ModelSwitch CLI entry point.

Usage:
    modelswitch --workspace /path/to/dir  # set workspace (default: ~/.modelswitch)
    modelswitch --install                 # install systemd/launchd service
    modelswitch --uninstall               # remove systemd/launchd service
    modelswitch --start                   # start the server (foreground)
    modelswitch --stop                    # stop system service
    modelswitch --restart                 # restart system service
    modelswitch --log                     # tail service logs
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

# systemd service paths
_SYSTEMD_UNIT_PATH = Path("/etc/systemd/system/modelswitch.service")


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
        "--stop",
        action="store_true",
        help="Stop the system service",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart the system service",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Tail service logs",
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
    elif args.stop:
        _cmd_stop()
    elif args.restart:
        _cmd_restart()
    elif args.log:
        _cmd_log(workspace)
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


# ── Service helpers ───────────────────────────────────────────────────


def _systemd_unit_path() -> Path:
    """Return the systemd unit file path."""
    return _SYSTEMD_UNIT_PATH


def _launchd_plist_name() -> str:
    return "com.modelswitch.gateway"


def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_launchd_plist_name()}.plist"


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
    """Generate and install systemd system unit file at /etc/systemd/system/."""
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
WantedBy=multi-user.target
"""
    unit_path = _systemd_unit_path()
    try:
        unit_path.write_text(unit_content)
    except PermissionError:
        print(f"ERROR: Permission denied writing to {unit_path}")
        print("Try running with sudo: sudo modelswitch --install")
        sys.exit(1)

    subprocess.run(["systemctl", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "enable", "modelswitch"], check=False)
    subprocess.run(["systemctl", "start", "modelswitch"], check=False)

    print(f"Installed systemd service: {unit_path}")
    print("Commands:")
    print("  modelswitch --status    # check status")
    print("  modelswitch --restart   # restart service")
    print("  modelswitch --stop      # stop service")
    print("  modelswitch --log       # view logs")
    print("  journalctl -u modelswitch -f")


def _install_launchd(workspace: Path) -> None:
    """Generate and install macOS launchd plist."""
    plist_name = _launchd_plist_name()
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
    plist_path = _launchd_plist_path()
    plist_path.write_text(plist_content)

    subprocess.run(["launchctl", "load", str(plist_path)], check=False)

    print(f"Installed launchd agent: {plist_path}")
    print("Commands:")
    print("  modelswitch --status    # check status")
    print("  modelswitch --restart   # restart service")
    print("  modelswitch --stop      # stop service")
    print("  modelswitch --log       # view logs")


# ── Uninstall ─────────────────────────────────────────────────────────


def _cmd_uninstall() -> None:
    """Remove installed system service."""
    system = platform.system()
    if system == "Linux":
        _cmd_stop()
        unit_path = _systemd_unit_path()
        if unit_path.exists():
            subprocess.run(["systemctl", "disable", "modelswitch"], check=False)
            try:
                unit_path.unlink()
            except PermissionError:
                print(f"ERROR: Permission denied removing {unit_path}")
                print("Try running with sudo: sudo modelswitch --uninstall")
                sys.exit(1)
            subprocess.run(["systemctl", "daemon-reload"], check=False)
            print(f"Removed systemd service: {unit_path}")
        else:
            print("No systemd service found.")
    elif system == "Darwin":
        plist_path = _launchd_plist_path()
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
            plist_path.unlink()
            print(f"Removed launchd agent: {plist_path}")
        else:
            print("No launchd agent found.")
    else:
        print(f"Service uninstall not supported on {platform.system()}")


# ── Stop ──────────────────────────────────────────────────────────────


def _cmd_stop() -> None:
    """Stop the system service."""
    system = platform.system()
    if system == "Linux":
        subprocess.run(["systemctl", "stop", "modelswitch"], check=False)
        print("Service stopped.")
    elif system == "Darwin":
        subprocess.run(["launchctl", "unload", str(_launchd_plist_path())], check=False)
        print("Service stopped.")
    else:
        print(f"Not supported on {system}")


# ── Restart ───────────────────────────────────────────────────────────


def _cmd_restart() -> None:
    """Restart the system service."""
    system = platform.system()
    if system == "Linux":
        subprocess.run(["systemctl", "daemon-reload"], check=False)
        subprocess.run(["systemctl", "restart", "modelswitch"], check=False)
        print("Service restarted.")
    elif system == "Darwin":
        subprocess.run(["launchctl", "unload", str(_launchd_plist_path())], check=False)
        subprocess.run(["launchctl", "load", str(_launchd_plist_path())], check=False)
        print("Service restarted.")
    else:
        print(f"Not supported on {system}")


# ── Log ───────────────────────────────────────────────────────────────


def _cmd_log(workspace: Path) -> None:
    """Tail service logs."""
    system = platform.system()
    if system == "Linux":
        os.execvp("journalctl", ["journalctl", "-u", "modelswitch", "-f"])
    elif system == "Darwin":
        log_file = workspace / "logs" / "launchd-stdout.log"
        if log_file.exists():
            os.execvp("tail", ["tail", "-f", str(log_file)])
        else:
            print(f"Log file not found: {log_file}")
            print("Make sure the service is running.")
    else:
        print(f"Not supported on {system}")


# ── Start (foreground) ────────────────────────────────────────────────


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
            ["systemctl", "is-active", "modelswitch"],
            capture_output=True,
            text=True,
        )
        status = result.stdout.strip() if result.returncode == 0 else "not installed"
        print(f"Service:   {status}")
        unit_path = _systemd_unit_path()
        if unit_path.exists():
            print(f"Unit file: {unit_path}")
    elif system == "Darwin":
        plist_path = _launchd_plist_path()
        if plist_path.exists():
            print(f"Service:   installed ({plist_path})")
        else:
            print("Service:   not installed")
