"""Launch the Streamlit session dashboard."""

from __future__ import annotations

import os
import subprocess
import sys

import typer

from nika.config import pkg_path
from nika.visualization.data import discover_sessions


def visualize_command(
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help="Open this session initially; defaults to the newest session.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Dashboard bind address."),
    port: int = typer.Option(8501, "--port", min=1, max=65535, help="Dashboard port."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not open a browser automatically."
    ),
) -> None:
    """Open an interactive dashboard for topology, failures, traces, and evaluation."""
    available_ids = {str(item["session_id"]) for item in discover_sessions()}
    if not available_ids:
        raise typer.BadParameter(
            "No sessions found. Run `nika env run <scenario>` first."
        )
    if session_id is not None and session_id not in available_ids:
        raise typer.BadParameter(f"Session '{session_id}' not found.")

    dashboard_path = pkg_path("visualization", "dashboard.py")
    env = os.environ.copy()
    if session_id:
        env["NIKA_VISUALIZE_SESSION_ID"] = session_id

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true" if no_browser else "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    typer.echo(f"Opening NIKA dashboard at http://{host}:{port}")
    try:
        subprocess.run(command, env=env, check=True)
    except KeyboardInterrupt:
        typer.echo("\nDashboard stopped.")
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(code=exc.returncode) from exc


if __name__ == "__main__":
    typer.run(visualize_command)
