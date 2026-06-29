"""Launch the Streamlit experiment studio."""

from __future__ import annotations

import subprocess
import sys

import typer

from nika.config import pkg_path


def studio_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Studio bind address."),
    port: int = typer.Option(8502, "--port", min=1, max=65535, help="Studio port."),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Do not open a browser automatically.",
    ),
) -> None:
    """Open the experiment runner UI for benchmark and evolution runs."""
    dashboard_path = pkg_path("visualization", "experiment_dashboard.py")
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
    typer.echo(f"Opening NIKA Experiment Studio at http://{host}:{port}")
    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        typer.echo("\nStudio stopped.")
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(code=exc.returncode) from exc
