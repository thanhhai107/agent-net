"""Root Typer application for the ``nika`` console script (``nika.codex_cli``)."""

import typer

from nika.codex_cli.commands.agent import agent_app
from nika.codex_cli.commands.benchmark import benchmark_app
from nika.codex_cli.commands.env import env_app
from nika.codex_cli.commands.evaluation import eval_app
from nika.codex_cli.commands.exec import exec_command
from nika.codex_cli.commands.failure import failure_app
from nika.codex_cli.commands.memory import memory_app
from nika.codex_cli.commands.session import session_app
from nika.codex_cli.commands.traffic import traffic_app
from nika.codex_cli.commands.tools import tools_app
from nika.codex_cli.commands.visualize import visualize_command

app = typer.Typer(help="NIKA network troubleshooting pipeline CLI.")
app.add_typer(session_app, name="session")
app.add_typer(env_app, name="env")
app.add_typer(failure_app, name="failure")
app.command("exec", context_settings={"allow_interspersed_args": False})(exec_command)
app.add_typer(agent_app, name="agent")
app.add_typer(eval_app, name="eval")
app.add_typer(memory_app, name="memory")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(traffic_app, name="traffic")
app.add_typer(tools_app, name="tools")
app.command("visualize")(visualize_command)


def main() -> None:
    """Console entrypoint for setuptools `[project.scripts]`."""
    app()


if __name__ == "__main__":
    main()
