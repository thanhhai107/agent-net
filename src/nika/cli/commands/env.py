"""Commands for starting, stopping, and listing network environments."""

import typer

env_app = typer.Typer(help="Kathara lab scenarios.")


@env_app.command("list")
def env_list() -> None:
    """Print scenario ids registered in the net env pool."""
    from nika.net_env.net_env_pool import list_all_net_envs

    for name in sorted(list_all_net_envs()):
        typer.echo(name)


@env_app.command("run")
def env_run(
    name: str = typer.Argument(..., metavar="NAME", help="Scenario id (see `nika env list`)."),
    tier: str | None = typer.Option(
        None,
        "-t",
        "--tier",
        help="Topology tier s, m, or l (required only for scalable scenarios).",
    ),
    no_redeploy: bool = typer.Option(False, "--no-redeploy", help="If set, do not redeploy when the lab already exists."),
    instance_tag: str | None = typer.Option(
        None,
        "--instance-tag",
        help="Optional tag for lab instance naming; required for human-friendly concurrent runs.",
    ),
) -> None:
    """Deploy one scenario and start a new session."""
    from nika.workflows.net_env_start import start_net_env

    session_id = start_net_env(name, tier, redeploy=not no_redeploy, instance_tag=instance_tag)
    typer.echo(f"session_id={session_id}")


@env_app.command("ps")
def env_ps() -> None:
    """List all currently running env instances."""
    from nika.utils.session_store import SessionStore

    sessions = SessionStore().list_running_sessions()
    if not sessions:
        typer.echo("No running env instances.")
        return
    store = SessionStore()
    for item in sessions:
        counts = store.count_failure_statuses(session_id=item["session_id"])
        if counts:
            failure_summary = ",".join(f"{status}:{count}" for status, count in sorted(counts.items()))
        else:
            failure_summary = "none"
        typer.echo(
            " | ".join(
                [
                    f"session_id={item.get('session_id')}",
                    f"lab={item.get('lab_name')}",
                    f"scenario={item.get('scenario_name')}",
                    f"tier={item.get('scenario_topo_size')}",
                    f"created_at={item.get('created_at')}",
                    f"failures={failure_summary}",
                ]
            )
        )


@env_app.command("stop")
def env_stop(
    session_id: str | None = typer.Option(None, "--session-id", help="Target session id (lab_hash)."),
    stop_all: bool = typer.Option(False, "--all", help="Stop all running sessions."),
) -> None:
    """Stop one or all network environment sessions."""
    from nika.workflows.net_env_stop import stop_net_env

    if stop_all and session_id is not None:
        raise typer.BadParameter("--all and --session-id cannot be used together.")
    try:
        stop_net_env(session_id=session_id, stop_all=stop_all)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
