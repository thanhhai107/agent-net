"""Commands for offline evaluation (metrics, judge, summary)."""

import typer

from nika.config import ENV_RESULT_DIR
from nika.utils.agent_config import ENV_JUDGE_MODEL, ENV_JUDGE_PROVIDER

eval_app = typer.Typer(help="Evaluate a completed agent session.")


@eval_app.command("metrics")
def eval_metrics(
    session_id: str | None = typer.Option(
        None, "--session_id", help="Target session id."
    ),
    result_dir: str | None = typer.Option(
        None,
        "--result_dir",
        envvar=ENV_RESULT_DIR,
        help="Results parent directory (default: results/). When set without --session_id, evaluate all closed sessions under this directory.",
    ),
) -> None:
    """Compute rule-based scores and trace stats on closed session(s); write eval_metrics.json."""
    from nika.workflows.eval.session import run_eval_metrics

    try:
        run_eval_metrics(session_id=session_id, result_dir=result_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@eval_app.command("judge")
def eval_judge(
    judge_provider: str | None = typer.Option(
        None,
        "-p",
        "--provider",
        envvar=ENV_JUDGE_PROVIDER,
        help="LLM provider for the judge (openai, ollama, deepseek).",
    ),
    judge_model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        envvar=ENV_JUDGE_MODEL,
        help="Judge model id.",
    ),
    session_id: str | None = typer.Option(
        None, "--session_id", help="Target session id."
    ),
    result_dir: str | None = typer.Option(
        None,
        "--result_dir",
        envvar=ENV_RESULT_DIR,
        help="Results parent directory (default: results/). When set without --session_id, judge all closed sessions under this directory.",
    ),
) -> None:
    """Run LLM-as-judge on closed session(s); write llm_judge.json."""
    from nika.utils.agent_config import resolve_judge_model, resolve_judge_provider
    from nika.workflows.eval.session import run_llm_judge

    judge_provider = resolve_judge_provider(judge_provider)
    judge_model = resolve_judge_model(judge_model)

    try:
        run_llm_judge(
            judge_provider, judge_model, session_id=session_id, result_dir=result_dir
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@eval_app.command("summary")
def eval_summary(
    output: str | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output CSV path (default: results/0_summary/evaluation_summary.csv).",
    ),
    problem: list[str] | None = typer.Option(
        None,
        "-p",
        "--problem",
        help="Include only sessions with this root-cause / problem id (repeatable).",
    ),
    env: list[str] | None = typer.Option(
        None,
        "-e",
        "--env",
        help="Include only sessions from this scenario / net env (repeatable).",
    ),
    category: list[str] | None = typer.Option(
        None,
        "-c",
        "--category",
        help="Include only sessions in this root-cause category (repeatable).",
    ),
    session_id: list[str] | None = typer.Option(
        None,
        "--session_id",
        help="Include only these session ids (repeatable).",
    ),
    agent: list[str] | None = typer.Option(
        None,
        "-a",
        "--agent",
        help="Include only sessions run with this agent type (repeatable).",
    ),
    model: list[str] | None = typer.Option(
        None,
        "--model",
        help="Include only sessions run with this model id (repeatable).",
    ),
    result_dir: str | None = typer.Option(
        None,
        "--result_dir",
        envvar=ENV_RESULT_DIR,
        help="Results parent directory (default: results/). Session output goes to {result_dir}/{session_id}.",
    ),
) -> None:
    """Aggregate finished sessions under the results directory into one CSV file."""
    from nika.workflows.eval.summary import run_eval_summary

    try:
        out_path = run_eval_summary(
            output_path=output,
            problems=problem,
            envs=env,
            categories=category,
            session_ids=session_id,
            agent_types=agent,
            models=model,
            results_dir=result_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Wrote summary CSV: {out_path}")


@eval_app.command("clean")
def eval_clean(
    yes: bool = typer.Option(
        False, "-y", "--yes", help="Skip the confirmation prompt."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Delete session files even when running sessions exist.",
    ),
) -> None:
    """Delete results/, runtime session JSON files, and the SQLite session index."""
    from nika.config import RESULTS_DIR, SESSIONS_DB, SESSIONS_DIR
    from nika.utils.session_store import SessionStore
    from nika.workflows.eval.clean import run_eval_clean

    running = SessionStore().list_running_sessions()
    if running and not force:
        ids = ", ".join(str(row.get("session_id", "?")) for row in running)
        raise typer.BadParameter(
            f"{len(running)} running session(s) found ({ids}). "
            "Close them with `nika session close` first, or pass --force."
        )

    label = (
        f"all files under {RESULTS_DIR}, session files under {SESSIONS_DIR}, "
        f"and the SQLite index at {SESSIONS_DB}"
    )
    if running and force:
        label += f" (including {len(running)} running session file(s))"
    if not yes:
        confirmed = typer.confirm(f"Delete {label}?", default=False)
        if not confirmed:
            raise typer.Abort()

    try:
        report = run_eval_clean(force=force)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        f"Removed {report.results_entries_removed} entr{'y' if report.results_entries_removed == 1 else 'ies'} "
        f"under {RESULTS_DIR}, {report.session_files_removed} session file(s) under {SESSIONS_DIR}, "
        f"and cleared the SQLite index at {SESSIONS_DB}."
    )
