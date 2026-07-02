"""Lazy-loaded Typer subcommands to keep ``nika`` CLI startup fast."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from gettext import gettext as _
from typing import Any

import typer
from typer import _click
from typer.core import TyperCommand, TyperGroup
from typer.main import get_command, get_group


@dataclass(frozen=True)
class LazyCommandSpec:
    module: str
    attr: str
    help: str = ""


LAZY_COMMANDS: dict[str, LazyCommandSpec] = {}


class LazyTyperGroup(TyperGroup):
    """Load subcommand modules on first use instead of at CLI import time."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loaded_commands: set[str] = set()

    def list_commands(self, ctx: _click.Context) -> list[str]:
        return sorted(set(super().list_commands(ctx)) | set(LAZY_COMMANDS.keys()))

    def _stub_lazy_command(self, cmd_name: str) -> TyperCommand:
        spec = LAZY_COMMANDS[cmd_name]
        help_text = spec.help or cmd_name

        def _stub_callback() -> None:
            raise RuntimeError(f"Lazy command {cmd_name!r} was not loaded.")

        return TyperCommand(
            name=cmd_name,
            help=help_text,
            short_help=help_text,
            callback=_stub_callback,
        )

    def _load_lazy_command(self, cmd_name: str) -> _click.Command:
        spec = LAZY_COMMANDS[cmd_name]
        module = importlib.import_module(spec.module)
        target = getattr(module, spec.attr)
        if not isinstance(target, typer.Typer):
            raise TypeError(
                f"Lazy command {cmd_name!r} must export a typer.Typer instance at "
                f"{spec.module}.{spec.attr}"
            )
        if (
            target.registered_callback
            or target.info.callback
            or target.registered_groups
            or len(target.registered_commands) > 1
        ):
            click_cmd = get_group(target)
        elif len(target.registered_commands) == 1 and target.registered_commands[0].name == cmd_name:
            # Root alias for a single-command Typer (e.g. exec_app -> `nika exec`).
            click_cmd = get_command(target)
        else:
            # Keep nested subcommands (e.g. benchmark_app -> `nika benchmark run`).
            click_cmd = get_group(target)
        self.add_command(click_cmd, name=cmd_name)
        self._loaded_commands.add(cmd_name)
        return click_cmd

    def get_command(self, ctx: _click.Context, cmd_name: str) -> _click.Command | None:
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        if cmd_name not in LAZY_COMMANDS:
            return None
        if getattr(self, "_defer_lazy_load", False):
            return self._stub_lazy_command(cmd_name)
        return self._load_lazy_command(cmd_name)

    def format_help(self, ctx: _click.Context, formatter: _click.HelpFormatter) -> None:
        self._defer_lazy_load = True
        try:
            super().format_help(ctx, formatter)
        finally:
            self._defer_lazy_load = False

    def format_commands(self, ctx: _click.Context, formatter: _click.HelpFormatter) -> None:
        self._defer_lazy_load = True
        try:
            commands: list[tuple[str, _click.Command | None, str | None]] = []
            for subcommand in self.list_commands(ctx):
                if subcommand in LAZY_COMMANDS and subcommand not in self._loaded_commands:
                    commands.append((subcommand, None, LAZY_COMMANDS[subcommand].help))
                    continue
                cmd = self.get_command(ctx, subcommand)
                if cmd is None or cmd.hidden:
                    continue
                commands.append((subcommand, cmd, None))

            if not commands:
                return

            limit = formatter.width - 6 - max(len(name) for name, _, _ in commands)
            rows: list[tuple[str, str]] = []
            for subcommand, cmd, lazy_help in commands:
                if cmd is not None:
                    help = cmd.get_short_help_str(limit)
                else:
                    help = lazy_help or ""
                rows.append((subcommand, help))

            with formatter.section(_("Commands")):
                formatter.write_dl(rows)
        finally:
            self._defer_lazy_load = False
