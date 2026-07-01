"""Compatibility wrapper for ``nika.codex_cli.main``."""

from nika.codex_cli.main import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
