"""Containerlab service API protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nika.service.lab.protocols import SupportsExec


@runtime_checkable
class SupportsSRL(SupportsExec, Protocol):
    pass


__all__ = ["SupportsExec", "SupportsSRL"]
