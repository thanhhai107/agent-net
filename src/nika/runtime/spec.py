"""Neutral topology model for lab runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeSpec:
    name: str
    image: str
    kind: str = "linux"
    binds: list[str] = field(default_factory=list)
    exec_cmds: list[str] = field(default_factory=list)


@dataclass
class LinkSpec:
    endpoints: tuple[str, str]


@dataclass
class LabSpec:
    name: str
    nodes: list[NodeSpec] = field(default_factory=list)
    links: list[LinkSpec] = field(default_factory=list)
