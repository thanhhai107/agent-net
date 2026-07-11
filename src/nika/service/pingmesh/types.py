"""Typed structures for PingMesh snapshot results."""

from __future__ import annotations

from typing import Literal, TypedDict

PingStatus = Literal["ok", "down", "unstable", "unknown"]
AnomalyType = Literal["unreachable", "packet_loss", "high_latency", "unknown"]


class PingStats(TypedDict, total=False):
    tx: int | None
    rx: int | None
    loss_percent: float | None
    time_ms: float | None
    rtt_min_ms: float | None
    rtt_avg_ms: float | None
    rtt_max_ms: float | None
    rtt_mdev_ms: float | None
    status: PingStatus


class PingPairResult(TypedDict, total=False):
    source: str
    target: str
    target_ip: str | None
    reachable: bool
    loss_percent: float | None
    rtt_min_ms: float | None
    rtt_avg_ms: float | None
    rtt_max_ms: float | None
    rtt_mdev_ms: float | None
    status: PingStatus


class PingAnomaly(TypedDict):
    source: str
    target: str
    type: AnomalyType
    detail: str


class PingMeshSummary(TypedDict):
    total_pairs: int
    reachable_pairs: int
    anomaly_count: int
    unreachable: int
    packet_loss: int
    high_latency: int
    unknown: int


class PingMeshSnapshot(TypedDict):
    timestamp: str
    endpoints: dict[str, str | None]
    sources: list[str]
    targets: list[str]
    results: list[PingPairResult]
    anomalies: list[PingAnomaly]
    summary: PingMeshSummary
