"""Shared initialization for orchestrator problem classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nika.net_env.net_env_pool import get_net_env_instance
from nika.runtime.base import LabRuntime
from nika.runtime.factory import runtime_for_net_env

if TYPE_CHECKING:
    from nika.net_env.base import NetworkEnvBase


def init_problem(
    scenario_name: str | None, **kwargs: Any
) -> tuple[NetworkEnvBase, LabRuntime]:
    """Resolve network environment and backend-neutral runtime for a problem."""
    net_env = get_net_env_instance(scenario_name, **kwargs)
    runtime = kwargs.get("runtime")
    if runtime is None:
        runtime = runtime_for_net_env(net_env)
        net_env.runtime = runtime
    return net_env, runtime
