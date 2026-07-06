"""Runtime factory helpers for backend resolution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nika.runtime.base import LabRuntime
from nika.runtime.containerlab.runtime import ContainerlabRuntime
from nika.runtime.kathara.runtime import KatharaRuntime

if TYPE_CHECKING:
    from nika.net_env.base import NetworkEnvBase


def resolve_backend(meta: dict[str, Any] | Any) -> str:
    """Return session backend; infer from scenario when metadata is missing."""
    if isinstance(meta, dict):
        backend = meta.get("backend")
        scenario_params = meta.get("scenario_params") or {}
        scenario_name = meta.get("scenario_name")
    else:
        backend = getattr(meta, "backend", None)
        scenario_params = getattr(meta, "scenario_params", None) or {}
        scenario_name = getattr(meta, "scenario_name", None)

    if backend:
        return str(backend)
    if isinstance(scenario_params, dict) and scenario_params.get("backend"):
        return str(scenario_params["backend"])

    if scenario_name:
        from nika.net_env.net_env_pool import scenario_supported_backends

        try:
            supported = scenario_supported_backends(str(scenario_name))
        except ValueError:
            pass
        else:
            if len(supported) == 1:
                return supported[0]

    return "kathara"


def _topology_file_from_meta(meta: dict[str, Any] | Any) -> Path | None:
    if isinstance(meta, dict):
        raw = meta.get("topology_file") or (meta.get("scenario_params") or {}).get("topology_file")
    else:
        raw = getattr(meta, "topology_file", None) or (getattr(meta, "scenario_params", None) or {}).get(
            "topology_file"
        )
    if not raw:
        return None
    return Path(str(raw))


def _runtime_workdir_from_meta(meta: dict[str, Any] | Any) -> Path | None:
    if isinstance(meta, dict):
        raw = meta.get("runtime_workdir") or (meta.get("scenario_params") or {}).get("runtime_workdir")
    else:
        raw = getattr(meta, "runtime_workdir", None) or (getattr(meta, "scenario_params", None) or {}).get(
            "runtime_workdir"
        )
    if not raw:
        return None
    return Path(str(raw))


def _lab_name_from_meta(meta: dict[str, Any] | Any) -> str:
    if isinstance(meta, dict):
        lab_name = meta.get("lab_name")
    else:
        lab_name = getattr(meta, "lab_name", None)
    if not lab_name:
        raise ValueError("Session metadata has no lab_name.")
    return str(lab_name)


def runtime_for_session(meta: dict[str, Any] | Any) -> LabRuntime:
    """Build a runtime from persisted session metadata."""
    backend = resolve_backend(meta)
    lab_name = _lab_name_from_meta(meta)
    if backend == "containerlab":
        topology_file = _topology_file_from_meta(meta)
        if topology_file is None:
            raise ValueError(f"Containerlab session {lab_name!r} has no topology_file.")
        return ContainerlabRuntime(
            lab_name=lab_name,
            topology_file=topology_file,
            runtime_workdir=_runtime_workdir_from_meta(meta),
        )
    from nika.net_env.net_env_pool import get_net_env_instance

    scenario_name = meta.get("scenario_name") if isinstance(meta, dict) else getattr(meta, "scenario_name", None)
    kwargs: dict[str, Any] = {"lab_name": lab_name, "backend": backend}
    if isinstance(meta, dict):
        scenario_params = dict(meta.get("scenario_params") or {})
    else:
        scenario_params = dict(getattr(meta, "scenario_params", None) or {})
    scenario_params.pop("backend", None)
    scenario_params.pop("topology_file", None)
    scenario_params.pop("runtime_workdir", None)
    if scenario_params.get("topo_size") is not None:
        kwargs["topo_size"] = scenario_params.pop("topo_size")
    kwargs.update(scenario_params)
    if scenario_name:
        net_env = get_net_env_instance(str(scenario_name), **kwargs)
        return KatharaRuntime(net_env)
    raise ValueError("Kathara runtime requires scenario_name in session metadata.")


def runtime_for_net_env(net_env: "NetworkEnvBase") -> LabRuntime:
    """Build a runtime for a network environment instance."""
    if net_env.backend == "containerlab":
        if net_env.topology_file is None:
            raise ValueError(f"Containerlab env {net_env.name!r} has no topology_file.")
        return ContainerlabRuntime(
            lab_name=net_env.name,
            topology_file=net_env.topology_file,
            runtime_workdir=net_env.runtime_workdir,
        )
    return KatharaRuntime(net_env)
