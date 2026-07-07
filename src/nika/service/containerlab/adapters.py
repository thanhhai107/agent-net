"""Containerlab runtime adapters."""

from __future__ import annotations

from nika.service.containerlab.srl_api import SRLAPIMixin
from nika.service.lab.adapters import LabRuntimeLabAPI

__all__ = ["LabRuntimeContainerlabAPI"]


class LabRuntimeContainerlabAPI(LabRuntimeLabAPI, SRLAPIMixin):
    """Containerlab lab API with SR Linux operations."""

    def uses_srl_router(self, device_name: str) -> bool:
        return SRLAPIMixin.uses_srl_router(self, device_name)
