from nika.service.lab.adapters import (
    LabRuntimeExecAdapter,
    LabRuntimeLabAPI,
    lab_api_for_runtime,
    node_status_from_container,
)
from nika.service.lab.frr_api import FRRAPIMixin
from nika.service.lab.intf_api import IntfAPIMixin
from nika.service.lab.nft_api import NFTableMixin
from nika.service.lab.protocols import SupportsExec
from nika.service.lab.semantic_mixin import SemanticOpsMixin
from nika.service.lab.tc_api import TCMixin

__all__ = [
    "FRRAPIMixin",
    "IntfAPIMixin",
    "LabRuntimeExecAdapter",
    "LabRuntimeLabAPI",
    "NFTableMixin",
    "SemanticOpsMixin",
    "SupportsExec",
    "TCMixin",
    "lab_api_for_runtime",
    "node_status_from_container",
]


def __getattr__(name: str):
    if name == "LabRuntimeContainerlabAPI":
        from nika.service.containerlab.adapters import LabRuntimeContainerlabAPI

        return LabRuntimeContainerlabAPI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
