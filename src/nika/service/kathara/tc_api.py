"""Kathara traffic-control API (re-exported from shared lab service)."""

from nika.service.kathara.base_api import KatharaBaseAPI
from nika.service.lab.tc_api import TCMixin

__all__ = ["KatharaTCAPI", "TCMixin"]


class KatharaTCAPI(KatharaBaseAPI, TCMixin):
    """Kathara Traffic Control API to manage traffic control settings on host intf_names."""

    pass
