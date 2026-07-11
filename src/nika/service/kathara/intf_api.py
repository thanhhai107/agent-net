"""Kathara interface API (re-exported from shared lab service)."""

from nika.service.kathara.base_api import KatharaBaseAPI
from nika.service.lab.intf_api import IntfAPIMixin

__all__ = ["IntfAPIMixin", "KatharaIntfAPI"]


class KatharaIntfAPI(KatharaBaseAPI, IntfAPIMixin):
    """Kathara interface API to manage host interfaces."""

    pass
