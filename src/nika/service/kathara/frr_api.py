"""Kathara FRR API (re-exported from shared lab service)."""

from nika.service.kathara.base_api import KatharaBaseAPI
from nika.service.lab.frr_api import FRRAPIMixin

__all__ = ["FRRAPIMixin", "KatharaFRRAPI"]


class KatharaFRRAPI(KatharaBaseAPI, FRRAPIMixin):
    pass
