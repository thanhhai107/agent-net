"""Kathara nftables API (re-exported from shared lab service)."""

from nika.service.kathara.base_api import KatharaBaseAPI
from nika.service.lab.nft_api import NFTableMixin

__all__ = ["KatharaNFTableAPI", "NFTableMixin"]


class KatharaNFTableAPI(KatharaBaseAPI, NFTableMixin):
    """Kathara API to manage nftables within Kathara labs."""

    pass
