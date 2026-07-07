"""Shared bases for Kathara scenario API smoke tests."""

from __future__ import annotations

from typing import ClassVar

from nika.runtime.factory import runtime_for_session
from nika.service.kathara import KatharaBaseAPI
from nika.service.kathara.frr_api import KatharaFRRAPI
from nika.service.kathara.intf_api import KatharaIntfAPI
from nika.service.kathara.nftable_api import KatharaNFTableAPI
from nika.service.kathara.tc_api import KatharaTCAPI
from tests.api_verify.helpers import ApiSmokeMixin
from tests.integration_base import SharedSessionTestCase


class KatharaScenarioApiSmokeTest(SharedSessionTestCase, ApiSmokeMixin):
    """One shared lab per class; subclasses set ``SCENARIO`` and optional ``ENV_RUN_ARGS``."""

    __test__ = False
    ENV_RUN_ARGS: ClassVar[list[str]] = []

    def _lab_name(self) -> str:
        return str(self._session_row(self.session_id)["lab_name"])

    def _runtime(self):
        return runtime_for_session(self._session_row(self.session_id))

    def _host_api(self) -> KatharaBaseAPI:
        return KatharaBaseAPI(lab_name=self._lab_name())

    def _frr_api(self) -> KatharaFRRAPI:
        return KatharaFRRAPI(lab_name=self._lab_name())

    def _intf_api(self) -> KatharaIntfAPI:
        return KatharaIntfAPI(lab_name=self._lab_name())

    def _tc_api(self) -> KatharaTCAPI:
        return KatharaTCAPI(lab_name=self._lab_name())

    def _nft_api(self) -> KatharaNFTableAPI:
        return KatharaNFTableAPI(lab_name=self._lab_name())
