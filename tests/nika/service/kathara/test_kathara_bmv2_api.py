"""Kathara BMv2 API smoke tests on ``p4_counter``.

Exercises P4 switch Thrift helpers and host discovery for BMv2 nodes.

Run:
  uv run python -m unittest tests.nika.service.kathara.test_kathara_bmv2_api -v
"""

from __future__ import annotations

import unittest

from nika.runtime.factory import resolve_backend
from nika.service.kathara.bmv2_api import KatharaBMv2API
from tests.support.prerequisites import docker_available
from tests.support.kathara_api_base import KatharaScenarioApiSmokeTest

SWITCH = "s1"
TABLE = "dmac_forward"
INGRESS_COUNTER = "ingress_port_counter"
EGRESS_COUNTER = "egress_port_counter"
EXPECTED_SWITCHES = ("s1", "s2", "s3", "s4")


@unittest.skipUnless(docker_available(), "Docker not available")
class KatharaBmv2ApiSmokeTest(KatharaScenarioApiSmokeTest):
    SCENARIO = "p4_counter"

    def _bmv2_api(self) -> KatharaBMv2API:
        return KatharaBMv2API(lab_name=self._lab_name())

    def test_session_backend(self) -> None:
        row = self._session_row(self.session_id)
        self.assertEqual(resolve_backend(row), "kathara")

    def test_kathara_bmv2_discovery(self) -> None:
        api = self._host_api()
        switches = self.smoke(
            "KatharaBaseAPI.get_bmv2_switches",
            api.get_bmv2_switches,
            expect_type=list,
        )
        self.assertEqual(set(switches), set(EXPECTED_SWITCHES))
        self.smoke("KatharaBaseAPI.load_machines", api.load_machines)
        self.assertEqual(set(api.bmv2_switches), set(EXPECTED_SWITCHES))

    def test_kathara_bmv2_switch_api(self) -> None:
        api = self._bmv2_api()
        self.smoke(
            "KatharaBMv2API.bmv2_get_log",
            lambda: api.bmv2_get_log(SWITCH, rows=20),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.read_p4_program",
            lambda: api.read_p4_program(SWITCH),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_switch_info",
            lambda: api.bmv2_switch_info(SWITCH),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_show_ports",
            lambda: api.bmv2_show_ports(SWITCH),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_show_tables",
            lambda: api.bmv2_show_tables(SWITCH),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_show_actions",
            lambda: api.bmv2_show_actions(SWITCH),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_table_info",
            lambda: api.bmv2_table_info(SWITCH, TABLE),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_table_dump",
            lambda: api.bmv2_table_dump(SWITCH, TABLE),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_table_show_actions",
            lambda: api.bmv2_table_show_actions(SWITCH, TABLE),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_table_num_entries",
            lambda: api.bmv2_table_num_entries(SWITCH, TABLE),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_get_counter_arrays",
            lambda: api.bmv2_get_counter_arrays(SWITCH),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_counter_read(ingress)",
            lambda: api.bmv2_counter_read(SWITCH, INGRESS_COUNTER, 0),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_counter_read(egress)",
            lambda: api.bmv2_counter_read(SWITCH, EGRESS_COUNTER, 0),
            min_len=1,
        )
        self.smoke(
            "KatharaBMv2API.bmv2_get_register_arrays",
            lambda: api.bmv2_get_register_arrays(SWITCH),
            min_len=1,
        )


if __name__ == "__main__":
    unittest.main()