"""Kathara telemetry API smoke tests on ``p4_int``.

Exercises InfluxDB query helpers and CSV-to-JSON parsing on the INT collector.

Run:
  uv run python -m unittest tests.api_verify.test_kathara_telemetry_api -v
"""

from __future__ import annotations

import json
import unittest

from nika.runtime.factory import resolve_backend
from nika.service.kathara.telemetry_api import KatharaTelemetryAPI
from tests.api_verify.helpers import p4_int_prerequisites
from tests.api_verify.kathara_base import KatharaScenarioApiSmokeTest

COLLECTOR = "collector"
MEASUREMENT = "flow_stat"


@unittest.skipUnless(
    p4_int_prerequisites(),
    "Docker or kathara/influxdb image not available",
)
class KatharaTelemetryApiSmokeTest(KatharaScenarioApiSmokeTest):
    SCENARIO = "p4_int"

    def _telemetry_api(self) -> KatharaTelemetryAPI:
        return KatharaTelemetryAPI(lab_name=self._lab_name())

    def test_session_backend(self) -> None:
        row = self._session_row(self.session_id)
        self.assertEqual(resolve_backend(row), "kathara")

    def test_kathara_telemetry_influx_api(self) -> None:
        api = self._telemetry_api()
        buckets = self.smoke(
            "KatharaTelemetryAPI.influx_list_buckets",
            lambda: api.influx_list_buckets(COLLECTOR),
            expect_type=list,
        )
        self.assertTrue(buckets)
        bucket_payload = buckets[0]
        self.assertIn("int_bucket", bucket_payload)

        measurements = self.smoke(
            "KatharaTelemetryAPI.influx_get_measurements",
            lambda: api.influx_get_measurements(COLLECTOR),
            expect_type=list,
        )
        self.assertTrue(measurements)

        count_rows = self.smoke(
            "KatharaTelemetryAPI.influx_count_measurements",
            lambda: api.influx_count_measurements(MEASUREMENT, host_name=COLLECTOR),
            expect_type=list,
        )
        self.assertEqual(len(count_rows), 1)
        json.loads(count_rows[0])

        sample_rows = self.smoke(
            "KatharaTelemetryAPI.influx_query_measurement",
            lambda: api.influx_query_measurement(
                MEASUREMENT, limit=5, offset=0, host_name=COLLECTOR
            ),
            expect_type=list,
        )
        self.assertEqual(len(sample_rows), 1)
        json.loads(sample_rows[0])


if __name__ == "__main__":
    unittest.main()
