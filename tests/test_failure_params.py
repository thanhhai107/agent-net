import unittest
from unittest.mock import patch

from nika.utils.failure_params import FailureParamSchema, get_failure_param_schema, resolve_failure_params


class FailureParamsTest(unittest.TestCase):
    def test_resolve_link_down_with_context_and_override(self) -> None:
        resolved = resolve_failure_params(
            "link_down",
            {"intf_name": "eth9"},
            context={"host_name": "pc1"},
        )
        self.assertEqual(resolved["host_name"], "pc1")
        self.assertEqual(resolved["intf_name"], "eth9")

    def test_resolve_rejects_unknown_key_for_typed_schema(self) -> None:
        with self.assertRaises(ValueError):
            resolve_failure_params("link_down", {"unknown": "x"}, context={"host_name": "pc1"})

    def test_unknown_problem_with_overrides_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_failure_params("new_future_problem", {"foo": "bar"})

    def test_schema_exists_for_link_down(self) -> None:
        self.assertIsNotNone(get_failure_param_schema("link_down"))

    def test_get_schema_prefers_problem_local_definition(self) -> None:
        problem_defined = FailureParamSchema(
            problem_name="sdn_controller_crash",
            summary="problem-local",
            fields=(),
            example="x",
        )
        with patch("nika.utils.failure_params._get_problem_defined_schema", return_value=problem_defined):
            schema = get_failure_param_schema("sdn_controller_crash")
        self.assertIsNotNone(schema)
        self.assertEqual(schema.summary, "problem-local")


if __name__ == "__main__":
    unittest.main()
