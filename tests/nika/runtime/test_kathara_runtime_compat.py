"""Unit tests for KatharaRuntime and factory defaults."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from docker.errors import APIError

from nika.runtime.factory import resolve_backend, runtime_for_net_env
from nika.runtime.base import LabCleanupError
from nika.runtime.kathara import KatharaRuntime
from nika.runtime.kathara import cleanup as kathara_cleanup
from nika.net_env.kathara.interdomain_routing.simple_bgp.lab import SimpleBGP
from nika.workflows.session import close as session_close


class KatharaRuntimeCompatTest(unittest.TestCase):
    def test_factory_default_backend(self) -> None:
        self.assertEqual(resolve_backend({}), "kathara")

    def test_runtime_for_kathara_net_env(self) -> None:
        env = SimpleBGP()
        runtime = runtime_for_net_env(env)
        self.assertIsInstance(runtime, KatharaRuntime)
        self.assertEqual(runtime.lab_name, env.name)

    def test_kathara_runtime_list_nodes_before_deploy(self) -> None:
        env = SimpleBGP()
        runtime = KatharaRuntime(env)
        nodes = runtime.list_nodes()
        self.assertIn("pc1", nodes)
        self.assertIn("router1", nodes)

    def test_global_cleanup_uses_official_backend_commands(self) -> None:
        kathara = Mock()
        kathara.get_machines_api_objects.return_value = []
        kathara.get_links_api_objects.return_value = []
        clab_result = Mock(returncode=0, stdout="", stderr="")

        with (
            patch.object(session_close.Kathara, "get_instance", return_value=kathara),
            patch.object(session_close.shutil, "which", return_value="/usr/bin/clab"),
            patch.object(
                session_close.subprocess,
                "run",
                return_value=clab_result,
            ) as run,
        ):
            session_close.clean_emulation_environment()

        kathara.wipe.assert_called_once_with(all_users=False)
        kathara.get_machines_api_objects.assert_called_once_with(all_users=False)
        kathara.get_links_api_objects.assert_called_once_with(all_users=False)
        run.assert_called_once_with(
            [
                "clab",
                "destroy",
                "--all",
                "--cleanup",
                "--yes",
                "--log-level",
                "error",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_lab_cleanup_retries_official_api_after_active_endpoint_race(self) -> None:
        kathara = Mock()
        response = Mock(status_code=403)
        transient_error = APIError(
            "network has active endpoints",
            response=response,
            explanation="network has active endpoints",
        )
        kathara.undeploy_lab.side_effect = [transient_error, None]
        kathara.get_machines_api_objects.side_effect = [[], []]
        kathara.get_links_api_objects.side_effect = [[object()], []]
        env = Mock()
        env.instance = kathara
        env.name = "lab-1"
        runtime = KatharaRuntime(env)

        with (
            patch.object(kathara_cleanup, "_CLEANUP_ATTEMPTS", 2),
            patch.object(kathara_cleanup.time, "sleep") as sleep,
        ):
            runtime.destroy()

        self.assertEqual(kathara.undeploy_lab.call_count, 2)
        kathara.undeploy_lab.assert_called_with(lab_name="lab-1")
        sleep.assert_called_once()

    def test_lab_cleanup_recovers_repeated_dangling_docker_endpoint(self) -> None:
        lab_name = "dc_clos_service__0716051913-0cb37d"
        lab_hash = kathara_cleanup.kathara_utils.generate_urlsafe_hash(lab_name)
        network_id = "68a6a551d36f285f100645ef52c0d557071a02a9cab750053e8dc73ff7a1db42"
        endpoint_name = f"kathara_user_leaf_router_3_4_{lab_hash}"
        response = Mock(
            status_code=403,
            url=f"http+docker://localhost/v1.55/networks/{network_id}",
            reason="Forbidden",
        )
        active_endpoint_error = APIError(
            "network removal failed",
            response=response,
            explanation=(
                "error while removing network: network has active endpoints "
                f'(name:"{endpoint_name}" id:"706a515f6031")'
            ),
        )

        network = Mock()
        network.id = network_id
        network.name = f"kathara_user_spine_leaf_{lab_hash}"
        network.attrs = {
            "Labels": {"app": "kathara", "lab_hash": lab_hash},
            # Reproduce the Docker/Kathara disagreement from experiment-02:
            # inspect has no container, but network deletion reports an endpoint.
            "Containers": {},
        }
        kathara = Mock()
        kathara.manager.client.networks.get.return_value = network
        kathara.undeploy_lab.side_effect = [
            active_endpoint_error,
            active_endpoint_error,
            None,
        ]
        kathara.get_machines_api_objects.return_value = []
        kathara.get_links_api_objects.side_effect = [
            [network],
            [network],
            [network],
            [],
        ]

        with (
            patch.object(kathara_cleanup, "_CLEANUP_ATTEMPTS", 3),
            patch.object(kathara_cleanup.time, "sleep"),
        ):
            kathara_cleanup.undeploy_kathara_lab(kathara, lab_name=lab_name)

        self.assertEqual(kathara.undeploy_lab.call_count, 3)
        network.disconnect.assert_called_once_with(endpoint_name, force=True)

    def test_lab_endpoint_recovery_ignores_foreign_endpoint(self) -> None:
        lab_name = "lab-1"
        lab_hash = kathara_cleanup.kathara_utils.generate_urlsafe_hash(lab_name)
        network_id = "68a6a551d36f285f100645ef52c0d557071a02a9cab750053e8dc73ff7a1db42"
        network = Mock()
        network.id = network_id
        network.attrs = {
            "Labels": {"app": "kathara", "lab_hash": lab_hash},
            "Containers": {
                "foreign-container-id": {"Name": "unrelated_container"},
            },
        }
        kathara = Mock()
        kathara.manager.client.networks.get.return_value = network
        kathara.get_links_api_objects.return_value = [network]
        response = Mock(status_code=403)
        error = APIError(
            (
                f"network {network_id} has active endpoints "
                '(name:"unrelated_container" id:"endpoint-id")'
            ),
            response=response,
            explanation="network has active endpoints",
        )

        recovered = kathara_cleanup._recover_lab_active_endpoints(
            kathara,
            lab_name=lab_name,
            error=error,
        )

        self.assertEqual(recovered, 0)
        network.disconnect.assert_not_called()

    def test_lab_cleanup_is_idempotent_when_lab_is_already_absent(self) -> None:
        kathara = Mock()
        kathara.get_machines_api_objects.return_value = []
        kathara.get_links_api_objects.return_value = []

        kathara_cleanup.undeploy_kathara_lab(kathara, lab_name="lab-1")
        kathara_cleanup.undeploy_kathara_lab(kathara, lab_name="lab-1")

        self.assertEqual(kathara.undeploy_lab.call_count, 2)
        self.assertEqual(kathara.get_machines_api_objects.call_count, 2)
        self.assertEqual(kathara.get_links_api_objects.call_count, 2)

    def test_lab_cleanup_does_not_retry_unrelated_api_errors(self) -> None:
        kathara = Mock()
        response = Mock(status_code=500)
        kathara.undeploy_lab.side_effect = APIError(
            "daemon failed", response=response, explanation="daemon failed"
        )
        env = Mock()
        env.instance = kathara
        env.name = "lab-1"
        runtime = KatharaRuntime(env)

        with self.assertRaisesRegex(LabCleanupError, "daemon failed"):
            runtime.destroy()

        kathara.undeploy_lab.assert_called_once_with(lab_name="lab-1")
        kathara.get_machines_api_objects.assert_not_called()

    def test_lab_cleanup_rejects_false_success_with_resources_remaining(self) -> None:
        kathara = Mock()
        kathara.get_machines_api_objects.return_value = []
        kathara.get_links_api_objects.return_value = [object()]
        env = Mock()
        env.instance = kathara
        env.name = "lab-1"
        runtime = KatharaRuntime(env)

        with (
            patch.object(kathara_cleanup, "_CLEANUP_ATTEMPTS", 2),
            patch.object(kathara_cleanup.time, "sleep"),
            self.assertRaisesRegex(LabCleanupError, "machines=0, links=1"),
        ):
            runtime.destroy()

        self.assertEqual(kathara.undeploy_lab.call_count, 2)

    def test_kathara_runtime_exists_when_only_links_remain(self) -> None:
        kathara = Mock()
        kathara.get_machines_api_objects.return_value = []
        kathara.get_links_api_objects.return_value = [object()]
        env = Mock()
        env.instance = kathara
        env.name = "lab-1"
        runtime = KatharaRuntime(env)

        self.assertTrue(runtime.exists())

    def test_global_cleanup_reports_both_backend_failures(self) -> None:
        clab_result = Mock(returncode=1, stdout="", stderr="clab error")

        with (
            patch.object(
                session_close,
                "wipe_kathara_user_labs",
                side_effect=LabCleanupError("kathara error"),
            ),
            patch.object(session_close.shutil, "which", return_value="/usr/bin/clab"),
            patch.object(session_close.subprocess, "run", return_value=clab_result),
            self.assertRaisesRegex(
                LabCleanupError,
                "kathara error; Containerlab cleanup failed: clab error",
            ),
        ):
            session_close.clean_emulation_environment()


if __name__ == "__main__":
    unittest.main()
