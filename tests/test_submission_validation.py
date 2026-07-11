import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, ToolMessage

from agent.langgraph.evidence import ToolObservation, observations_from_messages
from agent.langgraph.phases.submission import SubmissionPhase
from agent.langgraph.submission_validation import (
    ClaimVerdict,
    DiagnosisDraft,
    SubmissionVerification,
    SupportedClaim,
    draft_claims,
    evidence_records,
    validated_submission,
)


class SubmissionValidationTest(unittest.IsolatedAsyncioTestCase):
    def test_validation_drops_unsupported_and_invalid_values(self) -> None:
        records = evidence_records(
            [
                ToolObservation(
                    tool="check_interface",
                    tool_input='{"device": "r1"}',
                    summary="r1 eth0 state DOWN",
                )
            ]
        )
        draft = DiagnosisDraft(
            anomaly_status="present",
            anomaly_evidence_ids=["ev-001"],
            faulty_devices=[
                SupportedClaim(value="r1", evidence_ids=["ev-001"]),
                SupportedClaim(value="r2", evidence_ids=["missing"]),
            ],
            root_causes=[
                SupportedClaim(value="link_down", evidence_ids=["ev-001"]),
                SupportedClaim(value="invented_name", evidence_ids=["ev-001"]),
            ],
        )
        claims = draft_claims(
            draft,
            available_root_causes=["link_down"],
            valid_evidence_ids={records[0].evidence_id},
        )
        result = validated_submission(
            draft,
            claims,
            SubmissionVerification(
                verdicts=[
                    ClaimVerdict(claim_id="anomaly", supported=True),
                    ClaimVerdict(claim_id="device:1", supported=True),
                    ClaimVerdict(claim_id="root_cause:1", supported=False),
                ]
            ),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.faulty_devices, ("r1",))
        self.assertEqual(result.root_cause_name, ())
        self.assertEqual(result.unsupported_claims, ("root_cause:1",))

    def test_unverified_anomaly_produces_conservative_submission(self) -> None:
        draft = DiagnosisDraft(
            anomaly_status="present",
            anomaly_evidence_ids=["ev-001"],
        )
        claims = [
            {
                "claim_id": "anomaly",
                "kind": "anomaly",
                "value": "present",
                "evidence_ids": ["ev-001"],
            }
        ]
        result = validated_submission(
            draft,
            claims,
            SubmissionVerification(
                verdicts=[ClaimVerdict(claim_id="anomaly", supported=False)]
            ),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.is_anomaly)
        self.assertEqual(result.faulty_devices, ())
        self.assertEqual(result.root_cause_name, ())
        self.assertEqual(result.unsupported_claims, ("anomaly",))

    def test_message_evidence_preserves_tool_arguments(self) -> None:
        observations = observations_from_messages(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "check_interface",
                            "args": {"device": "r1", "interface": "eth0"},
                        }
                    ],
                ),
                ToolMessage(
                    content="r1 eth0 state DOWN",
                    name="check_interface",
                    tool_call_id="call-1",
                ),
            ]
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].tool, "check_interface")
        self.assertIn('"device": "r1"', observations[0].tool_input)
        self.assertIn('"interface": "eth0"', observations[0].tool_input)

    async def test_phase_lists_causes_and_commits_exactly_once(self) -> None:
        phase = SubmissionPhase.__new__(SubmissionPhase)
        list_tool = SimpleNamespace(
            name="list_avail_problems",
            ainvoke=AsyncMock(return_value="['link_down', 'dns_error']"),
        )
        submit_tool = SimpleNamespace(
            name="submit",
            ainvoke=AsyncMock(return_value=["Submission success."]),
        )
        phase.tools = [list_tool, submit_tool]
        phase.drafter = AsyncMock()
        phase.drafter.ainvoke.return_value = DiagnosisDraft(
            anomaly_status="present",
            anomaly_evidence_ids=["ev-001"],
            faulty_devices=[
                SupportedClaim(value="r1", evidence_ids=["ev-001"])
            ],
            root_causes=[
                SupportedClaim(value="link_down", evidence_ids=["ev-001"])
            ],
        )
        phase.verifier = AsyncMock()
        phase.verifier.ainvoke.return_value = SubmissionVerification(
            verdicts=[
                ClaimVerdict(claim_id="anomaly", supported=True),
                ClaimVerdict(claim_id="device:1", supported=True),
                ClaimVerdict(claim_id="root_cause:1", supported=True),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = await phase.submit_report(
                task_description="Diagnose the network.",
                diagnosis_report="r1 eth0 is down.",
                observations=[
                    ToolObservation(
                        tool="check_interface",
                        tool_input='{"device": "r1"}',
                        summary="r1 eth0 state DOWN",
                    )
                ],
                session_dir=str(Path(tmp)),
            )

        self.assertIsNotNone(result)
        list_tool.ainvoke.assert_awaited_once_with({})
        submit_tool.ainvoke.assert_awaited_once_with(
            {
                "is_anomaly": True,
                "faulty_devices": ["r1"],
                "root_cause_name": ["link_down"],
            }
        )

    async def test_phase_splits_newline_problem_catalog(self) -> None:
        phase = SubmissionPhase.__new__(SubmissionPhase)
        list_tool = SimpleNamespace(
            name="list_avail_problems",
            ainvoke=AsyncMock(return_value="link_down\ndns_error\nlink_down\n"),
        )
        submit_tool = SimpleNamespace(
            name="submit",
            ainvoke=AsyncMock(return_value=["Submission success."]),
        )
        phase.tools = [list_tool, submit_tool]
        phase.drafter = AsyncMock()
        phase.drafter.ainvoke.return_value = DiagnosisDraft(
            anomaly_status="present",
            anomaly_evidence_ids=["ev-001"],
            root_causes=[
                SupportedClaim(value="dns_error", evidence_ids=["ev-001"])
            ],
        )
        phase.verifier = AsyncMock()
        phase.verifier.ainvoke.return_value = SubmissionVerification(
            verdicts=[
                ClaimVerdict(claim_id="anomaly", supported=True),
                ClaimVerdict(claim_id="root_cause:1", supported=True),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            await phase.submit_report(
                task_description="Diagnose.",
                diagnosis_report="DNS check failed.",
                observations=[ToolObservation(tool="dig", summary="timed out")],
                session_dir=tmp,
            )

        submit_tool.ainvoke.assert_awaited_once_with(
            {
                "is_anomaly": True,
                "faulty_devices": [],
                "root_cause_name": ["dns_error"],
            }
        )

    async def test_phase_commits_false_when_draft_is_inconclusive(self) -> None:
        phase = SubmissionPhase.__new__(SubmissionPhase)
        list_tool = SimpleNamespace(
            name="list_avail_problems",
            ainvoke=AsyncMock(return_value="link_down"),
        )
        submit_tool = SimpleNamespace(
            name="submit",
            ainvoke=AsyncMock(return_value=["Submission success."]),
        )
        phase.tools = [list_tool, submit_tool]
        phase.drafter = AsyncMock()
        phase.drafter.ainvoke.return_value = DiagnosisDraft(
            anomaly_status="inconclusive"
        )
        phase.verifier = AsyncMock()

        with tempfile.TemporaryDirectory() as tmp:
            result = await phase.submit_report(
                task_description="Diagnose.",
                diagnosis_report="Evidence is inconclusive.",
                observations=[ToolObservation(tool="ping", summary="ambiguous")],
                session_dir=tmp,
            )

        self.assertIsNotNone(result)
        phase.verifier.ainvoke.assert_not_awaited()
        submit_tool.ainvoke.assert_awaited_once_with(
            {
                "is_anomaly": False,
                "faulty_devices": [],
                "root_cause_name": [],
            }
        )

    async def test_phase_submits_false_without_current_evidence(self) -> None:
        phase = SubmissionPhase.__new__(SubmissionPhase)
        list_tool = SimpleNamespace(
            name="list_avail_problems",
            ainvoke=AsyncMock(return_value="['link_down']"),
        )
        submit_tool = SimpleNamespace(name="submit", ainvoke=AsyncMock())
        phase.tools = [list_tool, submit_tool]

        with tempfile.TemporaryDirectory() as tmp:
            result = await phase.submit_report(
                task_description="Diagnose.",
                diagnosis_report="Probably a link fault.",
                observations=[],
                session_dir=tmp,
            )

        self.assertIsNotNone(result)
        list_tool.ainvoke.assert_not_awaited()
        submit_tool.ainvoke.assert_awaited_once_with(
            {
                "is_anomaly": False,
                "faulty_devices": [],
                "root_cause_name": [],
            }
        )


if __name__ == "__main__":
    unittest.main()
