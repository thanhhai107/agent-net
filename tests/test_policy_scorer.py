import unittest

from agent.memory.models import (
    ProceduralSkill,
    SkillExperience,
    SkillStep,
    SkillTransition,
)
from agent.memory.policy_scorer import (
    BehavioralReplayPolicyScorer,
    PolicyReplayDraft,
    PolicyReplayItem,
    StructuredReplayPolicyScorer,
)


def _skill(skill_id: str, tool: str) -> ProceduralSkill:
    return ProceduralSkill(
        skill_id=skill_id,
        title=skill_id,
        activation_condition="Use when the endpoint cannot reach its peer.",
        execution_steps=[
            SkillStep(order=1, action=f"Call {tool} and inspect the result.", tool_name=tool)
        ],
        termination_condition="Stop after the result is interpreted.",
        tools=[tool],
    )


def _experience() -> SkillExperience:
    return SkillExperience(
        experience_id="exp-1",
        session_id="session-1",
        reward=1.0,
        trajectory="The endpoint cannot reach its peer.",
        transitions=[
            SkillTransition(
                state="Endpoint reachability is unknown.",
                action="Call ping_pair and inspect the result.",
                tool_name="ping_pair",
                observation_summary="0% packet loss",
                done=True,
            )
        ],
    )


class PolicyScorerTest(unittest.TestCase):
    def test_structured_replay_prefers_matching_tool_policy(self) -> None:
        result = StructuredReplayPolicyScorer().score_batch(
            candidate=_skill("candidate", "ping_pair"),
            baseline=_skill("baseline", "cat_file"),
            experiences=[_experience()],
        )

        self.assertEqual(result.method, "structured_replay")
        self.assertGreater(
            result.scores[0].candidate_alignment,
            result.scores[0].baseline_alignment,
        )

    def test_behavioral_replay_requires_every_experience_id(self) -> None:
        class IncompleteModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, _prompt):
                return PolicyReplayDraft(scores=[])

        result = BehavioralReplayPolicyScorer(lambda: IncompleteModel()).score_batch(
            candidate=_skill("candidate", "ping_pair"),
            baseline=None,
            experiences=[_experience()],
        )

        self.assertEqual(result.method, "structured_replay")
        self.assertIn("omitted or invented", result.error)

    def test_behavioral_replay_reports_verified_batch(self) -> None:
        class CompleteModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, _prompt):
                return PolicyReplayDraft(
                    scores=[
                        PolicyReplayItem(
                            experience_id="exp-1",
                            candidate_alignment=0.8,
                            baseline_alignment=0.2,
                        )
                    ]
                )

        result = BehavioralReplayPolicyScorer(lambda: CompleteModel()).score_batch(
            candidate=_skill("candidate", "ping_pair"),
            baseline=_skill("baseline", "cat_file"),
            experiences=[_experience()],
        )

        self.assertEqual(result.method, "behavioral_replay")
        self.assertEqual(result.error, "")
        self.assertEqual(result.scores[0].candidate_alignment, 0.8)


if __name__ == "__main__":
    unittest.main()
