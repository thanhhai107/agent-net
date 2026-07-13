import unittest
from unittest.mock import patch

from agent.procedural_memory.models import (
    ProceduralSkill,
    SkillExperience,
    SkillStep,
    SkillTransition,
)
from agent.procedural_memory.policy_scorer import (
    BehavioralReplayPolicyScorer,
    PolicyLogprobScorer,
    PolicyReplayDraft,
    PolicyReplayItem,
    StructuredReplayPolicyScorer,
)
from agent.procedural_memory.policy_context import build_skill_policy_prefix


def _skill(skill_id: str, tool: str) -> ProceduralSkill:
    return ProceduralSkill(
        skill_id=skill_id,
        title=skill_id,
        activation_condition="Use when the endpoint cannot reach its peer.",
        execution_steps=[
            SkillStep(
                order=1, action=f"Call {tool} and inspect the result.", tool_name=tool
            )
        ],
        termination_condition="Stop after the result is interpreted.",
        tools=[tool],
    )


def _experience() -> SkillExperience:
    baseline = _skill("baseline", "cat_file")
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
                policy_context=build_skill_policy_prefix(
                    "Endpoint reachability is unknown.", baseline
                ),
            )
        ],
    )


class PolicyScorerTest(unittest.TestCase):
    def test_policy_logprob_rejects_transition_without_pre_action_context(self) -> None:
        experience = _experience().model_copy(deep=True)
        experience.transitions[0].policy_context = ""
        scorer = PolicyLogprobScorer(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
        )

        result = scorer.score_batch(
            candidate=_skill("candidate", "ping_pair"),
            baseline=_skill("baseline", "cat_file"),
            experiences=[experience],
        )

        self.assertEqual(result.method, "structured_replay")
        self.assertIn("no pre-action policy context", result.error)

    def test_policy_logprob_scorer_teacher_forces_target_actions(self) -> None:
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        def post(_url, *, json, **_kwargs):
            choices = []
            for index, prompt in enumerate(json["prompt"]):
                target_offset = prompt.rfind("Action:\n") + len("Action:\n")
                target_logprob = -1.0 if "Skill Name: candidate" in prompt else -2.0
                choices.append(
                    {
                        "index": index,
                        "logprobs": {
                            "text_offset": [0, target_offset],
                            "token_logprobs": [None, target_logprob],
                        },
                    }
                )
            return Response({"choices": choices})

        scorer = PolicyLogprobScorer(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
        )
        with patch(
            "agent.procedural_memory.policy_scorer.requests.post",
            side_effect=post,
        ) as request:
            result = scorer.score_batch(
                candidate=_skill("candidate", "ping_pair"),
                baseline=_skill("baseline", "cat_file"),
                experiences=[_experience()],
            )

        self.assertEqual(result.method, "policy_logprob")
        self.assertEqual(result.error, "")
        self.assertEqual(len(result.step_logprobs), 1)
        self.assertEqual(result.step_logprobs[0].candidate_logprob, -1.0)
        self.assertEqual(result.step_logprobs[0].baseline_logprob, -2.0)
        self.assertEqual(request.call_count, 2)

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
        prompts: list[str] = []

        class CompleteModel:
            def with_structured_output(self, _schema):
                return self

            def invoke(self, prompt):
                prompts.append(prompt)
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
        self.assertIn("Endpoint reachability is unknown", prompts[0])
        self.assertIn("Call ping_pair", prompts[0])
        self.assertNotIn("0% packet loss", prompts[0])

    def test_structured_replay_does_not_use_future_observation(self) -> None:
        first = _experience()
        second = first.model_copy(deep=True)
        second.experience_id = "exp-2"
        second.transitions[0].observation_summary = "A contradictory future result"
        scorer = StructuredReplayPolicyScorer()

        result = scorer.score_batch(
            candidate=_skill("candidate", "ping_pair"),
            baseline=None,
            experiences=[first, second],
        )

        self.assertEqual(
            result.scores[0].candidate_alignment,
            result.scores[1].candidate_alignment,
        )


if __name__ == "__main__":
    unittest.main()
