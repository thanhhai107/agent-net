"""Local Reflexion workflow using NIKA's diagnosis and submission phases."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from agent.composition import AgentRunConfig
from agent.extensions.workflow_base import ExtensionWorkflowBase
from agent.extensions.workflow_models import ReflexionEvaluation, ReflexionLesson

EVALUATOR_PROMPT = """\
Evaluate the diagnosis attempt using only the task, report, and visible tool
trajectory. Success requires an explicit anomaly decision, supported localization
when anomalous, and a supported root cause without unresolved contradictions.
Provide actionable feedback for any failure. Do not infer hidden ground truth.
"""

REFLECTION_PROMPT = """\
Turn the failed attempt and evaluator feedback into compact episodic guidance for
the next attempt. Identify reasoning or tool-selection mistakes and prescribe a
materially different evidence-gathering strategy. Do not solve the task and do not
promote unverified hypotheses to facts.
"""


class ReflexionAgent(ExtensionWorkflowBase):
    """Attempt, evaluate, reflect, retry, then submit the best report."""

    workflow_name = "reflexion"

    def __init__(self, config: AgentRunConfig) -> None:
        super().__init__(config)
        self.max_attempts = config.max_attempts
        self.evaluator = self.llm.with_structured_output(ReflexionEvaluation)
        self.reflector = self.llm.with_structured_output(ReflexionLesson)

    async def run(self, task_description: str) -> dict[str, Any]:
        reflections: list[ReflexionLesson] = []
        best_report = ""
        best_score = -1.0
        try:
            diagnosis_runner = self.prepare_diagnosis(task_description)
            for attempt in range(1, self.max_attempts + 1):
                prompt = json.dumps(
                    {
                        "task": task_description,
                        "attempt": attempt,
                        "prior_reflections": [
                            item.model_dump() for item in reflections
                        ],
                        "instruction": (
                            "Investigate with the available tools and return a complete "
                            "evidence-based diagnosis report. Do not submit results."
                        ),
                    },
                    ensure_ascii=False,
                )
                try:
                    result = await diagnosis_runner.ainvoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config={
                            "callbacks": [self.callback(f"attempt_{attempt}")],
                            "recursion_limit": self.max_steps,
                        },
                    )
                except GraphRecursionError:
                    return {
                        "diagnosis_report": "ERROR_MAX_STEPS_REACHED",
                        "is_max_steps_reached": True,
                    }
                except Exception as exc:
                    self.log_error(f"attempt_{attempt}", exc)
                    continue
                report = str(result["messages"][-1].content).strip()
                trajectory = [
                    {
                        "type": message.__class__.__name__,
                        "name": getattr(message, "name", None),
                        "content": str(getattr(message, "content", ""))[:4000],
                    }
                    for message in result["messages"][-24:]
                ]
                try:
                    raw_evaluation = await self.evaluator.ainvoke(
                        [
                            SystemMessage(content=EVALUATOR_PROMPT),
                            HumanMessage(
                                content=json.dumps(
                                    {
                                        "task": task_description,
                                        "report": report,
                                        "trajectory": trajectory,
                                    },
                                    ensure_ascii=False,
                                    default=str,
                                )
                            ),
                        ],
                        config={"callbacks": [self.callback(f"evaluator_{attempt}")]},
                    )
                    evaluation = ReflexionEvaluation.model_validate(raw_evaluation)
                except Exception as exc:
                    self.log_error(f"evaluator_{attempt}", exc)
                    evaluation = ReflexionEvaluation(
                        success=False,
                        quality_score=0.0,
                        evidence_sufficient=False,
                        feedback=["Evaluator output was unavailable or invalid."],
                    )
                if evaluation.quality_score > best_score:
                    best_score = evaluation.quality_score
                    best_report = report
                if evaluation.success:
                    break
                if attempt < self.max_attempts:
                    try:
                        raw_lesson = await self.reflector.ainvoke(
                            [
                                SystemMessage(content=REFLECTION_PROMPT),
                                HumanMessage(
                                    content=json.dumps(
                                        {
                                            "report": report,
                                            "evaluation": evaluation.model_dump(),
                                            "prior_reflections": [
                                                item.model_dump()
                                                for item in reflections
                                            ],
                                        },
                                        ensure_ascii=False,
                                    )
                                ),
                            ],
                            config={"callbacks": [self.callback(f"reflect_{attempt}")]},
                        )
                        reflections.append(ReflexionLesson.model_validate(raw_lesson))
                    except Exception as exc:
                        self.log_error(f"reflect_{attempt}", exc)
            self.record_terminal_diagnosis(best_report)
            await self.explore_tools(task_description)
            return await self.submit(best_report)
        finally:
            self.write_extension_snapshots()
