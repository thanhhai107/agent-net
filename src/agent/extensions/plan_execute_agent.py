"""Local Plan-and-Execute workflow using NIKA's diagnosis and submission phases."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from agent.composition import AgentRunConfig
from agent.extensions.workflow_base import ExtensionWorkflowBase
from agent.extensions.workflow_models import InvestigationPlan, ReplanDecision

PLANNER_PROMPT = """\
Create a concise ordered network-investigation plan for the supplied task.
Each step must be executable with diagnostic tools and state the evidence it
expects to collect. Do not diagnose the fault or propose remediation. Keep the
plan generic to the observed topology and task; do not assume hidden ground truth.
"""

SYNTHESIS_PROMPT = """\
Produce one diagnosis report from the task and collected investigation results.
State whether an anomaly exists, identify faulty devices when supported, identify
the root cause when supported, and cite concrete observations. Preserve uncertainty
when evidence is incomplete. Do not propose remediation and do not submit results.
"""

REPLANNER_PROMPT = """\
Review the task, completed investigation results, and remaining plan. Finish only
when anomaly status, localization, and root cause are adequately supported; then
return a concise diagnosis report. Otherwise return only the still-needed ordered
steps, without repeating completed work. Do not assume hidden ground truth.
"""


class PlanExecuteAgent(ExtensionWorkflowBase):
    """Plan, execute bounded steps, replan, then use upstream submission."""

    workflow_name = "plan-execute"

    def __init__(self, config: AgentRunConfig) -> None:
        super().__init__(config)
        self.planner = self.llm.with_structured_output(InvestigationPlan)
        self.replanner = self.llm.with_structured_output(ReplanDecision)

    async def run(self, task_description: str) -> dict[str, Any]:
        try:
            diagnosis_runner = self.prepare_diagnosis(task_description)
            try:
                raw_plan = await self.planner.ainvoke(
                    [
                        SystemMessage(content=PLANNER_PROMPT),
                        HumanMessage(content=task_description),
                    ],
                    config={"callbacks": [self.callback("planner")]},
                )
                plan = InvestigationPlan.model_validate(raw_plan)
            except Exception as exc:
                self.log_error("planner", exc)
                return {"diagnosis_report": "", "planning_failed": True}
            observations: list[dict[str, str]] = []
            remaining_steps = list(plan.steps)
            report = ""
            while remaining_steps and len(observations) < self.max_steps:
                step = remaining_steps.pop(0)
                prompt = (
                    f"Original task:\n{task_description}\n\n"
                    f"Investigation objective: {plan.objective}\n"
                    f"Assigned step {step.step_id}: {step.action}\n"
                    f"Expected evidence: {step.expected_evidence}\n\n"
                    "Perform this step with the available tools. Return the observed "
                    "evidence and its implication. Do not submit a final answer."
                )
                try:
                    result = await diagnosis_runner.ainvoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config={
                            "callbacks": [self.callback(f"executor_{step.step_id}")],
                            "recursion_limit": self.max_steps,
                        },
                    )
                except GraphRecursionError:
                    return {
                        "diagnosis_report": "ERROR_MAX_STEPS_REACHED",
                        "is_max_steps_reached": True,
                    }
                except Exception as exc:
                    self.log_error(f"executor_{step.step_id}", exc)
                    observations.append(
                        {
                            "step_id": step.step_id,
                            "action": step.action,
                            "expected_evidence": step.expected_evidence,
                            "result": f"Step failed: {type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                observations.append(
                    {
                        "step_id": step.step_id,
                        "action": step.action,
                        "expected_evidence": step.expected_evidence,
                        "result": str(result["messages"][-1].content),
                    }
                )

                try:
                    raw_decision = await self.replanner.ainvoke(
                        [
                            SystemMessage(content=REPLANNER_PROMPT),
                            HumanMessage(
                                content=json.dumps(
                                    {
                                        "task": task_description,
                                        "objective": plan.objective,
                                        "completed_results": observations,
                                        "remaining_plan": [
                                            item.model_dump()
                                            for item in remaining_steps
                                        ],
                                    },
                                    ensure_ascii=False,
                                    default=str,
                                )
                            ),
                        ],
                        config={"callbacks": [self.callback("replanner")]},
                    )
                    decision = ReplanDecision.model_validate(raw_decision)
                    if decision.completed:
                        report = decision.diagnosis_report.strip()
                        break
                    remaining_steps = list(decision.remaining_steps)
                except Exception as exc:
                    self.log_error("replanner", exc)

            synthesis_payload = json.dumps(
                {
                    "task": task_description,
                    "objective": plan.objective,
                    "investigation_results": observations,
                },
                ensure_ascii=False,
                default=str,
            )
            if not report:
                try:
                    synthesis = await self.llm.ainvoke(
                        [
                            SystemMessage(content=SYNTHESIS_PROMPT),
                            HumanMessage(content=synthesis_payload),
                        ],
                        config={"callbacks": [self.callback("synthesis")]},
                    )
                    report = str(synthesis.content).strip()
                except Exception as exc:
                    self.log_error("synthesis", exc)
                    report = synthesis_payload
            await self.explore_tools(task_description)
            return await self.submit(report)
        finally:
            self.write_extension_snapshots()
