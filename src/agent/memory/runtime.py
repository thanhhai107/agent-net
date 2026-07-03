"""Online Skill-Pro runtime that binds memory, prompt, skills, and tools.

The offline memory module learns reusable procedures after a session closes.
This runtime is the read-time counterpart: it injects retrieved procedures into
the diagnosis prompt, adds tool-specific skill hints to tool descriptions, and
keeps an active Skill-MDP option across tool calls.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field

from agent.memory.attributes import infer_memory_attributes
from agent.memory.models import MemoryQuery, SkillRetrieval
from agent.memory.safety import redact_oracle_markers
from agent.memory.service import ProceduralMemoryModule
from agent.tool_evolution.runtime import ToolEvolutionRuntime
from agent.utils.evidence import extract_link_down_devices
from agent.utils.loggers import MessageLogger

INTEGRATED_GUIDANCE_MARKER = "[Integrated learning guidance - not evidence]"
ENV_SKILL_META_CONTROLLER = "NIKA_SKILL_META_CONTROLLER"
ENV_SKILL_SELECTOR = "NIKA_SKILL_SELECTOR"
INTERNAL_TOOL_CALL_ID = "skill-runtime-internal"


def _short_text(value: Any, *, limit: int = 900) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def strip_integrated_learning_guidance(value: Any) -> str:
    text = str(value or "")
    if INTEGRATED_GUIDANCE_MARKER in text:
        text = text.split(INTEGRATED_GUIDANCE_MARKER, 1)[0]
    return text.strip()


def _compact_json(value: Any, *, limit: int = 900) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return _short_text(text, limit=limit)


def _compact_items(items: list[str], *, item_limit: int = 100) -> str:
    return "; ".join(_short_text(item, limit=item_limit) for item in items if item)


def _tool_input_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    payload = {
        key: value
        for key, value in kwargs.items()
        if key not in {"callbacks", "config", "run_manager"}
    }
    if payload:
        return payload
    if len(args) == 1:
        return args[0]
    if args:
        return {"args": list(args)}
    return {}


def _tool_input_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, "", [], {}):
        return {}
    return {"_value": value}


def _tool_result_content(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return getattr(result, "content", result)


def _append_followup_guidance(result: Any, guidance: str) -> Any:
    if isinstance(result, tuple) and len(result) == 2:
        content, artifact = result
        return (_append_guidance_to_content(content, guidance), artifact)
    return _append_guidance_to_content(result, guidance)


def _append_guidance_to_content(content: Any, guidance: str) -> Any:
    if isinstance(content, list):
        return [*content, {"type": "text", "text": guidance}]
    return f"{content}\n\n{guidance}"


def _tool_call_key(tool_name: str, tool_input: Any) -> str:
    return f"{tool_name}:{_compact_json(_tool_input_dict(tool_input), limit=500)}"


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


class SkillToolRuntime:
    """Runtime controller for online Skill-Pro usage during diagnosis."""

    def __init__(
        self,
        *,
        memory: ProceduralMemoryModule,
        memory_mode: str,
        session: Any,
        task_description: str,
        tools: list[BaseTool],
        session_dir: str | Path = "",
        tool_evolution_runtime: ToolEvolutionRuntime | None = None,
        top_k: int = 5,
        token_budget: int = 1500,
        max_skill_age: int = 4,
        selector_min_lcb: float = -0.05,
        selector_nominee_k: int = 3,
        meta_controller_llm: Any | None = None,
        meta_controller_mode: str | None = None,
        skill_selector_mode: str | None = None,
    ) -> None:
        self.memory = memory
        self.memory_mode = memory_mode
        self.session = session
        self.task_description = task_description
        self.tool_names = [tool.name for tool in tools]
        self.tool_evolution_runtime = tool_evolution_runtime
        self.top_k = top_k
        self.token_budget = token_budget
        self.max_skill_age = max(1, max_skill_age)
        self.selector_min_lcb = float(selector_min_lcb)
        self.selector_nominee_k = max(1, int(selector_nominee_k))
        self.meta_controller_llm = meta_controller_llm
        self.meta_controller_mode = (
            meta_controller_mode
            or os.getenv(ENV_SKILL_META_CONTROLLER, "heuristic")
        ).strip().lower()
        if self.meta_controller_mode not in {"heuristic", "llm"}:
            self.meta_controller_mode = "heuristic"
        self.skill_selector_mode = (
            skill_selector_mode
            or os.getenv(ENV_SKILL_SELECTOR, "lcb")
        ).strip().lower()
        if self.skill_selector_mode not in {"lcb", "llm_topk_lcb"}:
            self.skill_selector_mode = "lcb"
        self.active_skill: SkillRetrieval | None = None
        self.skill_age = 0
        self.prompt_selection_count = 0
        self.post_tool_selection_count = 0
        self.meta_controller_cache_hits = 0
        self.skill_cooldowns: dict[str, int] = {}
        self.recent_observations: list[str] = []
        self.recent_transitions: list[dict[str, Any]] = []
        self.pending_draft_explorations: dict[str, dict[str, Any]] = {}
        self.inflight_tool_calls = 0
        self._last_meta_controller_signature = ""
        self._last_meta_controller_reason = ""
        self._lock = Lock()
        self._metrics_lock = Lock()
        self.prompt_added_tokens = 0
        self.tool_description_added_tokens = 0
        self.followup_added_tokens = 0
        self.prompt_injection_count = 0
        self.tool_description_injection_count = 0
        self.followup_guidance_count = 0
        self._logger = (
            MessageLogger(
                agent="memory_agent",
                session_dir=str(session_dir),
                extra_fields={"phase": "skill_mdp_runtime"},
            )
            if session_dir
            else None
        )

    @property
    def scenario(self) -> str:
        return str(getattr(self.session, "scenario_name", "") or "")

    @property
    def topology_class(self) -> str:
        return str(getattr(self.session, "scenario_topo_size", "") or "")

    def prompt_suffix(self, *, activate_skill: bool = True) -> str:
        active_skill, retrieved = self._prepare_prompt_context(
            activate_skill=activate_skill
        )
        active_skill_id = active_skill.skill.skill_id if active_skill else ""
        context = self.memory.format_context(
            retrieved,
            active_skill_id=active_skill_id,
        )
        tool_candidates = sorted(self._contextual_tool_candidates(active_skill))
        active_block = self._active_skill_prompt_block(
            active_skill,
            tool_candidates=tool_candidates,
        )
        tool_links = self._active_skill_tool_links(
            active_skill,
            tool_candidates=tool_candidates,
        )
        draft_context = self._draft_prompt_context(tool_candidates)
        if activate_skill:
            sections = [
                "\n\nIntegrated Skill-Pro + DRAFT diagnosis loop:",
                "- Treat retrieved skills as advisory Skill-MDP options with initiation, policy, and termination.",
                (
                    "- The active Skill-MDP option below is a planning prior for the next LLM action; follow it when it improves evidence gathering."
                    if active_skill is not None
                    else "- No Skill-MDP option was activated for this step; use the candidate skill pool only as planning guidance."
                ),
                (
                    "- Before each tool call, prefer tools whose output can test or advance the active skill, but keep the normal ReAct evidence loop in control."
                    if active_skill is not None
                    else "- Before each tool call, pick evidence-gathering tools that can test the most relevant candidate skill initiation or policy step."
                ),
                "- After each tool result, decide whether the active skill should continue, terminate, or switch.",
                "- Skill termination only controls switching skills; it is not a final diagnosis stop condition.",
                "- Treat DRAFT tool docs and next-check suggestions as tool-use guidance, not as evidence.",
                "- Final diagnosis must cite current tool observations; memory and docs cannot replace evidence.",
            ]
            active_label = "Advisory Skill-MDP option selected before next LLM action"
        else:
            sections = [
                "\n\nIntegrated Skill-Pro + DRAFT read-only planning context:",
                "- Treat retrieved skills as candidate Skill-MDP options for planning only.",
                "- Do not treat a candidate as activated or reused until an execution step or tool call selects it.",
                "- Plan checks whose observations can test a skill initiation, advance a policy step, or verify termination.",
                "- Treat DRAFT tool docs and next-check suggestions as tool-use guidance, not as evidence.",
                "- Final diagnosis must cite current tool observations; memory and docs cannot replace evidence.",
            ]
            active_label = "Currently active Skill-MDP option from prior execution"
        if active_block:
            sections.append(
                f"\n{active_label}:\n"
                + active_block
            )
        if tool_links:
            sections.append("\nActive skill-tool links:\n" + tool_links)
        if draft_context:
            sections.append(
                "\nDRAFT tool documentation linked to Skill-Pro options:\n"
                + draft_context
            )
        if context:
            sections.append("\nSkill pool candidates:\n" + context)
        suffix = "\n".join(sections)
        added_tokens = self._record_added_tokens("prompt", suffix)
        self._log(
            "skill_prompt_context",
            {
                "activate_skill": activate_skill,
                "added_tokens": added_tokens,
                "active_skill_id": active_skill.skill.skill_id
                if active_skill
                else "",
                "retrieved_skills": [
                    item.skill.skill_id for item in retrieved[: self.top_k]
                ],
            },
        )
        return suffix

    def wrap_tools(self, tools: list[BaseTool]) -> list[BaseTool]:
        return [SkillAwareTool(wrapped_tool=tool, runtime=self) for tool in tools]

    def describe_tool(self, tool: BaseTool) -> str:
        description = _short_text(
            str(getattr(tool, "description", "") or "").strip(),
            limit=1200,
        )
        query = self._query(
            extra_text=f"tool:{tool.name} {description}",
            tools=[tool.name],
            top_k=2,
            token_budget=700,
        )
        retrieved = self.memory.retrieve(query=query)
        scoped_tools = self._retrieved_tool_scope(retrieved)
        if self.tool_evolution_runtime is not None:
            scoped_tools.update(self._draft_tool_scope())
        if scoped_tools and tool.name not in scoped_tools:
            return description
        skill_lines: list[str] = []
        for item in retrieved[:2]:
            if tool.name not in self._contextual_tool_candidates(item):
                continue
            skill = item.skill
            policy = "; ".join(step.action for step in skill.execution_steps[:2])
            skill_lines.append(
                redact_oracle_markers(
                    f"- {skill.skill_id}: use when {skill.activation_condition} Policy: {policy}"
                )
            )
        has_draft_guidance = (
            "DRAFT refined guidance:" in description
            or "DRAFT tool guidance:" in description
        )
        tool_guidance = ""
        draft_checks: list[str] = []
        if self.tool_evolution_runtime is not None:
            draft_checks = self.tool_evolution_runtime.next_checks(
                tool.name,
                limit=self.tool_evolution_runtime.next_checks_limit,
            )
            if not has_draft_guidance:
                tool_guidance = self.tool_evolution_runtime.tool_runtime_guidance(
                    tool.name,
                    max_chars=max(160, self.tool_evolution_runtime.tool_doc_chars // 2),
                )
        if not skill_lines and not tool_guidance and not draft_checks:
            return description
        additions = [
            "Integrated learning guidance:",
            (
                "Use this tool as part of the advisory Skill-MDP loop when its "
                "output can test a current hypothesis, cover an untested "
                "layer, or advance a skill policy step."
            ),
        ]
        if skill_lines:
            additions.append("Relevant procedural skills:\n" + "\n".join(skill_lines))
        if tool_guidance:
            additions.append("DRAFT tool guidance:\n" + tool_guidance)
        guidance = "\n".join(additions)
        self._record_added_tokens("tool_description", guidance)
        return (description + "\n\n" + guidance).strip()

    def before_tool(self, *, tool_name: str, tool_input: Any) -> SkillRetrieval | None:
        with self._lock:
            draft_exploration = self._planned_draft_exploration(
                tool_name=tool_name,
                tool_input=tool_input,
            )
            call_key = _tool_call_key(tool_name, tool_input)
            if draft_exploration is not None:
                self.pending_draft_explorations[call_key] = draft_exploration
            query = self._query(
                extra_text=f"next tool:{tool_name} input:{_compact_json(tool_input, limit=500)}",
                tools=[tool_name],
                top_k=max(3, self.top_k),
            )
            if self.active_skill is None:
                self._select_active_skill(query=query, source="tool_fallback")
            self.inflight_tool_calls += 1
            if self.active_skill is not None:
                self.skill_age += 1
                if not self._tool_matches_active_skill(
                    self.active_skill,
                    tool_name,
                ):
                    self._log(
                        "skill_policy_deviation",
                        {
                            "active_skill_id": self.active_skill.skill.skill_id,
                            "tool": tool_name,
                            "expected_tools": sorted(
                                self._contextual_tool_candidates(self.active_skill)
                            ),
                            "tool_input": tool_input,
                        },
                    )
            draft_payload = self._draft_log_payload(draft_exploration)
            self._log(
                "skill_activation",
                {
                    "source": "tool",
                    "active_skill_id": self.active_skill.skill.skill_id
                    if self.active_skill
                    else "",
                    "active_skill_score": round(self.active_skill.score, 6)
                    if self.active_skill
                    else 0.0,
                    "tool": tool_name,
                    "tool_input": tool_input,
                    "skill_age": self.skill_age,
                    **draft_payload,
                },
            )
            return self.active_skill

    def after_tool(
        self,
        *,
        tool_name: str,
        tool_input: Any,
        result: Any,
        status: str = "success",
    ) -> Any:
        text = _short_text(_tool_result_content(result), limit=1600)
        with self._lock:
            call_key = _tool_call_key(tool_name, tool_input)
            draft_exploration = self.pending_draft_explorations.pop(
                call_key,
                None,
            )
            observation = f"{tool_name}({_compact_json(tool_input, limit=300)}) -> {text}"
            self.recent_observations.append(observation)
            self.recent_observations = self.recent_observations[-12:]
            transition = {
                "tool": tool_name,
                "tool_input": tool_input,
                "status": status,
                "observation_summary": text,
            }
            transition.update(self._draft_log_payload(draft_exploration))
            self.recent_transitions.append(transition)
            self.recent_transitions = self.recent_transitions[-16:]
            self.inflight_tool_calls = max(0, self.inflight_tool_calls - 1)
            self._log(
                "skill_transition",
                {
                    "active_skill_id": self.active_skill.skill.skill_id
                    if self.active_skill
                    else "",
                    "tool": tool_name,
                    "tool_input": tool_input,
                    "status": status,
                    "observation_summary": text,
                    **self._draft_log_payload(draft_exploration),
                },
            )
            emit_followup_guidance = self.inflight_tool_calls == 0
            if emit_followup_guidance:
                self._refresh_active_skill_after_observation(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    observation_summary=text,
                    status=status,
                )
                guidance = self._followup_guidance(
                    tool_name,
                    draft_exploration=draft_exploration,
                )
            else:
                guidance = ""
        if not guidance:
            return result
        self._record_added_tokens("followup", guidance)
        return _append_followup_guidance(result, guidance)

    def snapshot(self) -> dict[str, Any]:
        return {
            "memory_mode": self.memory_mode,
            "bank_id": self.memory.bank_id,
            "active_skill_id": self.active_skill.skill.skill_id
            if self.active_skill
            else "",
            "skill_age": self.skill_age,
            "prompt_selection_count": self.prompt_selection_count,
            "post_tool_selection_count": self.post_tool_selection_count,
            "meta_controller_cache_hits": self.meta_controller_cache_hits,
            "meta_controller_mode": self.meta_controller_mode,
            "skill_selector_mode": self.skill_selector_mode,
            "config": {
                "top_k": self.top_k,
                "token_budget": self.token_budget,
                "max_skill_age": self.max_skill_age,
                "selector_min_lcb": self.selector_min_lcb,
                "selector_nominee_k": self.selector_nominee_k,
            },
            "skill_cooldowns": dict(self.skill_cooldowns),
            "tool_names": self.tool_names,
            "recent_observations": self.recent_observations,
            "recent_transitions": self.recent_transitions,
            "pending_draft_exploration_ids": [
                str(item.get("exploration_id") or "")
                for item in self.pending_draft_explorations.values()
                if item.get("exploration_id")
            ],
            "inflight_tool_calls": self.inflight_tool_calls,
            "prompt_added_tokens": self.prompt_added_tokens,
            "tool_description_added_tokens": self.tool_description_added_tokens,
            "followup_added_tokens": self.followup_added_tokens,
            "total_added_tokens": self.total_added_tokens,
            "prompt_injection_count": self.prompt_injection_count,
            "tool_description_injection_count": (
                self.tool_description_injection_count
            ),
            "followup_guidance_count": self.followup_guidance_count,
        }

    @property
    def total_added_tokens(self) -> int:
        return (
            self.prompt_added_tokens
            + self.tool_description_added_tokens
            + self.followup_added_tokens
        )

    def _record_added_tokens(self, bucket: str, text: str) -> int:
        added_tokens = _estimate_tokens(text)
        with self._metrics_lock:
            if bucket == "prompt":
                self.prompt_added_tokens += added_tokens
                self.prompt_injection_count += 1
            elif bucket == "tool_description":
                self.tool_description_added_tokens += added_tokens
                self.tool_description_injection_count += 1
            elif bucket == "followup":
                self.followup_added_tokens += added_tokens
                self.followup_guidance_count += 1
        return added_tokens

    def _planned_draft_exploration(
        self,
        *,
        tool_name: str,
        tool_input: Any,
    ) -> dict[str, Any] | None:
        if self.tool_evolution_runtime is None:
            return None
        return self.tool_evolution_runtime.match_planned_exploration(
            tool_name,
            _tool_input_dict(tool_input),
            diagnosis_only=True,
        )

    @staticmethod
    def _draft_log_payload(
        draft_exploration: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not draft_exploration:
            return {}
        return {
            "draft_exploration_id": str(
                draft_exploration.get("exploration_id") or ""
            ),
            "draft_next_exploration": str(
                draft_exploration.get("next_exploration") or ""
            ),
            "draft_planned_parameters": draft_exploration.get("parameters") or {},
        }

    def _prepare_prompt_context(
        self,
        *,
        activate_skill: bool = True,
    ) -> tuple[SkillRetrieval | None, list[SkillRetrieval]]:
        with self._lock:
            query = self._query(extra_text="decision prompt before next action")
            if not activate_skill:
                retrieved = self.memory.retrieve(query=query, session_id="")
                return self.active_skill, self._merge_active_with_retrieved(retrieved)
            termination_reason = (
                ""
                if self.active_skill is None
                else self._active_skill_termination_reason(
                    query,
                    allow_context_mismatch=True,
                    source="prompt",
                )
            )
            if self.active_skill is None or termination_reason:
                if termination_reason:
                    previous_skill_id = (
                        self.active_skill.skill.skill_id
                        if self.active_skill
                        else ""
                    )
                    self._cool_down_skill(previous_skill_id)
                    self._log(
                        "skill_termination",
                        {
                            "previous_skill_id": previous_skill_id,
                            "reason": termination_reason,
                            "source": "prompt",
                        },
                    )
                self._select_active_skill(query=query, source="prompt")
            retrieved = self.memory.retrieve(query=query, session_id="")
            return self.active_skill, self._merge_active_with_retrieved(retrieved)

    def _query(
        self,
        *,
        extra_text: str = "",
        tools: list[str] | None = None,
        top_k: int | None = None,
        token_budget: int | None = None,
    ) -> MemoryQuery:
        query_tools = tools or self.tool_names
        text = " ".join(
            item
            for item in [
                self.task_description,
                extra_text,
                self._draft_selection_context(tools=query_tools),
                " ".join(self.recent_observations[-3:]),
            ]
            if item
        )
        attrs = infer_memory_attributes(
            text,
            scenario=self.scenario,
            topology_class=self.topology_class,
            tools=query_tools,
        )
        return MemoryQuery(
            text=text,
            scenario=self.scenario,
            topology_class=self.topology_class,
            protocols=attrs.protocols,
            services=attrs.services,
            symptoms=attrs.symptoms,
            task_stage="diagnosis",
            tools=query_tools,
            top_k=top_k or self.top_k,
            token_budget=token_budget or self.token_budget,
        )

    def _draft_selection_context(self, *, tools: list[str]) -> str:
        if self.tool_evolution_runtime is None:
            return ""
        pieces: list[str] = []
        tool_filter = {tool for tool in tools if tool}
        planned = self.tool_evolution_runtime.planned_explorations(
            limit=self.tool_evolution_runtime.planned_checks,
            diagnosis_only=True,
        )
        planned_lines: list[str] = []
        for item in planned:
            tool_name = str(item.get("tool_name") or "")
            if tool_filter and tool_name not in tool_filter:
                continue
            planned_lines.append(
                " ".join(
                    part
                    for part in [
                        f"{tool_name}[{item.get('exploration_id') or ''}]",
                        _short_text(item.get("next_exploration") or "", limit=140),
                        f"parameters={_compact_json(item.get('parameters') or {}, limit=180)}"
                        if item.get("parameters")
                        else "",
                    ]
                    if part
                )
            )
        if planned_lines:
            pieces.append("DRAFT active exploration queue: " + " | ".join(planned_lines))
        for tool_name in sorted(tool_filter or set(self.tool_names))[:6]:
            checks = self.tool_evolution_runtime.next_checks(
                tool_name,
                limit=self.tool_evolution_runtime.next_checks_limit,
                diagnosis_only=True,
            )
            if checks:
                pieces.append(
                    f"DRAFT next checks for {tool_name}: "
                    + _compact_items(checks)
                )
        return _short_text(" ".join(pieces), limit=1400)

    def _draft_prompt_context(self, tool_names: list[str]) -> str:
        if self.tool_evolution_runtime is None:
            return ""
        return _short_text(
            self.tool_evolution_runtime.prompt_suffix(
                tool_names=tool_names,
                diagnosis_only=True,
            ).strip(),
            limit=1200,
        )

    def _active_skill_termination_reason(
        self,
        query: MemoryQuery,
        *,
        allow_context_mismatch: bool = True,
        source: str = "",
    ) -> str:
        if self.active_skill is None:
            return "no_active_skill"
        if self.skill_age >= self.max_skill_age:
            return "max_skill_age"
        if self._termination_condition_satisfied(self.active_skill.skill):
            return "termination_condition_satisfied"
        meta_reason = self._meta_controller_termination_reason(
            query=query,
            source=source,
        )
        if meta_reason:
            return meta_reason
        if not allow_context_mismatch:
            return ""
        active_id = self.active_skill.skill.skill_id
        for item in self.memory.retrieve(query=query, session_id=""):
            if item.skill.skill_id == active_id and item.score > 0.05:
                return ""
        return "context_mismatch"

    def _meta_controller_termination_reason(
        self,
        *,
        query: MemoryQuery,
        source: str,
    ) -> str:
        if (
            self.meta_controller_mode != "llm"
            or self.meta_controller_llm is None
            or self.active_skill is None
            or self.skill_age <= 0
        ):
            return ""
        skill = self.active_skill.skill
        state_signature = self._meta_controller_state_signature()
        if state_signature == self._last_meta_controller_signature:
            self.meta_controller_cache_hits += 1
            self._log(
                "skill_meta_controller",
                {
                    "source": source,
                    "active_skill_id": skill.skill_id,
                    "status": "cached",
                    "cached_reason": self._last_meta_controller_reason,
                },
            )
            return self._last_meta_controller_reason
        state_text = "\n".join(
            item
            for item in [
                query.text,
                "Recent observations:",
                "\n".join(self.recent_observations[-4:]),
            ]
            if item
        )
        prompt = (
            "[ROLE]\n"
            "You are a Skill-Pro meta-controller supervising a NIKA network "
            "diagnosis agent.\n\n"
            "[CURRENT STATE]\n"
            f"{state_text[:3000]}\n\n"
            "[ACTIVE OPTION]\n"
            f"Name: {redact_oracle_markers(skill.skill_id)}\n"
            f"Initiation: {redact_oracle_markers(skill.activation_condition)}\n"
            "Policy:\n"
            + "\n".join(
                f"- {redact_oracle_markers(step.action)}"
                for step in skill.execution_steps[:6]
            )
            + "\n"
            f"Termination: {redact_oracle_markers(skill.termination_condition)}\n\n"
            "[YOUR TASK]\n"
            "Return DONE if the termination condition is satisfied by current "
            "observations, or if the initiation condition no longer fits. "
            "Return CONTINUE if the option should keep controlling the next "
            "diagnostic step.\n\n"
            "[FORMAT]\n"
            "Output exactly one line:\n"
            "<status>DONE</status>\n"
            "or\n"
            "<status>CONTINUE</status>"
        )
        raw_text = ""
        try:
            response = self.meta_controller_llm.invoke(prompt)
            raw_text = str(getattr(response, "content", response) or "")
            status = self._parse_meta_controller_status(raw_text)
            self._log(
                "skill_meta_controller",
                {
                    "source": source,
                    "active_skill_id": skill.skill_id,
                    "status": status or "invalid",
                    "raw_response": _short_text(raw_text, limit=500),
                },
            )
            reason = "meta_controller_done" if status == "DONE" else ""
            self._last_meta_controller_signature = state_signature
            self._last_meta_controller_reason = reason
            return reason
        except Exception as exc:
            self._last_meta_controller_signature = state_signature
            self._last_meta_controller_reason = ""
            self._log(
                "skill_meta_controller",
                {
                    "source": source,
                    "active_skill_id": skill.skill_id,
                    "status": "error",
                    "error": _short_text(exc, limit=500),
                    "raw_response": _short_text(raw_text, limit=500),
                },
            )
            return ""

    def _meta_controller_state_signature(self) -> str:
        if self.active_skill is None:
            return ""
        return _compact_json(
            {
                "active_skill_id": self.active_skill.skill.skill_id,
                "skill_age": self.skill_age,
                "recent_observations": self.recent_observations[-4:],
            },
            limit=2200,
        )

    @staticmethod
    def _parse_meta_controller_status(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        tag = re.search(r"<status>\s*(DONE|CONTINUE)\s*</status>", text, re.I)
        if tag:
            return tag.group(1).upper()
        upper = text.upper()
        if "CONTINUE" in upper and "DONE" not in upper:
            return "CONTINUE"
        if "DONE" in upper and "CONTINUE" not in upper:
            return "DONE"
        return ""

    def _select_active_skill(
        self,
        *,
        query: MemoryQuery,
        source: str,
    ) -> None:
        session_id = str(getattr(self.session, "session_id", "") or "")
        if (
            self.skill_selector_mode == "llm_topk_lcb"
            and self.meta_controller_llm is not None
        ):
            self.active_skill = self.memory.select_skill_llm_topk_lcb(
                query=query,
                llm_agent=self.meta_controller_llm,
                session_id=session_id,
                top_k=max(3, self.top_k),
                nominee_k=self.selector_nominee_k,
                min_lcb=self.selector_min_lcb,
                exclude_skill_ids=self.skill_cooldowns,
                allow_excluded_fallback=True,
            )
        else:
            self.active_skill = self.memory.select_skill(
                query=query,
                session_id=session_id,
                top_k=max(3, self.top_k),
                min_lcb=self.selector_min_lcb,
                exclude_skill_ids=self.skill_cooldowns,
                allow_excluded_fallback=True,
            )
        self.skill_age = 0
        if source == "prompt" and self.active_skill is not None:
            self.prompt_selection_count += 1
        if source == "post_tool" and self.active_skill is not None:
            self.post_tool_selection_count += 1
        self._log(
            "skill_activation",
            {
                "source": source,
                "active_skill_id": self.active_skill.skill.skill_id
                if self.active_skill
                else "",
                "active_skill_score": round(self.active_skill.score, 6)
                if self.active_skill
                else 0.0,
                "skill_age": self.skill_age,
                "cooldown_exclusions": sorted(self.skill_cooldowns),
                "skill_selector_mode": self.skill_selector_mode,
            },
        )
        selected_id = self.active_skill.skill.skill_id if self.active_skill else ""
        self._decay_skill_cooldowns(selected_skill_id=selected_id)

    def _refresh_active_skill_after_observation(
        self,
        *,
        tool_name: str,
        tool_input: Any,
        observation_summary: str,
        status: str,
    ) -> None:
        if self.active_skill is None:
            query = self._query(
                extra_text=(
                    f"post-tool observation from {tool_name} status:{status} "
                    f"input:{_compact_json(tool_input, limit=300)} "
                    f"output:{observation_summary}"
                )
            )
            self._select_active_skill(query=query, source="post_tool")
            return
        query = self._query(
            extra_text=(
                f"post-tool observation from {tool_name} status:{status} "
                f"input:{_compact_json(tool_input, limit=300)} "
                f"output:{observation_summary}"
            )
        )
        termination_reason = self._active_skill_termination_reason(
            query,
            allow_context_mismatch=True,
            source="post_tool",
        )
        if not termination_reason:
            return
        previous_skill_id = self.active_skill.skill.skill_id
        self._cool_down_skill(previous_skill_id)
        self._log(
            "skill_termination",
            {
                "previous_skill_id": previous_skill_id,
                "reason": termination_reason,
                "source": "post_tool",
                "tool": tool_name,
                "status": status,
            },
        )
        self.active_skill = None
        self._select_active_skill(query=query, source="post_tool")

    def _cool_down_skill(self, skill_id: str, *, ttl: int = 1) -> None:
        if not skill_id:
            return
        self.skill_cooldowns[skill_id] = max(
            int(self.skill_cooldowns.get(skill_id, 0)),
            ttl,
        )

    def _decay_skill_cooldowns(self, *, selected_skill_id: str = "") -> None:
        next_cooldowns: dict[str, int] = {}
        for skill_id, ttl in self.skill_cooldowns.items():
            if skill_id == selected_skill_id:
                continue
            remaining = int(ttl) - 1
            if remaining > 0:
                next_cooldowns[skill_id] = remaining
        self.skill_cooldowns = next_cooldowns

    def _merge_active_with_retrieved(
        self,
        retrieved: list[SkillRetrieval],
    ) -> list[SkillRetrieval]:
        if self.active_skill is None:
            return retrieved
        active_id = self.active_skill.skill.skill_id
        merged = [self.active_skill]
        merged.extend(item for item in retrieved if item.skill.skill_id != active_id)
        return merged[: self.top_k]

    def _active_skill_prompt_block(
        self,
        active: SkillRetrieval | None,
        *,
        tool_candidates: list[str] | None = None,
    ) -> str:
        if active is None:
            return ""
        skill = active.skill
        lines = [
            f"Skill: {redact_oracle_markers(skill.skill_id)} ({redact_oracle_markers(skill.title)}) score={active.score:.3f}",
            f"Initiation: {redact_oracle_markers(skill.activation_condition)}",
            "Policy:",
        ]
        lines.extend(
            f"- {redact_oracle_markers(step.action)}"
            for step in skill.execution_steps[:6]
        )
        candidates = (
            list(tool_candidates)
            if tool_candidates is not None
            else sorted(self._contextual_tool_candidates(active))
        )
        if candidates:
            lines.append("Candidate tools: " + ", ".join(candidates[:8]))
        lines.append(
            "Skill termination condition (runtime only): "
            f"{redact_oracle_markers(skill.termination_condition)}"
        )
        return "\n".join(lines)

    def _active_skill_tool_links(
        self,
        active: SkillRetrieval | None,
        *,
        tool_candidates: list[str] | None = None,
    ) -> str:
        if active is None:
            return ""
        candidates = (
            list(tool_candidates)
            if tool_candidates is not None
            else sorted(self._contextual_tool_candidates(active))
        )
        lines: list[str] = []
        for tool_name in candidates[:6]:
            line = f"- {tool_name}: useful when it can advance the active policy or verify termination."
            if self.tool_evolution_runtime is not None:
                next_checks = self.tool_evolution_runtime.next_checks(
                    tool_name,
                    limit=self.tool_evolution_runtime.next_checks_limit,
                    diagnosis_only=True,
                )
                if next_checks:
                    line += " DRAFT checks: " + _compact_items(next_checks)
            lines.append(line)
        return "\n".join(lines)

    def _contextual_tool_candidates(
        self,
        retrieval: SkillRetrieval | None,
    ) -> set[str]:
        explicit = self._skill_tool_candidates(retrieval)
        if explicit:
            return explicit
        return self._fallback_tool_candidates()

    def _retrieved_tool_scope(self, retrieved: list[SkillRetrieval]) -> set[str]:
        explicit_scope: set[str] = set()
        saw_generic_skill = False
        for item in retrieved[: self.top_k]:
            explicit = self._skill_tool_candidates(item)
            if explicit:
                explicit_scope.update(explicit)
            else:
                saw_generic_skill = True
        if explicit_scope:
            return explicit_scope
        if saw_generic_skill:
            return self._fallback_tool_candidates()
        return set()

    def _draft_tool_scope(self) -> set[str]:
        if self.tool_evolution_runtime is None:
            return set()
        known = set(self.tool_names)
        scope = {
            str(item.get("tool_name") or "")
            for item in self.tool_evolution_runtime.planned_explorations(
                limit=self.tool_evolution_runtime.planned_checks * 2,
                diagnosis_only=True,
            )
        }
        return {tool for tool in scope if tool and tool in known}

    def _fallback_tool_candidates(self) -> set[str]:
        known = set(self.tool_names)
        if not known:
            return set()
        if len(known) <= 6:
            return set(self.tool_names)
        text = " ".join(
            [
                self.task_description,
                self.scenario,
                " ".join(self.recent_observations[-3:]),
            ]
        ).lower()
        recent_text = " ".join(self.recent_observations[-3:]).lower()
        candidates: set[str] = set()

        def add(*names: str) -> None:
            candidates.update(name for name in names if name in known)

        add("get_reachability", "ping_pair", "get_host_net_config")
        if any(
            marker in text
            for marker in (
                "pc_",
                "host",
                "interface",
                "link",
                "down",
                "carrier",
                "eth",
            )
        ):
            add("ethtool", "ip_addr_statistics")
        if self._host_link_or_reachability_symptom(recent_text):
            add("ethtool", "ip_addr_statistics")
            return candidates
        if self._bgp_or_route_symptom(recent_text):
            add("frr_show_bgp_summary", "frr_show_ip_route")
            if self._deep_bgp_symptom(recent_text):
                add("frr_get_bgp_conf")
        elif not candidates:
            add("frr_show_bgp_summary", "frr_show_ip_route")
        if "ospf" in text:
            add("frr_get_ospf_conf", "frr_exec")
        if any(
            marker in text
            for marker in ("latency", "loss", "bandwidth", "packet", "throughput")
        ):
            add("get_tc_statistics", "iperf_test")
        if any(marker in text for marker in ("service", "daemon", "frr service")):
            add("systemctl_ops")
        return candidates or set(self.tool_names[:6])

    def _skill_tool_candidates(self, retrieval: SkillRetrieval | None) -> set[str]:
        if retrieval is None:
            return set()
        skill = retrieval.skill
        candidates = {tool for tool in skill.tools if tool}
        candidates.update(step.tool_name for step in skill.execution_steps if step.tool_name)
        known = set(self.tool_names)
        skill_text = " ".join(
            [
                skill.activation_condition,
                " ".join(step.action for step in skill.execution_steps),
                skill.termination_condition,
            ]
        ).lower()
        for tool_name in known:
            if tool_name.lower() in skill_text:
                candidates.add(tool_name)
        return candidates & known if known else candidates

    def _tool_matches_active_skill(
        self,
        retrieval: SkillRetrieval,
        tool_name: str,
    ) -> bool:
        candidates = self._contextual_tool_candidates(retrieval)
        return not candidates or tool_name in candidates

    def _termination_condition_satisfied(self, skill) -> bool:
        if self.skill_age <= 0:
            return False
        condition = str(getattr(skill, "termination_condition", "") or "").lower()
        if any(
            marker in condition
            for marker in (
                "stop after one",
                "after one concrete",
                "after selecting the next",
                "after creating the initial",
                "after choosing exploration or exploitation",
            )
        ):
            return self.skill_age >= 1
        recent_successes = [
            item
            for item in self.recent_transitions[-self.max_skill_age :]
            if item.get("status") == "success" and item.get("tool")
        ]
        unique_tools = {str(item.get("tool")) for item in recent_successes}
        if any(
            marker in condition
            for marker in (
                "two independent",
                "at least two",
                "independent confirmation",
                "independent observations",
            )
        ):
            return len(unique_tools) >= 2
        if "evidence budget" in condition and self.skill_age >= self.max_skill_age:
            return True
        return False

    def _followup_guidance(
        self,
        tool_name: str,
        *,
        draft_exploration: dict[str, Any] | None = None,
    ) -> str:
        lines = [INTEGRATED_GUIDANCE_MARKER]
        if self.active_skill is not None:
            skill = self.active_skill.skill
            lines.append(
                "Active Skill-MDP option: "
                f"{redact_oracle_markers(skill.skill_id)} "
                f"({redact_oracle_markers(skill.title)})."
            )
        if draft_exploration is not None:
            draft_text = (
                f"{draft_exploration.get('exploration_id')}. "
                f"{draft_exploration.get('next_exploration') or ''}"
            ).strip()
            lines.append(f"DRAFT planned exploration advanced: {draft_text}")
        if self.tool_evolution_runtime is not None:
            next_checks = self.tool_evolution_runtime.next_checks(
                tool_name,
                limit=self.tool_evolution_runtime.next_checks_limit,
                diagnosis_only=True,
            )
            if next_checks:
                lines.append("DRAFT next checks: " + _compact_items(next_checks))
        evidence_guidance = self._evidence_guidance_after_tool(tool_name)
        if evidence_guidance:
            lines.append(evidence_guidance)
        if len(lines) == 1:
            return ""
        lines.append(
            "Use current tool output as evidence; use memory/DRAFT only to choose the next check."
        )
        return "\n".join(lines)

    @staticmethod
    def _host_link_or_reachability_symptom(text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            marker in lower
            for marker in (
                '"status":"unknown"',
                '"status": "unknown"',
                "destination host unreachable",
                "100% packet loss",
                "state down",
                "link detected: no",
                "ip_route is empty",
                "flags=4098",
            )
        )

    @staticmethod
    def _bgp_or_route_symptom(text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            marker in lower
            for marker in (
                "neighbor",
                "state/pfxrcd",
                "idle",
                "active",
                "connect",
                "no route",
                "network is unreachable",
                "rib-failure",
            )
        )

    @staticmethod
    def _deep_bgp_symptom(text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            marker in lower
            for marker in (
                "idle",
                "active",
                "connect",
                "establish",
                "configuration",
                "remote-as",
                "as mismatch",
                "neighbor down",
            )
        )

    def _evidence_guidance_after_tool(self, tool_name: str) -> str:
        text = "\n".join(self.recent_observations[-6:])
        lower = text.lower()
        down_hosts = self._host_link_down_devices(text)
        if down_hosts:
            devices = ", ".join(down_hosts)
            return (
                "Current observations are sufficient to support an endpoint "
                f"interface/link-down fault on {devices}. Unless a later "
                "observation directly contradicts this, stop calling diagnostic "
                "tools and write the final diagnosis using the current evidence. "
                "Use the exact root-cause id only after checking the available "
                "submission options."
            )
        if tool_name == "get_reachability" and self._host_link_or_reachability_symptom(
            lower
        ):
            return (
                "Reachability is abnormal or unresolved. Before deeper BGP or "
                "configuration checks, inspect endpoint host/link state with "
                "`get_host_net_config` and `ethtool` for the endpoint devices "
                "named in the failed reachability result."
            )
        if tool_name.startswith("frr_") and self._host_link_or_reachability_symptom(
            lower
        ):
            return (
                "Do not treat route or BGP prefix asymmetry as the root cause "
                "until endpoint host/link state has been checked. Prefer "
                "`get_host_net_config` and `ethtool` for affected endpoint "
                "devices next."
            )
        return ""

    @staticmethod
    def _host_link_down_devices(text: str) -> list[str]:
        return extract_link_down_devices(text)

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._logger is not None:
            self._logger.log(event, payload)


class SkillAwareTool(BaseTool):
    """Tool wrapper that keeps Skill-Pro state online across ReAct tool calls."""

    wrapped_tool: BaseTool = Field(exclude=True)
    runtime: SkillToolRuntime = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, *, wrapped_tool: BaseTool, runtime: SkillToolRuntime) -> None:
        super().__init__(
            name=wrapped_tool.name,
            description=runtime.describe_tool(wrapped_tool),
            args_schema=getattr(wrapped_tool, "args_schema", None),
            return_direct=getattr(wrapped_tool, "return_direct", False),
            response_format=getattr(wrapped_tool, "response_format", "content"),
            wrapped_tool=wrapped_tool,
            runtime=runtime,
        )
        self.handle_tool_error = getattr(wrapped_tool, "handle_tool_error", False)
        self.handle_validation_error = getattr(
            wrapped_tool,
            "handle_validation_error",
            False,
        )
        self.tags = getattr(wrapped_tool, "tags", None)
        self.metadata = getattr(wrapped_tool, "metadata", None)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        tool_input = _tool_input_from_call(args, kwargs)
        self.runtime.before_tool(tool_name=self.name, tool_input=tool_input)
        raw_result: Any = None
        try:
            raw_result = self._invoke_wrapped_tool(tool_input)
        except Exception as exc:
            self.runtime.after_tool(
                tool_name=self.name,
                tool_input=tool_input,
                result=str(exc),
                status="error",
            )
            raise
        result = self.runtime.after_tool(
            tool_name=self.name,
            tool_input=tool_input,
            result=raw_result,
        )
        return self._coerce_response_format(result, raw_result)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        tool_input = _tool_input_from_call(args, kwargs)
        self.runtime.before_tool(tool_name=self.name, tool_input=tool_input)
        raw_result: Any = None
        try:
            raw_result = await self._ainvoke_wrapped_tool(tool_input)
        except Exception as exc:
            self.runtime.after_tool(
                tool_name=self.name,
                tool_input=tool_input,
                result=str(exc),
                status="error",
            )
            raise
        result = self.runtime.after_tool(
            tool_name=self.name,
            tool_input=tool_input,
            result=raw_result,
        )
        return self._coerce_response_format(result, raw_result)

    def _invoke_wrapped_tool(self, tool_input: Any) -> Any:
        if (
            getattr(self.wrapped_tool, "response_format", "content")
            == "content_and_artifact"
        ):
            output = self.wrapped_tool.run(
                tool_input,
                callbacks=[],
                tool_call_id=INTERNAL_TOOL_CALL_ID,
            )
            return self._unwrap_tool_message(output)
        return self.wrapped_tool.invoke(tool_input, config={"callbacks": []})

    async def _ainvoke_wrapped_tool(self, tool_input: Any) -> Any:
        if (
            getattr(self.wrapped_tool, "response_format", "content")
            == "content_and_artifact"
        ):
            output = await self.wrapped_tool.arun(
                tool_input,
                callbacks=[],
                tool_call_id=INTERNAL_TOOL_CALL_ID,
            )
            return self._unwrap_tool_message(output)
        return await self.wrapped_tool.ainvoke(tool_input, config={"callbacks": []})

    @staticmethod
    def _unwrap_tool_message(output: Any) -> Any:
        if isinstance(output, ToolMessage):
            return output.content, output.artifact
        return output

    def _coerce_response_format(self, result: Any, raw_result: Any) -> Any:
        if (
            getattr(self, "response_format", "content") == "content_and_artifact"
            and not (isinstance(result, tuple) and len(result) == 2)
        ):
            return result, raw_result
        return result
