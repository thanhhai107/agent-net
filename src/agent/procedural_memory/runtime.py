"""Online Skill-Pro runtime that binds Procedural Memory, prompts, skills, and tools.

The offline Procedural Memory module learns reusable procedures after a session closes.
This runtime is the read-time counterpart: it injects retrieved procedures into
the diagnosis prompt, adds tool-specific skill hints to tool descriptions, and
keeps an active Skill-MDP option across tool calls.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field

from agent.module_config import module_defaults
from agent.procedural_memory.attributes import infer_procedural_memory_attributes
from agent.procedural_memory.models import ProceduralMemoryQuery, SkillRetrieval
from agent.procedural_memory.policy_context import (
    build_skill_policy_prefix,
    build_skill_policy_suffix,
)
from agent.procedural_memory.safety import redact_oracle_markers
from agent.procedural_memory.service import (
    GENERIC_SEED_SKILL_IDS,
    ProceduralMemoryModule,
)
from agent.tool_refinement.runtime import ToolRefinementRuntime
from agent.utils.loggers import MessageLogger
from agent.utils.tool_output import (
    INTEGRATED_GUIDANCE_MARKER,
    classify_tool_outcome,
    tool_output_content,
)

INTERNAL_TOOL_CALL_ID = "skill-runtime-internal"
_DEFAULTS = module_defaults().procedural_memory


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


def _tool_result_content(result: Any) -> Any:
    return tool_output_content(result)


def _append_followup_guidance(result: Any, guidance: str) -> Any:
    if isinstance(result, tuple) and len(result) == 2:
        content, artifact = result
        return (_append_guidance_to_content(content, guidance), artifact)
    return _append_guidance_to_content(result, guidance)


def _append_guidance_to_content(content: Any, guidance: str) -> Any:
    if isinstance(content, list):
        return [*content, {"type": "text", "text": guidance}]
    return f"{content}\n\n{guidance}"


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


class SkillToolRuntime:
    """Runtime controller for online Skill-Pro usage during diagnosis."""

    def __init__(
        self,
        *,
        procedural_memory: ProceduralMemoryModule,
        procedural_memory_mode: str,
        session: Any,
        task_description: str,
        tools: list[BaseTool],
        session_dir: str | Path = "",
        tool_refinement_runtime: ToolRefinementRuntime | None = None,
        top_k: int = _DEFAULTS.top_k,
        token_budget: int = _DEFAULTS.token_budget,
        max_skill_age: int = _DEFAULTS.max_skill_age,
        selection_epsilon: float = _DEFAULTS.selection_epsilon,
        meta_controller_llm: Any | None = None,
    ) -> None:
        self.procedural_memory = procedural_memory
        self.procedural_memory_mode = procedural_memory_mode
        self.session = session
        self.task_description = task_description
        self.tool_names = [tool.name for tool in tools]
        self.tool_descriptions = {
            tool.name: (getattr(tool, "description", "") or "") for tool in tools
        }
        self.tool_refinement_runtime = tool_refinement_runtime
        self.top_k = top_k
        self.token_budget = token_budget
        self.policy_token_budget = min(
            _DEFAULTS.policy_token_budget_max,
            max(
                _DEFAULTS.policy_token_budget_min,
                self.token_budget // _DEFAULTS.policy_token_budget_divisor,
            ),
        )
        self.followup_token_budget = _DEFAULTS.followup_token_budget
        self.max_skill_age = max(1, max_skill_age)
        self.selection_epsilon = max(0.0, min(1.0, selection_epsilon))
        self.selection_count = 0
        self.active_activation_id = ""
        self.meta_controller_llm = meta_controller_llm
        self.active_skill: SkillRetrieval | None = None
        self.skill_age = 0
        self.prompt_selection_count = 0
        self.post_tool_selection_count = 0
        self.meta_controller_cache_hits = 0
        self.skill_cooldowns: dict[str, int] = {}
        self.recent_observations: list[str] = []
        self.recent_transitions: list[dict[str, Any]] = []
        self.inflight_tool_calls = 0
        self._last_meta_controller_signature = ""
        self._last_meta_controller_reason = ""
        self._decision_policy_state = ""
        self._decision_policy_context = ""
        self._lock = Lock()
        self._metrics_lock = Lock()
        self.prompt_added_tokens = 0
        self.tool_description_added_tokens = 0
        self.followup_added_tokens = 0
        self.prompt_injection_count = 0
        self.tool_description_injection_count = 0
        self.followup_guidance_count = 0
        self._last_followup_signature = ""
        self._logger = (
            MessageLogger(
                agent="procedural_memory_agent",
                session_dir=str(session_dir),
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

    def prompt_suffix(
        self,
        *,
        activate_skill: bool = True,
        decision_context: str = "",
    ) -> str:
        active_skill, _ = self._prepare_prompt_context(
            activate_skill=activate_skill,
            decision_context=decision_context,
        )
        if active_skill is None:
            return ""
        self._capture_decision_policy_context(decision_context=decision_context)
        suffix = build_skill_policy_suffix(
            self._decision_policy_state,
            active_skill.skill,
            max_tokens=self.policy_token_budget,
            include_state=False,
        )
        added_tokens = self._record_added_tokens("prompt", suffix)
        self._log(
            "skill_prompt_context",
            {
                "activate_skill": activate_skill,
                "added_tokens": added_tokens,
                "active_skill_id": active_skill.skill.skill_id if active_skill else "",
                "retrieved_skills": [active_skill.skill.skill_id]
                if active_skill
                else [],
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
        tool_guidance = ""
        if self.tool_refinement_runtime is not None:
            tool_guidance = self.tool_refinement_runtime.tool_runtime_guidance(
                tool.name,
                max_chars=min(
                    _DEFAULTS.tool_guidance_char_budget,
                    self.tool_refinement_runtime.tool_doc_chars,
                ),
            )
        if not tool_guidance:
            return description
        guidance = "DRAFT contract notes (not evidence):\n" + tool_guidance
        capped_guidance = _short_text(guidance, limit=480)
        self._record_added_tokens("tool_description", capped_guidance)
        return (description + "\n\n" + capped_guidance).strip()

    def before_tool(self, *, tool_name: str, tool_input: Any) -> dict[str, str]:
        with self._lock:
            snapshot = {
                "active_skill_id": self.active_skill.skill.skill_id
                if self.active_skill
                else "",
                "policy_state": self._decision_policy_state,
                "policy_context": self._decision_policy_context,
                "activation_id": self.active_activation_id,
            }
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
                },
            )
            return snapshot

    def after_tool(
        self,
        *,
        tool_name: str,
        tool_input: Any,
        result: Any,
        status: str = "success",
        decision_snapshot: dict[str, str] | None = None,
    ) -> Any:
        if status != "error":
            status = classify_tool_outcome(result)
        text = _short_text(_tool_result_content(result), limit=1600)
        with self._lock:
            observation = (
                f"{tool_name}({_compact_json(tool_input, limit=300)}) -> {text}"
            )
            self.recent_observations.append(observation)
            self.recent_observations = self.recent_observations[-12:]
            decision_snapshot = decision_snapshot or {}
            active_skill_id = str(decision_snapshot.get("active_skill_id") or "")
            transition = {
                "active_skill_id": active_skill_id,
                "tool": tool_name,
                "tool_input": tool_input,
                "status": status,
                "observation_summary": text,
                "policy_state": str(decision_snapshot.get("policy_state") or ""),
                "policy_context": str(decision_snapshot.get("policy_context") or ""),
                "activation_id": str(decision_snapshot.get("activation_id") or ""),
            }
            self.recent_transitions.append(transition)
            self.recent_transitions = self.recent_transitions[-16:]
            self.inflight_tool_calls = max(0, self.inflight_tool_calls - 1)
            self._log(
                "skill_transition",
                {
                    "active_skill_id": active_skill_id,
                    "tool": tool_name,
                    "tool_input": tool_input,
                    "status": status,
                    "observation_summary": text,
                    "policy_state": transition["policy_state"],
                    "policy_context": transition["policy_context"],
                    "activation_id": transition["activation_id"],
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
                self._capture_decision_policy_context()
                guidance = self._followup_guidance(
                    tool_name,
                )
            else:
                guidance = ""
        if not guidance:
            return result
        capped_guidance = _short_text(
            guidance,
            limit=self.followup_token_budget * 4,
        )
        self._record_added_tokens("followup", capped_guidance)
        return _append_followup_guidance(result, capped_guidance)

    def _capture_decision_policy_context(self, *, decision_context: str = "") -> None:
        state = self.task_description
        if decision_context.strip():
            state += "\nCurrent decision context:\n" + _short_text(
                decision_context,
                limit=4000,
            )
        if self.recent_observations:
            state += "\nRecent observations:\n" + "\n".join(
                self.recent_observations[-4:]
            )
        skill = self.active_skill.skill if self.active_skill else None
        self._decision_policy_state = state
        self._decision_policy_context = build_skill_policy_prefix(state, skill)

    def _tool_refinement_policy_context(self) -> str:
        # DRAFT guidance is attached to the corresponding tool once. Repeating
        # the whole catalog in every policy prompt crowds out live evidence.
        return ""

    def snapshot(self) -> dict[str, Any]:
        attributed_transitions = sum(
            bool(item.get("active_skill_id")) for item in self.recent_transitions
        )
        transition_count = len(self.recent_transitions)
        return {
            "procedural_memory_mode": self.procedural_memory_mode,
            "bank_id": self.procedural_memory.bank_id,
            "active_skill_id": self.active_skill.skill.skill_id
            if self.active_skill
            else "",
            "active_activation_id": self.active_activation_id,
            "skill_age": self.skill_age,
            "prompt_selection_count": self.prompt_selection_count,
            "post_tool_selection_count": self.post_tool_selection_count,
            "meta_controller_cache_hits": self.meta_controller_cache_hits,
            "selection_policy": "epsilon_then_similarity_top_k_online_value",
            "selection_epsilon_initial": self.selection_epsilon,
            "meta_controller_available": self.meta_controller_llm is not None,
            "config": {
                "top_k": self.top_k,
                "token_budget": self.token_budget,
                "max_skill_age": self.max_skill_age,
            },
            "skill_cooldowns": dict(self.skill_cooldowns),
            "tool_names": self.tool_names,
            "recent_observations": self.recent_observations,
            "recent_transitions": self.recent_transitions,
            "attributed_transitions": attributed_transitions,
            "unattributed_transitions": transition_count - attributed_transitions,
            "active_skill_coverage": round(
                attributed_transitions / max(transition_count, 1),
                6,
            ),
            "inflight_tool_calls": self.inflight_tool_calls,
            "prompt_added_tokens": self.prompt_added_tokens,
            "tool_description_added_tokens": self.tool_description_added_tokens,
            "followup_added_tokens": self.followup_added_tokens,
            "total_added_tokens": self.total_added_tokens,
            "prompt_injection_count": self.prompt_injection_count,
            "tool_description_injection_count": (self.tool_description_injection_count),
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

    def _prepare_prompt_context(
        self,
        *,
        activate_skill: bool = True,
        decision_context: str = "",
    ) -> tuple[SkillRetrieval | None, list[SkillRetrieval]]:
        with self._lock:
            query = self._query(
                extra_text=" ".join(
                    item
                    for item in (
                        "decision prompt before next action",
                        _short_text(decision_context, limit=4000),
                    )
                    if item
                )
            )
            if not activate_skill:
                retrieved = self._retrieve(query)
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
                        self.active_skill.skill.skill_id if self.active_skill else ""
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
            retrieved = self._retrieve(query)
            return self.active_skill, self._merge_active_with_retrieved(retrieved)

    def _retrieve(self, query: ProceduralMemoryQuery) -> list[SkillRetrieval]:
        return self.procedural_memory.retrieve(
            query=query,
            session_id="",
            include_probationary=self.procedural_memory_mode == "evolve",
        )

    def _query(
        self,
        *,
        extra_text: str = "",
        tools: list[str] | None = None,
        top_k: int | None = None,
        token_budget: int | None = None,
    ) -> ProceduralMemoryQuery:
        query_tools = list(tools) if tools is not None else self._recent_tool_scope()
        text = " ".join(
            item
            for item in [
                self.task_description,
                extra_text,
                " ".join(self.recent_observations[-3:]),
            ]
            if item
        )
        attrs = infer_procedural_memory_attributes(
            text,
            scenario=self.scenario,
            topology_class=self.topology_class,
            tools=query_tools,
        )
        return ProceduralMemoryQuery(
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

    def _recent_tool_scope(self, *, limit: int = 6) -> list[str]:
        names: list[str] = []
        known_tools = set(self.tool_names)
        for transition in self.recent_transitions[-limit:]:
            name = str(transition.get("tool") or "")
            if name and name in known_tools and name not in names:
                names.append(name)
        return names

    def _active_skill_termination_reason(
        self,
        query: ProceduralMemoryQuery,
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
        for item in self._retrieve(query):
            if item.skill.skill_id == active_id and item.score > 0.05:
                return ""
        return "context_mismatch"

    def _meta_controller_termination_reason(
        self,
        *,
        query: ProceduralMemoryQuery,
        source: str,
    ) -> str:
        if (
            self.meta_controller_llm is None
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
        query: ProceduralMemoryQuery,
        source: str,
    ) -> None:
        session_id = str(getattr(self.session, "session_id", "") or "")
        self.active_skill = self.procedural_memory.select_skill(
            query=query,
            session_id=session_id,
            top_k=max(1, self.top_k),
            record_reuse=self.procedural_memory_mode == "evolve",
            exclude_skill_ids=self.skill_cooldowns,
            allow_excluded_fallback=False,
            exploration_epsilon=(
                self.procedural_memory.decayed_selection_epsilon(self.selection_epsilon)
                if self.procedural_memory_mode == "evolve"
                else 0.0
            ),
            exploration_key=f"{session_id}:{self.selection_count}",
            include_probationary=self.procedural_memory_mode == "evolve",
        )
        self.selection_count += 1
        self.active_activation_id = (
            f"{session_id}:{self.selection_count}"
            if self.active_skill is not None
            else ""
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
                "activation_id": self.active_activation_id,
                "skill_age": self.skill_age,
                "cooldown_exclusions": sorted(self.skill_cooldowns),
                "selection_policy": "epsilon_then_similarity_top_k_online_value",
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

    def _contextual_tool_candidates(
        self,
        retrieval: SkillRetrieval | None,
    ) -> set[str]:
        explicit = self._skill_tool_candidates(retrieval)
        if explicit:
            return explicit
        return self._fallback_tool_candidates()

    def _fallback_tool_candidates(self) -> set[str]:
        known_tools = list(dict.fromkeys(self.tool_names))
        if len(known_tools) <= 6:
            return set(known_tools)
        context = " ".join(
            [
                self.task_description,
                self.scenario,
                " ".join(self.recent_observations[-6:]),
            ]
        ).lower()
        context_tokens = {
            token for token in re.findall(r"[a-z0-9]+", context) if len(token) >= 3
        }
        recent_tools = {
            str(transition.get("tool") or "")
            for transition in self.recent_transitions[-6:]
        }
        ranked: list[tuple[float, int, str]] = []
        for index, tool_name in enumerate(known_tools):
            description = strip_integrated_learning_guidance(
                self.tool_descriptions.get(tool_name, "")
            ).lower()
            tool_tokens = {
                token
                for token in re.findall(
                    r"[a-z0-9]+",
                    f"{tool_name.replace('_', ' ')} {description}",
                )
                if len(token) >= 3
            }
            overlap = context_tokens & tool_tokens
            score = float(len(overlap))
            if tool_name.lower() in context:
                score += 3.0
            if tool_name in recent_tools:
                score += 2.0
            ranked.append((score, -index, tool_name))
        positive = [item for item in ranked if item[0] > 0]
        selected = sorted(positive or ranked, reverse=True)[:6]
        return {tool_name for _, _, tool_name in selected}

    def _skill_tool_candidates(self, retrieval: SkillRetrieval | None) -> set[str]:
        if retrieval is None:
            return set()
        skill = retrieval.skill
        candidates = {tool for tool in skill.tools if tool}
        candidates.update(
            step.tool_name for step in skill.execution_steps if step.tool_name
        )
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
    ) -> str:
        del tool_name
        if self.active_skill is None:
            return ""
        skill = self.active_skill.skill
        base_skill_id = skill.skill_id.split("__", 1)[0]
        if base_skill_id in GENERIC_SEED_SKILL_IDS:
            return ""
        next_index = (
            min(self.skill_age, len(skill.execution_steps) - 1)
            if skill.execution_steps
            else -1
        )
        signature = f"{skill.skill_id}:{next_index}"
        if signature == self._last_followup_signature:
            return ""
        self._last_followup_signature = signature
        lines = [
            INTEGRATED_GUIDANCE_MARKER,
            "Active Skill-MDP option: "
            f"{redact_oracle_markers(skill.skill_id)} "
            f"({redact_oracle_markers(skill.title)}).",
        ]
        if skill.execution_steps:
            next_step = skill.execution_steps[next_index].action
            lines.append("Next active policy step: " + redact_oracle_markers(next_step))
        lines.append(
            "Use the current tool output as evidence; the skill and DRAFT contract notes are guidance only."
        )
        return "\n".join(lines)

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        if self._logger is not None:
            self._logger.log(event, {"phase": "skill_mdp_runtime", **payload})


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
        decision_snapshot = self.runtime.before_tool(
            tool_name=self.name, tool_input=tool_input
        )
        raw_result: Any = None
        try:
            raw_result = self._invoke_wrapped_tool(tool_input)
        except Exception as exc:
            self.runtime.after_tool(
                tool_name=self.name,
                tool_input=tool_input,
                result=str(exc),
                status="error",
                decision_snapshot=decision_snapshot,
            )
            raise
        result = self.runtime.after_tool(
            tool_name=self.name,
            tool_input=tool_input,
            result=raw_result,
            decision_snapshot=decision_snapshot,
        )
        return self._coerce_response_format(result, raw_result)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        tool_input = _tool_input_from_call(args, kwargs)
        decision_snapshot = self.runtime.before_tool(
            tool_name=self.name, tool_input=tool_input
        )
        raw_result: Any = None
        try:
            raw_result = await self._ainvoke_wrapped_tool(tool_input)
        except Exception as exc:
            self.runtime.after_tool(
                tool_name=self.name,
                tool_input=tool_input,
                result=str(exc),
                status="error",
                decision_snapshot=decision_snapshot,
            )
            raise
        result = self.runtime.after_tool(
            tool_name=self.name,
            tool_input=tool_input,
            result=raw_result,
            decision_snapshot=decision_snapshot,
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
        if getattr(
            self, "response_format", "content"
        ) == "content_and_artifact" and not (
            isinstance(result, tuple) and len(result) == 2
        ):
            return result, raw_result
        return result
