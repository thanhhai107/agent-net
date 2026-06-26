"""Composable adapter that augments an existing troubleshooting workflow."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from agent.memory.models import MemoryQuery
from agent.memory.service import HybridMemoryModule
from agent.protocols import TroubleshootingAgent
from agent.utils.loggers import MessageLogger

_PROTOCOL_NAMES = (
    "bgp",
    "ospf",
    "rip",
    "dhcp",
    "dns",
    "arp",
    "icmp",
    "http",
    "mpls",
    "p4",
    "bmv2",
    "vpn",
)


class MemoryAugmentedAgent:
    """Add persistent memory to a workflow without changing its reasoning graph."""

    def __init__(
        self,
        agent: TroubleshootingAgent,
        memory: HybridMemoryModule,
        *,
        memory_mode: str,
        memory_top_k: int = 5,
        memory_token_budget: int = 1500,
    ) -> None:
        if memory_mode not in {"read", "evolve"}:
            raise ValueError("memory_mode must be read or evolve for the adapter")
        self.agent = agent
        self.memory = memory
        self.memory_mode = memory_mode
        self.memory_top_k = memory_top_k
        self.memory_token_budget = memory_token_budget
        self.session_id = agent.session_id

        session = getattr(agent, "session", None)
        if session is None:
            raise ValueError("memory-enabled workflows must expose their session")
        self.session = session
        self.session_dir = str(session.session_dir)

    def _protocols(self, task_description: str) -> list[str]:
        haystack = f"{self.session.scenario_name} {task_description}".lower()
        tokens = set(re.findall(r"[a-z0-9]+", haystack))
        return [name for name in _PROTOCOL_NAMES if name in tokens]

    async def run(self, task_description: str) -> dict[str, Any]:
        query = MemoryQuery(
            text=task_description,
            scenario=str(self.session.scenario_name),
            topology_class=str(self.session.scenario_topo_size or ""),
            protocols=self._protocols(task_description),
            task_stage="diagnosis",
            tools=list(getattr(self.agent, "diagnosis_tool_names", [])),
            top_k=self.memory_top_k,
            token_budget=self.memory_token_budget,
        )
        retrieved = await asyncio.to_thread(
            self.memory.retrieve,
            query=query,
            session_id=self.session_id,
        )
        MessageLogger(
            agent="memory_agent",
            session_dir=self.session_dir,
            extra_fields={"phase": "retrieval"},
        ).log(
            "memory_retrieval",
            {
                "bank_id": self.memory.bank_id,
                "memory_mode": self.memory_mode,
                "workflow": type(self.agent).__name__,
                "memory_ids": [item.memory.memory_id for item in retrieved],
                "scores": [round(item.score, 6) for item in retrieved],
            },
        )
        context = self.memory.format_context(retrieved)
        augmented_task = (
            f"{task_description}\n\n{context}" if context else task_description
        )
        return await self.agent.run(augmented_task)
