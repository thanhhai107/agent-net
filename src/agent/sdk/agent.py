"""Placeholder for an SDK-backed troubleshooting agent.

Select with ``nika agent run -a sdk`` once implemented.
"""

from typing import Any


class SdkAgent:
    """Two-phase troubleshooting agent backed by a vendor SDK."""

    def __init__(
        self,
        session_id: str,
        max_steps: int,
        llm_provider: str = "",
        model: str = "",
    ) -> None:
        self.session_id = session_id
        self.llm_provider = llm_provider
        self.model = model
        self.max_steps = max_steps

    async def run(self, task_description: str) -> dict[str, Any]:
        raise NotImplementedError(
            "SdkAgent is not implemented yet. "
            "Implement agent.sdk.agent.SdkAgent and register it in agent.registry."
        )
