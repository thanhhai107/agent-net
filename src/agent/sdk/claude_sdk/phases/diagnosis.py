"""Claude Agent SDK diagnosis phase."""

from agent.sdk.claude_sdk.worker import ClaudeSdkWorker
from agent.utils.template import OVERALL_DIAGNOSIS_PROMPT
from agent.utils.phases import DIAGNOSIS


class ClaudeSdkDiagnosisPhase:
    def __init__(
        self,
        session_id: str,
        session_dir: str,
        model: str,
        max_steps: int = 20,
        scenario_name: str = "",
        problem_names: list[str] | None = None,
    ) -> None:
        self._worker = ClaudeSdkWorker(
            session_id=session_id,
            session_dir=session_dir,
            phase=DIAGNOSIS,
            model=model,
            max_steps=max_steps,
            scenario_name=scenario_name,
            problem_names=problem_names,
            system_prompt=OVERALL_DIAGNOSIS_PROMPT,
        )

    async def run(self, task_description: str) -> str:
        prompt = f"{OVERALL_DIAGNOSIS_PROMPT}\n\nTask: {task_description}"
        return await self._worker.run(prompt)
