"""OpenAI Codex SDK diagnosis phase."""

from agent.sdk.codex_sdk.worker import CodexSdkWorker
from agent.utils.template import OVERALL_DIAGNOSIS_PROMPT
from agent.utils.phases import DIAGNOSIS


class CodexSdkDiagnosisPhase:
    def __init__(
        self,
        session_id: str,
        session_dir: str,
        model: str = "gpt-5.4-mini",
        reasoning_effort: str | None = None,
        scenario_name: str = "",
        problem_names: list[str] | None = None,
        *,
        stream_output: bool = True,
    ) -> None:
        self._worker = CodexSdkWorker(
            session_id=session_id,
            session_dir=session_dir,
            phase=DIAGNOSIS,
            model=model,
            reasoning_effort=reasoning_effort,
            scenario_name=scenario_name,
            problem_names=problem_names,
            system_prompt=OVERALL_DIAGNOSIS_PROMPT,
            stream_output=stream_output,
        )

    async def run(self, task_description: str) -> str:
        prompt = f"{OVERALL_DIAGNOSIS_PROMPT}\n\nTask: {task_description}"
        return await self._worker.run(prompt)
