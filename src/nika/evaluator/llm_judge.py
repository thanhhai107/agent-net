import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# from agent.llm.langchain_deepseek import DeepSeekLLM
from agent.llm.model_factory import load_model
from agent.module_config import module_defaults
from agent.utils.template import LLM_JUDGE_PROMPT_TEMPLATE

load_dotenv()


class Score(BaseModel):
    score: int = Field(..., ge=1, le=5, description="Score from 1 to 5.")
    comment: str = Field(
        ..., description="Comment explaining the rationale for the score."
    )


class Scores(BaseModel):
    relevance: Score = Field(
        ..., description="How relevant the agent's actions were to the problem."
    )
    correctness: Score = Field(
        ..., description="How correct the tools/commands and actions were."
    )
    efficiency: Score = Field(
        ..., description="How efficient and well-ordered the agent’s actions were."
    )
    clarity: Score = Field(
        ..., description="How clear and well-explained the agent’s reasoning was."
    )
    final_outcome: Score = Field(
        ...,
        description="Whether the final outcome existed and matched the ground truth.",
    )
    overall_score: Score = Field(
        ..., description="Overall final score summarizing the total performance."
    )


class JudgeResponse(BaseModel):
    scores: Scores = Field(
        ..., description="Per-criterion scores and evaluator comments."
    )
    overall_evaluation: str = Field(
        ..., description="High-level summary of strengths and weaknesses."
    )
    reasoning_for_overall_score: str = Field(
        ..., description="Explanation of why this overall score was given."
    )


class LLMJudge:
    def __init__(
        self,
        judge_llm_provider: str | None = None,
        judge_model: str | None = None,
    ):
        defaults = module_defaults().baseline
        judge_llm_provider = judge_llm_provider or defaults.judge_provider
        judge_model = judge_model or defaults.judge_model
        self.llm = load_model(llm_provider=judge_llm_provider, model=judge_model)
        self.llm = self.llm.with_structured_output(JudgeResponse)
        self.prompt = LLM_JUDGE_PROMPT_TEMPLATE

    def _parse_trace(self, trace: str) -> str:
        """Parse the agent's action history trace.
        1. Remove generation info and usage metadata.

        Args:
            trace: The raw trace string.

        Returns:
            str: The parsed trace.
        """
        new_trace = []
        for line in trace.splitlines():
            line = json.loads(line)
            if "event" in line:
                if line["event"] == "llm_start":
                    payload = line.get("prompts", "")
                    new_trace.append(
                        {
                            "timestamp": line.get("timestamp", ""),
                            "event": "LLM Prompt",
                            "payload": payload,
                        }
                    )
                elif line["event"] == "llm_end":
                    payload = line.get("text", "")
                    new_trace.append(
                        {
                            "timestamp": line.get("timestamp", ""),
                            "event": "LLM Response",
                            "payload": payload,
                        }
                    )
                else:
                    new_trace.append(line)
        return json.dumps(new_trace, ensure_ascii=False)

    def evaluate_agent(self, ground_truth: str, trace_path: str, save_path: str) -> str:
        """Evaluate the agent's performance based on the problem description, network environment info, and action history.

        Args:
            problem_description: Description of the problem.
            net_env_info: Information about the network environment.
            trace_path: Path to the file containing the agent's action history.
            save_path: Path to save the evaluation result.

        Returns:
            str: The evaluation result from the judge model.
            int: The score extracted from the evaluation result.
        """
        with open(trace_path, "r") as f:
            trace = f.read()
        trace = self._parse_trace(trace)

        self.prompt = self.prompt.format(
            ground_truth=ground_truth,
            trace=trace,
        )
        evaluation: JudgeResponse = self.llm.invoke(self.prompt)

        # Save evaluation result to file
        with open(save_path, "w+") as f:
            f.write(evaluation.model_dump_json(indent=2))

        return evaluation
