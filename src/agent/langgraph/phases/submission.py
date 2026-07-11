from dotenv import load_dotenv
import ast
import json
from typing import Any, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.llm.model_factory import DEFAULT_LLM_BACKEND, DEFAULT_MODEL, load_model
from agent.langgraph.evidence import ToolObservation
from agent.langgraph.submission_validation import (
    DiagnosisDraft,
    SubmissionVerification,
    ValidatedSubmission,
    draft_claims,
    evidence_records,
    submission_draft_prompt,
    validated_submission,
    verification_prompt,
)
from agent.utils.loggers import AgentCallbackLogger
from agent.utils.mcp_servers import MCPServerConfig
from agent.utils.phases import SUBMISSION

load_dotenv()


class SubmissionPhase:
    """Draft, verify, and commit one evidence-bound submission."""

    def __init__(
        self,
        session_id: str,
        llm_backend: str = DEFAULT_LLM_BACKEND,
        model: str = DEFAULT_MODEL,
    ):
        mcp_server_config = MCPServerConfig(session_id=session_id).load_config(if_submit=True)
        self.client = MultiServerMCPClient(connections=mcp_server_config)
        self.tools: list[BaseTool] = []
        self.llm = load_model(llm_backend=llm_backend, model=model)
        self.drafter = self.llm.with_structured_output(DiagnosisDraft)
        self.verifier = self.llm.with_structured_output(SubmissionVerification)

    async def load_tools(self):
        self.tools = await self.client.get_tools()
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True

    def _tool(self, name: str) -> BaseTool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise RuntimeError(f"Submission MCP tool is unavailable: {name}")

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if hasattr(value, "content"):
            value = value.content
        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                if len(value) == 1:
                    return SubmissionPhase._string_list(value[0])
                return [item.strip() for item in value if item.strip()]
            texts = [
                item.get("text", "")
                for item in value
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            if texts:
                return SubmissionPhase._string_list("\n".join(texts))
        if isinstance(value, str):
            text = value.strip()
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
                if parsed is not value:
                    return SubmissionPhase._string_list(parsed)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) > 1:
                return list(dict.fromkeys(lines))
            return [text] if text else []
        return []

    @staticmethod
    def _tool_error(value: Any) -> str:
        text = str(getattr(value, "content", value) or "").strip()
        lowered = text.lower()
        if lowered.startswith(("error", "toolerror", "validationerror")):
            return text[:1_000]
        return ""

    async def submit_report(
        self,
        *,
        task_description: str,
        diagnosis_report: str,
        observations: Sequence[ToolObservation],
        session_dir: str,
    ) -> dict[str, Any] | None:
        callback = AgentCallbackLogger(agent=SUBMISSION, session_dir=session_dir)
        records = evidence_records(observations)
        if not records:
            args = ValidatedSubmission(False, (), ()).to_tool_args()
            result = await self._tool("submit").ainvoke(args)
            if error := self._tool_error(result):
                raise RuntimeError(f"submit failed: {error}")
            callback._log(
                "submission_committed",
                {
                    "submission": args,
                    "unsupported_claims": [],
                    "evidence_count": 0,
                    "fallback_reason": "no_current_tool_evidence",
                },
            )
            return {
                "submission": args,
                "messages": [
                    AIMessage(content=f"Submission committed: {json.dumps(args)}")
                ],
                "tool_result": result,
            }

        available_raw = await self._tool("list_avail_problems").ainvoke({})
        if error := self._tool_error(available_raw):
            raise RuntimeError(f"list_avail_problems failed: {error}")
        available = self._string_list(available_raw)
        if not available:
            raise RuntimeError("list_avail_problems returned no valid root-cause ids")

        draft_raw = await self.drafter.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You draft evidence-bound network diagnosis submissions. "
                        "Follow the supplied evidence contract exactly."
                    )
                ),
                HumanMessage(
                    content=submission_draft_prompt(
                        task_description=task_description,
                        diagnosis_report=diagnosis_report,
                        records=records,
                        available_root_causes=available,
                    )
                ),
            ],
            config={"callbacks": [callback]},
        )
        draft = DiagnosisDraft.model_validate(draft_raw)
        claims = draft_claims(
            draft,
            available_root_causes=available,
            valid_evidence_ids={record.evidence_id for record in records},
        )
        callback._log(
            "submission_drafted",
            {
                "draft": draft.model_dump(),
                "structurally_valid_claims": len(claims),
                "evidence_count": len(records),
            },
        )
        if claims:
            verification_raw = await self.verifier.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "You are a conservative evidence verifier. Evaluate claims "
                            "independently from the diagnosis author."
                        )
                    ),
                    HumanMessage(
                        content=verification_prompt(records=records, claims=claims)
                    ),
                ],
                config={"callbacks": [callback]},
            )
            verification = SubmissionVerification.model_validate(verification_raw)
        else:
            verification = SubmissionVerification()
        submission = validated_submission(draft, claims, verification)
        callback._log(
            "submission_verified",
            {
                "verdicts": [item.model_dump() for item in verification.verdicts],
                "will_commit": submission is not None,
            },
        )
        args = submission.to_tool_args()
        result = await self._tool("submit").ainvoke(args)
        if error := self._tool_error(result):
            raise RuntimeError(f"submit failed: {error}")
        callback._log(
            "submission_committed",
            {
                "submission": args,
                "unsupported_claims": list(submission.unsupported_claims),
                "evidence_count": len(records),
            },
        )
        return {
            "submission": args,
            "messages": [AIMessage(content=f"Submission committed: {json.dumps(args)}")],
            "tool_result": result,
        }
