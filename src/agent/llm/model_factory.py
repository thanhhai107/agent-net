import os
import hashlib
import json
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

NETMIND_BASE_URL = "https://stream-netmind.viettel.vn/gateway/v1"
NETMIND_TIMEOUT_SECONDS = 90.0
NETMIND_MAX_RETRIES = 0
DEFAULT_LLM_BACKEND = "netmind"
DEFAULT_MODEL = "openai/gpt-oss-120b"
NETMIND_SUPPORTED_MODELS = (
    "MiniMax/MiniMax-M2.7",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "zai-org/GLM-4.7",
)

_GLM_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)
_GLM_ARG_PATTERN = re.compile(
    r"<arg_key>\s*(?P<key>.*?)\s*</arg_key>\s*"
    r"<arg_value>\s*(?P<value>.*?)\s*</arg_value>",
    re.DOTALL,
)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def _env_non_negative_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")
    return parsed


def _coerce_tool_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("tool call arguments must be a JSON object")


def _coerce_glm_xml_arg_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _make_glm_tool_call(
    name: str,
    args: dict[str, Any],
    call_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call payload must include a tool name")
    seed = json.dumps(
        {"name": name, "args": args},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    call_id = str(call_id or "")
    if not call_id:
        call_id = "call_glm_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return {
        "name": name.strip(),
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def _parse_glm_json_tool_call(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("tool call payload must be a JSON object")
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name") or payload.get("name") or payload.get("tool_name")
        raw_args = function.get(
            "arguments",
            payload.get("arguments", payload.get("args", payload.get("parameters"))),
        )
    else:
        name = payload.get("name") or payload.get("tool_name")
        raw_args = payload.get(
            "arguments",
            payload.get("args", payload.get("parameters")),
        )
    args = _coerce_tool_args(raw_args)
    return _make_glm_tool_call(name, args, payload.get("id"))


def _parse_glm_xml_tool_call(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ValueError("tool call payload must include a tool name")
    args: dict[str, Any] = {}
    arg_start = text.find("<arg_key>")
    if arg_start == -1:
        name = text
    else:
        name = text[:arg_start].strip()
        for match in _GLM_ARG_PATTERN.finditer(text[arg_start:]):
            key = match.group("key").strip()
            if not key:
                raise ValueError("tool call argument key must be non-empty")
            args[key] = _coerce_glm_xml_arg_value(match.group("value"))
        if not args:
            raise ValueError("tool call XML arguments are malformed")
    if "<" in name or ">" in name:
        raise ValueError("tool call XML payload has malformed tool name")
    return _make_glm_tool_call(name, args)


def _parse_glm_tool_call(raw: str) -> dict[str, Any]:
    try:
        return _parse_glm_json_tool_call(raw)
    except json.JSONDecodeError:
        return _parse_glm_xml_tool_call(raw)


def _extract_glm_tool_calls(content: Any) -> tuple[list[dict[str, Any]], str] | None:
    if not isinstance(content, str):
        return None
    matches = list(_GLM_TOOL_CALL_PATTERN.finditer(content))
    if not matches:
        return None
    calls: list[dict[str, Any]] = []
    for match in matches:
        try:
            calls.append(_parse_glm_tool_call(match.group(1)))
        except ValueError:
            return None
    cleaned = _GLM_TOOL_CALL_PATTERN.sub("", content).strip()
    return calls, cleaned


def _normalize_glm_tool_calls(result: ChatResult) -> ChatResult:
    for generation in result.generations:
        message = generation.message
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        extracted = _extract_glm_tool_calls(message.content)
        if extracted is None:
            continue
        tool_calls, cleaned_content = extracted
        generation.message = message.model_copy(
            update={
                "content": cleaned_content,
                "tool_calls": tool_calls,
                "invalid_tool_calls": [],
            }
        )
    return result


class GLM47ChatOpenAI(ChatOpenAI):
    """NetMind GLM-4.7 adapter for text-formatted tool calls.

    Some OpenAI-compatible GLM deployments emit tool calls as
    ``<tool_call>{...}</tool_call>`` in assistant content instead of using the
    OpenAI ``tool_calls`` field. LangChain agents only execute tools when the
    latter is populated, so normalize that response shape here.
    """

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return _normalize_glm_tool_calls(super()._generate(*args, **kwargs))

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return _normalize_glm_tool_calls(await super()._agenerate(*args, **kwargs))


def load_model(
    llm_backend: str = DEFAULT_LLM_BACKEND,
    model: str = DEFAULT_MODEL,
) -> BaseChatModel:
    if llm_backend == "ollama":
        return ChatOllama(
            model=model,
            temperature=0,
            validate_model_on_init=True,
            base_url=os.getenv("OLLAMA_API_URL"),
        )

    if llm_backend == "openai":
        return ChatOpenAI(
            model_name=model,
        )

    if llm_backend == "deepseek":
        return ChatDeepSeek(
            model=model,
            base_url="https://api.deepseek.com",
        )

    if llm_backend == "netmind":
        api_key = os.getenv("NETMIND_API_KEY")
        if not api_key:
            raise ValueError(
                "NETMIND_API_KEY is required when llm_backend is 'netmind'"
            )
        if model not in NETMIND_SUPPORTED_MODELS:
            supported = ", ".join(NETMIND_SUPPORTED_MODELS)
            raise ValueError(
                f"Unsupported NetMind model: {model!r}. "
                f"Supported models: {supported}"
            )
        chat_model = GLM47ChatOpenAI if model == "zai-org/GLM-4.7" else ChatOpenAI
        return chat_model(
            model=model,
            api_key=api_key,
            base_url=os.getenv("NETMIND_BASE_URL", NETMIND_BASE_URL),
            temperature=0,
            timeout=_env_float(
                "NETMIND_TIMEOUT_SECONDS",
                NETMIND_TIMEOUT_SECONDS,
            ),
            max_retries=_env_non_negative_int(
                "NETMIND_MAX_RETRIES",
                NETMIND_MAX_RETRIES,
            ),
        )

    raise ValueError(f"Unsupported llm backend: {llm_backend}")
