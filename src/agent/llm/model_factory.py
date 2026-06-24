import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

NETMIND_BASE_URL = "https://stream-netmind.viettel.vn/gateway/v1"
NETMIND_TIMEOUT_SECONDS = 90.0
NETMIND_MAX_RETRIES = 0
NETMIND_SUPPORTED_MODELS = (
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen/Qwen3.5-35B-A3B-FP8",
    "openai/gpt-oss-20b",
    "MiniMax/MiniMax-M2.7",
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


def load_model(llm_backend: str = "openai", model: str = "gpt-5-mini") -> BaseChatModel:
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
        return ChatOpenAI(
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
