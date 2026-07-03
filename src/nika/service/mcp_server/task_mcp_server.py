import json
import os
from difflib import get_close_matches
from typing import List

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from nika.orchestrator.problems.prob_pool import list_avail_problem_names as _list_avail_problems
from nika.service.mcp_server.mcp_session_context import get_session_dir
from nika.utils.errors import safe_tool

# Initialize FastMCP server
mcp = FastMCP(
    "task_mcp_server",
    instructions="This mcp server contains the apis to interact with tasks, for now using to submit your solution.",
)


class SubmissionFormat(BaseModel):
    is_anomaly: bool = Field(..., description="Indicates whether an anomaly was detected.")
    faulty_devices: List[str] = Field(
        ...,
        description=(
            "List of localized devices that are identified as faulty. "
            "Each item is a device name (string). "
            "Example: ['router_1', 'switch_2']"
        ),
    )
    root_cause_name: List[str] = Field(
        ...,
        description=(
            "The name(s) of the identified root cause(s) of the network anomaly. "
            "MUST be from the provided list of root cause names. "
            "Get the names from the 'list_avail_problems()' tool."
        ),
    )


@safe_tool
@mcp.tool()
def list_avail_problems() -> list[str]:
    """List all available root cause types.

    Returns:
        list[str]: A list of available root cause types.
    """
    return _list_avail_problems()


def _clean_list(values: List[str], field_name: str) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            raise ValueError(f"{field_name} cannot contain empty values.")
        if item not in cleaned:
            cleaned.append(item)
    return cleaned


def _validate_submission(
    *,
    is_anomaly: bool,
    faulty_devices: List[str],
    root_cause_name: List[str],
) -> SubmissionFormat:
    parsed = SubmissionFormat(
        is_anomaly=is_anomaly,
        faulty_devices=_clean_list(faulty_devices, "faulty_devices"),
        root_cause_name=_clean_list(root_cause_name, "root_cause_name"),
    )
    if not parsed.is_anomaly and (parsed.faulty_devices or parsed.root_cause_name):
        raise ValueError(
            "is_anomaly=False requires empty faulty_devices and root_cause_name."
        )
    if parsed.is_anomaly and not (parsed.faulty_devices or parsed.root_cause_name):
        raise ValueError(
            "is_anomaly=True requires at least one supported faulty_device or root_cause_name."
        )

    available = set(_list_avail_problems())
    invalid = [name for name in parsed.root_cause_name if name not in available]
    if invalid:
        suggestions = {
            name: get_close_matches(name, sorted(available), n=3, cutoff=0.6)
            for name in invalid
        }
        raise ValueError(
            "root_cause_name must be selected exactly from list_avail_problems(); "
            f"invalid={invalid}; suggestions={suggestions}"
        )
    return parsed


@safe_tool
@mcp.tool()
def submit(
    is_anomaly: bool,
    faulty_devices: List[str],
    root_cause_name: List[str],
) -> List[str]:
    """
    Submit a task solution.

    Args:
        is_anomaly: Indicates whether an anomaly was detected.
        faulty_devices: List of localized devices that are identified as faulty.
        root_cause_name: The name(s) of the identified root cause(s) of the network anomaly. MUST be selected from the result of 'list_avail_problems' tool.
    """
    parsed = _validate_submission(
        is_anomaly=is_anomaly,
        faulty_devices=faulty_devices,
        root_cause_name=root_cause_name,
    )
    submission_dict = {
        "is_anomaly": parsed.is_anomaly,
        "faulty_devices": parsed.faulty_devices,
        "root_cause_name": parsed.root_cause_name,
    }
    session_dir = get_session_dir()
    os.makedirs(session_dir, exist_ok=True)
    submission_path = os.path.join(session_dir, "submission.json")
    with open(submission_path, "w+", encoding="utf-8") as log_file:
        log_file.write(json.dumps(submission_dict))

    return ["Submission success."]


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
