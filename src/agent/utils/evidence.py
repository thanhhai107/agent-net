"""Evidence extraction helpers shared by diagnosis runtimes."""

from __future__ import annotations

import re

_DOWN_SIGNAL_RE = re.compile(
    r"state\s+down|link\s+detected:\s*no|flags=4098",
    re.I,
)
_TOOL_OBSERVATION_RE = re.compile(
    r"(?m)(?:^|\n)[A-Za-z_][A-Za-z0-9_]*\((?P<args>.*?)\)\s*->"
)
_DEVICE_FIELD_RE = re.compile(
    r"""["']?(?:host_name|host|router_name|device|node)["']?\s*[:=]\s*["'](?P<device>[A-Za-z][A-Za-z0-9_.-]*)["']""",
    re.I,
)
_INLINE_DOWN_RE = re.compile(
    r"\b(?P<device>[A-Za-z][A-Za-z0-9_.-]*)\b(?:\s+\S+){0,3}\s+"
    r"(?:state\s+down|link\s+detected:\s*no|flags=4098)",
    re.I,
)
_INTERFACE_RE = re.compile(r"^(?:eth|ens|enp|lo|docker|veth|br|bond)\d*(?:\.\d+)?$", re.I)


def _looks_like_device_token(token: str) -> bool:
    normalized = token.lower()
    if _INTERFACE_RE.fullmatch(normalized):
        return False
    if token.isdigit() or len(token) < 2:
        return False
    return any(char.isdigit() or "_" in token or "-" in token for char in token)


def _device_fields(text: str) -> set[str]:
    return {
        match.group("device").lower()
        for match in _DEVICE_FIELD_RE.finditer(text)
        if _looks_like_device_token(match.group("device"))
    }


def _tool_observation_segments(text: str) -> list[tuple[str, str]]:
    matches = list(_TOOL_OBSERVATION_RE.finditer(text))
    segments: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segments.append((match.group("args"), text[match.start() : end]))
    return segments


def extract_link_down_devices(text: str, *, window: int = 900) -> list[str]:
    """Return topology/device identifiers near concrete interface-down evidence."""
    value = str(text or "")
    devices: set[str] = set()
    for args, segment in _tool_observation_segments(value):
        if not _DOWN_SIGNAL_RE.search(segment):
            continue
        segment_devices = _device_fields(args)
        if not segment_devices:
            segment_devices = _device_fields(segment)
        if segment_devices:
            devices.update(segment_devices)
    if devices:
        return sorted(devices)
    for match in _INLINE_DOWN_RE.finditer(value):
        device = match.group("device")
        if _looks_like_device_token(device):
            devices.add(device.lower())
    return sorted(devices)
