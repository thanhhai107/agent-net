"""Render Containerlab topology templates with a dynamic lab name."""

from __future__ import annotations

from pathlib import Path

_LAB_NAME_PLACEHOLDER = "__LAB_NAME__"


def render_topology(
    template_path: Path,
    *,
    lab_name: str,
    output_path: Path,
    replacements: dict[str, str] | None = None,
) -> Path:
    """Render ``template_path`` into ``output_path``, substituting lab name placeholders."""
    content = template_path.read_text(encoding="utf-8")
    content = content.replace(_LAB_NAME_PLACEHOLDER, lab_name)
    for key, value in (replacements or {}).items():
        content = content.replace(key, value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
