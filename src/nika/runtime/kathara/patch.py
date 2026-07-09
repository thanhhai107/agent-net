"""Runtime fixes for the Kathara dependency."""

from __future__ import annotations

from typing import Optional


def patch_kathara_file_conversion() -> None:
    """Patch Kathara's text conversion helper to close file handles."""
    from Kathara import utils

    if getattr(utils.convert_win_2_linux, "_nika_closes_files", False):
        return

    def convert_win_2_linux(filename: str, write: bool = False) -> Optional[bytes]:
        if not utils.is_binary(filename):
            try:
                with open(filename, mode="r", encoding="utf-8-sig") as file_obj:
                    file_content = (
                        file_obj.read().replace("\n\r", "\n").replace("\r\n", "\n")
                    )
                if not write:
                    return file_content.encode("utf-8")
                with open(
                    filename, mode="w", encoding="utf-8", newline="\n"
                ) as file_obj_write:
                    file_obj_write.write(file_content)
                return None
            except Exception:
                pass

        if not write:
            with open(filename, mode="rb") as file_obj:
                return file_obj.read()
        return None

    convert_win_2_linux._nika_closes_files = True  # type: ignore[attr-defined]
    utils.convert_win_2_linux = convert_win_2_linux


patch_kathara_file_conversion()
