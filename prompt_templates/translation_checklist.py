"""Helpers to verify translated subtitle files preserve timestamps and block structure.

This module provides a small, importable function used by unit tests and CI to
ensure a model's output did not alter timestamps or block counts.
"""
import re
from typing import List, Tuple

_ts_re = re.compile(r"^.*-->.*$", flags=re.MULTILINE)


def extract_timestamps(text: str) -> List[str]:
    """Return all timestamp lines in order from the subtitle text."""
    return _ts_re.findall(text)


def split_blocks(text: str) -> List[str]:
    """Split subtitle file into blocks separated by blank lines."""
    return re.split(r"\n\s*\n", text.strip()) if text and text.strip() else []


def check_subtitles_preserve_timestamps(orig_text: str, translated_text: str) -> bool:
    """Return True if timestamps and block counts are preserved.

    - Verifies timestamp lines (those containing '-->') are identical and in the same order.
    - Verifies number of blocks (separated by blank lines) is unchanged.
    """
    orig_ts = extract_timestamps(orig_text)
    trans_ts = extract_timestamps(translated_text)
    if orig_ts != trans_ts:
        return False
    orig_blocks = split_blocks(orig_text)
    trans_blocks = split_blocks(translated_text)
    return len(orig_blocks) == len(trans_blocks)
