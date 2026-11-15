"""Deterministic normalization rules for Sanskrit text."""

import re
import unicodedata


_HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[^\S\r\n]+")


def normalize_text(text):
    """Apply Unicode NFC and standardize whitespace in a text value.

    Horizontal whitespace runs become one ordinary space. Line endings are
    standardized to ``\n`` and kept as line breaks so verse and paragraph
    boundaries are not flattened. Punctuation and non-string values are
    preserved.
    """
    if not isinstance(text, str):
        return text

    normalized_text = unicodedata.normalize("NFC", text)
    normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n")

    lines = normalized_text.split("\n")
    normalized_lines = []
    for line in lines:
        line = _HORIZONTAL_WHITESPACE_PATTERN.sub(" ", line)
        normalized_lines.append(line.strip())

    normalized_text = "\n".join(normalized_lines)
    return normalized_text.strip("\n")
