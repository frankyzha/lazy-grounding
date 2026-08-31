"""Conservative answer extraction and normalization."""

from __future__ import annotations

import re
import unicodedata

_ANSWER_BLOCK = re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_MARKDOWN = re.compile(r"[*`#]+")
_WHITESPACE = re.compile(r"\s+")


def extract_final_answer(text: str) -> str:
    """Extract the last explicit answer block, or clean the complete response."""

    blocks = _ANSWER_BLOCK.findall(text or "")
    return clean_short_answer(blocks[-1] if blocks else text)


def clean_short_answer(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", str(text or ""))
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = _MARKDOWN.sub("", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return re.sub(r"^[\s:;,.'\"-]+|[\s:;,.'\"-]+$", "", cleaned).strip()


def canonical_answer(text: str) -> str:
    """Normalize formatting for high-precision deterministic matching."""

    cleaned = clean_short_answer(text).casefold()
    cleaned = re.sub(r"[^\w\s.+%/-]", " ", cleaned, flags=re.UNICODE)
    return _WHITESPACE.sub(" ", cleaned).strip()


def answers_exactly_match(left: str, right: str) -> bool:
    left_normalized = canonical_answer(left)
    right_normalized = canonical_answer(right)
    return bool(left_normalized and left_normalized == right_normalized)
