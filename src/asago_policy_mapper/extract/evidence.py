"""Verification of evidence quotes against the source text they came from.

Grounding sends each chunk's built context (the chunk plus same-document
neighbours) to the model, so a returned quote can live in the core chunk or
in one of those neighbours. These helpers decide, for one quote, whether it
appears in a given set of source windows: exactly (``verified``), only after
normalization (``normalized``), or not at all (``not_found``). No fuzzy
matching.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, Sequence

VerificationStatus = Literal["verified", "normalized", "not_found"]

_CURLEY_TO_STRAIGHT = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "—": "-",
    "–": "-",
    "−": "-",
}


def normalize(text: str) -> str:
    """Normalize text for quote comparison.

    Applies NFKC compatibility normalization, joins words hyphenated
    across a line break ("exam-\\nple" becomes "example"; soft hyphens
    and U+2010/U+2011 wrap the same way; em/en dashes do not join),
    deletes stray soft hyphens, straightens curly quotes, guillemets and
    dashes, collapses all whitespace runs to single spaces and lowercases.
    """
    text = unicodedata.normalize("NFKC", text)
    # A soft hyphen directly before a line break is a wrap: delete it with
    # the break so the word halves join ("produc\u00ad\ntivity" becomes
    # "productivity"). Any other soft hyphen is a stray typing mark: it
    # deletes, leaving the word whole.
    text = re.sub(r"\u00ad\r?\n", "", text)
    text = text.replace("\u00ad", "")
    # A hyphen (or U+2010, which NFKC does not fold) directly before a line
    # break is a typographic wrap, so the two word halves join. This runs
    # before the dash mappings: an em/en dash before a break is punctuation,
    # not hyphenation, and must survive as a dash.
    text = re.sub(r"[\-\u2010]\r?\n", "", text)
    for curly, straight in _CURLEY_TO_STRAIGHT.items():
        text = text.replace(curly, straight)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def locate_quote(
    quote: str,
    windows: Sequence[tuple[int, str]],
) -> tuple[VerificationStatus, int | None]:
    """Locate a quote within (chunk_index, text) windows.

    Tries an exact substring match first, scanning windows in the order
    given; the first window containing the quote wins. If no exact match
    exists, retries with normalize() applied to both sides. Returns
    ("not_found", None) when the quote appears in no window.
    """
    for index, text in windows:
        if quote in text:
            return "verified", index

    normalized_quote = normalize(quote)
    if not normalized_quote:
        return "not_found", None
    for index, text in windows:
        if normalized_quote in normalize(text):
            return "normalized", index

    return "not_found", None
