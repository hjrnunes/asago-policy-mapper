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

# Characters that carry no width: deleting them lets quotes match text whose
# extraction scattered invisible marks through words.
_ZERO_WIDTH = "\u00ad\u200b\u200c\u200d\ufeff"

# Punctuation straightened for the normalized tier. U+2011 never survives
# NFKC (it folds to U+2010), but keeping the pair documents the intent.
_CURLY_TO_STRAIGHT = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
    "—": "-",
    "–": "-",
    "−": "-",
    "\u2010": "-",
    "\u2011": "-",
}


def normalize(text: str) -> str:
    """Normalize text for quote comparison.

    Applies NFKC compatibility normalization, deletes zero-width characters
    (soft hyphen, zero-width space/joiners, BOM), joins words hyphenated
    across a line break ("exam-\\nple" becomes "example"; em/en dashes do
    not join), straightens curly quotes, guillemets and dashes, collapses
    all whitespace runs to single spaces and lowercases.
    """
    text = unicodedata.normalize("NFKC", text)
    for zero_width in _ZERO_WIDTH:
        text = text.replace(zero_width, "")
    # A hyphen (or U+2010, which NFKC does not fold) directly before a line
    # break is a typographic wrap, so the two word halves join. This runs
    # before the dash mappings: an em/en dash before a break is punctuation,
    # not hyphenation, and must survive as a dash.
    text = re.sub(r"[\-\u2010]\r?\n", "", text)
    for curly, straight in _CURLY_TO_STRAIGHT.items():
        text = text.replace(curly, straight)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def locate_quote(
    quote: str,
    windows: Sequence[tuple[int, str]],
    preferred_index: int | None = None,
) -> tuple[VerificationStatus, int | None]:
    """Locate a quote within (chunk_index, text) windows.

    Tries an exact substring match first across all windows, then a
    normalized one, so an exact match always beats a merely normalized
    one. Within each tier the scan order puts ``preferred_index`` (the
    chunk whose context the model read) before the remaining windows in
    the order given, so a quote present in both the context's own chunk
    and a neighbour reports the context's own chunk, and otherwise the
    earliest window in the given order wins. Returns ("not_found", None)
    when the quote appears in no window.
    """
    ordered = list(windows)
    if preferred_index is not None:
        preferred = [w for w in windows if w[0] == preferred_index]
        if preferred:
            ordered = preferred + [w for w in windows if w[0] != preferred_index]

    for index, text in ordered:
        if quote in text:
            return "verified", index

    normalized_quote = normalize(quote)
    if not normalized_quote:
        return "not_found", None
    for index, text in ordered:
        if normalized_quote in normalize(text):
            return "normalized", index

    return "not_found", None
