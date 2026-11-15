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
    deletes stray soft hyphens, straightens curly quotes and dashes,
    collapses all whitespace runs to single spaces and lowercases.
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


def _earliest_match(
    needle: str,
    windows: Sequence[tuple[int, str]],
    exact: bool,
) -> int | None:
    """The lowest chunk index whose window contains the needle, or None.

    ``exact`` compares the raw needle; otherwise both sides go through
    normalize(). Ties across windows resolve on the chunk index (element
    value), never on the order the windows were given, so the contract
    holds for any caller ordering. An empty needle is a substring of any
    text (Python semantics), so it matches the lowest index present —
    callers guard the normalized tier's empty case themselves.
    """
    best: int | None = None
    for index, text in windows:
        haystack = text if exact else normalize(text)
        if needle in haystack and (best is None or index < best):
            best = index
    return best


def locate_quote(
    quote: str,
    windows: Sequence[tuple[int, str]],
    preferred_index: int | None = None,
) -> tuple[VerificationStatus, int | None]:
    """Locate a quote within (chunk_index, text) windows.

    Tries an exact substring match first across all windows, then a
    normalized one, so an exact match always beats a merely normalized
    one. Within each tier, when ``preferred_index`` (the chunk whose
    context the model read) contains the quote it wins; otherwise the
    earliest other window by chunk index wins — the lowest index among
    matching windows, independent of the order the windows were given.
    Returns ("not_found", None) when the quote appears in no window.
    """

    def _preferred_contains(needle: str, exact: bool) -> bool:
        for index, text in windows:
            if index != preferred_index:
                continue
            haystack = text if exact else normalize(text)
            if needle in haystack:
                return True
        return False

    exact_hit = _earliest_match(quote, windows, exact=True)
    if exact_hit is not None:
        if preferred_index is not None and _preferred_contains(quote, exact=True):
            return "verified", preferred_index
        return "verified", exact_hit

    normalized_quote = normalize(quote)
    if not normalized_quote:
        return "not_found", None
    normalized_hit = _earliest_match(normalized_quote, windows, exact=False)
    if normalized_hit is not None:
        if preferred_index is not None and _preferred_contains(normalized_quote, exact=False):
            return "normalized", preferred_index
        return "normalized", normalized_hit

    return "not_found", None
