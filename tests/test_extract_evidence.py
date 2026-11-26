"""Tests for evidence quote verification (extract/evidence.py)."""

from asago_policy_mapper.extract.evidence import locate_quote, normalize

# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_exact_quote_verified():
    """A quote that appears verbatim in a window verifies with its index."""
    windows = [(0, "Data minimization is required."), (1, "Bias monitoring runs monthly.")]
    status, found_index = locate_quote("Bias monitoring runs monthly.", windows)
    assert status == "verified"
    assert found_index == 1


def test_quote_not_found_reports_no_chunk():
    """A quote absent from every window reports not_found with no index."""
    windows = [(0, "Data minimization is required.")]
    status, found_index = locate_quote("Nothing matches this quote.", windows)
    assert status == "not_found"
    assert found_index is None


def test_quote_from_neighbour_chunk_records_found_index():
    """A quote that lives in a neighbour chunk reports that neighbour's index."""
    windows = [
        (0, "Retention is limited to 30 days."),
        (1, "Encryption at rest is mandatory."),
        (2, "Audits run yearly."),
    ]
    status, found_index = locate_quote("Encryption at rest is mandatory.", windows)
    assert status == "verified"
    assert found_index == 1


def test_whitespace_and_curly_quotes_normalized():
    """Whitespace and curly-punctuation differences verify as normalized."""
    source = "The  policy states “privacy’s a core value” — everywhere."
    # Exact match fails (the source doubles a space); normalized match must succeed.
    quote = "The policy states “privacy’s a core value” — everywhere."
    windows = [(0, "Unrelated text."), (1, source)]
    status, found_index = locate_quote(quote, windows)
    assert status == "normalized"
    assert found_index == 1


def test_normalize_collapses_whitespace_and_straightens_punctuation():
    """normalize applies NFKC, straightens quotes/dashes, collapses whitespace, lowercases."""
    text = "  Caf\u00e9 \u201cquotes\u201d \u2018single\u2019 \u2014 dash \u2013 en\u2009thin\u00a0gap\ttab\nnewline  "
    assert normalize(text) == "café \"quotes\" 'single' - dash - en thin gap tab newline"


def test_normalize_applies_nfkc_compatibility_forms():
    """NFKC folds compatibility characters: fullwidth, ligatures, circled digits."""
    assert normalize("Ａﬃ①") == "affi1"


def test_normalize_straightens_hyphen_characters():
    """U+2010 and U+2011 map to the ASCII hyphen."""
    assert normalize("non\u2010breaking and non\u2011breaking") == "non-breaking and non-breaking"


def test_normalize_deletes_zero_width_characters():
    """Soft hyphen, zero-width space/joiners and BOM vanish between word halves."""
    for zw in ("\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff"):
        assert normalize(f"pro{zw}cessing") == "processing"


def test_normalize_maps_guillemets_to_straight_quotes():
    """Guillemets compare like their straight double-quote counterparts."""
    assert normalize("«mot» and “mot”") == '"mot" and "mot"'


def test_normalize_joins_hyphenated_words_across_line_breaks():
    """A hyphen before a newline is a typographic wrap: the word joins."""
    assert normalize("exam-\nple") == "example"


def test_normalize_joins_u2010_wraps_and_crlf_wraps():
    """U+2010 wraps join like ASCII hyphens; CRLF line endings wrap too."""
    assert normalize("exam\u2010\r\nple") == "example"


def test_normalize_joins_soft_hyphen_wraps():
    """A soft hyphen before a line break is a wrap: the word joins.

    The soft-hyphen wrap join runs before zero-width deletion, which
    would otherwise strip the hyphen and leave the halves split.
    """
    assert normalize("raised produc\u00ad\ntivity targets.") == "raised productivity targets."
    assert normalize("exam\u00ad\r\nple") == "example"


def test_normalize_joins_u2011_wraps():
    """U+2011 wraps join too — NFKC folds it to U+2010 first."""
    assert normalize("exam\u2011\nple") == "example"


def test_normalize_still_deletes_mid_word_soft_hyphens():
    """A soft hyphen not acting as a wrap still deletes."""
    assert normalize("pro\u00adcessing") == "processing"


def test_normalize_does_not_join_em_dash_before_line_break():
    """An em/en dash before a break is punctuation, not hyphenation."""
    assert normalize("He said—\nand left") == "he said- and left"
    assert normalize("pp. 3–\n5") == "pp. 3- 5"


def test_locate_quote_matches_through_a_line_break_hyphen():
    """A reflowed quote matches a window whose word is wrapped mid-word.

    The window keeps the typesetting wrap ("produc-\\ntivity"); the model's
    quote carries the joined spelling ("productivity"). Both normalize to
    the same word.
    """
    status, found_index = locate_quote(
        "raised productivity targets", [(0, "The policy raised produc-\ntivity targets for AI systems.")]
    )
    assert status == "normalized"
    assert found_index == 0


def test_zero_width_only_quote_is_not_found():
    """A quote made only of zero-width characters normalizes to nothing."""
    status, found_index = locate_quote("\ufeff\u200b", [(0, "any text")])
    assert status == "not_found"
    assert found_index is None


def test_empty_quote_matches_first_window_by_substring_semantics():
    """An empty quote is a substring of any text, so it verifies in window order."""
    status, found_index = locate_quote("", [(7, "policy"), (8, "text")])
    assert status == "verified"
    assert found_index == 7


def test_whitespace_only_quote_not_found_against_real_text():
    """A whitespace-only quote normalizes to empty and never matches real text."""
    status, found_index = locate_quote("\t", [(7, "policy text")])
    assert status == "not_found"
    assert found_index is None


def test_nonempty_quote_skips_whitespace_only_window_text():
    """A real quote does not match a window whose text is only whitespace."""
    status, found_index = locate_quote("policy", [(7, "   "), (8, "policy text")])
    assert status == "verified"
    assert found_index == 8


def test_exact_match_wins_over_normalized():
    """The same text found exactly and in normalized form reports verified."""
    text = "Plain ascii sentence."
    windows = [(0, text.upper()), (1, text)]
    status, found_index = locate_quote(text, windows)
    assert status == "verified"
    assert found_index == 1


def test_first_window_wins_on_multiple_exact_matches():
    """Ties between windows resolve to the earliest chunk index.

    The windows are deliberately in non-ascending order, so this pins the
    value-based contract (lowest index wins), not the given sequence.
    """
    windows = [(3, "duplicated text."), (1, "duplicated text."), (2, "duplicated text.")]
    status, found_index = locate_quote("duplicated text.", windows)
    assert status == "verified"
    assert found_index == 1


def test_normalized_tie_resolves_to_earliest_index():
    """Normalized-tier ties also resolve to the lowest chunk index."""
    windows = [(5, "The  normalized   TEXT."), (2, "the  normalized text.")]
    status, found_index = locate_quote("the normalized text.", windows)
    assert status == "normalized"
    assert found_index == 2


def test_empty_windows_report_not_found():
    """No windows means the quote cannot be located."""
    status, found_index = locate_quote("any quote", [])
    assert status == "not_found"
    assert found_index is None


def test_normalized_match_searches_all_windows_after_exact_fails():
    """Exact failure in early windows still finds a normalized match later."""
    windows = [(0, "Exact different text."), (2, "The  normalized   TEXT.")]
    status, found_index = locate_quote("the normalized text.", windows)
    assert status == "normalized"
    assert found_index == 2


def test_preferred_window_wins_over_earlier_window():
    """A quote in several windows reports the context's own chunk first.

    Non-ascending windows: without the preference the lowest index (0)
    would win, so only the preferred_index mechanism can return 2.
    """
    windows = [(4, "shared text."), (0, "shared text."), (2, "shared text.")]
    status, found_index = locate_quote("shared text.", windows, preferred_index=2)
    assert status == "verified"
    assert found_index == 2


def test_preferred_window_wins_in_normalized_tier():
    """The preference also applies when only the normalized tier matches.

    Non-ascending windows, so the preference — not the given order —
    decides.
    """
    windows = [(3, "Shared  text."), (1, "Shared  text.")]
    status, found_index = locate_quote("shared text.", windows, preferred_index=3)
    assert status == "normalized"
    assert found_index == 3


def test_preferred_index_absent_from_windows_keeps_scan_order():
    """A preferred index that names no window changes nothing."""
    windows = [(3, "duplicated text."), (1, "duplicated text.")]
    status, found_index = locate_quote("duplicated text.", windows, preferred_index=9)
    assert status == "verified"
    assert found_index == 1


def test_preferred_window_without_the_quote_does_not_win():
    """The preference only applies when that window contains the quote."""
    windows = [(0, "the quote lives here."), (1, "unrelated text.")]
    status, found_index = locate_quote("the quote lives here.", windows, preferred_index=1)
    assert status == "verified"
    assert found_index == 0
