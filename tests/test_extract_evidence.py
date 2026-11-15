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
    """Ties between windows resolve to the earliest window in the sequence."""
    windows = [(3, "duplicated text."), (1, "duplicated text."), (2, "duplicated text.")]
    status, found_index = locate_quote("duplicated text.", windows)
    assert status == "verified"
    assert found_index == 3


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
