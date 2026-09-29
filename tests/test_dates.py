"""
Project Argus - Date sanity check tests
"""

from datetime import date

from src.utils.dates import sanitize_as_of_date

_TODAY = date(2026, 9, 12)


def test_past_year_passes_through():
    assert sanitize_as_of_date("2024", today=_TODAY) == "2024"


def test_current_year_passes_through():
    """A bare year compares by year number, not a specific day, so "this
    year" never spuriously flags as future partway through it."""
    assert sanitize_as_of_date("2026", today=_TODAY) == "2026"


def test_future_year_is_dropped():
    assert sanitize_as_of_date("2027", today=_TODAY) == ""


def test_past_iso_date_passes_through():
    assert sanitize_as_of_date("2026-01-15", today=_TODAY) == "2026-01-15"


def test_future_iso_date_is_dropped():
    assert sanitize_as_of_date("2026-12-25", today=_TODAY) == ""


def test_today_exactly_passes_through():
    assert sanitize_as_of_date("2026-09-12", today=_TODAY) == "2026-09-12"


def test_empty_string_passes_through():
    assert sanitize_as_of_date("", today=_TODAY) == ""


def test_unparseable_string_passes_through_unchanged():
    """Only confident future dates get dropped — a malformed string is a
    different problem this check isn't trying to solve."""
    assert sanitize_as_of_date("circa 2020s", today=_TODAY) == "circa 2020s"


def test_invalid_calendar_date_passes_through():
    assert sanitize_as_of_date("2026-13-99", today=_TODAY) == "2026-13-99"


def test_defaults_to_real_today_when_not_given():
    """Smoke test: no exception, and a clearly-past date survives against
    whatever the real clock says."""
    assert sanitize_as_of_date("2020-01-01") == "2020-01-01"
