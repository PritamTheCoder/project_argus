"""
Project Argus - Date sanity checks

Retrieval backends report as_of_date in whatever shape they have (bare year,
ISO prefix, empty) with no validation. A future date is almost always a
parsing bug — a misread table cell, a wrong field — not a real fact, so it
gets nulled rather than propagated as a false temporal signal.
"""

import logging
import re
from datetime import date, datetime

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"^\d{4}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def sanitize_as_of_date(raw: str, today: date | None = None) -> str:
    """Return ``raw`` unchanged unless it parses as a future date, in which
    case return "". Unparseable strings pass through as-is — this only
    catches dates it can confidently place in the future, not malformed ones."""
    if not raw:
        return raw
    today = today or date.today()

    try:
        if _YEAR_RE.match(raw):
            is_future = int(raw) > today.year  # bare year: only a later year counts as future
        elif _ISO_RE.match(raw):
            is_future = datetime.strptime(raw[:10], "%Y-%m-%d").date() > today
        else:
            return raw
    except ValueError:
        return raw

    if is_future:
        logger.warning("Dropped future as_of_date %r (today is %s)", raw, today)
        return ""
    return raw
