"""
Project Argus - Source Credibility Scoring

Scores a URL's domain against static authority tiers (1.0 down to 0.4) and
flags domains that look built to promote the query's own subject — worse than
an ordinary unrecognized source, not the same as one.
"""

import re
from urllib.parse import urlparse
from src.config import TRUSTED_DOMAINS, MAJOR_NEWS_DOMAINS, INDUSTRY_DOMAINS

# A domain containing one of these words alongside the query's subject
# (e.g. "<company>stock.com") is promotional by construction — registered to
# ride the subject's search traffic rather than to report on it.
_PROMOTIONAL_WORDS = {"stock", "shares", "ipo", "tracker", "price"}

# Proper nouns in the query — a cheap stand-in for "the query's subject"
# without needing an LLM call.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]{3,}\b")


def registrable_domain(url: str) -> str:
    """Lowercased hostname with a leading 'www.' stripped.

    Not a true public-suffix lookup (e.g. 'co.uk' isn't handled specially),
    but sufficient for comparing one registrable domain against another.
    """
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _is_promotional(domain: str, query: str) -> bool:
    """True if `domain` names the query's subject next to a promotional word."""
    if not query or not any(word in domain for word in _PROMOTIONAL_WORDS):
        return False
    subjects = _PROPER_NOUN_RE.findall(query)
    return any(subject.lower() in domain for subject in subjects)


def evaluate_source(url: str, query: str = "") -> dict:
    """
    Score a source's credibility (0.0-1.0) and classify its type from its domain.

    Args:
        url: the source URL.
        query: the research query this source was found for. Optional — used
            only to catch subject-promotional domains (see `_is_promotional`);
            scoring still works without it.

    Returns:
        dict: {"score": float, "type": str}
    """
    hostname = registrable_domain(url)
    if not hostname:
        return {"score": 0.4, "type": "Unverified/Web"}

    # Tier 1: Peer-reviewed / Academic (Score 1.0)
    for trusted in TRUSTED_DOMAINS:
        if hostname == trusted or hostname.endswith("." + trusted):
            return {"score": 1.0, "type": "Academic/Scientific"}

    if hostname.endswith(".edu"):
        return {"score": 1.0, "type": "Academic/Scientific"}

    # Tier 2: Government / Institutional (Score 0.9)
    if hostname.endswith(".gov"):
        return {"score": 0.9, "type": "Government/Institutional"}

    # Tier 2.5: Major News (Score 0.7)
    for news_domain in MAJOR_NEWS_DOMAINS:
        if hostname == news_domain or hostname.endswith("." + news_domain):
            return {"score": 0.7, "type": "Major News"}

    # Tier 3: Industry/Market Research (Score 0.65)
    for ind_domain in INDUSTRY_DOMAINS:
        if hostname == ind_domain or hostname.endswith("." + ind_domain):
            return {"score": 0.65, "type": "Industry/Market Research"}

    # Tier 4: .org domains (non-profits) get a slight credibility boost
    if hostname.endswith(".org"):
        return {"score": 0.6, "type": "News/Industry"}

    # Below every allowlisted tier: a domain built to promote the query's
    # subject scores worse than a merely-unrecognized one.
    if _is_promotional(hostname, query):
        return {"score": 0.2, "type": "Promotional/Unverified"}

    # Tier 5: Everything Else / Generic Web (Score 0.4)
    return {"score": 0.4, "type": "Unverified/Web"}
