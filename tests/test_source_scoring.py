"""
Project Argus - Source credibility scoring tests (pure, no network/LLM)
"""

from src.utils.source_scoring import evaluate_source, registrable_domain


# ── registrable_domain ───────────────────────────────────────────────────────

def test_registrable_domain_strips_www_and_lowercases():
    assert registrable_domain("https://WWW.Reuters.com/article/x") == "reuters.com"


def test_registrable_domain_no_www():
    assert registrable_domain("https://arxiv.org/abs/1234") == "arxiv.org"


def test_registrable_domain_handles_garbage_url():
    assert registrable_domain("not a url") == ""
    assert registrable_domain("") == ""


# ── Static tiers ──────────────────────────────────────────────────────────────

def test_trusted_domain_scores_top_tier():
    result = evaluate_source("https://www.nature.com/articles/x")
    assert result["score"] == 1.0
    assert result["type"] == "Academic/Scientific"


def test_edu_domain_scores_top_tier():
    assert evaluate_source("https://web.mit.edu/paper.pdf")["score"] == 1.0


def test_gov_domain_scores_government_tier():
    result = evaluate_source("https://www.sec.gov/filing/x")
    assert result["score"] == 0.9
    assert result["type"] == "Government/Institutional"


def test_major_news_domain_scores_news_tier():
    result = evaluate_source("https://www.reuters.com/business/x")
    assert result["score"] == 0.7
    assert result["type"] == "Major News"


def test_industry_domain_scores_industry_tier():
    assert evaluate_source("https://www.statista.com/stats/x")["score"] == 0.65


def test_org_domain_scores_slight_boost():
    assert evaluate_source("https://www.somefoundation.org/report")["score"] == 0.6


def test_unrecognized_domain_scores_generic_floor():
    result = evaluate_source("https://www.randomblog.info/post")
    assert result["score"] == 0.4
    assert result["type"] == "Unverified/Web"


def test_empty_url_scores_generic_floor():
    assert evaluate_source("")["score"] == 0.4


# ── Promotional/deny tier ────────────────────────────────────────────────────

def test_promotional_domain_scores_below_generic_floor():
    """A domain naming the query's subject + a commerce word (e.g.
    spacexstock.com for a SpaceX query) is worse than an unrecognized source."""
    result = evaluate_source(
        "https://spacexstock.com/valuation-analysis",
        query="Analyze SpaceX's valuation growth over time",
    )
    assert result["score"] == 0.2
    assert result["type"] == "Promotional/Unverified"


def test_promotional_check_requires_both_subject_and_commerce_word():
    # Has the subject but no commerce word — ordinary unrecognized source.
    result = evaluate_source("https://spacexnews.info/x", query="SpaceX valuation")
    assert result["score"] == 0.4


def test_promotional_check_requires_query_context():
    """Without a query, scoring still works — it just can't catch this case."""
    result = evaluate_source("https://spacexstock.com/valuation-analysis")
    assert result["score"] == 0.4


def test_promotional_check_never_demotes_an_allowlisted_domain():
    """Order matters: a trusted/news/gov domain is never re-scored downward,
    even in a contrived case where it happens to match the promotional pattern."""
    result = evaluate_source("https://www.statista.com/stock-market-data", query="stock market")
    assert result["score"] == 0.65  # Industry tier wins, not demoted to 0.2
