"""
Project Argus - Evidence grounding tests (pure functions)
"""

from src.utils.grounding import (
    normalize_text,
    quote_is_grounded,
    apply_quote_grounding,
    annotate_corroboration,
)


# ── quote_is_grounded ────────────────────────────────────────────────────────

def test_quote_grounded_exact():
    excerpt = "Solid-state batteries reached 400 Wh/kg in 2024 trials."
    assert quote_is_grounded("Solid-state batteries reached 400 Wh/kg", excerpt)


def test_quote_grounded_whitespace_and_case_insensitive():
    excerpt = "Energy   density   hit  400 Wh/kg."
    assert quote_is_grounded("energy density hit 400 wh/kg", excerpt)


def test_quote_not_grounded_when_absent():
    excerpt = "The study discusses lithium-ion chemistry."
    assert not quote_is_grounded("achieved 900 Wh/kg breakthrough", excerpt)


def test_quote_rejected_when_too_short():
    excerpt = "It is 400 Wh/kg."
    assert not quote_is_grounded("400", excerpt)  # below min_len


# ── apply_quote_grounding ────────────────────────────────────────────────────

def test_apply_grounding_keeps_grounded_fact():
    facts = [{
        "claim": "Reached 400 Wh/kg",
        "source_excerpt": "The cell reached 400 Wh/kg in lab testing.",
        "support_quote": "reached 400 Wh/kg in lab testing",
        "support_level": "SUPPORTED",
        "confidence": 0.9,
    }]
    apply_quote_grounding(facts)
    assert facts[0]["support_level"] == "SUPPORTED"
    assert facts[0]["grounding_failed"] is False


def test_apply_grounding_drops_ungrounded_fact():
    facts = [{
        "claim": "Reached 900 Wh/kg",
        "source_excerpt": "The cell reached 400 Wh/kg in lab testing.",
        "support_quote": "reached 900 Wh/kg breakthrough record",
        "support_level": "SUPPORTED",
        "confidence": 0.95,
    }]
    apply_quote_grounding(facts)
    assert facts[0]["support_level"] == "NOT_SUPPORTED"
    assert facts[0]["confidence"] <= 0.3
    assert facts[0]["grounding_failed"] is True


def test_apply_grounding_drops_when_quote_missing():
    facts = [{
        "claim": "X",
        "source_excerpt": "Some supporting text that is long enough.",
        "support_quote": "",
        "support_level": "SUPPORTED",
        "confidence": 0.8,
    }]
    apply_quote_grounding(facts)
    assert facts[0]["support_level"] == "NOT_SUPPORTED"


def test_apply_grounding_ignores_not_supported():
    facts = [{
        "claim": "X",
        "source_excerpt": "irrelevant",
        "support_quote": "",
        "support_level": "NOT_SUPPORTED",
        "confidence": 0.2,
    }]
    apply_quote_grounding(facts)
    # Untouched (no grounding_failed key added for non-supported).
    assert facts[0]["support_level"] == "NOT_SUPPORTED"
    assert "grounding_failed" not in facts[0]


# ── annotate_corroboration ───────────────────────────────────────────────────

def test_corroboration_two_sources_no_cap():
    emb = [1.0] + [0.0] * 383
    facts = [
        {"claim": "A", "source_url": "http://a", "embedding": emb, "support_level": "SUPPORTED", "confidence": 0.9},
        {"claim": "A2", "source_url": "http://b", "embedding": emb, "support_level": "SUPPORTED", "confidence": 0.9},
    ]
    annotate_corroboration(facts)
    assert facts[0]["corroboration_count"] == 2
    assert facts[0]["single_source_warning"] is False
    assert facts[0]["confidence"] == 0.9  # not capped


def test_corroboration_single_source_caps_confidence():
    emb = [1.0] + [0.0] * 383
    facts = [
        {"claim": "A", "source_url": "http://a", "embedding": emb, "support_level": "SUPPORTED", "confidence": 0.95},
    ]
    annotate_corroboration(facts)
    assert facts[0]["corroboration_count"] == 1
    assert facts[0]["single_source_warning"] is True
    assert facts[0]["confidence"] == 0.7  # capped


def test_corroboration_same_source_twice_is_single():
    emb = [1.0] + [0.0] * 383
    facts = [
        {"claim": "A", "source_url": "http://a", "embedding": emb, "support_level": "SUPPORTED", "confidence": 0.9},
        {"claim": "A2", "source_url": "http://a", "embedding": emb, "support_level": "SUPPORTED", "confidence": 0.9},
    ]
    annotate_corroboration(facts)
    # Same URL twice → still a single distinct source.
    assert facts[0]["corroboration_count"] == 1
    assert facts[0]["single_source_warning"] is True


def test_corroboration_dissimilar_claims_not_grouped():
    e1 = [1.0] + [0.0] * 383
    e2 = [0.0, 1.0] + [0.0] * 382
    facts = [
        {"claim": "A", "source_url": "http://a", "embedding": e1, "support_level": "SUPPORTED", "confidence": 0.9},
        {"claim": "B", "source_url": "http://b", "embedding": e2, "support_level": "SUPPORTED", "confidence": 0.9},
    ]
    annotate_corroboration(facts)
    assert facts[0]["corroboration_count"] == 1
    assert facts[1]["corroboration_count"] == 1
