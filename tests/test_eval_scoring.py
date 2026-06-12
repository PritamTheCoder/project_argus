"""
Project Argus - Eval scoring tests

Pure-function coverage for the evaluation harness scoring. No network/keys.
"""

from eval.scoring import (
    keyword_coverage,
    trap_violations,
    citation_integrity,
    source_stats,
    verified_fact_stats,
    score_case,
)


def test_keyword_coverage_partial():
    cov = keyword_coverage("Solid-state batteries reach 400 Wh/kg.", ["solid-state", "Wh/kg", "dendrite"])
    assert cov["score"] == 2 / 3
    assert "dendrite" in cov["missing"]


def test_keyword_coverage_empty_requirements():
    assert keyword_coverage("anything", [])["score"] == 1.0


def test_trap_violations_detected():
    assert trap_violations("contains lorem ipsum text", ["lorem ipsum"]) == ["lorem ipsum"]
    assert trap_violations("clean report", ["lorem ipsum"]) == []


def test_citation_integrity_ok():
    report = "Claim one [1]. Claim two [2]."
    source_map = {"[1]": {}, "[2]": {}}
    result = citation_integrity(report, source_map)
    assert result["ok"]
    assert result["cited_ids"] == ["1", "2"]


def test_citation_integrity_unresolved():
    report = "Claim one [1]. Dangling [5]."
    source_map = {"[1]": {}}
    result = citation_integrity(report, source_map)
    assert not result["ok"]
    assert result["unresolved"] == ["5"]


def test_source_stats_avg_credibility():
    source_map = {
        "[1]": {"credibility_score": 1.0},
        "[2]": {"credibility_score": 0.5},
    }
    stats = source_stats(source_map)
    assert stats["source_count"] == 2
    assert stats["avg_credibility"] == 0.75


def test_verified_fact_stats_counts_supported_only():
    facts = [
        {"support_level": "SUPPORTED"},
        {"support_level": "PARTIALLY_SUPPORTED"},
        {"support_level": "NOT_SUPPORTED"},
        {"support_level": "UNCERTAIN"},
    ]
    stats = verified_fact_stats(facts)
    assert stats["verified_fact_count"] == 2
    assert stats["total_fact_count"] == 4


def test_score_case_pass():
    case = {
        "id": "c1",
        "must_include": ["solid-state", "Wh/kg"],
        "must_not_include": [],
        "min_coverage": 0.6,
        "min_credibility": 0.5,
    }
    final_state = {
        "report": "Solid-state cells now exceed 400 Wh/kg per multiple sources [1][2]. " * 5,
        "source_map": {"[1]": {"credibility_score": 1.0}, "[2]": {"credibility_score": 0.6}},
        "verified_facts": [{"support_level": "SUPPORTED"}, {"support_level": "SUPPORTED"}],
    }
    result = score_case(case, final_state)
    assert result["passed"]
    assert result["citation_ok"]
    assert result["verified_fact_count"] == 2


def test_score_case_fails_on_unresolved_citation():
    case = {"id": "c2", "must_include": ["x"], "min_coverage": 0.0, "min_credibility": 0.0}
    final_state = {
        "report": "Body referencing a missing source [9]. " * 10,
        "source_map": {"[1]": {"credibility_score": 1.0}},
        "verified_facts": [],
    }
    result = score_case(case, final_state)
    assert not result["passed"]
    assert result["unresolved_citations"] == ["9"]


def test_score_case_fails_on_trap():
    case = {
        "id": "c3",
        "must_include": [],
        "must_not_include": ["hallucinated"],
        "min_coverage": 0.0,
        "min_credibility": 0.0,
    }
    final_state = {
        "report": "This contains a hallucinated claim. " * 10,
        "source_map": {"[1]": {"credibility_score": 1.0}},
        "verified_facts": [],
    }
    result = score_case(case, final_state)
    assert not result["passed"]
    assert result["trap_violations"] == ["hallucinated"]
