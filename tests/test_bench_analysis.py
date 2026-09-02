"""
Project Argus - Benchmark analysis tests (pure, no network/LLM).

The bench report drives decisions about what to fix next, so its arithmetic has
to be right before a 30-minute run is spent producing it.
"""

import pytest

from bench.analysis import (
    diagnose, domain_rollup, efficiency, funnel, node_costs, overall_summary, source_tiers,
)


# ── source_tiers ─────────────────────────────────────────────────────────────

def test_source_tiers_buckets_by_credibility():
    source_map = {
        "[1]": {"url": "https://www.sec.gov/a", "credibility_score": 0.9},
        "[2]": {"url": "https://reuters.com/b", "credibility_score": 0.7},
        "[3]": {"url": "https://pitchbook.com/c", "credibility_score": 0.65},
        "[4]": {"url": "https://blog.info/d", "credibility_score": 0.4},
        "[5]": {"url": "https://xstock.com/e", "credibility_score": 0.2},
    }
    t = source_tiers(source_map)
    assert t["counts"] == {"primary": 1, "major_news": 1, "industry": 1,
                           "generic_web": 1, "promotional": 1}
    assert t["total"] == 5
    assert t["authoritative_share"] == 0.4      # primary + major_news
    assert t["domain_diversity"] == 1.0         # five distinct sites


def test_source_tiers_detects_domain_concentration():
    """Several pages from one site must not read as several sources."""
    source_map = {
        f"[{i}]": {"url": f"https://farm.com/page{i}", "credibility_score": 0.4}
        for i in range(4)
    }
    t = source_tiers(source_map)
    assert t["distinct_domains"] == 1
    assert t["domain_diversity"] == 0.25


def test_source_tiers_empty():
    t = source_tiers({})
    assert t["total"] == 0
    assert t["authoritative_share"] == 0.0
    assert t["domain_diversity"] == 0.0


# ── funnel ───────────────────────────────────────────────────────────────────

def _state(**kw):
    base = {"source_map": {}, "structured_evidence": [], "verified_facts": [],
            "contradictions": [], "consensus_findings": []}
    base.update(kw)
    return base


def test_funnel_tracks_dropoff_and_yields():
    st = _state(
        source_map={"[1]": {"url": "http://a"}, "[2]": {"url": "http://b"}},
        structured_evidence=[{}] * 10,
        verified_facts=[
            {"support_level": "SUPPORTED", "corroboration_count": 2},
            {"support_level": "SUPPORTED", "corroboration_count": 1},
            {"support_level": "PARTIALLY_SUPPORTED", "corroboration_count": 3},
            {"support_level": "NOT_SUPPORTED"},
        ],
    )
    f = funnel(st)
    assert f["sources"] == 2
    assert f["facts_extracted_last_pass"] == 10
    assert f["facts_verified"] == 4
    assert f["facts_supported"] == 3
    assert f["facts_corroborated"] == 2
    assert f["grounding_survival"] == 0.75      # 3 of 4 survived
    assert f["corroboration_rate"] == 0.667     # 2 of 3 supported
    assert f["facts_per_source"] == 1.5


def test_funnel_handles_empty_run():
    f = funnel(_state())
    assert f["facts_supported"] == 0
    assert f["grounding_survival"] == 0.0       # no division by zero
    assert f["corroboration_rate"] == 0.0


# ── efficiency ───────────────────────────────────────────────────────────────

def test_efficiency_computes_unit_cost():
    m = {"llm_calls": 30, "llm_errors": 6, "input_tokens": 40000,
         "output_tokens": 10000, "total_tokens": 50000}
    e = efficiency(m, {"facts_supported": 25}, latency_s=200.0)
    assert e["failover_rate"] == 0.2
    assert e["tokens_per_supported_fact"] == 2000.0
    assert e["calls_per_supported_fact"] == 1.2
    assert e["seconds_per_supported_fact"] == 8.0


def test_efficiency_with_no_facts_does_not_divide_by_zero():
    e = efficiency({"llm_calls": 5, "total_tokens": 100}, {"facts_supported": 0}, 10.0)
    assert e["tokens_per_supported_fact"] == 0.0


def test_efficiency_tolerates_missing_metrics():
    e = efficiency(None, {"facts_supported": 3}, 1.0)
    assert e["llm_calls"] == 0
    assert e["failover_rate"] == 0.0


# ── node_costs ───────────────────────────────────────────────────────────────

def test_node_costs_sorted_by_tokens():
    deltas = {
        "refiner": {"llm_calls": 2, "llm_errors": 1, "total_tokens": 40000},
        "writer":  {"llm_calls": 1, "llm_errors": 0, "total_tokens": 5000},
    }
    rows = node_costs(deltas, {"refiner": 45.0, "writer": 7.0, "scout": 110.0})
    assert rows[0]["node"] == "refiner"
    # A node with wall-clock but no tokens (scout is network-bound) still appears.
    assert any(r["node"] == "scout" and r["tokens"] == 0 and r["seconds"] == 110.0 for r in rows)


# ── rollup + diagnosis ───────────────────────────────────────────────────────

def _case(cid, domain, passed=True, cred=0.8, auth=0.6, div=0.9,
          survival=0.8, corrob=0.5, tpf=2000, fail=0.05):
    return {
        "id": cid, "domain": domain, "passed": passed, "avg_credibility": cred,
        "tiers": {"authoritative_share": auth, "domain_diversity": div},
        "funnel": {"facts_supported": 20, "grounding_survival": survival,
                   "corroboration_rate": corrob},
        "efficiency": {"tokens_per_supported_fact": tpf, "failover_rate": fail,
                       "latency_s": 200, "total_tokens": 40000, "llm_calls": 30},
    }


def test_domain_rollup_groups_and_averages():
    rows = domain_rollup([
        _case("a", "hard_science", cred=0.9),
        _case("b", "hard_science", cred=0.7),
        _case("c", "finance_private", cred=0.5),
    ])
    by = {r["domain"]: r for r in rows}
    assert by["hard_science"]["cases"] == 2
    assert by["hard_science"]["avg_credibility"] == 0.8
    assert by["finance_private"]["cases"] == 1


def test_domain_rollup_excludes_errored_cases_from_averages():
    rows = domain_rollup([
        _case("a", "d1", cred=0.8),
        {"id": "b", "domain": "d1", "passed": False, "error": "boom"},
    ])
    r = rows[0]
    assert r["cases"] == 2 and r["errors"] == 1
    assert r["avg_credibility"] == 0.8   # the errored case must not drag it to 0.4


def test_diagnose_flags_weak_retrieval_and_failover():
    results = [_case("a", "d1", auth=0.2, fail=0.3)]
    findings = diagnose(domain_rollup(results), overall_summary(results))
    areas = {f["area"] for f in findings}
    assert "retrieval" in areas
    assert "reliability" in areas
    assert findings[0]["severity"] == "high"   # high severity sorted first


def test_diagnose_silent_when_everything_is_healthy():
    results = [_case("a", "d1", auth=0.8, div=0.95, survival=0.9,
                     corrob=0.6, tpf=1500, fail=0.0)]
    assert diagnose(domain_rollup(results), overall_summary(results)) == []


def test_diagnose_flags_token_cost_only_above_threshold():
    healthy = [_case("a", "d1", auth=0.8, div=0.95, survival=0.9, corrob=0.6, tpf=3000, fail=0.0)]
    assert not any(f["area"] == "efficiency" for f in
                   diagnose(domain_rollup(healthy), overall_summary(healthy)))

    costly = [_case("a", "d1", auth=0.8, div=0.95, survival=0.9, corrob=0.6, tpf=9000, fail=0.0)]
    assert any(f["area"] == "efficiency" for f in
               diagnose(domain_rollup(costly), overall_summary(costly)))


def test_overall_summary_empty_when_all_cases_errored():
    assert overall_summary([{"id": "a", "error": "boom"}]) == {}
