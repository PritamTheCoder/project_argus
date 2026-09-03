"""
Project Argus - Benchmark analysis tests (pure, no network/LLM).

The bench report drives decisions about what to fix next, so its arithmetic has
to be right before a 30-minute run is spent producing it.
"""

import pytest

from bench.analysis import (
    backend_rollup, diagnose, domain_rollup, efficiency, funnel, node_costs, overall_summary, source_tiers,
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
          survival=0.8, corrob=0.5, tpf=2000, fail=0.05, rejected=0.0):
    return {
        "id": cid, "domain": domain, "passed": passed, "avg_credibility": cred,
        "tiers": {"authoritative_share": auth, "domain_diversity": div},
        "funnel": {"facts_supported": 20, "grounding_survival": survival,
                   "corroboration_rate": corrob},
        "efficiency": {"tokens_per_supported_fact": tpf, "failover_rate": fail,
                       "billed_failure_rate": fail, "rejected_rate": rejected,
                       "wasted_tokens": int(40000 * fail),
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


# ── backend health ───────────────────────────────────────────────────────────

def _bh(**kw):
    base = {"calls": 0, "failures": 0, "empty_calls": 0, "results": 0, "degraded": False}
    base.update(kw)
    return base


def test_backend_rollup_flags_a_backend_that_never_returns_results():
    """The dangerous failure: every call 'succeeds' but yields nothing."""
    results = [
        {"backend_health": {"semantic_scholar_search": _bh(calls=3, results=0, empty_calls=3, degraded=True),
                            "arxiv_search": _bh(calls=2, results=14)}},
        {"backend_health": {"semantic_scholar_search": _bh(calls=2, results=0, empty_calls=2, degraded=True)}},
    ]
    rows = backend_rollup(results)
    by = {r["backend"]: r for r in rows}
    ss = by["semantic_scholar_search"]
    assert ss["calls"] == 5 and ss["results"] == 0
    assert ss["degraded"] is True
    assert ss["cases_degraded"] == 2
    assert by["arxiv_search"]["degraded"] is False
    assert by["arxiv_search"]["results_per_call"] == 7.0
    assert rows[0]["backend"] == "semantic_scholar_search"  # degraded sorted first


def test_backend_rollup_empty():
    assert backend_rollup([{"id": "a"}]) == []


def test_diagnose_reports_a_degraded_backend_as_high_severity():
    healthy = [_case("a", "d1", auth=0.8, div=0.95, survival=0.9, corrob=0.6, tpf=1500, fail=0.0)]
    backends = [{"backend": "semantic_scholar_search", "calls": 5, "results": 0,
                 "empty_calls": 5, "cases_degraded": 2, "results_per_call": 0.0, "degraded": True}]
    findings = diagnose(domain_rollup(healthy), overall_summary(healthy), backends)
    assert [f["area"] for f in findings] == ["backend"]
    assert findings[0]["severity"] == "high"
    assert "semantic_scholar_search" in findings[0]["finding"]


def test_diagnose_ignores_healthy_backends():
    healthy = [_case("a", "d1", auth=0.8, div=0.95, survival=0.9, corrob=0.6, tpf=1500, fail=0.0)]
    backends = [{"backend": "arxiv_search", "calls": 5, "results": 30,
                 "empty_calls": 0, "cases_degraded": 0, "results_per_call": 6.0, "degraded": False}]
    assert diagnose(domain_rollup(healthy), overall_summary(healthy), backends) == []


# ── relevance vs credibility ─────────────────────────────────────────────────

def test_source_tiers_separates_relevance_from_credibility():
    """The failure this exists to catch: peer-reviewed sources on the wrong
    subject. Credibility is perfect, relevance is not."""
    source_map = {
        "[1]": {"url": "http://arxiv.org/a", "credibility_score": 1.0, "relevance_score": 0.02},
        "[2]": {"url": "http://arxiv.org/b", "credibility_score": 1.0, "relevance_score": 0.05},
        "[3]": {"url": "http://europepmc.org/c", "credibility_score": 0.95, "relevance_score": 0.81},
    }
    t = source_tiers(source_map)
    assert t["authoritative_share"] == 1.0      # all credible...
    assert t["off_topic_share"] == 0.667        # ...but two thirds off-topic
    assert t["avg_relevance"] == 0.293


def test_source_tiers_ignores_sources_without_a_relevance_score():
    """Sources from before this signal existed must not count as zero."""
    source_map = {
        "[1]": {"url": "http://a.com/x", "credibility_score": 0.9, "relevance_score": 0.8},
        "[2]": {"url": "http://b.com/y", "credibility_score": 0.9},          # not scored
        "[3]": {"url": "http://c.com/z", "credibility_score": 0.9, "relevance_score": None},
    }
    t = source_tiers(source_map)
    assert t["relevance_scored"] == 1
    assert t["avg_relevance"] == 0.8
    assert t["off_topic_share"] == 0.0


def test_source_tiers_relevance_absent_entirely():
    t = source_tiers({"[1]": {"url": "http://a.com", "credibility_score": 0.9}})
    assert t["avg_relevance"] == 0.0
    assert t["relevance_scored"] == 0


def test_diagnose_flags_credible_but_off_topic_retrieval():
    """High credibility must not mask the wrong corpus."""
    results = [_case("a", "biomedical", auth=0.9, div=0.95, survival=0.9, corrob=0.6,
                     tpf=1500, fail=0.0)]
    results[0]["tiers"]["off_topic_share"] = 0.7
    results[0]["tiers"]["avg_relevance"] = 0.08
    findings = diagnose(domain_rollup(results), overall_summary(results))
    assert [f["area"] for f in findings] == ["relevance"]
    assert findings[0]["severity"] == "high"


def test_diagnose_quiet_when_sources_are_on_topic():
    results = [_case("a", "d1", auth=0.8, div=0.95, survival=0.9, corrob=0.6, tpf=1500, fail=0.0)]
    results[0]["tiers"]["off_topic_share"] = 0.0
    results[0]["tiers"]["avg_relevance"] = 0.72
    assert diagnose(domain_rollup(results), overall_summary(results)) == []


# ── Failure accounting: rejected (free) vs billed (paid twice) ───────────────
# A rejected attempt (quota/outage) costs nothing; blending it with billed
# failures into one rate hides a capacity problem behind a schema warning.

def test_quota_rejection_is_classified_as_unbilled():
    from eval.metrics import classify_llm_error
    real = Exception(
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
        "`openai/gpt-oss-120b` in organization `org_01kw1` service tier `on_demand` "
        "on tokens per day (TPD): Limit 200000, Used 199341'}}"
    )
    assert classify_llm_error(real) == "rejected"


def test_generated_then_invalid_is_classified_as_billed():
    from eval.metrics import classify_llm_error
    assert classify_llm_error(Exception("tool_use_failed: model emitted a bare array")) == "billed"


def test_unknown_failure_counts_as_billed():
    """Overstating cost is the safer error for a spend metric."""
    from eval.metrics import classify_llm_error
    assert classify_llm_error(ValueError("something odd")) == "billed"


def test_outage_and_auth_failures_are_unbilled():
    from eval.metrics import classify_llm_error
    for msg in ("503 Service Unavailable", "401 authentication error", "APITimeoutError"):
        assert classify_llm_error(Exception(msg)) == "rejected", msg


def test_rejected_attempts_do_not_count_as_wasted_tokens():
    from bench.analysis import efficiency
    metrics = {"llm_calls": 100, "llm_errors": 73, "llm_rejected": 73,
               "llm_billed_failures": 0, "total_tokens": 27000}
    e = efficiency(metrics, {"facts_supported": 10}, 100.0)
    assert e["rejected_rate"] == 0.73
    assert e["billed_failure_rate"] == 0.0
    assert e["wasted_tokens"] == 0, "a refused request is never billed"


def test_billed_failures_are_priced_at_the_average_successful_call():
    from bench.analysis import efficiency
    # 10 successful calls at 1000 tokens each; 5 generated-then-unusable.
    metrics = {"llm_calls": 15, "llm_errors": 5, "llm_rejected": 0,
               "llm_billed_failures": 5, "total_tokens": 10000}
    e = efficiency(metrics, {"facts_supported": 10}, 100.0)
    assert e["wasted_tokens"] == 5000


def test_legacy_results_without_the_split_are_treated_as_billed():
    """Result files predating this split must keep reporting as they did."""
    from bench.analysis import efficiency
    e = efficiency({"llm_calls": 10, "llm_errors": 4, "total_tokens": 6000},
                   {"facts_supported": 5}, 50.0)
    assert e["billed_failures"] == 4
    assert e["rejected_attempts"] == 0


def test_diagnose_separates_capacity_from_reliability():
    results = [_case("a", "d1", fail=0.0, rejected=0.7)]
    findings = diagnose(domain_rollup(results), overall_summary(results))
    areas = {f["area"] for f in findings}
    assert "capacity" in areas
    assert "reliability" not in areas, "a quota ceiling is not a schema defect"


def test_legacy_result_file_backfills_billed_failures():
    """A result file from before the split has failover_rate but no
    billed_failure_rate/rejected_rate keys at all. Reading those as missing-so-0
    would report a 73%-failover run as 0% failure. It must fall back to the old
    blended number instead of silently dropping it."""
    legacy = _case("a", "d1", fail=0.73)
    del legacy["efficiency"]["billed_failure_rate"]
    del legacy["efficiency"]["rejected_rate"]
    del legacy["efficiency"]["wasted_tokens"]

    overall = overall_summary([legacy])
    assert overall["billed_failure_rate"] == 0.73
    assert overall["rejected_rate"] == 0.0

    rows = domain_rollup([legacy])
    assert rows[0]["billed_failure_rate"] == 0.73


def test_report_only_does_not_zero_out_an_old_high_failover_run():
    """End-to-end: the actual failure that motivated this fix."""
    legacy = {
        "id": "x", "domain": "d1", "passed": True, "avg_credibility": 0.7,
        "tiers": {"authoritative_share": 0.5, "domain_diversity": 0.5},
        "funnel": {"facts_supported": 10, "grounding_survival": 0.8, "corroboration_rate": 0.2},
        "efficiency": {"tokens_per_supported_fact": 3000, "failover_rate": 0.73,
                       "failovers": 20, "latency_s": 300, "total_tokens": 60000, "llm_calls": 27},
    }
    findings = diagnose(domain_rollup([legacy]), overall_summary([legacy]))
    assert any(f["area"] == "reliability" for f in findings), (
        "a pre-split run with real failures must still surface a reliability finding"
    )
