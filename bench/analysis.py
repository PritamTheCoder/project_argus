"""
Project Argus - Benchmark analysis (pure functions, no I/O).

The eval harness answers "did this change break anything?". The bench answers
two harder questions:

  1. WHERE does the pipeline lose information?  -> `funnel()`
     Sources gathered are not sources used; facts extracted are not facts that
     survive grounding. The funnel makes each drop-off visible so the weakest
     stage is a number rather than a guess.

  2. IS optimisation actually needed, and WHERE?  -> `efficiency()` / `node_costs()`
     Tokens-per-supported-fact is the honest unit cost of research. Node-level
     attribution says which stage to optimise instead of optimising by vibes.

Everything here is a pure function over a finished run, so it is unit-tested
without network access or API keys.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

_SUPPORTED = ("SUPPORTED", "PARTIALLY_SUPPORTED")

# Below this query-similarity a source is counted as off-topic regardless of
# how credible its domain is. Mirrors LOW_RELEVANCE_THRESHOLD in src/config.
OFF_TOPIC_RELEVANCE = 0.35

# Credibility buckets, aligned with src/utils/source_scoring.py tiers.
_TIERS = (
    ("primary", 0.85),        # .gov, academic, peer-reviewed
    ("major_news", 0.68),     # Reuters/Bloomberg-class reporting
    ("industry", 0.55),       # trade press, company/funding databases
    ("generic_web", 0.35),    # unrecognised domains
    ("promotional", 0.0),     # subject-promotional domains (deny tier)
)


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def source_tiers(source_map: Dict[str, Any]) -> Dict[str, Any]:
    """Distribution of sources across credibility tiers, plus domain diversity.

    An average credibility hides its own shape: 0.5 could be "half primary, half
    junk" or "everything mediocre". Those need different fixes.
    """
    counts = {name: 0 for name, _ in _TIERS}
    domains: set[str] = set()
    for v in source_map.values():
        score = float(v.get("credibility_score", 0.0) or 0.0)
        domains.add(_domain_of(v.get("url", "")))
        for name, floor in _TIERS:
            if score >= floor:
                counts[name] += 1
                break
    total = len(source_map)

    # Relevance is scored separately from credibility, because they are
    # independent: a peer-reviewed paper on the wrong subject is highly credible
    # and useless. Sources predating this signal have no score and are excluded
    # rather than counted as zero.
    rel = [v.get("relevance_score") for v in source_map.values()]
    rel = [r for r in rel if r is not None]

    return {
        "counts": counts,
        "total": total,
        "distinct_domains": len(domains - {""}),
        # 1.0 = every source from a different site; low values mean one site dominates.
        "domain_diversity": round(len(domains - {""}) / total, 3) if total else 0.0,
        "authoritative_share": round((counts["primary"] + counts["major_news"]) / total, 3) if total else 0.0,
        "avg_relevance": round(sum(rel) / len(rel), 3) if rel else 0.0,
        "off_topic_share": round(sum(1 for r in rel if r < OFF_TOPIC_RELEVANCE) / len(rel), 3) if rel else 0.0,
        "relevance_scored": len(rel),
    }


def funnel(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Information retained at each pipeline stage, with per-stage yield.

    Reading it: a low `grounding_survival` means the Verifier is rejecting most
    extracted claims (extraction quality, or sources without quotable prose). A
    low `corroboration_rate` means sources are not overlapping — usually too few
    or too homogeneous.

    Note on scope: `verified_facts` ACCUMULATES across research loop iterations
    (it carries an `operator.add` reducer), while `structured_evidence` is
    overwritten by each Refiner pass. So `facts_extracted_last_pass` covers only
    the final iteration and must NOT be compared against the verified counts —
    doing so reads as "more facts survived than were extracted". The ratios
    below only compare fields of the same scope.
    """
    sources = len(final_state.get("source_map", {}) or {})
    extracted_last = len(final_state.get("structured_evidence", []) or [])
    verified = final_state.get("verified_facts", []) or []
    supported = [f for f in verified if f.get("support_level") in _SUPPORTED]
    corroborated = [f for f in supported if (f.get("corroboration_count") or 1) >= 2]

    def _ratio(a: int, b: int) -> float:
        return round(a / b, 3) if b else 0.0

    return {
        "sources": sources,
        "facts_extracted_last_pass": extracted_last,
        "facts_verified": len(verified),
        "facts_supported": len(supported),
        "facts_corroborated": len(corroborated),
        "contradictions": len(final_state.get("contradictions", []) or []),
        "consensus_findings": len(final_state.get("consensus_findings", []) or []),
        # Yields — same-scope comparisons only (see the note above).
        "facts_per_source": _ratio(len(supported), sources),
        "grounding_survival": _ratio(len(supported), len(verified)),
        "corroboration_rate": _ratio(len(corroborated), len(supported)),
    }


def efficiency(metrics: Optional[Dict[str, Any]], fnl: Dict[str, Any], latency_s: float) -> Dict[str, Any]:
    """Unit economics of a single case.

    `tokens_per_supported_fact` is the number to watch: it is what a unit of
    trustworthy output actually costs.

    Failures split two ways, because they cost very different amounts. A
    *rejected* attempt (429, auth, 5xx) never generated, so it is free and
    signals a quota or availability problem. A *billed* failure generated
    output that was then unusable, so it is paid for twice and signals a
    schema or prompt problem.
    """
    m = metrics or {}
    calls = m.get("llm_calls", 0)
    errors = m.get("llm_errors", 0)
    # Older result files predate the split; treat their failures as billed,
    # which is how they were already being reported.
    rejected = m.get("llm_rejected", 0)
    billed_failures = m.get("llm_billed_failures", errors - rejected)
    total_tok = m.get("total_tokens", 0)
    supported = fnl.get("facts_supported", 0)

    def _per_fact(v: float) -> float:
        return round(v / supported, 1) if supported else 0.0

    # Failed attempts report no usage, so price them at the average successful
    # call. Only billed failures are counted — rejected ones cost nothing.
    ok_calls = calls - errors
    avg_call_tokens = (total_tok / ok_calls) if ok_calls > 0 else 0.0

    return {
        "llm_calls": calls,
        "failovers": errors,
        "failover_rate": round(errors / calls, 3) if calls else 0.0,
        "rejected_attempts": rejected,
        "rejected_rate": round(rejected / calls, 3) if calls else 0.0,
        "billed_failures": billed_failures,
        "billed_failure_rate": round(billed_failures / calls, 3) if calls else 0.0,
        "wasted_tokens": int(billed_failures * avg_call_tokens),
        "input_tokens": m.get("input_tokens", 0),
        "output_tokens": m.get("output_tokens", 0),
        "total_tokens": total_tok,
        "latency_s": round(latency_s, 1),
        "tokens_per_supported_fact": _per_fact(total_tok),
        "calls_per_supported_fact": _per_fact(calls),
        "seconds_per_supported_fact": _per_fact(latency_s),
    }


def _backfill_legacy_efficiency(result: Dict[str, Any]) -> Dict[str, Any]:
    """Fill billed/rejected fields on a result written before the split existed.

    `--report-only` reads the persisted `efficiency` dict straight off disk
    rather than recomputing it, so an older result file has no
    `billed_failure_rate`/`rejected_rate` keys at all. Left alone, `_avg()`'s
    `.get(sub, 0)` would silently read that as 0% failure. Triggers on the key
    being absent, not falsy, so a genuine 0.0 is never touched.
    """
    eff = result.get("efficiency")
    if not isinstance(eff, dict) or "billed_failure_rate" in eff:
        return result
    eff = dict(eff)  # don't mutate the caller's dict
    calls = eff.get("llm_calls", 0)
    failovers = eff.get("failovers", 0)
    eff["billed_failures"] = failovers
    eff["billed_failure_rate"] = eff.get("failover_rate", 0.0)
    eff["rejected_attempts"] = 0
    eff["rejected_rate"] = 0.0
    eff.setdefault("wasted_tokens", 0)
    result = dict(result)
    result["efficiency"] = eff
    return result


def node_costs(node_deltas: Dict[str, Dict[str, int]], node_seconds: Dict[str, float]) -> List[Dict[str, Any]]:
    """Per-node cost table, sorted by token spend.

    Node-level tuning needs this: a slow node and an expensive node are often
    not the same node.
    """
    nodes = set(node_deltas) | set(node_seconds)
    rows = []
    for n in nodes:
        d = node_deltas.get(n, {})
        rows.append({
            "node": n,
            "llm_calls": d.get("llm_calls", 0),
            "failovers": d.get("llm_errors", 0),
            "tokens": d.get("total_tokens", 0),
            "seconds": round(node_seconds.get(n, 0.0), 1),
        })
    return sorted(rows, key=lambda r: (-r["tokens"], -r["seconds"]))


def backend_rollup(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Retrieval-backend health summed across the suite.

    A backend that answers every call but returns nothing is the failure this
    exists to surface: the pipeline silently falls through to whatever else
    responded — possibly the wrong corpus — while still reporting high
    credibility, because credibility and relevance are different properties.
    """
    agg: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for name, h in (r.get("backend_health") or {}).items():
            a = agg.setdefault(name, {"backend": name, "calls": 0, "failures": 0,
                                      "empty_calls": 0, "results": 0, "cases_degraded": 0})
            a["calls"] += h.get("calls", 0)
            a["failures"] += h.get("failures", 0)
            a["empty_calls"] += h.get("empty_calls", 0)
            a["results"] += h.get("results", 0)
            a["cases_degraded"] += 1 if h.get("degraded") else 0

    for a in agg.values():
        a["results_per_call"] = round(a["results"] / a["calls"], 2) if a["calls"] else 0.0
        a["degraded"] = a["results"] == 0 and a["calls"] > 0
    return sorted(agg.values(), key=lambda a: (not a["degraded"], -a["calls"]))


def domain_rollup(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate case results by domain — the per-domain view the bench exists for."""
    results = [_backfill_legacy_efficiency(r) for r in results]
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_domain.setdefault(r.get("domain", "unknown"), []).append(r)

    rows = []
    for domain, rs in sorted(by_domain.items()):
        ok = [r for r in rs if not r.get("error")]
        n = len(ok)

        def _avg(path: str, sub: str) -> float:
            vals = [r[path].get(sub, 0) or 0 for r in ok if isinstance(r.get(path), dict)]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        rows.append({
            "domain": domain,
            "cases": len(rs),
            "passed": sum(1 for r in rs if r.get("passed")),
            "errors": sum(1 for r in rs if r.get("error")),
            "avg_credibility": round(sum(r.get("avg_credibility", 0) for r in ok) / n, 3) if n else 0.0,
            "authoritative_share": _avg("tiers", "authoritative_share"),
            "domain_diversity": _avg("tiers", "domain_diversity"),
            "avg_relevance": _avg("tiers", "avg_relevance"),
            "off_topic_share": _avg("tiers", "off_topic_share"),
            "facts_supported": _avg("funnel", "facts_supported"),
            "grounding_survival": _avg("funnel", "grounding_survival"),
            "corroboration_rate": _avg("funnel", "corroboration_rate"),
            "tokens_per_fact": _avg("efficiency", "tokens_per_supported_fact"),
            "failover_rate": _avg("efficiency", "failover_rate"),
            "billed_failure_rate": _avg("efficiency", "billed_failure_rate"),
            "rejected_rate": _avg("efficiency", "rejected_rate"),
            "latency_s": _avg("efficiency", "latency_s"),
        })
    return rows


# ── Diagnosis ────────────────────────────────────────────────────────────────
# Thresholds are deliberately explicit rather than tuned: each one encodes a
# claim about what "good" means for this pipeline, and should be argued with.

_THRESHOLDS = {
    "authoritative_share": 0.4,     # <40% primary/major-news sources = weak retrieval
    "grounding_survival": 0.5,      # <50% of extracted facts surviving = extraction problem
    "corroboration_rate": 0.3,      # <30% corroborated = too few/too homogeneous sources
    "domain_diversity": 0.7,        # <0.7 = one site dominating the evidence
    "off_topic_share": 0.25,        # >25% off-topic = retrieval reaching the wrong corpus
    "billed_failure_rate": 0.1,     # >10% generated-then-unusable = schema/prompt problem
    "rejected_rate": 0.1,           # >10% refused unbilled = quota/availability ceiling
    "tokens_per_fact": 4000,        # >4k tokens per trustworthy fact = optimisation worth doing
}


def diagnose(rollup: List[Dict[str, Any]], overall: Dict[str, Any],
             backends: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Turn measurements into ranked, actionable findings.

    Each finding names the metric, the observed value, the threshold it breached,
    and which domains are worst — so the report says what to fix, not just what
    the numbers were.
    """
    findings: List[Dict[str, Any]] = []

    # Checked first: a dead backend explains downstream symptoms, so reporting
    # it alongside them rather than above them buries the cause.
    for b in (backends or []):
        if b.get("degraded"):
            findings.append({
                "severity": "high", "area": "backend",
                "finding": f"`{b['backend']}` was called {b['calls']}x and returned 0 results all suite.",
                "worst": "all domains routed to it",
                "action": ("Sources silently came from other backends, which may be the wrong "
                           "corpus for those queries. Check the backend's auth/rate limits."),
            })

    def _worst(metric: str, higher_is_better: bool = True) -> str:
        if not rollup:
            return "—"
        s = sorted(rollup, key=lambda r: r.get(metric, 0), reverse=not higher_is_better)
        return ", ".join(f"{r['domain']} ({r.get(metric, 0)})" for r in s[:3])

    if overall.get("off_topic_share", 0.0) > _THRESHOLDS["off_topic_share"]:
        findings.append({
            "severity": "high", "area": "relevance",
            "finding": (f"{overall['off_topic_share']:.0%} of sources are off-topic "
                        f"(avg query similarity {overall.get('avg_relevance', 0):.2f})."),
            "worst": _worst("off_topic_share", higher_is_better=False),
            "action": ("Sources are credible but not about the question — retrieval is reaching "
                       "the wrong corpus. Check backend routing for those domains before "
                       "reading any credibility score as quality."),
        })
    if overall.get("authoritative_share", 1.0) < _THRESHOLDS["authoritative_share"]:
        findings.append({
            "severity": "high", "area": "retrieval",
            "finding": f"Only {overall['authoritative_share']:.0%} of sources are primary or major-news tier.",
            "worst": _worst("authoritative_share"),
            "action": "Route more queries to primary-source backends (EDGAR/academic/institutional) and extend the domain tier tables.",
        })
    if overall.get("grounding_survival", 1.0) < _THRESHOLDS["grounding_survival"]:
        findings.append({
            "severity": "high", "area": "extraction",
            "finding": f"Only {overall['grounding_survival']:.0%} of extracted facts survive verbatim-quote grounding.",
            "worst": _worst("grounding_survival"),
            "action": "The Refiner is producing claims it cannot quote. Tighten its schema and require excerpts copied verbatim from the document.",
        })
    if overall.get("corroboration_rate", 1.0) < _THRESHOLDS["corroboration_rate"]:
        findings.append({
            "severity": "medium", "area": "coverage",
            "finding": f"Only {overall['corroboration_rate']:.0%} of supported facts are corroborated by 2+ sources.",
            "worst": _worst("corroboration_rate"),
            "action": "Gather more sources per sub-query, or widen retrieval — most claims currently rest on a single source.",
        })
    if overall.get("domain_diversity", 1.0) < _THRESHOLDS["domain_diversity"]:
        findings.append({
            "severity": "medium", "area": "retrieval",
            "finding": f"Domain diversity is {overall['domain_diversity']}; evidence is concentrated in few sites.",
            "worst": _worst("domain_diversity"),
            "action": "Review the per-domain source cap and broaden query variety.",
        })
    # Two distinct faults, so two findings: a rejected attempt costs nothing and
    # points at capacity; a billed failure costs tokens and points at the prompt.
    if overall.get("billed_failure_rate", 0.0) > _THRESHOLDS["billed_failure_rate"]:
        findings.append({
            "severity": "high", "area": "reliability",
            "finding": (
                f"{overall['billed_failure_rate']:.0%} of LLM attempts generated output "
                f"that was then unusable (~{overall.get('wasted_tokens', 0):,} tokens)."
            ),
            "worst": _worst("billed_failure_rate", higher_is_better=False),
            "action": "These prompts are paid for twice. Fix the failing schema/payload rather than relying on the ladder.",
        })
    if overall.get("rejected_rate", 0.0) > _THRESHOLDS["rejected_rate"]:
        findings.append({
            "severity": "high", "area": "capacity",
            "finding": (
                f"{overall['rejected_rate']:.0%} of LLM attempts were refused before generating "
                f"(rate limit, auth or provider outage). These cost no tokens."
            ),
            "worst": _worst("rejected_rate", higher_is_better=False),
            "action": (
                "A quota or availability ceiling, not a code defect. Results are skewed toward "
                "whichever provider absorbed the load — check the by-model split before trusting "
                "this run's quality numbers."
            ),
        })
    if overall.get("tokens_per_fact", 0) > _THRESHOLDS["tokens_per_fact"]:
        findings.append({
            "severity": "medium", "area": "efficiency",
            "finding": f"{overall['tokens_per_fact']:,.0f} tokens per supported fact.",
            "worst": _worst("tokens_per_fact", higher_is_better=False),
            "action": "Token optimisation is justified. Target the highest-token node in the node-cost table first.",
        })

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings, key=lambda f: order.get(f["severity"], 3))


def overall_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Suite-wide averages used by `diagnose` and the report header."""
    results = [_backfill_legacy_efficiency(r) for r in results]
    ok = [r for r in results if not r.get("error")]
    n = len(ok)
    if not n:
        return {}

    def _avg(path: str, sub: str) -> float:
        vals = [r[path].get(sub, 0) or 0 for r in ok if isinstance(r.get(path), dict)]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "cases": len(results),
        "passed": sum(1 for r in results if r.get("passed")),
        "errors": sum(1 for r in results if r.get("error")),
        "avg_credibility": round(sum(r.get("avg_credibility", 0) for r in ok) / n, 3),
        "authoritative_share": _avg("tiers", "authoritative_share"),
        "domain_diversity": _avg("tiers", "domain_diversity"),
        "avg_relevance": _avg("tiers", "avg_relevance"),
        "off_topic_share": _avg("tiers", "off_topic_share"),
        "grounding_survival": _avg("funnel", "grounding_survival"),
        "corroboration_rate": _avg("funnel", "corroboration_rate"),
        "failover_rate": _avg("efficiency", "failover_rate"),
        "billed_failure_rate": _avg("efficiency", "billed_failure_rate"),
        "rejected_rate": _avg("efficiency", "rejected_rate"),
        "wasted_tokens": sum((r.get("efficiency") or {}).get("wasted_tokens", 0) for r in ok),
        "tokens_per_fact": _avg("efficiency", "tokens_per_supported_fact"),
        "total_tokens": sum((r.get("efficiency") or {}).get("total_tokens", 0) for r in ok),
        "total_calls": sum((r.get("efficiency") or {}).get("llm_calls", 0) for r in ok),
        "total_latency_s": round(sum((r.get("efficiency") or {}).get("latency_s", 0) for r in ok), 1),
    }
