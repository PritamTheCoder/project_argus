"""
Project Argus - Per-domain research benchmark.

Deeper and slower than `eval/` (which is the fast regression check). This runs
the real graph across every research domain the system claims to serve, and
reports where quality actually comes from and where it leaks.

Designed to survive a long run:
  - each case is isolated; one failure never aborts the suite
  - each result is written to disk the moment it finishes, so a crash or a
    Ctrl-C keeps everything already earned
  - `--resume` skips cases already on disk for that run id
  - `--pace` sleeps between cases to stay under free-tier RPM/TPD limits

Usage:
    python -m bench.run_bench                       # full suite
    python -m bench.run_bench --domain finance_public
    python -m bench.run_bench --only fin_private_valuation
    python -m bench.run_bench --resume 20260902_170000
    python -m bench.run_bench --report-only 20260902_170000

Outputs:
    bench/results/<run_id>/<case_id>.json   raw per-case data
    bench/report/<run_id>.md                synthesised report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench.analysis import (
    _backfill_legacy_efficiency, backend_rollup, diagnose, domain_rollup,
    efficiency, funnel, node_costs, overall_summary, source_tiers,
)
from eval.metrics import RunMetrics
from eval.scoring import score_case

logger = logging.getLogger("argus.bench")

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = BENCH_DIR / "cases.json"
RESULTS_DIR = BENCH_DIR / "results"
REPORT_DIR = BENCH_DIR / "report"

# A single case that hangs must not consume the whole run's budget.
CASE_TIMEOUT_S = 900


def _load_cases(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(m: RunMetrics) -> Dict[str, int]:
    s = m.summary()
    return {
        "llm_calls": s["llm_calls"],
        "llm_errors": s["llm_errors"],
        "total_tokens": s["total_tokens"],
    }


def _delta(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    return {k: after[k] - before.get(k, 0) for k in after}


async def _run_one(case: dict) -> tuple[dict, RunMetrics, Dict[str, Dict[str, int]]]:
    """Run the graph for one case.

    Returns the final state, the usage metrics, and per-node usage deltas.
    The deltas come from snapshotting the shared counters at each node boundary —
    the graph runs nodes sequentially, so a delta is attributable to the node
    that just finished, which is what makes node-level cost attributable.
    """
    from src.graph.kg import kg_store
    from src.graph.pipeline import astream_research

    metrics = RunMetrics()
    kg_store.clear_scratchpad()

    final_state: dict = {}
    node_deltas: Dict[str, Dict[str, int]] = {}
    prev = _snapshot(metrics)

    async for node_name, state_update in astream_research(case["query"], callbacks=[metrics]):
        if node_name == "__final__":
            final_state = state_update
            continue
        now = _snapshot(metrics)
        d = _delta(now, prev)
        prev = now
        acc = node_deltas.setdefault(node_name, {"llm_calls": 0, "llm_errors": 0, "total_tokens": 0})
        for k, v in d.items():  # a looped node is visited more than once
            acc[k] += v

    return final_state, metrics, node_deltas


def _analyse(case: dict, final_state: dict, metrics: RunMetrics,
             node_deltas: Dict[str, Dict[str, int]], latency_s: float) -> dict:
    """Score + analyse one finished case into the record written to disk."""
    result = score_case(case, final_state)
    result["domain"] = case.get("domain", "unknown")
    result["query"] = case.get("query", "")
    result["notes"] = case.get("notes", "")
    result["error"] = None

    source_map = final_state.get("source_map", {}) or {}
    fnl = funnel(final_state)
    result["tiers"] = source_tiers(source_map)
    result["funnel"] = fnl
    result["efficiency"] = efficiency(metrics.summary(), fnl, latency_s)
    result["node_costs"] = node_costs(node_deltas, final_state.get("node_seconds", {}) or {})
    result["by_model"] = metrics.summary().get("by_model", {})
    result["backend_health"] = final_state.get("backend_health", {}) or {}
    result["sources"] = [
        {"url": v.get("url", ""), "credibility": v.get("credibility_score", 0),
         "type": v.get("source_type", ""), "relevance": v.get("relevance_score")}
        for v in source_map.values()
    ]
    return result


async def run_bench(cases: List[dict], run_dir: Path, pace_s: float = 0.0,
                    resume: bool = False) -> List[dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []

    for i, case in enumerate(cases, start=1):
        cid = case.get("id", "?")
        out_path = run_dir / f"{cid}.json"

        if resume and out_path.exists():
            print(f"[{i}/{len(cases)}] {cid}: already done, skipping", flush=True)
            results.append(json.loads(out_path.read_text(encoding="utf-8")))
            continue

        print(f"\n[{i}/{len(cases)}] === {cid}  ({case.get('domain', '?')}) ===", flush=True)
        t0 = time.perf_counter()
        try:
            final_state, metrics, node_deltas = await asyncio.wait_for(
                _run_one(case), timeout=CASE_TIMEOUT_S
            )
            result = _analyse(case, final_state, metrics, node_deltas, time.perf_counter() - t0)
        except asyncio.TimeoutError:
            logger.error("Case %s exceeded %ss", cid, CASE_TIMEOUT_S)
            result = {"id": cid, "domain": case.get("domain", "unknown"), "passed": False,
                      "error": f"timeout after {CASE_TIMEOUT_S}s"}
        except Exception as e:  # one bad case must never abort the suite
            logger.exception("Case %s failed", cid)
            result = {"id": cid, "domain": case.get("domain", "unknown"), "passed": False,
                      "error": f"{type(e).__name__}: {e}"}
        result.setdefault("latency_s", round(time.perf_counter() - t0, 1))

        # Written immediately: a later crash never costs work already done.
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(result)
        _print_case(result)

        if pace_s and i < len(cases):
            print(f"    pacing {pace_s:.0f}s before next case...", flush=True)
            await asyncio.sleep(pace_s)

    return results


def _print_case(r: dict) -> None:
    if r.get("error"):
        print(f"    [ERROR] {r['error']}", flush=True)
        return
    e, f, t = r.get("efficiency", {}), r.get("funnel", {}), r.get("tiers", {})
    print(
        f"    [{'PASS' if r['passed'] else 'FAIL'}] cred={r.get('avg_credibility')} "
        f"authoritative={t.get('authoritative_share')} "
        f"facts={f.get('facts_supported')} corrob={f.get('corroboration_rate')}",
        flush=True,
    )
    print(
        f"           {e.get('total_tokens', 0):,} tok "
        f"({e.get('tokens_per_supported_fact', 0):,.0f}/fact) "
        f"{e.get('llm_calls', 0)} calls, {e.get('billed_failures', 0)} billed-fail, "
        f"{e.get('rejected_attempts', 0)} rejected, "
        f"{e.get('latency_s', 0)}s",
        flush=True,
    )


# ── Report ───────────────────────────────────────────────────────────────────

def _md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def build_report(results: List[dict], run_id: str) -> str:
    # Backfilled once so the per-case appendix reads the same billed/rejected
    # split as the headline and domain tables.
    results = [_backfill_legacy_efficiency(r) for r in results]
    overall = overall_summary(results)
    rollup = domain_rollup(results)
    backends = backend_rollup(results)
    findings = diagnose(rollup, overall, backends)

    L = [f"# Argus Research Benchmark — {run_id}", ""]

    if not overall:
        return "\n".join(L + ["No case completed successfully."])

    L += [
        f"**{overall['passed']}/{overall['cases']} cases passed**"
        + (f" · {overall['errors']} errored" if overall["errors"] else ""),
        "",
        "## Headline",
        "",
    ]
    L += _md_table(
        ["Metric", "Value", "Reading"],
        [
            ["Avg source credibility", overall["avg_credibility"], "0–1; tier-weighted quality of evidence"],
            ["Avg query relevance", overall.get("avg_relevance", 0), "0–1; is the source actually about the question"],
            ["Off-topic share", f"{overall.get('off_topic_share', 0):.0%}", "credible sources on the wrong subject"],
            ["Authoritative share", f"{overall['authoritative_share']:.0%}", "primary + major-news sources"],
            ["Domain diversity", overall["domain_diversity"], "1.0 = every source a different site"],
            ["Grounding survival", f"{overall['grounding_survival']:.0%}", "extracted facts that survive verbatim-quote checks"],
            ["Corroboration rate", f"{overall['corroboration_rate']:.0%}", "supported facts backed by 2+ sources"],
            ["Billed failures", f"{overall.get('billed_failure_rate', 0):.0%}", "generated output that was unusable — paid for twice"],
            ["Rejected attempts", f"{overall.get('rejected_rate', 0):.0%}", "refused before generating (quota/outage) — cost nothing"],
            ["Tokens per supported fact", f"{overall['tokens_per_fact']:,.0f}", "unit cost of trustworthy output"],
            ["Total", f"{overall['total_tokens']:,} tok / {overall['total_calls']} calls / {overall['total_latency_s']}s", "whole suite"],
        ],
    )

    # ── Findings ────────────────────────────────────────────────────────────
    L += ["", "## What's missing", ""]
    if findings:
        for f in findings:
            L += [
                f"### [{f['severity'].upper()}] {f['area']} — {f['finding']}",
                "",
                f"- **Worst domains:** {f['worst']}",
                f"- **Action:** {f['action']}",
                "",
            ]
    else:
        L += ["No threshold breaches. Every measured dimension is within target.", ""]

    # ── Backend health ──────────────────────────────────────────────────────
    if backends:
        L += ["", "## Retrieval backend health", "",
              "A backend answering every call while returning nothing is the "
              "dangerous case: the pipeline falls through to whatever else "
              "responded, which may be the wrong corpus.", ""]
        L += _md_table(
            ["Backend", "Calls", "Results", "Results/call", "Empty calls", "Status"],
            [[b["backend"], b["calls"], b["results"], b["results_per_call"],
              b["empty_calls"], "**DEGRADED**" if b["degraded"] else "ok"]
             for b in backends],
        )

    # ── Per domain ──────────────────────────────────────────────────────────
    L += ["", "## Per-domain quality", ""]
    L += _md_table(
        ["Domain", "Pass", "Cred", "Relevance", "Off-topic", "Authoritative", "Diversity", "Facts", "Grounding", "Corrob"],
        [[r["domain"], f"{r['passed']}/{r['cases']}", r["avg_credibility"],
          r.get("avg_relevance", 0), f"{r.get('off_topic_share', 0):.0%}",
          f"{r['authoritative_share']:.0%}", r["domain_diversity"], r["facts_supported"],
          f"{r['grounding_survival']:.0%}", f"{r['corroboration_rate']:.0%}"] for r in rollup],
    )

    L += ["", "## Per-domain cost", ""]
    L += _md_table(
        ["Domain", "Tokens/fact", "Billed fail", "Rejected", "Latency (s)"],
        [[r["domain"], f"{r['tokens_per_fact']:,.0f}", f"{r.get('billed_failure_rate', 0):.0%}",
          f"{r.get('rejected_rate', 0):.0%}", r["latency_s"]]
         for r in rollup],
    )

    # ── Node costs, summed across the suite ─────────────────────────────────
    node_tot: Dict[str, Dict[str, float]] = {}
    for r in results:
        for row in r.get("node_costs", []) or []:
            a = node_tot.setdefault(row["node"], {"tokens": 0, "llm_calls": 0, "failovers": 0, "seconds": 0.0})
            a["tokens"] += row["tokens"]; a["llm_calls"] += row["llm_calls"]
            a["failovers"] += row["failovers"]; a["seconds"] += row["seconds"]
    if node_tot:
        grand = sum(a["tokens"] for a in node_tot.values()) or 1
        L += [
            "", "## Where the cost is (all cases)", "",
            "Optimise from the top of this table. A slow node and an expensive "
            "node are frequently not the same node.", "",
        ]
        L += _md_table(
            ["Node", "Tokens", "% of tokens", "Calls", "Failover", "Seconds"],
            [[n, f"{a['tokens']:,}", f"{a['tokens'] / grand:.0%}", a["llm_calls"],
              a["failovers"], f"{a['seconds']:.0f}"]
             for n, a in sorted(node_tot.items(), key=lambda kv: -kv[1]["tokens"])],
        )

    # ── Verdict on optimisation ─────────────────────────────────────────────
    L += ["", "## Do we need token / call optimisation?", ""]
    tpf = overall["tokens_per_fact"]
    bfr = overall.get("billed_failure_rate", 0.0)
    rr = overall.get("rejected_rate", 0.0)
    waste = overall.get("wasted_tokens", 0)

    if bfr > 0.1:
        L += [f"**Yes — but fix reliability first.** {bfr:.0%} of attempts generated output that "
              f"was then unusable, costing roughly **{waste:,} tokens**. That is the cheapest "
              "reduction available and needs no quality trade-off.", ""]
    if rr > 0.1:
        L += [f"**Read this run with care.** {rr:.0%} of attempts were refused before generating "
              "(rate limit, auth or outage). Those cost no tokens, but the work shifted onto "
              "whichever provider still had capacity — so the quality and latency numbers "
              "describe that provider, not the intended configuration. Check the by-model split.", ""]
    if tpf > 4000:
        L += [f"**Token optimisation is justified**: {tpf:,.0f} tokens per supported fact. "
              "Target the top row of the node-cost table.", ""]
    elif bfr <= 0.1 and rr <= 0.1:
        L += [f"**Not a priority.** {tpf:,.0f} tokens per supported fact with a "
              f"{bfr:.0%} billed-failure rate is within target; spend effort on retrieval "
              "quality instead.", ""]

    # ── Per-case appendix ───────────────────────────────────────────────────
    L += ["", "## Per-case detail", ""]
    for r in results:
        L.append(f"### {r['id']} ({r.get('domain', '?')})")
        if r.get("error"):
            L += ["", f"**ERROR:** {r['error']}", ""]
            continue
        t, f, e = r.get("tiers", {}), r.get("funnel", {}), r.get("efficiency", {})
        L += [
            "",
            f"- **{'PASS' if r['passed'] else 'FAIL'}** · coverage {r.get('coverage_score')} · "
            f"citations {'ok' if r.get('citation_ok') else 'BROKEN'} · {r.get('report_len', 0):,} chars",
            f"- Sources: {t.get('total', 0)} across {t.get('distinct_domains', 0)} domains — {t.get('counts', {})}",
            f"- Funnel: {f.get('sources')} sources → {f.get('facts_verified')} verified → "
            f"{f.get('facts_supported')} supported → {f.get('facts_corroborated')} corroborated "
            f"(last Refiner pass extracted "
            f"{f.get('facts_extracted_last_pass', f.get('facts_extracted', 'n/a'))}; "
            f"verified counts accumulate across loop iterations)",
            f"- Cost: {e.get('total_tokens', 0):,} tok, {e.get('llm_calls', 0)} calls, "
            f"{e.get('billed_failures', 0)} billed-fail, "
            f"{e.get('rejected_attempts', 0)} rejected, {e.get('latency_s', 0)}s",
        ]
        if r.get("coverage_missing"):
            L.append(f"- Missing keywords: {r['coverage_missing']}")
        if r.get("notes"):
            L.append(f"- _Case note: {r['notes']}_")
        L.append("")

    return "\n".join(L)


def _write_report(results: List[dict], run_id: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{run_id}.md"
    out.write_text(build_report(results, run_id), encoding="utf-8")
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-20s | %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Run the Argus per-domain research benchmark.")
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--only", type=str, help="Run only this case id.")
    p.add_argument("--domain", type=str, help="Run only cases in this domain.")
    p.add_argument("--pace", type=float, default=0.0,
                   help="Seconds to sleep between cases (free-tier RPM relief).")
    p.add_argument("--resume", type=str, metavar="RUN_ID",
                   help="Continue a previous run id, skipping cases already on disk.")
    p.add_argument("--report-only", type=str, metavar="RUN_ID",
                   help="Rebuild the report from existing results without running anything.")
    args = p.parse_args()

    if args.report_only:
        run_dir = RESULTS_DIR / args.report_only
        if not run_dir.is_dir():
            print(f"No results directory: {run_dir}", file=sys.stderr)
            return 2
        results = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(run_dir.glob("*.json"))]
        print(f"Report: {_write_report(results, args.report_only)}")
        return 0

    cases = _load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c.get("id") == args.only]
    if args.domain:
        cases = [c for c in cases if c.get("domain") == args.domain]
    if not cases:
        print("No matching cases.", file=sys.stderr)
        return 2

    run_id = args.resume or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_id
    print(f"Run {run_id} — {len(cases)} case(s) → {run_dir}", flush=True)

    results = asyncio.run(run_bench(cases, run_dir, pace_s=args.pace, resume=bool(args.resume)))

    report = _write_report(results, run_id)
    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{'=' * 60}")
    print(f"Bench complete: {passed}/{len(results)} passed")
    print(f"Results: {run_dir}")
    print(f"Report:  {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
