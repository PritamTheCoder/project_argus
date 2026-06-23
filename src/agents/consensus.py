"""
Project Argus - Consensus & Contradiction Agent

Runs once on the final verified evidence, between the Critic loop and the Writer.

It groups semantically-similar SUPPORTED claims, then asks an LLM to judge each
multi-source group as either:
  - CONSENSUS:      independent sources agree → a single, higher-trust finding
  - CONTRADICTION:  sources disagree on the same point → surfaced explicitly

This is what turns isolated facts ("Source A: 400 Wh/kg") into analyst-grade
statements ("3 independent sources report 380-420 Wh/kg") and prevents two
conflicting SUPPORTED facts from silently both reaching the report.

It also computes a per-run quality score from the final evidence.

Clustering and scoring are pure functions (unit-tested without an LLM).
"""

import logging
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from src.schema.state import AgentState
from src.config import CRITIC_MODEL, CRITIC_PROVIDER
from src.utils.llm_factory import get_llm
from src.utils.grounding import _cosine, _fact_source_key

logger = logging.getLogger(__name__)

_SUPPORTED_LEVELS = ("SUPPORTED", "PARTIALLY_SUPPORTED")

# Only clusters with at least this many distinct sources are sent to the LLM
# for consensus/contradiction judgement (a single source can't corroborate).
_MIN_SOURCES_FOR_JUDGEMENT = 2
# Cap how many clusters we send to the LLM to bound cost.
_MAX_CLUSTERS_JUDGED = 25


# ── Pure helpers ─────────────────────────────────────────────────────────────

def cluster_facts(facts: List[Dict[str, Any]], sim_threshold: float = 0.85) -> List[List[Dict[str, Any]]]:
    """
    Greedily cluster facts by cosine similarity of their ``embedding``.

    Each fact joins the first existing cluster whose seed it is similar enough
    to; otherwise it seeds a new cluster. Facts without an embedding become
    their own singleton cluster. Deterministic given input order.
    """
    clusters: List[List[Dict[str, Any]]] = []
    seeds: List[List[float]] = []

    for fact in facts:
        emb = fact.get("embedding")
        if emb is None:
            clusters.append([fact])
            seeds.append(None)
            continue

        placed = False
        for ci, seed in enumerate(seeds):
            if seed is not None and _cosine(emb, seed) >= sim_threshold:
                clusters[ci].append(fact)
                placed = True
                break
        if not placed:
            clusters.append([fact])
            seeds.append(emb)

    return clusters


def distinct_sources(cluster: List[Dict[str, Any]]) -> List[str]:
    """Distinct, non-empty source keys within a cluster (preserves first-seen order)."""
    seen = []
    for f in cluster:
        key = _fact_source_key(f)
        if key and key not in seen:
            seen.append(key)
    return seen


def compute_quality_score(
    verified_facts: List[Dict[str, Any]],
    source_map: Dict[str, Any],
    contradiction_count: int,
) -> Dict[str, Any]:
    """Deterministic quality summary of a finished run."""
    supported = [f for f in verified_facts if f.get("support_level") in _SUPPORTED_LEVELS]
    n = len(supported)

    creds = [float(f.get("credibility_score", 0.4) or 0.0) for f in supported]
    avg_cred = round(sum(creds) / len(creds), 3) if creds else 0.0

    sources = {f.get("source_url", "") for f in supported if f.get("source_url")}
    corroborated = sum(1 for f in supported if (f.get("corroboration_count") or 1) >= 2)

    return {
        "verified_fact_count": n,
        "avg_source_credibility": avg_cred,
        "distinct_source_count": len(sources),
        "corroborated_fact_count": corroborated,
        "single_source_fact_count": n - corroborated,
        "contradiction_count": contradiction_count,
    }


# ── LLM judgement schema ─────────────────────────────────────────────────────

class ClusterJudgment(BaseModel):
    index: int = Field(description="The cluster index from the input, to map the result back.")
    relationship: Literal["CONSENSUS", "CONTRADICTION", "UNRELATED"] = Field(
        description="CONSENSUS if the claims agree, CONTRADICTION if they conflict, UNRELATED if they aren't really about the same point."
    )
    statement: str = Field(
        description="One-sentence synthesis. For CONSENSUS, state the agreed finding and the range if numeric. For CONTRADICTION, state what the sources disagree on."
    )


class ConsensusBatch(BaseModel):
    judgments: List[ClusterJudgment] = Field(default_factory=list)


def _build_cluster_prompt(candidates: List[List[Dict[str, Any]]]) -> str:
    blocks = []
    for idx, cluster in enumerate(candidates):
        lines = [f"Cluster {idx}:"]
        for f in cluster:
            src = f.get("source_url", "") or f.get("source_id", "")
            cred = f.get("credibility_score", "?")
            lines.append(f'  - "{f.get("claim", "")}" (source: {src}, credibility: {cred})')
        blocks.append("\n".join(lines))
    body = "\n\n".join(blocks)
    return (
        "You are a research analyst detecting agreement and conflict across sources.\n"
        "Each cluster below contains claims from DIFFERENT sources that are about a similar point.\n"
        "For EACH cluster decide the `relationship`:\n"
        "  - CONSENSUS: the sources broadly agree (synthesize the agreed finding; if numeric, give the range).\n"
        "  - CONTRADICTION: the sources give conflicting/incompatible values or conclusions.\n"
        "  - UNRELATED: the claims are not actually about the same specific point.\n"
        "Return one judgment per cluster, echoing its `index`.\n\n"
        f"{body}"
    )


# ── Node ─────────────────────────────────────────────────────────────────────

def consensus_node(state: AgentState) -> dict:
    """
    Detect consensus and contradictions across the final verified evidence and
    compute the run's quality score.
    """
    logger.info("Consensus: Analysing agreement/conflict across verified evidence...")

    verified_facts = state.get("verified_facts", []) or []
    source_map = state.get("source_map", {}) or {}

    supported = [f for f in verified_facts if f.get("support_level") in _SUPPORTED_LEVELS]

    consensus_findings: List[Dict[str, Any]] = []
    contradictions: List[Dict[str, Any]] = []

    if len(supported) >= 2:
        clusters = cluster_facts(supported)
        # Candidate clusters: multi-source groups worth judging.
        candidates = [
            c for c in clusters
            if len(distinct_sources(c)) >= _MIN_SOURCES_FOR_JUDGEMENT
        ][:_MAX_CLUSTERS_JUDGED]

        if candidates:
            try:
                llm = get_llm(CRITIC_MODEL, CRITIC_PROVIDER, temperature=0)
                structured = llm.with_structured_output(ConsensusBatch)
                result: ConsensusBatch = structured.invoke(_build_cluster_prompt(candidates))

                for judgment in result.judgments:
                    if not (0 <= judgment.index < len(candidates)):
                        continue
                    cluster = candidates[judgment.index]
                    entry = {
                        "statement": judgment.statement,
                        "claims": [f.get("claim", "") for f in cluster],
                        "sources": distinct_sources(cluster),
                        "source_count": len(distinct_sources(cluster)),
                    }
                    if judgment.relationship == "CONSENSUS":
                        consensus_findings.append(entry)
                    elif judgment.relationship == "CONTRADICTION":
                        contradictions.append(entry)
            except Exception as e:
                logger.error(f"Consensus: LLM judgement failed ({e}). Continuing without consensus/contradiction.")

    quality_score = compute_quality_score(verified_facts, source_map, len(contradictions))

    logger.info(
        f"Consensus: {len(consensus_findings)} consensus finding(s), "
        f"{len(contradictions)} contradiction(s). Quality: {quality_score}"
    )

    return {
        "consensus_findings": consensus_findings,
        "contradictions": contradictions,
        "quality_score": quality_score,
        "active_node": "consensus",
    }
