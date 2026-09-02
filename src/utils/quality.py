"""
Project Argus - Run quality scoring (pure function, no I/O, no LLM).

Called twice in one run:
  - by the Verifier, after each research iteration, so the router can act on
    source quality (see `route_after_critic`) before the loop exits.
  - by the Consensus node, once at the end, with the real `contradiction_count`
    (contradictions require the cross-source clustering only Consensus does).
"""

from typing import Any, Dict, List

_SUPPORTED_LEVELS = ("SUPPORTED", "PARTIALLY_SUPPORTED")


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
