"""
Project Argus - Evidence grounding utilities (pure functions)

Two correctness guards used by the Verifier:

1. Verbatim-quote grounding (Chain-of-Verification): a claim may only be
   treated as supported if the verifier produced a `support_quote` that appears
   verbatim in the source excerpt. No quote → the claim is dropped to
   NOT_SUPPORTED. This stops a fluent-but-ungrounded excerpt from passing.

2. Cross-source corroboration: a claim supported by only a single source is
   flagged and its confidence capped, so the Writer can hedge single-source
   claims instead of asserting them as established fact.

Both are pure (no I/O, no LLM) so they are deterministic and unit-tested.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List

_WS_RE = re.compile(r"\s+")

# Support levels that count as "the source backs this claim".
_SUPPORTED_LEVELS = ("SUPPORTED", "PARTIALLY_SUPPORTED")


def normalize_text(s: str) -> str:
    """Lowercase and collapse whitespace for tolerant verbatim matching."""
    return _WS_RE.sub(" ", (s or "").lower()).strip()


def quote_is_grounded(quote: str, source_excerpt: str, min_len: int = 12) -> bool:
    """
    True if ``quote`` appears (case-/whitespace-insensitively) as a contiguous
    span of ``source_excerpt``. Quotes shorter than ``min_len`` characters are
    rejected as too weak to constitute evidence.
    """
    q = normalize_text(quote)
    if len(q) < min_len:
        return False
    return q in normalize_text(source_excerpt)


def apply_quote_grounding(facts: List[Dict[str, Any]], min_len: int = 12) -> List[Dict[str, Any]]:
    """
    Downgrade any SUPPORTED/PARTIALLY_SUPPORTED fact whose ``support_quote`` is
    not grounded in its ``source_excerpt``. Mutates and returns ``facts``.

    Adds ``grounding_failed`` (bool) to every supported fact for observability.
    """
    for f in facts:
        if f.get("support_level") not in _SUPPORTED_LEVELS:
            continue
        quote = f.get("support_quote", "")
        if quote_is_grounded(quote, f.get("source_excerpt", ""), min_len=min_len):
            f["grounding_failed"] = False
        else:
            f["support_level"] = "NOT_SUPPORTED"
            f["confidence"] = min(float(f.get("confidence", 0.0) or 0.0), 0.3)
            f["grounding_failed"] = True
            reason = f.get("reasoning", "") or ""
            f["reasoning"] = (reason + " [Dropped: support quote not found verbatim in source excerpt.]").strip()
    return facts


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _fact_source_key(fact: Dict[str, Any]) -> str:
    """Identity of the source a fact came from (url preferred, id as fallback)."""
    return fact.get("source_url", "") or fact.get("source_id", "")


def merged_source_keys(fact: Dict[str, Any]) -> List[str]:
    """
    Every distinct source backing a fact.

    The Verifier folds near-duplicate claims from other sources into the
    survivor's ``merged_sources``; those sources still corroborate the claim, so
    they must count here. Falls back to the fact's own source when it was never
    merged.
    """
    merged = fact.get("merged_sources")
    if merged:
        return [k for k in merged if k]
    key = _fact_source_key(fact)
    return [key] if key else []


def annotate_corroboration(
    facts: List[Dict[str, Any]],
    sim_threshold: float = 0.85,
    single_source_conf_cap: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    For each fact carrying an ``embedding``, count how many *distinct sources*
    make a semantically-similar claim (cosine >= ``sim_threshold``). Annotates:

      - ``corroboration_count``: number of distinct sources in the claim's cluster
      - ``single_source_warning``: True when fewer than 2 distinct sources agree

    Single-source supported claims have their confidence capped at
    ``single_source_conf_cap``. Facts without an embedding are conservatively
    flagged as single-source. Mutates and returns ``facts``.
    """
    embs = [f.get("embedding") for f in facts]
    n = len(facts)

    for i, f in enumerate(facts):
        # Sources already folded into this claim by the Verifier's merge step
        # count even when no embedding is available to cluster on.
        sources = set(merged_source_keys(f))

        if embs[i] is not None:
            for j in range(n):
                if j == i or embs[j] is None:
                    continue
                if _cosine(embs[i], embs[j]) >= sim_threshold:
                    sources.update(merged_source_keys(facts[j]))

        count = max(len(sources), 1)
        f["corroboration_count"] = count
        if count < 2:
            f["single_source_warning"] = True
            if f.get("support_level") in _SUPPORTED_LEVELS:
                f["confidence"] = min(float(f.get("confidence", 1.0) or 0.0), single_source_conf_cap)
        else:
            f["single_source_warning"] = False

    return facts
