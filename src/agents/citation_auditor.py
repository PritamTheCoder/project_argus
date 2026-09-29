"""Citation auditor: checks the Writer's finished sentences against their own
cited evidence (Phase 10.A). The Verifier grounds facts to verbatim quotes
before writing, but nothing previously checked whether the Writer's prose
stayed within what its citations actually support — a fast prose model is
exactly the kind of model most likely to quietly broaden a claim."""

import logging
import re
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from src.schema.state import AgentState, CitationAuditBatch
from src.config import (
    CITATION_AUDITOR_MODEL, CITATION_AUDITOR_PROVIDER, CITATION_AUDITOR_FALLBACK_CHAIN,
    CITATION_AUDIT_BATCH_SIZE,
)
from src.utils.llm_factory import get_llm_with_fallbacks
from src.utils.retry import retry_on_rate_limit
from src.agents.writer import build_citation_remap, resolve_fact_citation_id

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\[])")
_REFERENCES_MARKER = "\n\n---\n### References\n"


def _split_citable_sentences(report: str) -> List[str]:
    """Sentences carrying at least one [n] marker, from the report body only
    (the References list itself is excluded — its "- **[1]**: url" lines
    aren't claims to check). Lines with no marker (headers, table rows,
    the quality banner) have nothing to check them against, so they're
    skipped rather than sent to the judge for no reason."""
    body = report.split(_REFERENCES_MARKER)[0]
    sentences = []
    for line in body.split("\n"):
        line = line.strip()
        if not line or line.startswith(("#", "|", ">", "---")):
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip("-* ").strip()
            if sentence and _CITATION_RE.search(sentence):
                sentences.append(sentence)
    return sentences


def _group_facts_by_citation_id(facts: List[dict], old_to_new_id_map: dict) -> Dict[str, List[dict]]:
    """The same source_id/URL resolution the Writer uses to number its
    citations, so a sentence's [n] can be traced back to the fact(s) it
    actually came from."""
    grouped: Dict[str, List[dict]] = {}
    for fact in facts:
        if fact.get("support_level") not in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
            continue
        new_sid = resolve_fact_citation_id(fact, old_to_new_id_map)
        if new_sid is None:
            continue
        grouped.setdefault(new_sid, []).append(fact)
    return grouped


def _render_batch(sentences: List[str], evidence_by_id: Dict[str, List[dict]]) -> str:
    blocks = []
    for i, sentence in enumerate(sentences, start=1):
        cited_ids = [f"[{n}]" for n in _CITATION_RE.findall(sentence)]
        evidence_lines = []
        for cid in cited_ids:
            facts = evidence_by_id.get(cid, [])
            if not facts:
                evidence_lines.append(f"{cid}: (no evidence text available for this citation)")
                continue
            for fact in facts:
                evidence_lines.append(f"{cid}: {fact.get('claim', '')}")
        blocks.append(
            f"Sentence {i}: \"{sentence}\"\n"
            f"Cited IDs: {', '.join(cited_ids) or '(none)'}\n"
            "Evidence:\n" + "\n".join(f"- {line}" for line in evidence_lines)
        )
    return "\n\n".join(blocks)


_SYSTEM_PROMPT = (
    "You are a strict citation-entailment checker. For each numbered sentence "
    "below, decide whether the evidence listed under its cited ID(s) actually "
    "supports the sentence AS WRITTEN — not merely whether the topic is related.\n\n"
    "SUPPORTED: the evidence fully backs the sentence's claim.\n"
    "PARTIALLY_SUPPORTED: the evidence backs part of the claim, but the sentence "
    "adds an unsupported number, degree, or certainty the evidence doesn't state.\n"
    "UNSUPPORTED: the evidence doesn't address this claim, or none was found for the cited ID.\n"
    "CONTRADICTED: the evidence directly contradicts the sentence.\n\n"
    "Return exactly one verdict per sentence, in the same order given, with `cited_ids` "
    "copied from the sentence's \"Cited IDs\" line and `sentence` copied verbatim."
)


def citation_auditor_node(state: AgentState) -> dict:
    """Post-write pass: verify every cited sentence in the finished report
    against the evidence its citations actually point to."""
    logger.info("CitationAuditor: Checking report sentences against their citations...")

    report = state.get("report", "")
    sentences = _split_citable_sentences(report)
    if not sentences:
        logger.info("CitationAuditor: No cited sentences found; skipping.")
        return {"citation_audit": [], "active_node": "citation_auditor"}

    new_source_map, old_to_new_id_map = build_citation_remap(state.get("source_map", {}))
    evidence_by_id = _group_facts_by_citation_id(state.get("verified_facts", []), old_to_new_id_map)

    structured_llm = get_llm_with_fallbacks(
        CITATION_AUDITOR_MODEL, CITATION_AUDITOR_PROVIDER,
        fallback_chain=CITATION_AUDITOR_FALLBACK_CHAIN,
        temperature=0,
        structured_schema=CitationAuditBatch,
    )
    chain = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "{input}"),
    ]) | structured_llm

    all_results: List[dict] = []
    for start in range(0, len(sentences), CITATION_AUDIT_BATCH_SIZE):
        batch = sentences[start:start + CITATION_AUDIT_BATCH_SIZE]
        batch_text = _render_batch(batch, evidence_by_id)
        try:
            result: CitationAuditBatch = retry_on_rate_limit(chain.invoke, {"input": batch_text})
            all_results.extend(v.model_dump() for v in result.results)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"CitationAuditor: batch failed ({e}); skipping {len(batch)} sentences.")

    if not all_results:
        return {"citation_audit": [], "active_node": "citation_auditor"}

    unsupported = sum(1 for r in all_results if r["verdict"] in ("UNSUPPORTED", "CONTRADICTED"))
    logger.info(
        f"CitationAuditor: {len(all_results)} sentences checked, {unsupported} not fully supported."
    )

    # Same banner-prepend pattern the Writer already uses for quality_score
    # (writer.py) — a visible integrity note, not a silent rewrite. Full
    # per-sentence detail lives in citation_audit, surfaced via the API.
    fully_supported = len(all_results) - unsupported
    banner = f"> **Citation Integrity** — {fully_supported}/{len(all_results)} cited sentences fully supported"
    if unsupported:
        banner += f" ({unsupported} flagged — see citation_audit for detail)"
    report = banner + "\n\n" + report

    return {"report": report, "citation_audit": all_results, "active_node": "citation_auditor"}
