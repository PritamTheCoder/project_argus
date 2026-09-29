"""Writer agent: synthesizes the report from verified evidence, with a
citation id on each claim. The citation auditor checks those sentences
against the same evidence after this node."""

import logging
from langchain_core.prompts import ChatPromptTemplate
from src.schema.state import AgentState
from src.config import WRITER_MODEL, WRITER_PROVIDER, WRITER_FALLBACK_CHAIN
from src.utils.llm_factory import get_llm_with_fallbacks

logger = logging.getLogger(__name__)


def _as_text(content) -> str:
    """response.content is a str for most providers, but a list of blocks for
    some (e.g. Gemini). Flatten either into one string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text", ""))
    return "".join(parts)


def build_citation_remap(source_map: dict) -> tuple[dict, dict]:
    """Collapse duplicate URLs in ``source_map`` into single renumbered
    citation IDs (two source_ids pointing at the same URL become one ``[n]``).

    Returns ``(new_source_map, old_to_new_id_map)``. ``old_to_new_id_map`` is
    keyed by both the original source_id (e.g. ``"[4]"``) and the URL itself,
    so callers can resolve a fact by whichever one it carries — shared by
    ``writer_node`` and the citation_auditor, which both need to map a fact
    back to the report's final citation numbering.
    """
    unique_urls = {}       # url -> new_source_id (e.g. "[1]")
    new_source_map = {}    # new_source_id -> {"url": url, ...}
    old_to_new_id_map = {} # old_source_id or url -> new_source_id

    next_id = 1
    for old_id, data in source_map.items():
        url = data.get("url")
        if url not in unique_urls:
            new_id = f"[{next_id}]"
            unique_urls[url] = new_id
            new_source_map[new_id] = data
            next_id += 1

        old_to_new_id_map[old_id] = unique_urls[url]
        if url:
            old_to_new_id_map[url] = unique_urls[url]

    return new_source_map, old_to_new_id_map


def resolve_fact_citation_id(fact: dict, old_to_new_id_map: dict) -> str | None:
    """Resolve one fact's final citation ID through the collapsed remap,
    falling back from its stored source_id to its source_url."""
    new_sid = old_to_new_id_map.get(fact.get("source_id", "?"))
    if new_sid is None:
        new_sid = old_to_new_id_map.get(fact.get("source_url", ""))
    return new_sid


def writer_node(state: AgentState) -> dict:
    """Write the final report."""
    logger.info("Writer: Synthesizing report...")

    query = state["query"]
    evidence = state.get("verified_facts", [])
    source_map = state.get("source_map", {})

    new_source_map, old_to_new_id_map = build_citation_remap(source_map)

    evidence_text = ""
    for fact in evidence:
        if fact.get("support_level") not in ["SUPPORTED", "PARTIALLY_SUPPORTED"]:
            continue
        new_sid = resolve_fact_citation_id(fact, old_to_new_id_map)

        if new_sid is None:
            new_sid = fact.get("source_id", "?")
            logger.warning(f"Writer: Unresolvable source_id={new_sid}")

        claim = fact.get("claim", fact.get("text", ""))
        score = fact.get("credibility_score", 0.4)
        stype = fact.get("source_type", "Unknown")

        # Grounding/temporal/corroboration annotations for the Writer to honour.
        annotations = f"[Source: {new_sid}] [Credibility: {score}, Type: {stype}]"
        corro = fact.get("corroboration_count")
        if corro is not None:
            annotations += f" [Sources agreeing: {corro}]"
        if fact.get("single_source_warning"):
            annotations += " [SINGLE-SOURCE]"
        as_of = fact.get("as_of_date", "")
        if as_of:
            annotations += f" [As of: {as_of}]"

        evidence_text += f"- {claim} {annotations}\n"

    # Cross-source consensus and contradictions from the Consensus node.
    consensus_findings = state.get("consensus_findings", []) or []
    contradictions = state.get("contradictions", []) or []

    consensus_text = "(none detected)"
    if consensus_findings:
        consensus_text = "\n".join(
            f"- {c.get('statement', '')} (agreed by {c.get('source_count', 0)} independent sources)"
            for c in consensus_findings
        )

    contradiction_text = "(none detected)"
    if contradictions:
        contradiction_text = "\n".join(
            f"- {c.get('statement', '')} (conflicting sources: {', '.join(c.get('sources', []))})"
            for c in contradictions
        )

    sid_counts = {}
    for fact in evidence:
        sid = fact.get("source_id", "?")
        sid_counts[sid] = sid_counts.get(sid, 0) + 1
    logger.info(f"Writer: Citation distribution across source_ids: {sid_counts}")
    logger.info(f"Writer: Total unique sources in source_map: {len(source_map)}")
    logger.info(f"Writer: Total unique URLs after dedup: {len(new_source_map)}")
        
    system_prompt = (
        "You are an elite technical Ghostwriter. Your objective is to write a highly detailed, comprehensive, and exhaustive report answering the user's query.\n\n"
        "IMPORTANT GUIDELINES FOR LENGTH & STRUCTURE:\n"
        "- Write a LONG, in-depth report. Unpack all details thoroughly.\n"
        "- Use a clear structure: Introduction, well-reasoned Body sections with descriptive subheaders, and a strong Conclusion.\n"
        "- DO NOT summarize away important technical details. Expand upon them deeply based on the evidence.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided evidence. Do not use outside knowledge.\n"
        "2. Cite every claim using the source ID (e.g., [1], [3]). Almost every factual sentence MUST be cited.\n"
        "3. **EVIDENCE-GATING & HEDGING (CRITICAL)**:\n"
        "   - For sources with credibility >= 0.7 (Academic/Government/Major News): Assert facts confidently.\n"
        "   - For sources with credibility 0.5-0.69 (Industry/Market Research): State as reported findings or projections.\n"
        "   - For sources with credibility < 0.5 (Unverified/Web): Lightly hedge (e.g., 'according to industry sources' or 'unverified reports suggest').\n"
        "   - IMPORTANT: Do NOT excessively prefix every sentence with 'unverified reports suggest'. Use hedging sparingly and naturally, only where required.\n"
        "   - In cases of contradiction across sources, explicitly prioritize the claim from the higher credibility source, noting the lower-credibility contention.\n"
        "4. **EXHAUSTIVE CITATION**: You must integrate and cite ALL provided evidence. Attempt to cite every unique source provided to ensure maximum coverage.\n"
        "5. **CORROBORATION & SINGLE-SOURCE**: Claims tagged [Sources agreeing: N] with N>=2 are independently corroborated — present them confidently. Claims tagged [SINGLE-SOURCE] rest on one source — hedge them explicitly (e.g., 'a single source reports…').\n"
        "6. **CONSENSUS**: The 'Consensus Findings' provided are agreements across independent sources. Lead with them and present them as the most reliable conclusions.\n"
        "7. **CONTRADICTIONS (CRITICAL)**: For every item in 'Contradictions' provided, you MUST surface the disagreement explicitly in a dedicated 'Conflicting Evidence' subsection — never silently pick one side. Note which source is more credible, but do not hide the conflict.\n"
        "8. **TEMPORAL SCOPING**: When a fact carries an [As of: DATE] tag, scope the claim in time ('as of DATE, …'). When sources conflict on a time-sensitive metric, prefer the most recent and say so.\n"
        "9. **COMPARATIVE TABLES**: When comparing multiple entities across a shared dimension (e.g., companies vs. a metric), render a Markdown table (e.g., | Entity | Metric | Value | As of | Source |) instead of dense prose.\n"
        "10. Do NOT generate a 'References' section manually; just use the [ID] markers in text.\n"
        "11. If evidence is missing for a specific topic, simply omit that section entirely.\n"
    )

    critique = state.get("critique", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Query: {query}\n\n"
         "Critique/Caveats:\n{critique}\n\n"
         "Consensus Findings (corroborated across independent sources):\n{consensus}\n\n"
         "Contradictions (MUST be surfaced explicitly):\n{contradictions}\n\n"
         "Evidence:\n{evidence}")
    ])

    invoke_payload = {
        "query": query,
        "critique": critique,
        "consensus": consensus_text,
        "contradictions": contradiction_text,
        "evidence": evidence_text,
    }

    # The Writer may be configured to use a heavier model (e.g. Nemotron). The
    # cross-provider ladder guards the call so a provider/rate-limit error never
    # blocks report generation — it fails over through Kimi then Gemini.
    llm = get_llm_with_fallbacks(
        WRITER_MODEL, WRITER_PROVIDER,
        fallback_chain=WRITER_FALLBACK_CHAIN,
        temperature=0.7,
    )
    response = (prompt | llm).invoke(invoke_payload)

    report_content = _as_text(response.content)

    # Prepend a compact research-quality banner so the trust signals are visible
    # at a glance (also useful for PDF/UI headers downstream).
    qs = state.get("quality_score", {}) or {}
    if qs:
        banner = (
            "> **Research Quality** — "
            f"{qs.get('verified_fact_count', 0)} verified facts · "
            f"{qs.get('distinct_source_count', 0)} sources · "
            f"avg credibility {qs.get('avg_source_credibility', 0)} · "
            f"{qs.get('corroborated_fact_count', 0)} corroborated · "
            f"{qs.get('contradiction_count', 0)} contradiction(s)\n\n"
        )
        report_content = banner + report_content

    if new_source_map:
        report_content += "\n\n---\n### References\n"
        sorted_ids = sorted(new_source_map.keys(), key=lambda x: int(x.strip("[]")) if x.strip("[]").isdigit() else 0)
        
        for sid in sorted_ids:
            data = new_source_map[sid]
            url = data.get("url", "#")
            score = data.get("credibility_score", "N/A")
            stype = data.get("source_type", "Unknown")
            report_content += f"- **{sid}**: {url} *(Credibility: {score}, {stype})*\n"
            
    logger.info("Writer: Report generation complete.")
    
    return {
        "report": report_content,
        "active_node": "writer"
    }
