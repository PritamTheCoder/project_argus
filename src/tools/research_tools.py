"""
LangChain structured tools the acquisition agent can call, wrapping the
provider backends plus two utility tools (KG memory lookup, calculator). Each
tool is registered in the shared `registry` with tags so the agent can be given
the right subset and so new tools are pluggable without touching the graph.

Search tools return a list of "candidate" dicts (each with a `url`) that the
Scout feeds into the document pipeline. `kg_lookup` and `calculator` return
informational payloads the model reasons over but that don't produce documents.
"""

from __future__ import annotations

import ast
import contextvars
import logging
import operator as _op
from typing import Any, Dict, List

from langchain_core.tools import tool

from src.config import ACADEMIC_MAX_RESULTS, EDGAR_MAX_RESULTS, KG_LOOKUP_GLOBAL
from src.tools import providers
from src.tools.registry import ToolSpec, registry

logger = logging.getLogger(__name__)

# The current run's session id, set by the acquisition flow before the model may
# invoke kg_lookup. A contextvar (task-local) is used because kg_lookup is called
# indirectly by the LLM and can't take session_id as an argument. Default "" ⇒ no
# scope available (kg_lookup then returns nothing rather than leaking global facts).
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "kg_lookup_session_id", default=""
)

# The current run's owner (API key hash), so a KG_LOOKUP_GLOBAL lookup can be
# scoped to "this owner's facts" rather than every owner's — the KG has no
# other tenant boundary. Empty for the CLI/Streamlit paths, which have no
# owner concept; those callers implicitly share one "local" bucket.
_current_owner_key_hash: contextvars.ContextVar[str] = contextvars.ContextVar(
    "kg_lookup_owner_key_hash", default=""
)


def set_kg_session(session_id: str) -> contextvars.Token:
    """Bind the session id kg_lookup should scope to. Returns a token for reset."""
    return _current_session_id.set(session_id or "")


def reset_kg_session(token: contextvars.Token) -> None:
    _current_session_id.reset(token)


def set_kg_owner(owner_key_hash: str) -> contextvars.Token:
    """Bind the owner kg_lookup's global mode should scope to. Returns a token for reset."""
    return _current_owner_key_hash.set(owner_key_hash or "")


def reset_kg_owner(token: contextvars.Token) -> None:
    _current_owner_key_hash.reset(token)


# ── Search tools (yield retrieval candidates) ────────────────────────────────

@tool
async def web_search(query: str, limit: int = 8, category: str = "") -> List[Dict[str, Any]]:
    """Search the general web (news, company sites, blogs, market data).
    Best for current events, market/commercial information, product announcements,
    and non-academic topics.
    Set `category` to steer results toward a source type when the question calls
    for one: "news", "company", "financial report", "publication", "people", or
    "personal site". Leave it empty for a general search.
    Returns candidate pages to be scraped."""
    results = await providers.web_search(query, limit=limit, category=category or None)
    return [r.to_candidate() for r in results]


@tool
async def sec_edgar_search(
    query: str, limit: int = EDGAR_MAX_RESULTS, forms: str = "", cik: str = ""
) -> List[Dict[str, Any]]:
    """Search official SEC EDGAR filings — US regulatory primary sources.
    Best for company financials, funding rounds, ownership, and risk disclosures:
    these are the filings themselves, not news coverage about them.

    STRONGLY PREFER passing `cik` (the company's SEC identifier) or its stock
    ticker: a plain name search mostly returns SPVs and feeder funds set up to
    invest in the company, whose amounts are NOT the company's funding rounds.
    A ticker or exact registered name is resolved to a CIK automatically.

    `forms` optionally filters by type, e.g. "D" (private placements / funding
    rounds), "10-K" (annual report), "8-K" (material events), "S-1" (IPO).
    Note: Form D reports the amount raised, NOT the company's valuation — no SEC
    filing states a private company's valuation."""
    results = await providers.edgar_search(query, limit=limit, forms=forms or None, cik=cik or None)
    return [r.to_candidate() for r in results]


@tool
async def semantic_scholar_search(query: str, limit: int = ACADEMIC_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search 200M+ peer-reviewed academic papers (Semantic Scholar) with abstracts.
    Best for scientific, technical, or medical questions where authoritative,
    citable literature matters. Returns papers with abstracts (no scraping needed)."""
    results = await providers.semantic_scholar_search(query, limit=limit)
    return [r.to_candidate() for r in results]


@tool
async def arxiv_search(query: str, limit: int = ACADEMIC_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search arXiv preprints (physics, CS, math, quantitative biology, etc.).
    Best for the very latest technical research that may not yet be peer-reviewed.
    Returns papers with abstracts (no scraping needed)."""
    results = await providers.arxiv_search(query, limit=limit)
    return [r.to_candidate() for r in results]


@tool
async def europe_pmc_search(query: str, limit: int = ACADEMIC_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search Europe PMC — PubMed/MEDLINE, PMC full text, and biomedical preprints.
    THE tool for clinical, medical, drug, and life-sciences questions: trials,
    outcomes, treatments, biology. arXiv does NOT cover these subjects, so use
    this rather than a general academic search for anything medical.
    Returns papers with abstracts (no scraping needed)."""
    results = await providers.europe_pmc_search(query, limit=limit)
    return [r.to_candidate() for r in results]


@tool
async def crossref_search(query: str, limit: int = ACADEMIC_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search Crossref DOI metadata across academic publishers (journals, conferences).
    Best for locating formally published work and its bibliographic details.
    Returns papers with abstracts/metadata (no scraping needed)."""
    results = await providers.crossref_search(query, limit=limit)
    return [r.to_candidate() for r in results]


# ── Utility tools ────────────────────────────────────────────────────────────

@tool
def kg_lookup(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Look up facts already verified in this run's knowledge graph.
    Call this FIRST to check what is already known before searching the web, to
    avoid redundant research. Returns previously verified claims with their sources."""
    try:
        # Resolve scope first (before the embedding cost). Session-scoped by
        # default — consistent with the Critic/Reflector retrieval — so kg_lookup
        # can't leak prior runs' facts. `global` opt-in enables cross-run memory,
        # scoped to the current owner — the KG has no other tenant boundary, so
        # an unscoped global lookup would leak one API key's research to another.
        owner_key_hash = None
        if KG_LOOKUP_GLOBAL:
            session_id = None
            owner_key_hash = _current_owner_key_hash.get()
        else:
            session_id = _current_session_id.get()
            if not session_id:
                return []  # no run scope bound → nothing this run knows yet

        from src.utils.embeddings import get_embeddings
        from src.graph.kg import kg_store
        emb = get_embeddings([query])[0]
        facts = kg_store.retrieve_relevant_facts(emb, k=k, session_id=session_id, owner_key_hash=owner_key_hash)
        return [
            {
                "claim": f.get("claim", ""),
                "source_url": f.get("source_url", ""),
                "support_level": f.get("support_level", ""),
                "credibility_score": f.get("credibility_score", 0.4),
            }
            for f in facts
        ]
    except Exception as e:  # noqa: BLE001
        logger.warning("kg_lookup failed: %s", e)
        return []


# Safe arithmetic evaluation (no eval()): supports + - * / ** and unary minus.
_ALLOWED_BINOPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.Pow: _op.pow, ast.Mod: _op.mod,
    ast.FloorDiv: _op.floordiv,
}
_ALLOWED_UNARY = {ast.UAdd: _op.pos, ast.USub: _op.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression")


def evaluate_expression(expression: str) -> Dict[str, Any]:
    """Pure, safe arithmetic evaluator (unit-tested)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return {"expression": expression, "result": _safe_eval(tree), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"expression": expression, "result": None, "error": str(e)}


@tool
def calculator(expression: str) -> Dict[str, Any]:
    """Evaluate a basic arithmetic expression (+, -, *, /, **, %).
    Use to sanity-check numeric claims, unit conversions, or magnitudes found in
    sources. Example: '400 * 1.25' or '(450-380)/380'."""
    return evaluate_expression(expression)


# ── Registration ─────────────────────────────────────────────────────────────

def _register_default_tools() -> None:
    """Register all built-in tools once (idempotent)."""
    defs = [
        ToolSpec(name="web_search", tool=web_search, tags=["retrieval", "web"],
                 when_to_use="news, market/commercial data, company info, non-academic topics",
                 yields_candidates=True),
        ToolSpec(name="semantic_scholar_search", tool=semantic_scholar_search, tags=["retrieval", "academic"],
                 when_to_use="peer-reviewed science/tech/medical literature", yields_candidates=True),
        ToolSpec(name="arxiv_search", tool=arxiv_search, tags=["retrieval", "academic"],
                 when_to_use="latest preprints in physics/CS/math", yields_candidates=True),
        ToolSpec(name="crossref_search", tool=crossref_search, tags=["retrieval", "academic"],
                 when_to_use="formally published papers across publishers", yields_candidates=True),
        ToolSpec(name="europe_pmc_search", tool=europe_pmc_search, tags=["retrieval", "academic", "biomedical"],
                 when_to_use="clinical trials, medicine, drugs, biology — PubMed/MEDLINE and PMC",
                 yields_candidates=True),
        ToolSpec(name="sec_edgar_search", tool=sec_edgar_search, tags=["retrieval", "financial"],
                 when_to_use="company financials, funding rounds, ownership — official SEC filings as primary sources",
                 yields_candidates=True),
        ToolSpec(name="kg_lookup", tool=kg_lookup, tags=["memory"],
                 when_to_use="check already-known facts before searching", yields_candidates=False),
        ToolSpec(name="calculator", tool=calculator, tags=["compute"],
                 when_to_use="verify numeric claims / unit math", yields_candidates=False),
    ]
    for spec in defs:
        if not registry.has(spec.name):
            registry.register(spec)


_register_default_tools()
