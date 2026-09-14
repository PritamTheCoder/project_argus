"""Scout agent: iterates through the plan, calls the search/scrape tools, and
builds the raw data payload."""

import asyncio
import logging
from src.schema.state import AgentState
from src.tools.scraper import scrape_urls, shared_crawler
from src.agents.acquisition import gather_sources_for_query
from src.tools.tool_telemetry import ToolCallTracer
from src.utils.embeddings import get_embeddings
from src.utils.rerank import rerank_chunks
from src.utils.rrf import reciprocal_rank_fusion
from src.utils.grounding import _cosine
from src.utils.dates import sanitize_as_of_date
from src.utils.source_scoring import evaluate_source, registrable_domain
from src.config import (
    MAX_CHUNK_TOKENS, TOP_K_CHUNKS, SCOUT_CONCURRENCY, MAX_SOURCES_PER_DOMAIN,
    AUTHORITATIVE_CREDIBILITY, LOW_RELEVANCE_THRESHOLD, RELEVANCE_SAMPLE_CHARS,
    RELEVANCE_GATE_ENABLED, RELEVANCE_GATE_MIN_SOURCES,
)

logger = logging.getLogger(__name__)

# When the embedding model is unavailable we cannot rank chunks, so we fall back
# to passing raw scraped text straight to the Refiner. Cap it to avoid flooding
# the extraction context when many documents take the degraded path at once.
DEGRADED_MAX_CHARS = 12000


def _parse_intent(intent) -> tuple[str, str]:
    """Extract (query, mode) from a plan entry (dict or bare string)."""
    if isinstance(intent, dict):
        return intent.get("query", str(intent)), intent.get("mode", "MIXED")
    return str(intent), "MIXED"


def _relevance_to_query(content: str, query_emb) -> float | None:
    """Cosine similarity between the query and a source's content, 0..1.

    Credibility answers "is this source trustworthy"; it says nothing about
    whether the source is *about* the question, and retrieval only ranks
    candidates against each other so it always yields a "best" result no
    matter how poor the pool. This is the absolute check.

    Returns None when no query embedding is available (the Scout's degraded
    path), so callers can tell "not measured" from "measured as irrelevant".
    """
    if query_emb is None or not (content or "").strip():
        return None
    try:
        # The local embedding model truncates anyway; cap the input so a long
        # document doesn't cost more than the signal is worth.
        emb = get_embeddings([content[:RELEVANCE_SAMPLE_CHARS]])[0]
        return round(max(0.0, min(1.0, _cosine(query_emb, emb))), 3)
    except Exception as e:  # noqa: BLE001 — a missing score must not fail a run
        logger.warning("Scout: relevance scoring failed: %s", e)
        return None


def _align_reranked_to_docs(
    ranked_chunks: list[str],
    candidate_pairs: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """
    Map reranked chunk strings back to their originating ``(doc_id, chunk)`` pairs.

    Each candidate pair is consumed at most once, so when the same chunk text
    appears in multiple documents it is never silently collapsed to a single
    (arbitrary) document — each ranked occurrence resolves to a distinct source.
    Preserves the order of ``ranked_chunks``.
    """
    remaining = list(candidate_pairs)
    aligned: list[tuple[int, str]] = []
    for rc in ranked_chunks:
        for i, (doc_id, chunk) in enumerate(remaining):
            if chunk == rc:
                aligned.append((doc_id, chunk))
                remaining.pop(i)
                break
    return aligned


async def scout_node(state: AgentState) -> dict:
    """Execute the research plan by searching and scraping for each query."""
    logger.info("Scout: Starting research execution...")
    from src.graph.kg import kg_store
    
    queries = state["plan"]
    session_id = state.get("session_id", "")
    owner_key_hash = state.get("owner_key_hash", "")
    all_scraped_data = []
    source_map = state.get("source_map", {}).copy()
    
    # Parse keys like "[1]" to find the max integer and continue numbering from there.
    current_ids = [
        int(k.strip("[]")) 
        for k in source_map.keys() 
        if k.startswith("[") and k.endswith("]") and k.strip("[]").isdigit()
    ]
    next_id = max(current_ids) + 1 if current_ids else 1

    # url -> candidate metadata gathered by the acquisition agent (credibility
    # hints from academic backends, publication dates, etc.).
    candidate_meta: dict[str, dict] = {}

    # URLs already turned into sources. Seeded with anything from a prior round so
    # a URL is registered once even though parallel gathers share a seen snapshot.
    registered_urls: set[str] = {v["url"] for v in source_map.values()}

    # Sources registered per domain so far, so one site (or its subpages) can't
    # crowd out source diversity. Seeded from any prior round for the same reason.
    domain_counts: dict[str, int] = {}
    for v in source_map.values():
        d = registrable_domain(v["url"])
        domain_counts[d] = domain_counts.get(d, 0) + 1

    # Sources held back as off-topic, best first. Kept so the gate can never
    # leave the run with no evidence at all.
    deferred_off_topic: list[tuple[float, str, str, str]] = []

    def _register_source(
        url: str, content: str, query: str, query_emb=None,
        enforce_gate: bool = True, relevance: float | None = None,
    ) -> None:
        """``relevance`` is passed only when already known (the restore path
        below), so a deferred source isn't re-embedded to learn what it scored."""
        nonlocal next_id
        if url in registered_urls:
            return  # first occurrence wins (matches the old per-query seen_urls dedup)

        # Measured before anything is claimed, so a rejected source doesn't
        # consume a slot in the per-domain cap.
        if relevance is None:
            relevance = _relevance_to_query(content, query_emb)
        off_topic = relevance is not None and relevance < LOW_RELEVANCE_THRESHOLD
        if off_topic:
            logger.info(
                "Scout: %s is only %.2f similar to the query — authoritative but "
                "possibly off-topic.", url, relevance,
            )
            if RELEVANCE_GATE_ENABLED and enforce_gate:
                deferred_off_topic.append((relevance, url, content, query))
                return

        base = evaluate_source(url, query)
        score = base["score"]
        stype = base["type"]
        as_of = ""

        # The per-domain cap stops one content farm's subpages crowding out
        # diversity, but must not throttle an authoritative domain — sec.gov and
        # journal publishers legitimately supply many documents.
        domain = registrable_domain(url)
        if score < AUTHORITATIVE_CREDIBILITY and domain_counts.get(domain, 0) >= MAX_SOURCES_PER_DOMAIN:
            return

        registered_urls.add(url)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Prefer the backend's credibility hint when it is higher than the
        # domain heuristic (e.g. a Semantic Scholar paper at a generic .org host).
        meta = candidate_meta.get(url)
        if meta:
            hint_score = meta.get("credibility_hint")
            hint_type = meta.get("source_type_hint")
            if hint_score is not None and hint_score > score:
                score = hint_score
                if hint_type:
                    stype = hint_type
            as_of = sanitize_as_of_date(meta.get("as_of_date", "") or "")

        source_id = f"[{next_id}]"
        source_map[source_id] = {
            "url": url,
            "snippet": (content[:200] + "...") if content else "",
            "credibility_score": score,
            "source_type": stype,
            "as_of_date": as_of,
            "relevance_score": relevance,
        }
        all_scraped_data.append({
            "source_id": source_id,
            "url": url,
            "content": content,
            "query": query,
        })
        next_id += 1

    # ── Phase A: gather + scrape every sub-query CONCURRENTLY ─────────────────
    # Both are network-bound and independent across queries, so overlap them
    # (bounded by SCOUT_CONCURRENCY). All gathers share the same seen-URL snapshot;
    # cross-query duplicates are collapsed at registration in Phase B. The
    # per-provider rate limiter still caps actual API RPM.
    initial_seen = {v["url"] for v in source_map.values()}
    sem = asyncio.Semaphore(max(SCOUT_CONCURRENCY, 1))
    # One tracer for every sub-query: a backend that returns nothing for the
    # whole run is only visible at run scope, not per-query.
    tracer = ToolCallTracer()

    async def _gather_and_scrape(intent, crawler) -> tuple[str, list, list]:
        q, mode_str = _parse_intent(intent)
        async with sem:
            logger.info(f"Scout: Processing query '{q}' with mode '{mode_str}'")
            candidates = await gather_sources_for_query(
                q, mode_str, seen_urls=initial_seen, session_id=session_id,
                owner_key_hash=owner_key_hash, tracer=tracer,
            )
            if not candidates:
                logger.info(f"Scout: No new candidate sources for query '{q}'.")
                return q, [], []

            scrape_targets = [c["url"] for c in candidates if c.get("needs_scrape")]
            prefetched = [c for c in candidates if not c.get("needs_scrape") and (c.get("content") or "").strip()]
            logger.info(
                f"Scout: {len(candidates)} candidate(s) for '{q}' "
                f"({len(prefetched)} prefetched, {len(scrape_targets)} to scrape)."
            )
            scraped = await scrape_urls(scrape_targets, query=q, crawler=crawler) if scrape_targets else []
            results = [
                {"url": c["url"], "content": c["content"], "success": True}
                for c in prefetched
            ]
            results.extend(scraped)
            return q, candidates, results

    # One browser shared across every concurrent sub-query (see shared_crawler()).
    async with shared_crawler() as crawler:
        gathered = await asyncio.gather(*[_gather_and_scrape(intent, crawler) for intent in queries])

    # ── Phase B: index + hierarchically retrieve + register (SEQUENTIAL) ──────
    # Kept sequential so shared state (next_id, source_map, registered_urls, and
    # KG writes) stays deterministic and race-free.
    for q, candidates, results in gathered:
        if not candidates:
            continue

        # Record candidate metadata (credibility/source hints, dates) for later.
        for c in candidates:
            candidate_meta[c["url"]] = c

        try:
            query_emb = get_embeddings([q])[0]
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            query_emb = None
        
        # Process successful results
        for res in results:
            if not res["success"] or not res["content"]:
                continue
                
            content = res["content"]
            
            from src.utils.chunking import extract_summary, chunk_document
            summary = extract_summary(content)
            chunks = chunk_document(content)  # <= MAX_CHUNK_TOKENS (~300)

            if not chunks:
                continue

            if query_emb is not None:
                try:
                    summary_emb = get_embeddings([summary])[0]
                    chunk_embs = get_embeddings(chunks)
                    doc_id = kg_store.store_document_and_chunks(
                        url=res["url"],
                        query=q,
                        summary=summary,
                        summary_embedding=summary_emb,
                        chunks=chunks,
                        chunk_embeddings=chunk_embs,
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.error(f"Failed to index document: {e}")
            else:
                logger.warning("Skipping indexing due to missing query embedding.")

        if query_emb is not None:
            logger.info("Scout: Performing Hierarchical Retrieval (vector + BM25)...")
            top_doc_ids = kg_store.retrieve_top_docs(query_emb, k=10, session_id=session_id)
            # Fuse dense (vector) and sparse (BM25) chunk candidates before reranking.
            vector_chunks = kg_store.retrieve_top_chunks(query_emb, top_doc_ids, k=TOP_K_CHUNKS * 2)
            bm25_chunks = kg_store.retrieve_top_chunks_bm25(q, top_doc_ids, k=TOP_K_CHUNKS * 2)
            candidate_pairs = reciprocal_rank_fusion([vector_chunks, bm25_chunks])[:TOP_K_CHUNKS * 2]
            top_stage2_chunks = [chunk for doc_id, chunk in candidate_pairs]

            best_chunks = top_stage2_chunks[:TOP_K_CHUNKS]
            try:
                top_stage3 = rerank_chunks(q, top_stage2_chunks, top_k=TOP_K_CHUNKS)
                if top_stage3:
                    best_chunks = top_stage3
            except Exception as e:
                logger.error(f"Reranking failed: {e}")

            # Parent Document Retrieval — map ranked chunks back to their exact
            # source docs (consuming each occurrence so duplicate chunk text across
            # documents is never misattributed to a single arbitrary source).
            best_pairs = _align_reranked_to_docs(best_chunks, candidate_pairs)
            best_doc_ids = list({doc_id for doc_id, _ in best_pairs})

            doc_all_chunks = kg_store.get_all_chunks_for_docs(best_doc_ids)

            source_content_map = {}
            for doc_id in best_doc_ids:
                meta = kg_store.get_doc_metadata(doc_id)
                url = meta.get("url", "unknown")
                if url not in source_content_map:
                    source_content_map[url] = []
                source_content_map[url].extend(doc_all_chunks.get(doc_id, []))

            for url, matched_chunks in source_content_map.items():
                best_content = "\n\n".join(matched_chunks)
                _register_source(url, best_content, q, query_emb=query_emb)
        else:
            # Graceful degradation: embeddings are unavailable so we cannot rank
            # or index. Rather than dropping every scraped page for this query,
            # pass the (already pre-filtered) raw content straight to the Refiner.
            logger.warning(
                "Scout: Query embedding unavailable — degrading to raw-content "
                "extraction for this query (no ranking/indexing)."
            )
            for res in results:
                if not res.get("success") or not res.get("content"):
                    continue
                content = res["content"][:DEGRADED_MAX_CHARS]
                _register_source(res["url"], content, q)

    # An empty source map would starve the Refiner and hide the failure. If the
    # gate rejected everything, restore the closest matches so the Critic sees
    # weak evidence and can re-search, rather than seeing none.
    if deferred_off_topic and len(source_map) < RELEVANCE_GATE_MIN_SOURCES:
        deferred_off_topic.sort(key=lambda d: d[0], reverse=True)
        restore = deferred_off_topic[: RELEVANCE_GATE_MIN_SOURCES - len(source_map)]
        logger.warning(
            "Scout: every source scored below the relevance floor; restoring the "
            "%d closest of %d so the run has evidence to critique.",
            len(restore), len(deferred_off_topic),
        )
        for score, url, content, query in restore:
            _register_source(url, content, query, enforce_gate=False, relevance=score)
    elif deferred_off_topic:
        logger.info(
            "Scout: dropped %d off-topic source(s) below relevance %.2f.",
            len(deferred_off_topic), LOW_RELEVANCE_THRESHOLD,
        )

    logger.info(f"Scout: Collected {len(all_scraped_data)} highly relevant chunk sets across queries.")
    logger.info(f"Scout: Source map has {len(source_map)} entries:")
    for sid, data in source_map.items():
        logger.info(f"  {sid}: {data.get('url', '?')} (cred={data.get('credibility_score', '?')}, type={data.get('source_type', '?')})")
    
    health = tracer.backend_health()
    degraded = [name for name, h in health.items() if h["degraded"]]
    if degraded:
        logger.warning(
            "Scout: retrieval backend(s) returned NO results all run: %s. "
            "Sources came from the remaining backends, which may be the wrong corpus.",
            ", ".join(sorted(degraded)),
        )

    return {
        "scraped_data": all_scraped_data,
        "source_map": source_map,
        "backend_health": health,
        "active_node": "scout"
    }
