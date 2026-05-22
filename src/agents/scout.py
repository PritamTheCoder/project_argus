"""
Project Argus - Scout Agent

Role: Research Execution
Responsibility: Iterates through the plan, calls the search/scrape tools,
and builds the raw data payload.
"""

import logging
import numpy as np
from src.schema.state import AgentState
from src.tools.scout import run_scout
from src.tools.search import search_ddg
from src.tools.scraper import scrape_urls
from src.utils.embeddings import get_embeddings
from src.utils.rerank import rerank_chunks
from src.utils.source_scoring import evaluate_source
from src.config import MAX_CHUNK_TOKENS, TOP_K_CHUNKS

logger = logging.getLogger(__name__)


async def scout_node(state: AgentState) -> dict:
    """
    Execute the research plan by searching and scraping for each query.
    
    Args:
        state: AgentState containing `plan` (list of strings).
        
    Returns:
        dict: Updates `scraped_data` and `source_map`.
    """
    logger.info("Scout: Starting research execution...")
    from src.graph.kg import kg_store
    
    queries = state["plan"]
    all_scraped_data = []
    source_map = state.get("source_map", {}).copy()
    
    # Determine the next available source ID (e.g. 1, 2, 3...)
    # We parse keys like "[1]" to find the max integer.
    current_ids = [
        int(k.strip("[]")) 
        for k in source_map.keys() 
        if k.startswith("[") and k.endswith("]") and k.strip("[]").isdigit()
    ]
    next_id = max(current_ids) + 1 if current_ids else 1
    
    # Execute searches concurrently for speed, but sequentially per query to avoid 
    # hitting rate limits too hard or overwhelming the scraper.
    # For robust production, we might want to parallelize this further, 
    # but let's go linear-by-query, parallel-within-query (run_scout does this) for now.
    
    for intent in queries:
        if isinstance(intent, dict):
            q = intent.get("query", str(intent))
            mode_str = intent.get("mode", "MIXED")
        else:
            q = str(intent)
            mode_str = "MIXED"
            
        logger.info(f"Scout: Processing query '{q}' with mode '{mode_str}'")
        
        try:
            from src.tools.search import DomainPolicy
            policy = DomainPolicy[mode_str]
        except (KeyError, ValueError):
            from src.tools.search import DomainPolicy
            policy = DomainPolicy.MIXED
        
        # We need to extract the search+scrape steps from run_scout so we can intercept 
        # the URLs before scraping to apply the Global Seen-Set.
        
        # 1. Search
        urls = await search_ddg(q, policy=policy)
        if not urls:
            continue
            
        # 2. Apply Global Seen-Set Deduplication
        seen_urls = {v["url"] for v in source_map.values()}
        new_urls = [u for u in urls if u not in seen_urls]
        
        if not new_urls:
            logger.info(f"Scout: All {len(urls)} URLs for query '{q}' were already seen.")
            continue
            
        logger.info(f"Scout: Found {len(urls)} URLs, {len(new_urls)} are new. Scraping...")
        
        # 3. Scrape ONLY the new URLs
        results = await scrape_urls(new_urls, query=q)
        

        # Get query embedding for Stage 2 & 3
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
            
            # 1. Synthesize Coarse Document Summary
            from src.utils.chunking import extract_summary, chunk_document
            summary = extract_summary(content)
            
            # 2. Chunking to <= MAX_CHUNK_TOKENS (~300)
            chunks = chunk_document(content)

            if not chunks:
                continue

            # 3. Embed & Store in Hierarchical Index
            if query_emb is not None:
                try:
                    summary_emb = get_embeddings([summary])[0]
                    chunk_embs = get_embeddings(chunks)
                    
                    # Store in Local Vector DB
                    doc_id = kg_store.store_document_and_chunks(
                        url=res["url"], 
                        query=q, 
                        summary=summary, 
                        summary_embedding=summary_emb, 
                        chunks=chunks, 
                        chunk_embeddings=chunk_embs
                    )
                except Exception as e:
                    logger.error(f"Failed to index document: {e}")
            else:
                logger.warning("Skipping indexing due to missing query embedding.")

        # 4. Hierarchical Retrieval (cross-document)
        if query_emb is not None:
            logger.info("Scout: Performing Hierarchical Retrieval...")
            # Step A: Top N Docs (Coarse)
            top_doc_ids = kg_store.retrieve_top_docs(query_emb, k=10)
            
            # Step B: Top M Chunks from those Docs (Fine)
            top_retrieved = kg_store.retrieve_top_chunks(query_emb, top_doc_ids, k=TOP_K_CHUNKS * 2) # Fetch extra for reranker
            
            top_stage2_chunks = [chunk for doc_id, chunk in top_retrieved]
            chunk_to_doc_map = {chunk: doc_id for doc_id, chunk in top_retrieved}
            
            # Step C: Cross Encoder Reranking
            best_chunks = top_stage2_chunks[:TOP_K_CHUNKS]
            try:
                top_stage3 = rerank_chunks(q, top_stage2_chunks, top_k=TOP_K_CHUNKS)
                if top_stage3:
                    best_chunks = top_stage3
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
            
            # Parent Document Retrieval
            # Identify which documents contained the best chunks
            best_doc_ids = list({chunk_to_doc_map[chunk] for chunk in best_chunks if chunk in chunk_to_doc_map})
            
            # Retrieve the full text (all chunks) for these highly relevant documents
            doc_all_chunks = kg_store.get_all_chunks_for_docs(best_doc_ids)
            
            # Compile full document arrays back to their source URLs
            source_content_map = {}
            for doc_id in best_doc_ids:
                meta = kg_store.get_doc_metadata(doc_id)
                url = meta.get("url", "unknown")
                if url not in source_content_map:
                    source_content_map[url] = []
                
                # Expand the extraction context to the entire document
                source_content_map[url].extend(doc_all_chunks.get(doc_id, []))

            # Build Source Map and Scraped Data Payload
            for url, matched_chunks in source_content_map.items():
                best_content = "\n\n".join(matched_chunks)
                
                # Evaluate source credibility
                cred_info = evaluate_source(url)

                source_id = f"[{next_id}]"
                source_map[source_id] = {
                    "url": url,
                    "snippet": best_content[:200] + "...",
                    "credibility_score": cred_info["score"],
                    "source_type": cred_info["type"]
                }
                next_id += 1
                
                all_scraped_data.append({
                    "source_id": source_id,
                    "url": url,
                    "content": best_content,
                    "query": q
                })
        else:
            logger.warning("Query embedding missing, skipping retrieval phase.")
            
    logger.info(f"Scout: Collected {len(all_scraped_data)} highly relevant chunk sets across queries.")
    
    # Audit log the source_map before propagating to the state
    logger.info(f"Scout: Source map has {len(source_map)} entries:")
    for sid, data in source_map.items():
        logger.info(f"  {sid}: {data.get('url', '?')} (cred={data.get('credibility_score', '?')}, type={data.get('source_type', '?')})")
    
    return {
        "scraped_data": all_scraped_data,
        "source_map": source_map,
        "active_node": "scout"
    }
