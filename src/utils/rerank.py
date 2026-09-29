"""
Project Argus - Reranker Utility

Local CrossEncoder (ms-marco-MiniLM-L-6-v2) — runs fully offline, no API key.
Used by the Scout's hierarchical retrieval to score and select the most
relevant chunks from scraped documents.
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)

_local_model = None


def _get_local_reranker():
    """Lazy-load the local CrossEncoder reranker."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import CrossEncoder
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        logger.info("Loading local CrossEncoder ms-marco-MiniLM-L-6-v2...")
        _local_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    return _local_model


def get_reranker():
    """Return the local CrossEncoder for backward compat (used by cold-start scripts)."""
    return _get_local_reranker()


def rerank_chunks(query: str, chunks: List[str], top_k: int = 5) -> List[str]:
    """
    Rerank a list of text chunks against the query using the local CrossEncoder.

    Returns:
        List of top_k chunks sorted by relevance (most relevant first).
    """
    if not chunks:
        return []

    try:
        model = _get_local_reranker()
        pairs = [[query, chunk] for chunk in chunks]
        scores = model.predict(pairs)

        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        reranked = [chunk for _, chunk in scored_chunks[:top_k]]
        logger.info(f"Local Reranker: returned {len(reranked)}/{len(chunks)} chunks.")
        return reranked
    except Exception as e:
        logger.error(f"Local CrossEncoder reranking also failed: {e}")
        return chunks[:top_k]
