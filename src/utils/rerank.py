"""
Project Argus - Reranker Utility

Primary: NVIDIA NVIDIARerank (nv-rerank-qa-mistral-4b:1) — cloud-based, high quality.
Fallback: Local CrossEncoder (ms-marco-MiniLM-L-6-v2) — runs fully offline.

The reranker is used by the Scout's hierarchical retrieval pipeline 
to score and select the most relevant chunks from scraped documents.
"""

import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_nvidia_reranker = None
_local_model = None


def _get_nvidia_reranker():
    """Lazy-load the NVIDIA Reranker client."""
    global _nvidia_reranker
    if _nvidia_reranker is None:
        try:
            from langchain_nvidia_ai_endpoints import NVIDIARerank
            api_key = os.environ.get("NVIDIA_API_KEY", "")
            if not api_key:
                logger.warning("NVIDIA_API_KEY not set. NVIDIA Reranker unavailable.")
                return None
            _nvidia_reranker = NVIDIARerank(
                model="nv-rerank-qa-mistral-4b:1",
                api_key=api_key,
            )
            logger.info("NVIDIA Reranker (nv-rerank-qa-mistral-4b:1) initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize NVIDIA Reranker: {e}")
            return None
    return _nvidia_reranker


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
    Rerank a list of text chunks against the query.

    Strategy:
      1. Try NVIDIA NVIDIARerank (cloud) — highest quality.
      2. If NVIDIA fails or is unavailable, fall back to local CrossEncoder.

    Returns:
        List of top_k chunks sorted by relevance (most relevant first).
    """
    if not chunks:
        return []

    # Try NVIDIA first
    nvidia = _get_nvidia_reranker()
    if nvidia is not None:
        try:
            from langchain_core.documents import Document
            docs = [Document(page_content=chunk) for chunk in chunks]
            results = nvidia.compress_documents(query=query, documents=docs)
            # Results come pre-sorted by relevance (highest first)
            reranked = [doc.page_content for doc in results[:top_k]]
            logger.info(f"NVIDIA Reranker: returned {len(reranked)}/{len(chunks)} chunks.")
            return reranked
        except Exception as e:
            logger.warning(f"NVIDIA Reranker failed: {e}. Falling back to local CrossEncoder.")

    # Fallback: local CrossEncoder
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
