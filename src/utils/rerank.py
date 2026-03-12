import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Load model once at module level to avoid reloading it pointlessly
_model = None

def get_reranker():
    global _model
    if _model is None:
        import os
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        logger.info("Loading CrossEncoder ms-marco-MiniLM-L-6-v2...")
        _model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    return _model

def rerank_chunks(query: str, chunks: list, top_k: int = 5) -> list:
    """
    Rerank a list of chunks against the query using CrossEncoder.
    Batches the query-chunk pairs automatically.
    """
    if not chunks:
        return []
        
    model = get_reranker()
    pairs = [[query, chunk] for chunk in chunks]
    
    # Predict generates scores for the batch
    scores = model.predict(pairs)
    
    # Sort descending
    scored_chunks = list(zip(scores, chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    return [chunk for score, chunk in scored_chunks[:top_k]]
