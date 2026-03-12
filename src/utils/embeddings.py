import os
import json
import hashlib
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODE, LOCAL_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Local cache file
CACHE_FILE = "embedding_cache_local.json"

_local_model = None
_client = None

def get_local_model():
    global _local_model
    if _local_model is None:
        logger.info(f"Loading local embedding model {LOCAL_EMBEDDING_MODEL}...")
        _local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _local_model

def get_openai_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client

def _get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def _save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)

def get_embeddings(texts: List[str], model="text-embedding-3-large") -> List[List[float]]:
    if not texts:
        return []

    cache = _load_cache()
    embeddings = [None] * len(texts)
    to_embed_indices = []
    to_embed_texts = []
    
    for i, text in enumerate(texts):
        # Remove linebreaks to follow best practices
        cleaned_text = text.replace("\n", " ")
        
        # Include mode/model in hash to avoid collision between local and openai caches
        hash_input = f"{EMBEDDING_MODE}:{model}:{LOCAL_EMBEDDING_MODEL}:{cleaned_text}"
        h = _get_hash(hash_input)
        if h in cache:
            embeddings[i] = cache[h]
        else:
            to_embed_indices.append(i)
            to_embed_texts.append(cleaned_text)
            
    if to_embed_texts:
        if EMBEDDING_MODE == "openai":
            try:
                client = get_openai_client()
                BATCH_SIZE = 1000
                for i in range(0, len(to_embed_texts), BATCH_SIZE):
                    batch_texts = to_embed_texts[i:i+BATCH_SIZE]
                    response = client.embeddings.create(input=batch_texts, model=model)
                    
                    for j, data in enumerate(response.data):
                        idx = to_embed_indices[i + j]
                        emb = data.embedding
                        embeddings[idx] = emb
                        
                        hash_input = f"{EMBEDDING_MODE}:{model}:{LOCAL_EMBEDDING_MODEL}:{to_embed_texts[i+j]}"
                        cache[_get_hash(hash_input)] = emb
            except Exception as e:
                logger.warning(f"OpenAI embedding failed: {e}. Falling back to local model.")
                _get_local_embeddings(to_embed_texts, to_embed_indices, embeddings, cache)
        else:
            _get_local_embeddings(to_embed_texts, to_embed_indices, embeddings, cache)

        _save_cache(cache)
        
    return embeddings

def _get_local_embeddings(to_embed_texts, to_embed_indices, embeddings, cache):
    try:
        local_model = get_local_model()
        # sentence-transformers encodes in batches automatically
        encoded_embs = local_model.encode(to_embed_texts, show_progress_bar=False, convert_to_numpy=True)
        
        for i, emb in enumerate(encoded_embs):
            idx = to_embed_indices[i]
            emb_list = emb.tolist()
            embeddings[idx] = emb_list
            
            # Using same hash schema as above, default embedding target "text-embedding-3-large" for cache compat
            # We mock the model name here just for the cache key to remain consistent if passed
            hash_input = f"{EMBEDDING_MODE}:text-embedding-3-large:{LOCAL_EMBEDDING_MODEL}:{to_embed_texts[i]}"
            cache[_get_hash(hash_input)] = emb_list
            
    except Exception as e:
        logger.error(f"Local embedding failed: {e}")
        raise e
    
    return embeddings
