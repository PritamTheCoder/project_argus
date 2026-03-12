"""
Project Argus - Chunking & Summarization Utilities

Extracts the chunking and summarization logic from scout.py into
testable, reusable functions.
"""

import logging
from src.config import MAX_CHUNK_TOKENS

logger = logging.getLogger(__name__)

# Rough token estimate: 1 word ~ 1.3 tokens
DEFAULT_MAX_WORDS = int(MAX_CHUNK_TOKENS / 1.3)  # ~230 words for 300 tokens


def extract_summary(content: str, max_len: int = 400) -> str:
    """
    Extract a 1-sentence coarse summary from the document content.
    Falls back to the first 200 characters if no clean sentence boundary is found.
    
    Args:
        content: Raw document text.
        max_len: Maximum character length of the summary.
    
    Returns:
        A short extractive summary string.
    """
    if not content or not content.strip():
        return ""
    
    # Normalize whitespace
    cleaned = content.replace("  ", " ").strip()
    
    # Split on sentence boundaries
    sentences = cleaned.split(". ")
    
    summary = sentences[0].rstrip(".") + "." if sentences else cleaned[:200]
    
    # If the first "sentence" is too short (e.g., a header or URL), extend
    if len(summary.split()) < 5 and len(sentences) > 1:
        summary = sentences[0] + ". " + sentences[1] + "."
    
    return summary[:max_len]


def chunk_document(content: str, max_words: int = DEFAULT_MAX_WORDS, min_paragraph_chars: int = 50) -> list[str]:
    """
    Break document content into chunks of approximately <= max_words words each.
    
    Strategy:
    1. Split on double-newlines to get paragraphs.
    2. Filter out very short paragraphs (< min_paragraph_chars).  
    3. If NO paragraphs survive the filter, fall back to splitting the raw
       content on single newlines or by word count.
    4. Accumulate paragraphs into chunks until adding the next paragraph
       would exceed max_words.
    5. If a single paragraph exceeds max_words, hard-split it by word count.
    
    Args:
        content: Raw document text.
        max_words: Maximum words per chunk (~230 for 300 tokens).
        min_paragraph_chars: Minimum characters for a paragraph to qualify.
    
    Returns:
        List of chunk strings, each approximately <= max_words words.
    """
    if not content or not content.strip():
        return []
    
    # Step 1: Try splitting on double-newline
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > min_paragraph_chars]
    
    # Step 2: Fallback — if no paragraphs survived, try single newlines
    if not paragraphs:
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > min_paragraph_chars]
    
    # Step 3: Last resort — treat entire content as one block
    if not paragraphs and len(content.strip()) > min_paragraph_chars:
        paragraphs = [content.strip()]
    
    if not paragraphs:
        return []
    
    chunks = []
    current_chunk = ""
    current_words = 0
    
    for p in paragraphs:
        p_words = len(p.split())
        
        # If a single paragraph exceeds max_words, hard-split it
        if p_words > max_words:
            # Flush any accumulated chunk first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_words = 0
            
            # Hard-split the oversized paragraph
            words = p.split()
            for i in range(0, len(words), max_words):
                sub_chunk = " ".join(words[i:i + max_words])
                chunks.append(sub_chunk)
            continue
        
        # Normal accumulation
        if current_words + p_words > max_words and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = p + " "
            current_words = p_words
        else:
            current_chunk += p + " "
            current_words += p_words
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks
