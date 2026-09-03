import sqlite3
import os
import logging
from pathlib import Path
from src.config import DB_PATH

logger = logging.getLogger(__name__)

def init_db():
    """Initializes the SQLite database and creates the cache tables if absent."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS url_cache (
            url TEXT PRIMARY KEY,
            markdown_content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Metered search APIs bill per request, so an identical query costs real
    # money every time — and re-running the same research query is exactly what
    # development looks like. Cache the raw response body keyed by query+params.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_cache (
            key TEXT PRIMARY KEY,
            payload TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def get_cached_search(key: str, ttl_hours: int) -> str | None:
    """Cached raw search response for ``key``, if it is younger than ttl_hours.

    ttl_hours <= 0 disables the cache entirely (always a miss).
    """
    if ttl_hours <= 0:
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT payload FROM search_cache "
            "WHERE key = ? AND timestamp > datetime('now', ?)",
            (key, f"-{int(ttl_hours)} hours"),
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logger.warning(f"Search cache read failed: {e}")
        return None


def set_cached_search(key: str, payload: str) -> None:
    """Store a raw search response body, replacing any existing entry."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "REPLACE INTO search_cache (key, payload, timestamp) "
            "VALUES (?, ?, CURRENT_TIMESTAMP)",
            (key, payload),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Search cache write failed: {e}")

def get_cached_markdown(url: str) -> str | None:
    """
    Retrieves the cached markdown for a given URL.
    Returns None if the URL is not found in the cache.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT markdown_content FROM url_cache WHERE url = ?", (url,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        return None
    except Exception as e:
        logger.warning(f"Cache read failed for {url}: {e}")
        return None

def set_cached_markdown(url: str, markdown: str) -> None:
    """
    Stores the markdown content for a given URL into the cache.
    Replaces any existing entry for that URL.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "REPLACE INTO url_cache (url, markdown_content) VALUES (?, ?)", 
            (url, markdown)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Cache write failed for {url}: {e}")
