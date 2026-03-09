import sqlite3
import os
import logging
from pathlib import Path
from src.config import DB_PATH

logger = logging.getLogger(__name__)

def init_db():
    """Initializes the SQLite database and creates the url_cache table if it doesn't exist."""
    # Ensure the directory exists
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
    conn.commit()
    conn.close()

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
