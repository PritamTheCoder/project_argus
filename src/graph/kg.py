import sqlite3
import sqlite_vec
import json
import logging
import struct
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_db()

    def _init_db(self):
        # Create metadata table for facts
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim TEXT,
            source_url TEXT,
            source_excerpt TEXT,
            support_level TEXT,
            confidence REAL,
            credibility_score REAL,
            source_type TEXT,
            session_id TEXT,
            support_quote TEXT,
            corroboration_count INTEGER,
            as_of_date TEXT
        )
        """)
        # Migrate older DBs that pre-date newer columns (idempotent).
        self._ensure_column("facts", "session_id", "TEXT")
        self._ensure_column("facts", "support_quote", "TEXT")
        self._ensure_column("facts", "corroboration_count", "INTEGER")
        self._ensure_column("facts", "as_of_date", "TEXT")
        # Index to keep per-session retrieval fast as the fact store grows.
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id)")
        # Create virtual table for vector embeddings (all-MiniLM-L6-v2 is 384 dims)
        self.db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
            embedding float[384]
        );
        """)
        
        # New tables for Scout hierarchical retrieval
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            query TEXT,
            summary TEXT
        )
        """)
        self.db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs USING vec0(
            embedding float[384]
        );
        """)
        
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            content TEXT,
            FOREIGN KEY (doc_id) REFERENCES docs(id)
        )
        """)
        self.db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[384]
        );
        """)

        self.db.commit()
    
    def _ensure_column(self, table: str, column: str, col_type: str):
        """Idempotently add a column to an existing table (lightweight migration)."""
        cursor = self.db.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if column not in existing:
            logger.info(f"KG migration: adding column {table}.{column} ({col_type}).")
            self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            self.db.commit()

    def _serialize_f32(self, vector: List[float]) -> bytes:
        return struct.pack(f"{len(vector)}f", *vector)

    def store_facts(self, facts_with_embeddings: List[Dict[str, Any]], session_id: str = ""):
        if not facts_with_embeddings:
            return

        cursor = self.db.cursor()
        for fact in facts_with_embeddings:
            cursor.execute("""
            INSERT INTO facts (claim, source_url, source_excerpt, support_level, confidence, credibility_score, source_type, session_id, support_quote, corroboration_count, as_of_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fact["claim"], fact["source_url"], fact["source_excerpt"], fact["support_level"], fact["confidence"], fact.get("credibility_score", 0.4), fact.get("source_type", "Unverified/Web"), session_id or fact.get("session_id", ""), fact.get("support_quote", ""), fact.get("corroboration_count"), fact.get("as_of_date", "")))

            fact_id = cursor.lastrowid

            # Insert embedding into vec table
            cursor.execute("""
            INSERT INTO vec_facts(rowid, embedding) VALUES (?, ?)
            """, (fact_id, self._serialize_f32(fact["embedding"])))

        self.db.commit()
        logger.info(f"Stored {len(facts_with_embeddings)} facts into Knowledge Graph (session={session_id or 'global'}).")

    def retrieve_relevant_facts(self, query_embedding: List[float], k: int = 5, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Semantic search over stored facts.

        When ``session_id`` is provided, results are restricted to facts gathered
        in that run. Because sqlite-vec applies the KNN ``MATCH`` cut before the
        JOIN, we over-fetch a larger candidate pool and filter by session in
        Python so the requested ``k`` is honoured per-session rather than globally.
        """
        # Over-fetch when scoping so the session filter still yields up to k results.
        fetch_k = k if not session_id else max(k * 10, 100)

        cursor = self.db.cursor()
        cursor.execute("""
            SELECT
                f.id,
                f.claim,
                f.source_url,
                f.source_excerpt,
                f.support_level,
                f.confidence,
                f.credibility_score,
                f.source_type,
                f.session_id,
                DISTANCE
            FROM vec_facts v
            JOIN facts f ON f.id = v.rowid
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance ASC
        """, (self._serialize_f32(query_embedding), fetch_k))

        results = []
        for row in cursor.fetchall():
            if session_id and row[8] != session_id:
                continue
            results.append({
                "id": row[0],
                "claim": row[1],
                "source_url": row[2],
                "source_excerpt": row[3],
                "support_level": row[4],
                "confidence": row[5],
                "credibility_score": row[6],
                "source_type": row[7],
                "session_id": row[8],
                "distance": row[9]
            })
            if len(results) >= k:
                break
        return results

    def store_document_and_chunks(self, url: str, query: str, summary: str, summary_embedding: List[float], chunks: List[str], chunk_embeddings: List[List[float]]) -> int:
        cursor = self.db.cursor()
        cursor.execute("""
        INSERT INTO docs (url, query, summary)
        VALUES (?, ?, ?)
        """, (url, query, summary))
        
        doc_id = cursor.lastrowid
        cursor.execute("""
        INSERT INTO vec_docs(rowid, embedding) VALUES (?, ?)
        """, (doc_id, self._serialize_f32(summary_embedding)))
        
        for chunk, emb in zip(chunks, chunk_embeddings):
            if not chunk.strip():
                continue
            cursor.execute("""
            INSERT INTO chunks (doc_id, content) VALUES (?, ?)
            """, (doc_id, chunk))
            chunk_id = cursor.lastrowid
            cursor.execute("""
            INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)
            """, (chunk_id, self._serialize_f32(emb)))
            
        self.db.commit()
        return doc_id
        
    def retrieve_top_docs(self, query_embedding: List[float], k: int = 5) -> List[int]:
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT rowid FROM vec_docs
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance ASC
        """, (self._serialize_f32(query_embedding), k))
        return [row[0] for row in cursor.fetchall()]

    def retrieve_top_chunks(self, query_embedding: List[float], doc_ids: List[int], k: int = 8) -> List[tuple[int, str]]:
        if not doc_ids:
            return []
        
        cursor = self.db.cursor()
        placeholders = ",".join(["?"] * len(doc_ids))
        
        query = f"""
            SELECT c.doc_id, c.content 
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.rowid
            WHERE c.doc_id IN ({placeholders})
              AND v.embedding MATCH ? AND v.k = ?
            ORDER BY v.distance ASC
        """
        params = tuple(doc_ids) + (self._serialize_f32(query_embedding), k * 3) # overfetch slightly to account for doc_id filter selectivity
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append((row[0], row[1]))
            if len(results) >= k:
                break
                
        return results

    def get_doc_metadata(self, doc_id: int) -> dict:
        cursor = self.db.cursor()
        cursor.execute("SELECT url, query, summary FROM docs WHERE id=?", (doc_id,))
        row = cursor.fetchone()
        if row:
            return {"url": row[0], "query": row[1], "summary": row[2]}
        return {}
        
    def get_all_chunks_for_docs(self, doc_ids: List[int]) -> Dict[int, List[str]]:
        """Retrieve all chunks for the given document IDs, ordered by their original sequence."""
        if not doc_ids:
            return {}
            
        cursor = self.db.cursor()
        placeholders = ",".join(["?"] * len(doc_ids))
        
        query = f"""
            SELECT doc_id, content 
            FROM chunks 
            WHERE doc_id IN ({placeholders})
            ORDER BY doc_id, id ASC
        """
        cursor.execute(query, tuple(doc_ids))
        
        results = {doc_id: [] for doc_id in doc_ids}
        for row in cursor.fetchall():
            results[row[0]].append(row[1])
            
        return results

    def find_gaps(self, query_embedding: List[float], k: int = 20, session_id: Optional[str] = None) -> List[str]:
        """
        Return claims near the query vector that have weak or no evidential support.
        These represent concrete knowledge gaps the Reflector should fill.

        When ``session_id`` is provided, only gaps from the current run are
        considered (avoids resurfacing stale gaps from earlier sessions).
        """
        fetch_k = k if not session_id else max(k * 10, 100)
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT f.claim, f.support_level, f.confidence, f.session_id
                FROM vec_facts v
                JOIN facts f ON f.id = v.rowid
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance ASC
            """, (self._serialize_f32(query_embedding), fetch_k))

            gaps = []
            for row in cursor.fetchall():
                claim, support_level, confidence, row_session = row
                if session_id and row_session != session_id:
                    continue
                if support_level in ("NOT_SUPPORTED", "UNCERTAIN") or (confidence is not None and confidence < 0.4):
                    gaps.append(claim)
                if len(gaps) >= k:
                    break
            return gaps
        except Exception as e:
            logger.warning(f"find_gaps() failed: {e}")
            return []

    def clear_scratchpad(self):
        """Clear out temporary documents and chunks from the vector database. Keeps verified facts."""
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM docs")
        cursor.execute("DELETE FROM vec_docs")
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM vec_chunks")
        self.db.commit()
        logger.info("Cleared Vector DB scratchpad (docs and chunks).")

# Global singleton KG instance
kg_store = KnowledgeGraph()

