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
            confidence REAL
        )
        """)
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
    
    def _serialize_f32(self, vector: List[float]) -> bytes:
        return struct.pack(f"{len(vector)}f", *vector)
    
    def store_facts(self, facts_with_embeddings: List[Dict[str, Any]]):
        if not facts_with_embeddings:
            return

        cursor = self.db.cursor()
        for fact in facts_with_embeddings:
            cursor.execute("""
            INSERT INTO facts (claim, source_url, source_excerpt, support_level, confidence)
            VALUES (?, ?, ?, ?, ?)
            """, (fact["claim"], fact["source_url"], fact["source_excerpt"], fact["support_level"], fact["confidence"]))
            
            fact_id = cursor.lastrowid
            
            # Insert embedding into vec table
            cursor.execute("""
            INSERT INTO vec_facts(rowid, embedding) VALUES (?, ?)
            """, (fact_id, self._serialize_f32(fact["embedding"])))
            
        self.db.commit()
        logger.info(f"Stored {len(facts_with_embeddings)} facts into Knowledge Graph.")

    def retrieve_relevant_facts(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT 
                f.id,
                f.claim,
                f.source_url,
                f.source_excerpt,
                f.support_level,
                f.confidence,
                DISTANCE
            FROM vec_facts v
            JOIN facts f ON f.id = v.rowid
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance ASC
        """, (self._serialize_f32(query_embedding), k))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "claim": row[1],
                "source_url": row[2],
                "source_excerpt": row[3],
                "support_level": row[4],
                "confidence": row[5],
                "distance": row[6]
            })
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

# Global singleton KG instance
kg_store = KnowledgeGraph()
