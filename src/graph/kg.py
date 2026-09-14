import re
import sqlite3
import sqlite_vec
import json
import logging
import struct
import threading
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.fts_available = True
        # sqlite3.Connection isn't thread-safe; API jobs run in separate threads.
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
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
        self._ensure_column("facts", "disputed", "INTEGER DEFAULT 0")
        self._ensure_column("facts", "dispute_reason", "TEXT DEFAULT ''")
        self._ensure_column("facts", "owner_key_hash", "TEXT DEFAULT ''")
        # Index to keep per-session retrieval fast as the fact store grows.
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id)")
        # all-MiniLM-L6-v2 embeddings are 384-dim
        self.db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
            embedding float[384]
        );
        """)

        # docs/chunks support Scout's hierarchical (doc -> chunk) retrieval
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            query TEXT,
            summary TEXT
        )
        """)
        # Scopes doc/chunk retrieval to the run that scraped them (same as `facts`).
        self._ensure_column("docs", "session_id", "TEXT")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_docs_session ON docs(session_id)")
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

        # BM25 keyword index over chunks, fused with vec_chunks at query time so
        # exact tokens (model numbers, tickers) aren't lost to embeddings alone.
        # Degrades to vector-only if the SQLite build lacks FTS5.
        try:
            self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content, tokenize='porter unicode61'
            );
            """)
        except sqlite3.OperationalError as e:
            self.fts_available = False
            logger.warning(f"FTS5 unavailable ({e}); BM25 chunk retrieval disabled, vector-only.")

        # Evidence-graph tables: sources, contradictions, consensus findings,
        # and gaps, so this data survives past the run that computed it.
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            source_id TEXT,
            url TEXT,
            credibility_score REAL,
            source_type TEXT,
            relevance_score REAL,
            as_of_date TEXT,
            snippet TEXT
        )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_sources_session ON sources(session_id)")

        self.db.execute("""
        CREATE TABLE IF NOT EXISTS contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            statement TEXT,
            claims TEXT,
            sources TEXT,
            source_count INTEGER
        )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_contradictions_session ON contradictions(session_id)")

        self.db.execute("""
        CREATE TABLE IF NOT EXISTS consensus_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            statement TEXT,
            claims TEXT,
            sources TEXT,
            source_count INTEGER
        )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_consensus_session ON consensus_findings(session_id)")

        # gap_type: "knowledge_gap" (a claim that failed verification) or
        # "coverage_gap" (a planned sub-question no fact ever answered).
        # iteration is the research-loop pass that produced it, so a gap
        # closed by a later re-search pass is still visible as history rather
        # than silently overwritten.
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            gap_type TEXT,
            description TEXT,
            iteration INTEGER
        )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_gaps_session ON gaps(session_id)")

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

    def store_facts(self, facts_with_embeddings: List[Dict[str, Any]], session_id: str = "", owner_key_hash: str = ""):
        if not facts_with_embeddings:
            return

        with self._lock:
            cursor = self.db.cursor()
            for fact in facts_with_embeddings:
                cursor.execute("""
                INSERT INTO facts (claim, source_url, source_excerpt, support_level, confidence, credibility_score, source_type, session_id, support_quote, corroboration_count, as_of_date, owner_key_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (fact["claim"], fact["source_url"], fact["source_excerpt"], fact["support_level"], fact["confidence"], fact.get("credibility_score", 0.4), fact.get("source_type", "Unverified/Web"), session_id or fact.get("session_id", ""), fact.get("support_quote", ""), fact.get("corroboration_count"), fact.get("as_of_date", ""), owner_key_hash))

                fact_id = cursor.lastrowid

                cursor.execute("""
                INSERT INTO vec_facts(rowid, embedding) VALUES (?, ?)
                """, (fact_id, self._serialize_f32(fact["embedding"])))

            self.db.commit()
        logger.info(f"Stored {len(facts_with_embeddings)} facts into Knowledge Graph (session={session_id or 'global'}).")

    def store_sources(self, sources: List[Dict[str, Any]], session_id: str = "") -> None:
        """Persist source_map entries. Each entry needs a ``source_id`` (the
        "[n]" citation label) added by the caller, since source_map's dict
        values don't carry their own key.

        Idempotent per (session_id, source_id) — source_map accumulates across
        research-loop iterations, so a naive insert would duplicate sources
        already written on an earlier pass."""
        if not sources:
            return
        with self._lock:
            cursor = self.db.cursor()
            cursor.execute("SELECT source_id FROM sources WHERE session_id = ?", (session_id,))
            already_stored = {row[0] for row in cursor.fetchall()}
            new_sources = [s for s in sources if s.get("source_id", "") not in already_stored]
            for s in new_sources:
                cursor.execute("""
                INSERT INTO sources (session_id, source_id, url, credibility_score, source_type, relevance_score, as_of_date, snippet)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, s.get("source_id", ""), s.get("url", ""),
                      s.get("credibility_score", 0.0), s.get("source_type", ""),
                      s.get("relevance_score"), s.get("as_of_date", ""), s.get("snippet", "")))
            self.db.commit()

    def store_contradictions(self, entries: List[Dict[str, Any]], session_id: str = "") -> None:
        """``entries`` is Consensus's contradiction list: statement, claims
        (list of claim text), sources (list of URLs), source_count."""
        self._store_cluster_entries("contradictions", entries, session_id)

    def store_consensus_findings(self, entries: List[Dict[str, Any]], session_id: str = "") -> None:
        """Same shape as ``store_contradictions``, for agreeing clusters."""
        self._store_cluster_entries("consensus_findings", entries, session_id)

    def _store_cluster_entries(self, table: str, entries: List[Dict[str, Any]], session_id: str) -> None:
        if not entries:
            return
        with self._lock:
            cursor = self.db.cursor()
            for e in entries:
                cursor.execute(f"""
                INSERT INTO {table} (session_id, statement, claims, sources, source_count)
                VALUES (?, ?, ?, ?, ?)
                """, (session_id, e.get("statement", ""),
                      json.dumps(e.get("claims", [])), json.dumps(e.get("sources", [])),
                      e.get("source_count", 0)))
            self.db.commit()

    def store_gaps(self, gaps: List[Dict[str, Any]], session_id: str = "") -> None:
        """``gaps`` entries: {"gap_type": "knowledge_gap"|"coverage_gap",
        "description": str, "iteration": int}."""
        if not gaps:
            return
        with self._lock:
            cursor = self.db.cursor()
            for g in gaps:
                cursor.execute("""
                INSERT INTO gaps (session_id, gap_type, description, iteration)
                VALUES (?, ?, ?, ?)
                """, (session_id, g.get("gap_type", ""), g.get("description", ""), g.get("iteration", 0)))
            self.db.commit()

    def flag_fact(self, session_id: str, fact_id: int, reason: str) -> bool:
        """Mark a fact disputed and zero its confidence, so existing
        confidence-threshold consumers (Librarian's cross-run reuse, the
        Critic's KG lookup) stop trusting it without any new filtering logic.
        Returns False if no such fact exists in this session."""
        with self._lock:
            cursor = self.db.cursor()
            cursor.execute(
                "UPDATE facts SET disputed = 1, dispute_reason = ?, confidence = 0.0 "
                "WHERE session_id = ? AND id = ?",
                (reason, session_id, fact_id),
            )
            self.db.commit()
            return cursor.rowcount > 0

    def get_evidence_graph(self, session_id: str) -> Dict[str, Any]:
        """Everything the evidence-graph API needs for one run: facts, sources,
        contradictions, consensus findings, and gaps, all scoped to session_id."""
        with self._lock:
            cursor = self.db.cursor()

            cursor.execute(
                "SELECT id, claim, source_url, source_excerpt, support_level, confidence, "
                "credibility_score, source_type, support_quote, corroboration_count, as_of_date, "
                "disputed, dispute_reason "
                "FROM facts WHERE session_id = ?", (session_id,),
            )
            cols = [d[0] for d in cursor.description]
            facts = [dict(zip(cols, row)) for row in cursor.fetchall()]

            cursor.execute(
                "SELECT id, source_id, url, credibility_score, source_type, relevance_score, "
                "as_of_date, snippet FROM sources WHERE session_id = ?", (session_id,),
            )
            cols = [d[0] for d in cursor.description]
            sources = [dict(zip(cols, row)) for row in cursor.fetchall()]

            contradictions = self._read_cluster_entries(cursor, "contradictions", session_id)
            consensus_findings = self._read_cluster_entries(cursor, "consensus_findings", session_id)

            cursor.execute(
                "SELECT id, gap_type, description, iteration FROM gaps WHERE session_id = ?",
                (session_id,),
            )
            cols = [d[0] for d in cursor.description]
            gaps = [dict(zip(cols, row)) for row in cursor.fetchall()]

        return {
            "facts": facts, "sources": sources,
            "contradictions": contradictions, "consensus_findings": consensus_findings,
            "gaps": gaps,
        }

    def _read_cluster_entries(self, cursor, table: str, session_id: str) -> List[Dict[str, Any]]:
        cursor.execute(
            f"SELECT id, statement, claims, sources, source_count FROM {table} WHERE session_id = ?",
            (session_id,),
        )
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        for r in rows:
            r["claims"] = json.loads(r["claims"]) if r["claims"] else []
            r["sources"] = json.loads(r["sources"]) if r["sources"] else []
        return rows

    def get_fact_detail(self, session_id: str, fact_id: int) -> Optional[Dict[str, Any]]:
        """One fact plus its source's full metadata and any contradiction
        entries that name its claim. Corroboration is reported only as the
        count already stored on the fact — the specific corroborating rows
        are not persisted, so a list of them cannot be honestly reconstructed."""
        with self._lock:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT id, claim, source_url, source_excerpt, support_level, confidence, "
                "credibility_score, source_type, support_quote, corroboration_count, as_of_date, "
                "disputed, dispute_reason "
                "FROM facts WHERE session_id = ? AND id = ?", (session_id, fact_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cursor.description]
            fact = dict(zip(cols, row))

            cursor.execute(
                "SELECT source_id, url, credibility_score, source_type, relevance_score, as_of_date "
                "FROM sources WHERE session_id = ? AND url = ? LIMIT 1",
                (session_id, fact["source_url"]),
            )
            src_row = cursor.fetchone()
            source = dict(zip([d[0] for d in cursor.description], src_row)) if src_row else None

            contradictions = [
                c for c in self._read_cluster_entries(cursor, "contradictions", session_id)
                if fact["claim"] in c["claims"]
            ]

        fact["source"] = source
        fact["contradictions"] = contradictions
        return fact

    def copy_session(self, source_session_id: str, target_session_id: str) -> None:
        """Copy one session's evidence-graph rows to another. A forked thread
        gets its own session_id, so without this its evidence graph would
        only show what's found after the fork, missing everything inherited.

        Embeddings (``vec_facts``) are skipped — that index only serves
        mid-run corroboration lookups, not evidence-graph display."""
        with self._lock:
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO facts (claim, source_url, source_excerpt, support_level, confidence,
                    credibility_score, source_type, session_id, support_quote, corroboration_count, as_of_date,
                    disputed, dispute_reason)
                SELECT claim, source_url, source_excerpt, support_level, confidence,
                    credibility_score, source_type, ?, support_quote, corroboration_count, as_of_date,
                    disputed, dispute_reason
                FROM facts WHERE session_id = ?
            """, (target_session_id, source_session_id))
            cursor.execute("""
                INSERT INTO sources (session_id, source_id, url, credibility_score, source_type,
                    relevance_score, as_of_date, snippet)
                SELECT ?, source_id, url, credibility_score, source_type,
                    relevance_score, as_of_date, snippet
                FROM sources WHERE session_id = ?
            """, (target_session_id, source_session_id))
            for table in ("contradictions", "consensus_findings"):
                cursor.execute(f"""
                    INSERT INTO {table} (session_id, statement, claims, sources, source_count)
                    SELECT ?, statement, claims, sources, source_count
                    FROM {table} WHERE session_id = ?
                """, (target_session_id, source_session_id))
            cursor.execute("""
                INSERT INTO gaps (session_id, gap_type, description, iteration)
                SELECT ?, gap_type, description, iteration FROM gaps WHERE session_id = ?
            """, (target_session_id, source_session_id))
            self.db.commit()

    def retrieve_relevant_facts(
        self, query_embedding: List[float], k: int = 5,
        session_id: Optional[str] = None, owner_key_hash: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over stored facts.

        ``session_id`` restricts results to one run. Leave it unset for a
        global, cross-run lookup — then pass ``owner_key_hash`` to restrict
        results to that owner's own facts, since the KG has no other way to
        keep one API key's research from leaking into another's.

        sqlite-vec applies the KNN ``MATCH`` cut before the JOIN, so we
        over-fetch a larger candidate pool and filter in Python — otherwise
        the scoping filter could leave fewer than ``k`` results.
        """
        # Over-fetch when scoping so the filter still yields up to k results.
        fetch_k = k if not (session_id or owner_key_hash) else max(k * 10, 100)

        with self._lock:
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
                    f.disputed,
                    f.dispute_reason,
                    f.owner_key_hash,
                    DISTANCE
                FROM vec_facts v
                JOIN facts f ON f.id = v.rowid
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance ASC
            """, (self._serialize_f32(query_embedding), fetch_k))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            if session_id and row[8] != session_id:
                continue
            if not session_id and owner_key_hash and row[11] != owner_key_hash:
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
                "disputed": row[9],
                "dispute_reason": row[10],
                "owner_key_hash": row[11],
                "distance": row[12]
            })
            if len(results) >= k:
                break
        return results

    def store_document_and_chunks(
        self, url: str, query: str, summary: str, summary_embedding: List[float],
        chunks: List[str], chunk_embeddings: List[List[float]], session_id: str = "",
    ) -> int:
        with self._lock:
            cursor = self.db.cursor()
            cursor.execute("""
            INSERT INTO docs (url, query, summary, session_id)
            VALUES (?, ?, ?, ?)
            """, (url, query, summary, session_id))

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
                if self.fts_available:
                    cursor.execute("""
                    INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)
                    """, (chunk_id, chunk))

            self.db.commit()
            return doc_id
        
    def retrieve_top_docs(self, query_embedding: List[float], k: int = 5, session_id: Optional[str] = None) -> List[int]:
        """Scoped like ``retrieve_relevant_facts``: over-fetch then filter by
        session in Python, since the vec0 KNN cut happens before any JOIN."""
        fetch_k = k if not session_id else max(k * 10, 100)
        with self._lock:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT v.rowid, d.session_id
                FROM vec_docs v
                JOIN docs d ON d.id = v.rowid
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance ASC
            """, (self._serialize_f32(query_embedding), fetch_k))
            rows = cursor.fetchall()

        results = []
        for doc_id, row_session in rows:
            if session_id and row_session != session_id:
                continue
            results.append(doc_id)
            if len(results) >= k:
                break
        return results

    def retrieve_top_chunks(self, query_embedding: List[float], doc_ids: List[int], k: int = 8) -> List[tuple[int, str]]:
        if not doc_ids:
            return []

        with self._lock:
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
            params = tuple(doc_ids) + (self._serialize_f32(query_embedding), k * 3)  # overfetch: doc_id filter is applied after the KNN MATCH
            cursor.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append((row[0], row[1]))
            if len(results) >= k:
                break

        return results

    @staticmethod
    def _fts5_match_expr(text: str) -> str:
        """Quote each token individually so punctuation in ``text`` can't break FTS5 syntax."""
        tokens = re.findall(r"\w+", text)
        return " OR ".join(f'"{t}"' for t in tokens)

    def retrieve_top_chunks_bm25(self, query_text: str, doc_ids: List[int], k: int = 8) -> List[tuple[int, str]]:
        """Sparse (BM25) counterpart to ``retrieve_top_chunks``. Returns ``[]`` on
        any failure so callers degrade to vector-only instead of erroring."""
        if not doc_ids or not self.fts_available:
            return []
        match_expr = self._fts5_match_expr(query_text)
        if not match_expr:
            return []

        placeholders = ",".join(["?"] * len(doc_ids))
        # MATCH must use the FTS5 table's real name, not its join alias `f`
        # (SQLite raises "no such column: f" otherwise) — `f.rank` is fine.
        query = f"""
            SELECT c.doc_id, c.content
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ? AND c.doc_id IN ({placeholders})
            ORDER BY f.rank
            LIMIT ?
        """
        try:
            with self._lock:
                cursor = self.db.cursor()
                cursor.execute(query, (match_expr, *doc_ids, k))
                return [(row[0], row[1]) for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            logger.warning(f"BM25 chunk retrieval failed ({e}); continuing vector-only.")
            return []

    def get_doc_metadata(self, doc_id: int) -> dict:
        with self._lock:
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

        with self._lock:
            cursor = self.db.cursor()
            placeholders = ",".join(["?"] * len(doc_ids))

            query = f"""
                SELECT doc_id, content
                FROM chunks
                WHERE doc_id IN ({placeholders})
                ORDER BY doc_id, id ASC
            """
            cursor.execute(query, tuple(doc_ids))
            rows = cursor.fetchall()

        results = {doc_id: [] for doc_id in doc_ids}
        for row in rows:
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
            with self._lock:
                cursor = self.db.cursor()
                cursor.execute("""
                    SELECT f.claim, f.support_level, f.confidence, f.session_id
                    FROM vec_facts v
                    JOIN facts f ON f.id = v.rowid
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance ASC
                """, (self._serialize_f32(query_embedding), fetch_k))
                rows = cursor.fetchall()

            gaps = []
            for row in rows:
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
        """Clear all temporary documents and chunks (every session). Keeps
        verified facts. Manual/admin use only, not called by the pipeline."""
        with self._lock:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM docs")
            cursor.execute("DELETE FROM vec_docs")
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM vec_chunks")
            if self.fts_available:
                cursor.execute("DELETE FROM chunks_fts")
            self.db.commit()
        logger.info("Cleared Vector DB scratchpad (docs and chunks).")

kg_store = KnowledgeGraph()

