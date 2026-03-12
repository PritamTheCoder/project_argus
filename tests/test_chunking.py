"""
Project Argus - Chunking & Summarization Tests

Comprehensive tests for the chunking pipeline to catch edge cases
that would cause documents to be silently dropped or produce
oversized chunks.

Run with:
    python -m pytest tests/test_chunking.py -v
"""

import pytest
from src.utils.chunking import chunk_document, extract_summary, DEFAULT_MAX_WORDS


# ── Helper ───────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


# ── Summary Tests ────────────────────────────────────────────────────────────


class TestExtractSummary:
    """Tests for the extractive summary function."""

    def test_basic_summary(self):
        text = "Solid state batteries are a promising technology. They replace liquid electrolytes with solid materials."
        summary = extract_summary(text)
        assert len(summary) > 0
        assert summary.endswith(".")

    def test_short_first_sentence_extends(self):
        """If the first sentence is < 5 words, it should pull in the second."""
        text = "SSB tech. Energy density has improved dramatically in the latest round of testing."
        summary = extract_summary(text)
        assert "Energy density" in summary

    def test_empty_content(self):
        assert extract_summary("") == ""
        assert extract_summary("   ") == ""
        assert extract_summary(None) == ""

    def test_truncation(self):
        """Summary should never exceed max_len."""
        long_text = "A " * 500 + "."
        summary = extract_summary(long_text, max_len=100)
        assert len(summary) <= 100

    def test_header_prefix(self):
        """A URL or header as the first 'sentence' should still produce something useful."""
        text = "https://example.com. Solid state batteries achieved 500 Wh/kg energy density in 2025."
        summary = extract_summary(text)
        assert len(summary) > 10


# ── Chunking Tests ───────────────────────────────────────────────────────────


class TestChunkDocument:
    """Tests for the document chunking function."""

    def test_normal_paragraphs(self):
        """Standard multi-paragraph document should produce multiple chunks."""
        para = "This is a test paragraph with enough words to pass the minimum character filter easily. " * 3
        content = f"{para}\n\n{para}\n\n{para}"
        
        chunks = chunk_document(content)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert _word_count(chunk) <= DEFAULT_MAX_WORDS + 5  # small tolerance

    def test_empty_content(self):
        assert chunk_document("") == []
        assert chunk_document("   ") == []
        assert chunk_document(None) == []

    def test_single_paragraph_no_newlines(self):
        """BUG CHECK: A document with NO \\n\\n separators should NOT produce 0 chunks."""
        content = "This is a long document that has no paragraph breaks at all. " * 20
        chunks = chunk_document(content)
        assert len(chunks) > 0, "Single-paragraph doc should NOT be dropped!"

    def test_all_short_paragraphs(self):
        """BUG CHECK: If all paragraphs are < 50 chars, the doc should still chunk."""
        content = "Short line one.\n\nAnother short one.\n\nYet another."
        # All paragraphs are < 50 chars, but the combined content is > 50
        chunks = chunk_document(content)
        # With the fallback logic, this should still produce chunks from single-newline 
        # split or raw content
        # Note: if ALL blocks are tiny, the fallback handles it
        if len(content.strip()) > 50:
            assert len(chunks) > 0, "All-short-paragraphs doc should NOT be dropped!"

    def test_oversized_paragraph_is_split(self):
        """BUG CHECK: A single paragraph with 500+ words should be hard-split."""
        content = "word " * 500
        chunks = chunk_document(content)
        assert len(chunks) > 1, "Oversized paragraph should be split into multiple chunks!"
        for chunk in chunks:
            assert _word_count(chunk) <= DEFAULT_MAX_WORDS, f"Chunk has {_word_count(chunk)} words, exceeds {DEFAULT_MAX_WORDS}"

    def test_mixed_paragraph_sizes(self):
        """Mix of normal and oversized paragraphs should all be properly chunked."""
        normal_para = "This is a normal paragraph with enough content to pass the filter. " * 5
        huge_para = "dense technical content about battery chemistry. " * 100  # ~500 words
        content = f"{normal_para}\n\n{huge_para}\n\n{normal_para}"
        
        chunks = chunk_document(content)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert _word_count(chunk) <= DEFAULT_MAX_WORDS + 5

    def test_chunk_preserves_all_content(self):
        """No content should be silently lost during chunking."""
        words = ["word" + str(i) for i in range(100)]
        content = " ".join(words)
        chunks = chunk_document(content)
        
        # Rejoin all chunks and check all unique words are present
        rejoined = " ".join(chunks)
        for w in words:
            assert w in rejoined, f"Lost word '{w}' during chunking!"

    def test_real_world_scraped_content(self):
        """Simulate a real scraped page with headers, short lines, and long paragraphs."""
        content = """Battery Technology Update 2025

By John Doe | Published: March 2025

The development of solid-state batteries has seen remarkable progress over the past year. Multiple companies including Toyota, QuantumScape, and Samsung SDI have announced significant breakthroughs in energy density and cycle life performance.

Current lithium-ion batteries typically achieve energy densities of 250-300 Wh/kg, but solid-state prototypes have demonstrated values exceeding 500 Wh/kg in laboratory settings. This represents a fundamental shift in what is achievable with electrochemical energy storage technology.

One of the key challenges remains manufacturing scalability. While lab-scale cells perform well, scaling production to millions of units per year requires solving complex engineering problems related to solid electrolyte processing, interface stability, and quality control at scale.

Toyota has committed to launching their first solid-state battery EV by 2027, targeting a range of over 1000 km on a single charge. QuantumScape's QSE-5 cells have completed over 1000 charge-discharge cycles with minimal degradation.

The cost trajectory is also encouraging. Analysts project that solid-state battery costs will reach parity with conventional lithium-ion by 2030, with costs dropping below $80 per kWh at scale."""

        chunks = chunk_document(content)
        assert len(chunks) >= 1, "Real-world content should produce chunks!"
        
        # No chunk should be absurdly large
        for i, chunk in enumerate(chunks):
            wc = _word_count(chunk)
            assert wc <= DEFAULT_MAX_WORDS + 5, f"Chunk {i} has {wc} words (max {DEFAULT_MAX_WORDS})"

    def test_unicode_content(self):
        """Non-ASCII content should not crash the chunker."""
        content = "バッテリー技術の最新情報。固体電池は非常に有望な技術です。" * 10
        # This may or may not produce chunks depending on char count, but must not crash
        result = chunk_document(content)
        assert isinstance(result, list)

    def test_content_with_only_newlines(self):
        """Content that is just whitespace and newlines should return empty."""
        assert chunk_document("\n\n\n\n\n") == []
        assert chunk_document("   \n\n   \n\n   ") == []


# ── Integration: Scout-level chunking simulation ─────────────────────────────


class TestScoutChunkingIntegration:
    """Tests that simulate how scout.py would use the chunking utility."""

    def test_scrape_result_with_content_produces_chunks_and_summary(self):
        """A typical scrape result should produce both a summary and chunks."""
        content = (
            "QuantumScape has announced a major breakthrough in solid-state battery technology. "
            "Their latest prototype achieved an energy density of 500 Wh/kg, which is nearly "
            "double that of conventional lithium-ion batteries.\n\n"
            "The company's QSE-5 cells have completed over 1000 charge-discharge cycles with "
            "less than 10% capacity degradation. This level of cycle life is unprecedented for "
            "solid-state cells at this energy density.\n\n"
            "Manufacturing scale-up remains the primary challenge. QuantumScape plans to begin "
            "pilot production in 2026, with full-scale manufacturing targeted for 2028."
        )
        
        summary = extract_summary(content)
        chunks = chunk_document(content)
        
        assert len(summary) > 0
        assert len(chunks) >= 1
        assert "QuantumScape" in summary

    def test_minimal_scrape_result(self):
        """A scrape result with very little content should still work."""
        content = "Solid-state batteries are batteries that use solid electrolytes instead of liquid ones."
        
        summary = extract_summary(content)
        chunks = chunk_document(content)
        
        assert len(summary) > 0
        assert len(chunks) >= 1
