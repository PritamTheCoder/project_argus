"""
Project Argus - Reranker tests (local CrossEncoder only)
"""

from unittest.mock import MagicMock, patch

import src.utils.rerank as rerank


def _reset():
    rerank._local_model = None


def test_rerank_chunks_orders_by_score_descending():
    _reset()
    with patch("src.utils.rerank._get_local_reranker") as mock_local:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock_local.return_value = mock_model
        result = rerank.rerank_chunks("q", ["low", "high", "mid"], top_k=2)
    assert result == ["high", "mid"]


def test_rerank_chunks_respects_top_k():
    _reset()
    with patch("src.utils.rerank._get_local_reranker") as mock_local:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.2, 0.3, 0.4]
        mock_local.return_value = mock_model
        result = rerank.rerank_chunks("q", ["a", "b", "c", "d"], top_k=1)
    assert result == ["d"]


def test_empty_chunks_returns_empty_without_loading_the_model():
    _reset()
    with patch("src.utils.rerank._get_local_reranker") as mock_local:
        result = rerank.rerank_chunks("q", [], top_k=5)
    assert result == []
    mock_local.assert_not_called()


def test_local_model_failure_degrades_to_original_order():
    """A broken local model must not crash the run — return the first top_k
    chunks unscored rather than propagate the exception."""
    _reset()
    with patch("src.utils.rerank._get_local_reranker", side_effect=RuntimeError("model load failed")):
        result = rerank.rerank_chunks("q", ["a", "b", "c"], top_k=2)
    assert result == ["a", "b"]


def test_get_reranker_returns_the_local_model():
    _reset()
    with patch("src.utils.rerank._get_local_reranker", return_value="the-model") as mock_local:
        result = rerank.get_reranker()
    assert result == "the-model"
    mock_local.assert_called_once()
