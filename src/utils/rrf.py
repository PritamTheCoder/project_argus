"""Reciprocal Rank Fusion — combines ranked retrieval lists (e.g. dense vector +
BM25 keyword search) without needing to normalize scores on incompatible scales."""

from typing import Hashable, List, Sequence, TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[T]], k: int = 60) -> List[T]:
    """score(item) = sum(1 / (k + rank)) across every list it appears in (1-indexed
    rank). Items are deduped by equality/hash. Returns items sorted by score, descending."""
    scores: dict[T, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda item: scores[item], reverse=True)
