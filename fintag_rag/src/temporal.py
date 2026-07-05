"""Temporal Compatibility Module (thesis Section 3.5.4, Algorithm 4).

Computes T(d, q) = |T(d) intersect T(q)| / |T(q)| for each retrieved passage
and discards passages below the compatibility threshold delta, preventing
cross-fiscal-year contamination of the operands that reach the symbolic
reasoning engine (Section 3.5.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import FinTAGRAGConfig
from .ontology import EnrichedQuery
from .retrieval import RetrievedChunk


@dataclass
class TemporalScoredChunk:
    retrieved: RetrievedChunk
    temporal_score: float
    query_years: List[int]
    chunk_years: List[int]


def temporal_compatibility(query_years: List[int], chunk_years: List[int]) -> float:
    """T(d, q) = |T(d) ^ T(q)| / |T(q)|.

    When the query carries no explicit fiscal-year constraint (|T(q)| = 0),
    there is nothing to validate against, so the passage is not penalized.
    """
    if not query_years:
        return 1.0
    overlap = set(query_years) & set(chunk_years)
    return len(overlap) / len(query_years)


class TemporalCompatibilityModule:
    def __init__(self, config: FinTAGRAGConfig = None):
        self.config = config or FinTAGRAGConfig()

    def score(self, enriched_query: EnrichedQuery, chunks: List[RetrievedChunk]) -> List[TemporalScoredChunk]:
        query_years = enriched_query.explicit_years
        return [
            TemporalScoredChunk(
                retrieved=r,
                temporal_score=temporal_compatibility(query_years, r.chunk.fiscal_years),
                query_years=query_years,
                chunk_years=r.chunk.fiscal_years,
            )
            for r in chunks
        ]

    def filter(self, enriched_query: EnrichedQuery, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Algorithm 4: discard passages with T(d, q) < delta."""
        scored = self.score(enriched_query, chunks)
        return [s.retrieved for s in scored if s.temporal_score >= self.config.temporal_compatibility_threshold]
