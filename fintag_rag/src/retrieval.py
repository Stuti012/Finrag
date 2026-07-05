"""Hybrid Retrieval + Context Filtering (thesis Sections 3.5.2 and 3.5.3).

Implements Algorithm 2 (Hybrid Retrieval): dense retrieval via FAISS, sparse
retrieval via BM25, Reciprocal Rank Fusion of the two rankings, and
Cross-Encoder re-ranking of the fused candidates -- followed by Algorithm 3
(Context Filtering): dropping passages below a relevance threshold and
removing near-duplicate passages.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import FinTAGRAGConfig
from .data import Chunk
from .ontology import EnrichedQuery


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # sigmoid-scaled cross-encoder relevance score in [0, 1]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HybridRetriever:
    """Dense (FAISS) + sparse (BM25) retrieval with RRF fusion and CE rerank."""

    def __init__(self, chunks: List[Chunk], config: FinTAGRAGConfig = None, show_progress: bool = True):
        self.chunks = chunks
        self.config = config or FinTAGRAGConfig()
        self._embedder = None
        self._cross_encoder = None
        self._show_progress = show_progress
        self._build_dense_index()
        self._build_sparse_index()

    # -- lazy model loading keeps import-time cheap for unit tests --
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.config.cross_encoder_model)
        return self._cross_encoder

    def _build_dense_index(self) -> None:
        import faiss

        texts = [c.text for c in self.chunks]
        self.embeddings = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=self._show_progress,
            normalize_embeddings=True,
        ).astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        if len(self.embeddings):
            self.index.add(self.embeddings)

    def _build_sparse_index(self) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [c.text.lower().split() for c in self.chunks]
        self._bm25_corpus = tokenized
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def dense_search(self, query_text: str, top_k: int) -> List[Tuple[int, float]]:
        if not len(self.chunks):
            return []
        q_emb = self.embedder.encode([query_text], normalize_embeddings=True).astype("float32")
        top_k = min(top_k, len(self.chunks))
        scores, idxs = self.index.search(q_emb, top_k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def sparse_search(self, query_text: str, top_k: int) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query_text.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

    @staticmethod
    def reciprocal_rank_fusion(
        dense: List[Tuple[int, float]], sparse: List[Tuple[int, float]], k: int = 60
    ) -> List[Tuple[int, float]]:
        """RRF(d) = sum over rankers of 1 / (k + rank(d))."""
        rrf_scores = defaultdict(float)
        for rank, (idx, _) in enumerate(sorted(dense, key=lambda x: -x[1])):
            rrf_scores[idx] += 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(sorted(sparse, key=lambda x: -x[1])):
            rrf_scores[idx] += 1.0 / (k + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: -x[1])

    def rerank(self, query_text: str, candidate_idxs: List[int]) -> List[Tuple[int, float]]:
        if not candidate_idxs:
            return []
        pairs = [(query_text, self.chunks[i].text) for i in candidate_idxs]
        raw_scores = self.cross_encoder.predict(pairs)
        scored = [(idx, _sigmoid(float(s))) for idx, s in zip(candidate_idxs, raw_scores)]
        return sorted(scored, key=lambda x: -x[1])

    def retrieve(self, enriched_query: EnrichedQuery) -> List[RetrievedChunk]:
        """Run the full hybrid-retrieval pipeline for one enriched query."""
        query_text = enriched_query.expanded_text()
        dense = self.dense_search(query_text, self.config.top_k_dense)
        sparse = self.sparse_search(query_text, self.config.top_k_sparse)
        fused = self.reciprocal_rank_fusion(dense, sparse, self.config.rrf_k)
        candidate_idxs = [idx for idx, _ in fused[: self.config.rerank_top_k]]
        reranked = self.rerank(enriched_query.normalized, candidate_idxs)
        return [RetrievedChunk(chunk=self.chunks[i], score=score) for i, score in reranked]

    def dense_only_retrieve(self, enriched_query: EnrichedQuery, top_k: int) -> List[RetrievedChunk]:
        """Dense-retrieval-only baseline (no BM25, no reranking)."""
        dense = self.dense_search(enriched_query.expanded_text(), top_k)
        # dense inner-product scores on normalized vectors already live in [-1, 1]
        return [RetrievedChunk(chunk=self.chunks[i], score=max(0.0, s)) for i, s in dense]

    def sparse_only_retrieve(self, enriched_query: EnrichedQuery, top_k: int) -> List[RetrievedChunk]:
        """BM25-only baseline."""
        sparse = self.sparse_search(enriched_query.expanded_text(), top_k)
        max_score = max((s for _, s in sparse), default=1.0) or 1.0
        return [RetrievedChunk(chunk=self.chunks[i], score=s / max_score) for i, s in sparse]

    def embedding_of(self, chunk_id: str) -> np.ndarray:
        for i, c in enumerate(self.chunks):
            if c.chunk_id == chunk_id:
                return self.embeddings[i]
        return None


def filter_context(
    retrieved: List[RetrievedChunk],
    retriever: HybridRetriever,
    config: FinTAGRAGConfig = None,
) -> List[RetrievedChunk]:
    """Algorithm 3: Context Filtering.

    Drops chunks below the relevance threshold tau, then removes near-duplicate
    chunks (cosine similarity above the dedup threshold) to reduce redundancy
    in the evidence passed to downstream reasoning.
    """
    config = config or FinTAGRAGConfig()
    kept: List[RetrievedChunk] = [r for r in retrieved if r.score >= config.context_relevance_threshold]

    deduped: List[RetrievedChunk] = []
    kept_embeddings: List[np.ndarray] = []
    id_to_idx = {c.chunk_id: i for i, c in enumerate(retriever.chunks)}
    for r in kept:
        idx = id_to_idx.get(r.chunk.chunk_id)
        emb = retriever.embeddings[idx] if idx is not None else None
        is_dup = False
        if emb is not None:
            for prior_emb in kept_embeddings:
                sim = float(np.dot(emb, prior_emb))
                if sim >= config.dedup_cosine_threshold:
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(r)
            if emb is not None:
                kept_embeddings.append(emb)

    return deduped[: config.final_top_k]
