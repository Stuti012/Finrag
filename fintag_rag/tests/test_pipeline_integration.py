"""End-to-end pipeline wiring test using a fake retriever, so the full
FinTAG-RAG orchestration (query processing -> context filtering -> temporal
compatibility -> symbolic reasoning -> answer generation) can be exercised
without downloading any embedding/LLM models.
"""

import numpy as np

from src.config import FinTAGRAGConfig
from src.data import Chunk, Fact
from src.ontology import EnrichedQuery
from src.pipeline import FinTAGRAGPipeline
from src.retrieval import RetrievedChunk


class FakeRetriever:
    """Duck-types the parts of HybridRetriever the pipeline touches."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.embeddings = np.zeros((len(chunks), 4), dtype="float32")

    def retrieve(self, enriched_query: EnrichedQuery):
        # Pretend every chunk was retrieved and cross-encoder-scored highly.
        return [RetrievedChunk(chunk=c, score=0.9) for c in self.chunks]

    def sparse_only_retrieve(self, enriched_query, top_k):
        return [RetrievedChunk(chunk=c, score=0.5) for c in self.chunks[:top_k]]

    def dense_only_retrieve(self, enriched_query, top_k):
        return [RetrievedChunk(chunk=c, score=0.5) for c in self.chunks[:top_k]]


def _build_chunks():
    facts = [Fact(metric="net income", year=2019, value=100.0), Fact(metric="net income", year=2020, value=150.0)]
    return [Chunk(chunk_id="docA::table", doc_id="docA", text="net income (2019): 100; net income (2020): 150",
                  kind="table", fiscal_years=[2019, 2020], facts=facts)]


def test_full_mode_end_to_end_symbolic_answer():
    retriever = FakeRetriever(_build_chunks())
    pipeline = FinTAGRAGPipeline(retriever=retriever, config=FinTAGRAGConfig(), load_generator=False)
    result = pipeline.answer("What was the percentage increase in net income from 2019 to 2020?", mode="full")

    assert result.symbolic_result is not None
    assert result.symbolic_result.success
    assert abs(result.numeric_answer - 50.0) < 1e-6
    assert result.answer_text == "50.0"
    assert len(result.temporally_filtered) == 1


def test_bm25_and_dense_modes_skip_symbolic_reasoning():
    retriever = FakeRetriever(_build_chunks())
    pipeline = FinTAGRAGPipeline(retriever=retriever, config=FinTAGRAGConfig(), load_generator=False)
    for mode in ("bm25", "dense"):
        result = pipeline.answer("What was net income in 2020?", mode=mode)
        assert result.symbolic_result is None
        assert result.numeric_answer is None  # no generator loaded -> LLM path never ran
        assert len(result.retrieved) == 1


def test_llm_only_mode_has_no_retrieval():
    retriever = FakeRetriever(_build_chunks())
    pipeline = FinTAGRAGPipeline(retriever=retriever, config=FinTAGRAGConfig(), load_generator=False)
    result = pipeline.answer("What was net income in 2020?", mode="llm_only", gold_context_text="net income was 150 in 2020")
    assert result.retrieved == []
    assert result.symbolic_result is None
