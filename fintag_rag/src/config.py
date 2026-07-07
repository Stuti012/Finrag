"""Central configuration for the FinTAG-RAG pipeline.

Starting point: the hyperparameters reported in Chapter 4 of the thesis
(Table 4.3): top_k_dense=20, top_k_sparse=20, rrf_k=60,
context_relevance_threshold=0.35, temporal_compatibility_threshold=0.5,
final_top_k=5. The candidate-pool sizes below (top_k_*, rerank_top_k,
final_top_k) have since been widened beyond those literal values: this
from-scratch implementation uses a general-purpose MS-MARCO cross-encoder
(not fine-tuned on financial text, unlike the thesis's system) and no
company-name-aware retrieval index, so a wider candidate pool measurably
improves the odds that the correct document survives to the symbolic
reasoning stage. The filtering *thresholds* (relevance tau, temporal delta)
are left at the thesis's reported values.
"""

from dataclasses import dataclass


@dataclass
class FinTAGRAGConfig:
    # -- Chunking (Section 3.4 of the thesis) --
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 32

    # -- Hybrid retrieval (Section 3.5.2 / Table 4.3) --
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_dense: int = 40
    top_k_sparse: int = 40
    rrf_k: int = 60
    rerank_top_k: int = 40

    # -- Context filtering (Section 3.5.3) --
    context_relevance_threshold: float = 0.35
    dedup_cosine_threshold: float = 0.95
    final_top_k: int = 8

    # -- Temporal compatibility (Section 3.5.4) --
    temporal_compatibility_threshold: float = 0.5

    # -- Answer generation --
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_temperature: float = 0.0
    llm_max_new_tokens: int = 300
    use_openai: bool = False
    openai_model: str = "gpt-3.5-turbo"

    # -- Numerical tolerance for exact/approximate match evaluation --
    numeric_tolerance: float = 0.01

    seed: int = 42
