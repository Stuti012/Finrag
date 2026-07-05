"""Central configuration for the FinTAG-RAG pipeline.

Default values mirror the hyperparameters reported in Chapter 4 of the thesis
(Table 4.3), so that a run of this implementation is directly comparable to
the thesis's reported numbers.
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
    top_k_dense: int = 20
    top_k_sparse: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 20

    # -- Context filtering (Section 3.5.3) --
    context_relevance_threshold: float = 0.35
    dedup_cosine_threshold: float = 0.95
    final_top_k: int = 5

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
