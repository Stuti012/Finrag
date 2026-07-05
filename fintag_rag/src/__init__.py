"""FinTAG-RAG: Temporal-Aware Hybrid Retrieval and Symbolic Reasoning for Financial QA.

A clean, self-contained implementation of the six-module architecture described
in the FinTAG-RAG thesis:

    1. Query Processing      (fintag_rag.src.ontology)
    2. Hybrid Retrieval       (fintag_rag.src.retrieval)
    3. Context Filtering      (fintag_rag.src.retrieval)
    4. Temporal Compatibility (fintag_rag.src.temporal)
    5. Symbolic Reasoning     (fintag_rag.src.symbolic)
    6. Answer Generation      (fintag_rag.src.generation)

Plus a causal reasoning extension (fintag_rag.src.causal) requested alongside
the thesis's context / temporal / numerical accuracy dimensions.

See fintag_rag.src.pipeline.FinTAGRAGPipeline for the orchestrator that ties
all six stages together, and fintag_rag.src.evaluation for the metrics used
to report context, temporal, numerical, and causal accuracy against baselines.
"""

__version__ = "0.1.0"
