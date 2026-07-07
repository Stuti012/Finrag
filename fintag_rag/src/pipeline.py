"""FinTAG-RAG pipeline orchestrator (thesis Algorithm 7: Overall Framework).

Ties the six modules together in the order specified by the thesis:

    query processing -> hybrid retrieval -> context filtering ->
    temporal compatibility -> symbolic reasoning -> answer generation

`mode` also exposes the ablation / baseline configurations used in the
thesis's evaluation (Section 4.3): "bm25", "dense", and "llm_only" skip one or
more of these stages so that the value each stage adds can be measured
directly, matching Table 5.1-5.4 of the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .causal import CausalExtractor, CausalRelation, is_causal_question
from .config import FinTAGRAGConfig
from .generation import AnswerGenerator, extract_final_number
from .ontology import EnrichedQuery, QueryProcessor
from .retrieval import HybridRetriever, RetrievedChunk, filter_context
from .symbolic import SymbolicReasoner, SymbolicResult
from .temporal import TemporalCompatibilityModule


@dataclass
class PipelineResult:
    question: str
    mode: str
    enriched_query: EnrichedQuery
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    context_filtered: List[RetrievedChunk] = field(default_factory=list)
    temporally_filtered: List[RetrievedChunk] = field(default_factory=list)
    symbolic_result: Optional[SymbolicResult] = None
    causal_relations: List[CausalRelation] = field(default_factory=list)
    answer_text: str = ""
    numeric_answer: Optional[float] = None


class FinTAGRAGPipeline:
    """The full six-module FinTAG-RAG pipeline, plus baseline modes for comparison."""

    def __init__(self, retriever: HybridRetriever, config: FinTAGRAGConfig = None, load_generator: bool = True):
        self.retriever = retriever
        self.config = config or FinTAGRAGConfig()
        self.query_processor = QueryProcessor()
        self.temporal_module = TemporalCompatibilityModule(self.config)
        self.symbolic_reasoner = SymbolicReasoner(self.config)
        self.causal_extractor = CausalExtractor()
        self.generator = AnswerGenerator(self.config) if load_generator else None

    def answer(
        self,
        question: str,
        mode: str = "full",
        gold_context_text: Optional[str] = None,
    ) -> PipelineResult:
        enriched = self.query_processor.process(question)
        result = PipelineResult(question=question, mode=mode, enriched_query=enriched)

        if mode == "llm_only":
            # No retrieval at all -- the caller supplies the example's own
            # source document directly (the thesis's "gold-standard context"
            # upper-bound setting for this baseline, Section 5.5.2).
            evidence_texts = [gold_context_text] if gold_context_text else []
            if self.generator is not None:
                raw = self.generator.generate_llm_only(question, evidence_texts)
                result.answer_text = raw
                result.numeric_answer = extract_final_number(raw)
            return result

        if mode == "bm25":
            result.retrieved = self.retriever.sparse_only_retrieve(enriched, self.config.final_top_k)
        elif mode == "dense":
            result.retrieved = self.retriever.dense_only_retrieve(enriched, self.config.final_top_k)
        else:  # "full" (FinTAG-RAG)
            result.retrieved = self.retriever.retrieve(enriched)
            result.context_filtered = filter_context(result.retrieved, self.retriever, self.config)
            result.temporally_filtered = self.temporal_module.filter(enriched, result.context_filtered)

        if mode in ("bm25", "dense"):
            # Baselines: raw retrieved passages go straight to the LLM, which
            # must perform its own arithmetic -- no symbolic engine, no
            # temporal filtering (this isolates what hybrid retrieval +
            # temporal validation + symbolic reasoning each contribute).
            evidence_texts = [c.chunk.text for c in result.retrieved]
            if self.generator is not None:
                raw = self.generator.generate_llm_only(question, evidence_texts)
                result.answer_text = raw
                result.numeric_answer = extract_final_number(raw)
            return result

        # "full" mode: symbolic reasoning over temporally-validated operands.
        result.symbolic_result = self.symbolic_reasoner.reason_with_fallback(
            enriched,
            [
                ("temporally_filtered", result.temporally_filtered),
                ("context_filtered", result.context_filtered),
                ("retrieved", result.retrieved),
            ],
        )
        if result.symbolic_result.success:
            result.numeric_answer = result.symbolic_result.value

        if is_causal_question(question):
            combined_text = " ".join(c.chunk.text for c in result.temporally_filtered)
            result.causal_relations = self.causal_extractor.extract(combined_text)

        if self.generator is not None:
            result.answer_text = self.generator.generate(
                enriched, result.temporally_filtered, result.symbolic_result, result.causal_relations
            )
        elif result.symbolic_result and result.symbolic_result.success:
            result.answer_text = str(result.symbolic_result.value)

        return result
