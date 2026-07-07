"""Evaluation harness (thesis Chapter 4 / Section 4.4): computes real, run-time
metrics for the four accuracy dimensions -- context, temporal, numerical, and
causal -- and reproduces the thesis's baseline comparison (BM25 / Dense /
LLM-only / FinTAG-RAG) empirically rather than by copying the thesis's
reported figures.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .causal import CausalExtractor, evaluate_causal_accuracy
from .data import Chunk, FinQAExample, parse_financial_number
from .pipeline import FinTAGRAGPipeline, PipelineResult
from .temporal import temporal_compatibility


def _doc_id_of(example: FinQAExample) -> str:
    return example.id.rsplit("-", 1)[0] if "-" in example.id else example.id


def parse_gold_answer(example: FinQAExample) -> Optional[float]:
    return parse_financial_number(str(example.answer))


def answers_match(pred: Optional[float], gold: Optional[float], tolerance: float = 0.01) -> Optional[bool]:
    """Numeric match within tolerance, also treating a 100x scale difference
    (25.0 vs 0.25) as equivalent -- FinQA mixes percentage and decimal answer
    conventions across examples."""
    if pred is None or gold is None:
        return None
    if gold == 0:
        return abs(pred) <= tolerance
    if abs(pred - gold) / abs(gold) <= tolerance:
        return True
    if abs(pred - gold * 100) / abs(gold * 100) <= tolerance:
        return True
    if abs(pred - gold / 100) / abs(gold / 100 or 1e-9) <= tolerance:
        return True
    return False


def _gold_evidence_coverage(gold_evidence: List[str], chunks: List[Chunk]) -> Optional[float]:
    """Fraction of gold evidence snippets (FinQA `gold_inds`) whose content
    token-overlaps (>=60%) with the pooled retrieved text. A proxy for recall
    since our chunk linearization reformats raw table cells / sentences."""
    if not gold_evidence:
        return None
    pooled_tokens = set(re.findall(r"[a-z0-9]+", " ".join(c.text.lower() for c in chunks)))
    covered = 0
    counted = 0
    for snippet in gold_evidence:
        tokens = set(re.findall(r"[a-z0-9]+", str(snippet).lower()))
        if not tokens:
            continue
        counted += 1
        if len(tokens & pooled_tokens) / len(tokens) >= 0.6:
            covered += 1
    return covered / counted if counted else None


@dataclass
class EvalRecord:
    example_id: str
    mode: str
    question: str
    predicted_numeric: Optional[float] = None
    gold_numeric: Optional[float] = None
    is_correct: Optional[bool] = None
    context_precision: Optional[float] = None  # fraction of retrieved chunks from the correct source doc
    context_recall: Optional[float] = None  # fraction of gold evidence snippets covered
    temporal_alignment: Optional[float] = None  # fraction of final evidence matching the query's fiscal years
    n_evidence: int = 0
    time_seconds: float = 0.0
    error: Optional[str] = None


def evaluate_example_with_trace(
    pipeline: FinTAGRAGPipeline, example: FinQAExample, mode: str = "full"
) -> "tuple[EvalRecord, PipelineResult]":
    """Like `evaluate_example`, but also returns the full PipelineResult so a
    caller (e.g. the committee-demo notebook) can display the module-by-module
    reasoning trace without running the pipeline a second time."""
    start = time.time()
    doc_id = _doc_id_of(example)

    gold_context_text = None
    if mode == "llm_only":
        gold_context_text = (example.table_text + "\n" + example.context_text).strip()

    result: PipelineResult = pipeline.answer(example.question, mode=mode, gold_context_text=gold_context_text)

    gold_numeric = parse_gold_answer(example)
    is_correct = answers_match(result.numeric_answer, gold_numeric, pipeline.config.numeric_tolerance)

    evidence_chunks = result.temporally_filtered if mode == "full" else result.retrieved
    context_precision = None
    context_recall = None
    temporal_alignment = None
    if mode != "llm_only":
        if evidence_chunks:
            context_precision = sum(1 for c in evidence_chunks if c.chunk.doc_id == doc_id) / len(evidence_chunks)
        recall = _gold_evidence_coverage(example.gold_evidence, [c.chunk for c in evidence_chunks])
        context_recall = recall

        query_years = result.enriched_query.explicit_years
        if evidence_chunks:
            temporal_alignment = sum(
                1 for c in evidence_chunks if temporal_compatibility(query_years, c.chunk.fiscal_years) > 0 or not query_years
            ) / len(evidence_chunks)
        elif not query_years:
            temporal_alignment = None  # no evidence, nothing to score

    record = EvalRecord(
        example_id=example.id,
        mode=mode,
        question=example.question,
        predicted_numeric=result.numeric_answer,
        gold_numeric=gold_numeric,
        is_correct=is_correct,
        context_precision=context_precision,
        context_recall=context_recall,
        temporal_alignment=temporal_alignment,
        n_evidence=len(evidence_chunks),
        time_seconds=time.time() - start,
        error=result.symbolic_result.error if (result.symbolic_result and not result.symbolic_result.success) else None,
    )
    return record, result


def evaluate_example(pipeline: FinTAGRAGPipeline, example: FinQAExample, mode: str = "full") -> EvalRecord:
    record, _ = evaluate_example_with_trace(pipeline, example, mode=mode)
    return record


def _mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _median(values: List[float]) -> Optional[float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    n = len(values)
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2


def aggregate_metrics(records: List[EvalRecord]) -> Dict[str, float]:
    scored = [r for r in records if r.is_correct is not None]
    numeric_errors = [
        abs(r.predicted_numeric - r.gold_numeric)
        for r in records
        if r.predicted_numeric is not None and r.gold_numeric is not None
    ]
    n_total = len(records) or 1
    n_correct = sum(1 for r in scored if r.is_correct)
    return {
        "n_total": len(records),
        "n_scored": len(scored),
        # Accuracy conditional on the system having produced an answer at all
        # (the standard "exact match among answered questions" framing).
        "numerical_accuracy": (n_correct / len(scored)) if scored else 0.0,
        # Accuracy over every question, whether or not an answer was attempted
        # -- reported alongside the conditional figure so a low answer_rate
        # can't make the system look more accurate than it really is.
        "numerical_accuracy_unconditional": n_correct / n_total,
        "answer_rate": sum(1 for r in records if r.predicted_numeric is not None) / len(records) if records else 0.0,
        "mae": _mean(numeric_errors) or 0.0,
        "median_ae": _median(numeric_errors) or 0.0,
        "context_precision": _mean([r.context_precision for r in records]) or 0.0,
        "context_recall": _mean([r.context_recall for r in records]) or 0.0,
        "temporal_alignment": _mean([r.temporal_alignment for r in records]) or 0.0,
        "mean_time_seconds": _mean([r.time_seconds for r in records]) or 0.0,
    }


def evaluate_dataset(
    pipeline: FinTAGRAGPipeline,
    examples: List[FinQAExample],
    mode: str = "full",
    max_examples: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    subset = examples[:max_examples] if max_examples else examples
    records = []
    for i, ex in enumerate(subset):
        rec = evaluate_example(pipeline, ex, mode=mode)
        records.append(rec)
        if verbose and (i + 1) % 25 == 0:
            print(f"  [{mode}] {i + 1}/{len(subset)} evaluated")
    return {"records": records, "metrics": aggregate_metrics(records)}


def run_baseline_comparison(
    pipeline: FinTAGRAGPipeline,
    examples: List[FinQAExample],
    max_examples: int = 150,
    modes: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Reproduces the thesis's Table 5.1-5.4 style comparison, computed live
    on this run rather than copied from the paper."""
    modes = modes or ["bm25", "dense", "llm_only", "full"]
    results = {}
    for mode in modes:
        if verbose:
            print(f"Running baseline: {mode} ({max_examples} examples)...")
        results[mode] = evaluate_dataset(pipeline, examples, mode=mode, max_examples=max_examples, verbose=verbose)
        if verbose:
            m = results[mode]["metrics"]
            print(
                f"  {mode:10s} -> accuracy(answered)={m['numerical_accuracy']:.1%}  "
                f"accuracy(all)={m['numerical_accuracy_unconditional']:.1%}  "
                f"answer_rate={m['answer_rate']:.1%}  "
                f"context_precision={m['context_precision']:.1%}  "
                f"temporal_alignment={m['temporal_alignment']:.1%}  "
                f"median_ae={m['median_ae']:.2f}  mae={m['mae']:.2f}"
            )
    return results


def full_four_dimension_report(pipeline: FinTAGRAGPipeline, examples: List[FinQAExample], max_examples: int = 150) -> Dict:
    """Convenience wrapper producing the four headline numbers for the
    committee: context, temporal, numerical (FinTAG-RAG full mode) and causal
    (small hand-labeled set, see fintag_rag.src.causal)."""
    full_eval = evaluate_dataset(pipeline, examples, mode="full", max_examples=max_examples, verbose=True)
    causal_eval = evaluate_causal_accuracy(CausalExtractor())
    return {
        "context_precision": full_eval["metrics"]["context_precision"],
        "context_recall": full_eval["metrics"]["context_recall"],
        "temporal_alignment": full_eval["metrics"]["temporal_alignment"],
        "numerical_accuracy": full_eval["metrics"]["numerical_accuracy"],
        "numerical_accuracy_unconditional": full_eval["metrics"]["numerical_accuracy_unconditional"],
        "answer_rate": full_eval["metrics"]["answer_rate"],
        "mae": full_eval["metrics"]["mae"],
        "median_ae": full_eval["metrics"]["median_ae"],
        "causal_span_f1": causal_eval["mean_span_f1"],
        "causal_detection_rate": causal_eval["detection_rate"],
        "full_eval": full_eval,
        "causal_eval": causal_eval,
    }
