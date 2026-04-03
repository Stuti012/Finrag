"""Evaluation metrics for the Financial QA system.

Provides comprehensive metrics for:
1. Numerical reasoning accuracy (program + execution accuracy)
2. Context filtering quality (retrieval precision/recall)
3. Causality detection performance
4. Implicit temporal reasoning accuracy
5. Overall QA performance
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.financial_utils import answers_match, normalize_answer, parse_financial_number


class NumericalReasoningMetrics:
    """Metrics for evaluating numerical reasoning performance."""

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def program_accuracy(
        self, predicted_steps: List[Dict], gold_program: List[str]
    ) -> Dict[str, float]:
        """Evaluate if the generated program structure matches the gold program.

        Measures:
        - Operation match rate: % of operations correctly identified
        - Step-wise accuracy: % of steps with correct operation and arguments
        """
        if not gold_program or not predicted_steps:
            return {
                "operation_match_rate": 0.0,
                "step_accuracy": 0.0,
                "program_length_match": gold_program is not None and predicted_steps is not None,
            }

        gold_ops = []
        for step in gold_program:
            match = re.match(r"(\w+)\(", step.strip())
            if match:
                gold_ops.append(match.group(1).lower())

        pred_ops = [s.get("operation", "").lower() for s in predicted_steps]

        # Operation match rate
        if gold_ops:
            matches = sum(
                1 for g, p in zip(gold_ops, pred_ops) if g == p
            )
            op_match = matches / len(gold_ops)
        else:
            op_match = 0.0

        # Step accuracy (exact match per step)
        min_len = min(len(gold_ops), len(pred_ops))
        step_matches = sum(1 for i in range(min_len) if gold_ops[i] == pred_ops[i])
        step_acc = step_matches / max(len(gold_ops), 1)

        return {
            "operation_match_rate": op_match,
            "step_accuracy": step_acc,
            "program_length_match": len(gold_ops) == len(pred_ops),
            "gold_steps": len(gold_ops),
            "predicted_steps": len(pred_ops),
        }

    def execution_accuracy(
        self, predicted_answer: str, gold_answer: str
    ) -> Dict[str, float]:
        """Evaluate if the computed answer matches the gold answer.

        Returns exact match, numerical closeness, and relative error.
        """
        exact_match = answers_match(predicted_answer, gold_answer, self.tolerance)

        result = {
            "exact_match": float(exact_match),
            "predicted": predicted_answer,
            "gold": gold_answer,
        }

        # Compute relative error for numerical answers
        try:
            pred_val = float(normalize_answer(predicted_answer))
            gold_val = float(normalize_answer(gold_answer))
            if gold_val != 0:
                result["relative_error"] = abs(pred_val - gold_val) / abs(gold_val)
            else:
                result["relative_error"] = abs(pred_val)
            result["absolute_error"] = abs(pred_val - gold_val)
        except (ValueError, TypeError):
            result["relative_error"] = None
            result["absolute_error"] = None

        return result

    def evaluate_batch(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate numerical reasoning across a batch of results."""
        execution_scores = []
        program_scores = []
        relative_errors = []

        for r in results:
            # Execution accuracy
            if r.get("predicted_answer") and r.get("gold_answer"):
                ea = self.execution_accuracy(r["predicted_answer"], r["gold_answer"])
                execution_scores.append(ea["exact_match"])
                if ea.get("relative_error") is not None:
                    relative_errors.append(ea["relative_error"])

            # Program accuracy
            num_info = r.get("numerical", {})
            if num_info.get("program_steps") and r.get("gold_answer"):
                # We need the gold program from the example
                pass

        return {
            "execution_accuracy": np.mean(execution_scores) if execution_scores else 0.0,
            "num_evaluated": len(execution_scores),
            "mean_relative_error": np.mean(relative_errors) if relative_errors else None,
            "median_relative_error": np.median(relative_errors) if relative_errors else None,
        }


class ContextFilteringMetrics:
    """Metrics for evaluating context retrieval and filtering quality."""

    def retrieval_precision_recall(
        self,
        retrieved_docs: List[str],
        gold_evidence: List[str],
    ) -> Dict[str, float]:
        """Compute precision and recall for retrieved context.

        Args:
            retrieved_docs: Retrieved document texts.
            gold_evidence: Gold standard evidence texts.
        """
        if not gold_evidence:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Compute overlap using substring matching
        relevant_retrieved = 0
        for retrieved in retrieved_docs:
            r_lower = retrieved.lower().strip()
            for gold in gold_evidence:
                g_lower = gold.lower().strip()
                # Check if there's significant overlap
                if g_lower in r_lower or r_lower in g_lower:
                    relevant_retrieved += 1
                    break
                # Partial overlap check (jaccard on words)
                r_words = set(r_lower.split())
                g_words = set(g_lower.split())
                if r_words and g_words:
                    jaccard = len(r_words & g_words) / len(r_words | g_words)
                    if jaccard > 0.3:
                        relevant_retrieved += 1
                        break

        precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0
        recall = relevant_retrieved / len(gold_evidence) if gold_evidence else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def context_sufficiency(
        self,
        retrieved_context: str,
        question: str,
        gold_answer: str,
    ) -> Dict[str, float]:
        """Check if retrieved context contains enough info to answer the question."""
        if not retrieved_context:
            return {"has_numbers": False, "has_answer_value": False, "sufficiency_score": 0.0}

        ctx_lower = retrieved_context.lower()

        # Check if answer value appears in context
        answer_norm = normalize_answer(gold_answer)
        has_answer = answer_norm in ctx_lower

        # Check if relevant numbers are present
        from ..utils.financial_utils import extract_numbers_from_text
        ctx_numbers = extract_numbers_from_text(retrieved_context)
        has_numbers = len(ctx_numbers) > 0

        # Compute sufficiency score
        score = 0.0
        if has_numbers:
            score += 0.3
        if has_answer:
            score += 0.5

        # Check keyword overlap with question
        q_words = set(question.lower().split()) - {"the", "a", "is", "was", "of", "in", "what", "how"}
        ctx_words = set(ctx_lower.split())
        overlap = len(q_words & ctx_words) / max(len(q_words), 1)
        score += 0.2 * overlap

        return {
            "has_numbers": has_numbers,
            "has_answer_value": has_answer,
            "keyword_overlap": overlap,
            "sufficiency_score": min(1.0, score),
        }

    def evaluate_batch(
        self, results: List[Dict[str, Any]], examples: List[Any] = None
    ) -> Dict[str, float]:
        """Evaluate context filtering across a batch."""
        precisions = []
        recalls = []
        f1s = []
        sufficiency_scores = []

        for i, r in enumerate(results):
            retrieval = r.get("retrieval", {})
            all_retrieved = []
            for ctx in retrieval.get("table_contexts", []):
                all_retrieved.append(ctx.get("text", ""))
            for ctx in retrieval.get("text_contexts", []):
                all_retrieved.append(ctx.get("text", ""))

            gold_evidence = []
            if examples and i < len(examples):
                gold_evidence = examples[i].gold_evidence

            if all_retrieved:
                pr = self.retrieval_precision_recall(all_retrieved, gold_evidence)
                precisions.append(pr["precision"])
                recalls.append(pr["recall"])
                f1s.append(pr["f1"])

                full_ctx = " ".join(all_retrieved)
                suff = self.context_sufficiency(
                    full_ctx, r.get("question", ""), r.get("gold_answer", "")
                )
                sufficiency_scores.append(suff["sufficiency_score"])

        return {
            "mean_precision": np.mean(precisions) if precisions else 0.0,
            "mean_recall": np.mean(recalls) if recalls else 0.0,
            "mean_f1": np.mean(f1s) if f1s else 0.0,
            "mean_sufficiency": np.mean(sufficiency_scores) if sufficiency_scores else 0.0,
            "num_evaluated": len(precisions),
        }


class CausalityDetectionMetrics:
    """Metrics for evaluating causality detection performance."""

    def causal_detection_accuracy(
        self, detected_relations: List[Dict], is_causal_question: bool
    ) -> Dict[str, float]:
        """Evaluate causality detection quality."""
        has_detections = len(detected_relations) > 0

        result = {
            "is_causal_question": is_causal_question,
            "num_relations_detected": len(detected_relations),
            "detection_triggered": has_detections,
        }

        if detected_relations:
            confidences = [r.get("confidence", 0) for r in detected_relations]
            result["mean_confidence"] = np.mean(confidences)
            result["max_confidence"] = max(confidences)

            # Check diversity of causal types
            types = set(r.get("relation_type", "unknown") for r in detected_relations)
            result["relation_types"] = list(types)
            result["type_diversity"] = len(types)
        else:
            result["mean_confidence"] = 0.0
            result["max_confidence"] = 0.0
            result["relation_types"] = []
            result["type_diversity"] = 0

        return result

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate causality detection across a batch."""
        causal_questions = 0
        correctly_detected = 0
        all_confidences = []
        total_relations = 0

        for r in results:
            causal_info = r.get("causal", {})
            is_causal = causal_info.get("is_causal", False)

            if is_causal:
                causal_questions += 1
                relations = causal_info.get("causal_relations", [])
                if relations:
                    correctly_detected += 1
                    total_relations += len(relations)
                    all_confidences.extend(
                        rel.get("confidence", 0) for rel in relations
                    )

        return {
            "causal_questions_found": causal_questions,
            "detection_rate": correctly_detected / max(causal_questions, 1),
            "mean_confidence": np.mean(all_confidences) if all_confidences else 0.0,
            "total_relations_detected": total_relations,
            "avg_relations_per_question": total_relations / max(causal_questions, 1),
        }


class TemporalReasoningMetrics:
    """Metrics for evaluating temporal reasoning performance."""

    def temporal_extraction_accuracy(
        self,
        extracted_entities: List[Dict],
        question: str,
        table: List[List[str]],
    ) -> Dict[str, float]:
        """Evaluate temporal entity extraction quality."""
        # Extract ground truth years from question and table
        year_pattern = re.compile(r"\b(19\d{2}|20\d{2})\b")
        gt_years = set()
        for m in year_pattern.finditer(question):
            gt_years.add(int(m.group(1)))

        # Extracted years
        extracted_years = set()
        for entity in extracted_entities:
            if entity.get("type") == "year":
                extracted_years.add(entity["value"])

        # Precision and recall on year extraction
        if not gt_years and not extracted_years:
            return {"year_precision": 1.0, "year_recall": 1.0, "year_f1": 1.0}

        correct = len(gt_years & extracted_years)
        precision = correct / len(extracted_years) if extracted_years else 0.0
        recall = correct / len(gt_years) if gt_years else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "year_precision": precision,
            "year_recall": recall,
            "year_f1": f1,
            "gt_years": sorted(gt_years),
            "extracted_years": sorted(extracted_years),
        }

    def temporal_reasoning_accuracy(
        self,
        result: Dict[str, Any],
        gold_answer: str,
    ) -> Dict[str, float]:
        """Evaluate temporal reasoning contribution to correct answer."""
        temporal_info = result.get("temporal", {})

        score = 0.0
        components = {}

        # Did we correctly identify temporal question type?
        temporal_types = temporal_info.get("temporal_type", {})
        has_temporal_type = any(temporal_types.values())
        components["type_detected"] = has_temporal_type

        # Did we extract relevant entities?
        entities = temporal_info.get("temporal_entities", [])
        components["entities_extracted"] = len(entities)

        # Did trend analysis produce useful results?
        trend = temporal_info.get("trend_analysis")
        if trend and trend.get("trend") != "insufficient_data":
            components["trend_detected"] = True
            score += 0.3
        else:
            components["trend_detected"] = False

        if has_temporal_type:
            score += 0.3
        if entities:
            score += 0.2
        if temporal_info.get("temporal_context"):
            score += 0.2

        components["temporal_reasoning_score"] = min(1.0, score)
        return components

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate temporal reasoning across a batch."""
        temporal_scores = []
        entity_counts = []
        trend_detected = 0
        temporal_questions = 0

        for r in results:
            classification = r.get("classification", {})
            if "temporal" in classification.get("active_modules", []):
                temporal_questions += 1

                temporal_info = r.get("temporal", {})
                entities = temporal_info.get("temporal_entities", [])
                entity_counts.append(len(entities))

                trend_analysis = temporal_info.get("trend_analysis")
                if trend_analysis is not None and trend_analysis.get("trend", "") != "insufficient_data":
                    trend_detected += 1

                # Score temporal reasoning
                tr_acc = self.temporal_reasoning_accuracy(r, r.get("gold_answer", ""))
                temporal_scores.append(tr_acc.get("temporal_reasoning_score", 0))

        return {
            "temporal_questions_found": temporal_questions,
            "mean_temporal_score": np.mean(temporal_scores) if temporal_scores else 0.0,
            "mean_entities_extracted": np.mean(entity_counts) if entity_counts else 0.0,
            "trend_detection_rate": trend_detected / max(temporal_questions, 1),
        }


class FinQAEvaluator:
    """Comprehensive evaluator for the Financial QA system."""

    def __init__(self, tolerance: float = 0.01):
        self.numerical_metrics = NumericalReasoningMetrics(tolerance)
        self.context_metrics = ContextFilteringMetrics()
        self.causality_metrics = CausalityDetectionMetrics()
        self.temporal_metrics = TemporalReasoningMetrics()

    def evaluate(
        self,
        results: List[Dict[str, Any]],
        examples: List[Any] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation across all metric categories.

        Returns a comprehensive report with:
        - overall_accuracy: % of questions answered correctly
        - numerical_metrics: program and execution accuracy
        - context_metrics: retrieval precision, recall, sufficiency
        - causality_metrics: causal detection rate and confidence
        - temporal_metrics: temporal reasoning quality
        """
        report = {
            "num_examples": len(results),
            "overall": {},
            "numerical_reasoning": {},
            "context_filtering": {},
            "causality_detection": {},
            "temporal_reasoning": {},
        }

        # Overall accuracy
        correct = 0
        total = 0
        for r in results:
            pred = r.get("predicted_answer", "")
            gold = r.get("gold_answer", "")
            if pred and gold:
                total += 1
                if answers_match(pred, gold):
                    correct += 1

        report["overall"] = {
            "accuracy": correct / max(total, 1),
            "correct": correct,
            "total": total,
        }

        # Module-specific metrics
        report["numerical_reasoning"] = self.numerical_metrics.evaluate_batch(results)
        report["context_filtering"] = self.context_metrics.evaluate_batch(results, examples)
        report["causality_detection"] = self.causality_metrics.evaluate_batch(results)
        report["temporal_reasoning"] = self.temporal_metrics.evaluate_batch(results)

        # Per-question-type breakdown
        type_results = defaultdict(list)
        for r in results:
            primary = r.get("classification", {}).get("primary_type", "unknown")
            type_results[primary].append(r)

        report["per_type_accuracy"] = {}
        for qtype, type_res in type_results.items():
            type_correct = sum(
                1 for r in type_res
                if answers_match(r.get("predicted_answer", ""), r.get("gold_answer", ""))
            )
            report["per_type_accuracy"][qtype] = {
                "accuracy": type_correct / max(len(type_res), 1),
                "count": len(type_res),
                "correct": type_correct,
            }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Pretty-print an evaluation report."""
        print("\n" + "=" * 70)
        print("FINANCIAL QA SYSTEM - EVALUATION REPORT")
        print("=" * 70)

        print(f"\nExamples evaluated: {report['num_examples']}")

        # Overall
        overall = report["overall"]
        print(f"\n--- Overall Performance ---")
        print(f"  Accuracy: {overall['accuracy']:.4f} ({overall['correct']}/{overall['total']})")

        # Numerical
        num = report["numerical_reasoning"]
        print(f"\n--- Numerical Reasoning ---")
        print(f"  Execution Accuracy: {num.get('execution_accuracy', 0):.4f}")
        if num.get("mean_relative_error") is not None:
            print(f"  Mean Relative Error: {num['mean_relative_error']:.4f}")
        print(f"  Examples evaluated: {num.get('num_evaluated', 0)}")

        # Context
        ctx = report["context_filtering"]
        print(f"\n--- Context Filtering ---")
        print(f"  Precision: {ctx.get('mean_precision', 0):.4f}")
        print(f"  Recall: {ctx.get('mean_recall', 0):.4f}")
        print(f"  F1: {ctx.get('mean_f1', 0):.4f}")
        print(f"  Sufficiency: {ctx.get('mean_sufficiency', 0):.4f}")

        # Causality
        causal = report["causality_detection"]
        print(f"\n--- Causality Detection ---")
        print(f"  Causal questions: {causal.get('causal_questions_found', 0)}")
        print(f"  Detection rate: {causal.get('detection_rate', 0):.4f}")
        print(f"  Mean confidence: {causal.get('mean_confidence', 0):.4f}")

        # Temporal
        temp = report["temporal_reasoning"]
        print(f"\n--- Temporal Reasoning ---")
        print(f"  Temporal questions: {temp.get('temporal_questions_found', 0)}")
        print(f"  Mean temporal score: {temp.get('mean_temporal_score', 0):.4f}")
        print(f"  Trend detection rate: {temp.get('trend_detection_rate', 0):.4f}")

        # Per-type breakdown
        print(f"\n--- Per-Type Accuracy ---")
        for qtype, info in report.get("per_type_accuracy", {}).items():
            print(f"  {qtype}: {info['accuracy']:.4f} ({info['correct']}/{info['count']})")

        print("\n" + "=" * 70)
