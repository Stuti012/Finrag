"""Research-grade evaluation metrics for Financial QA."""

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from ..utils.financial_utils import answers_match, normalize_answer


class NumericalReasoningMetrics:
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def execution_accuracy(self, predicted_answer: str, gold_answer: str) -> Dict[str, float]:
        exact_match = answers_match(predicted_answer, gold_answer, self.tolerance)
        result = {"exact_match": float(exact_match), "predicted": predicted_answer, "gold": gold_answer}
        try:
            pred_val = float(normalize_answer(predicted_answer))
            gold_val = float(normalize_answer(gold_answer))
            result["absolute_error"] = abs(pred_val - gold_val)
            result["relative_error"] = abs(pred_val - gold_val) / abs(gold_val) if gold_val != 0 else abs(pred_val)
        except Exception:
            result["absolute_error"] = None
            result["relative_error"] = None
        return result

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        matches, rel_err = [], []
        for r in results:
            pred, gold = r.get("predicted_answer", ""), r.get("gold_answer", "")
            if pred and gold:
                row = self.execution_accuracy(pred, gold)
                matches.append(row["exact_match"])
                if row["relative_error"] is not None:
                    rel_err.append(row["relative_error"])
        return {
            "execution_accuracy": float(np.mean(matches)) if matches else 0.0,
            "num_evaluated": len(matches),
            "mean_relative_error": float(np.mean(rel_err)) if rel_err else None,
            "median_relative_error": float(np.median(rel_err)) if rel_err else None,
        }


class ContextFilteringMetrics:
    @staticmethod
    def _extract_tokens(text: str) -> List[str]:
        words = re.findall(r"[a-z]{3,}", (text or "").lower())
        nums = re.findall(r"-?\d[\d,.]*", text or "")
        return list(set(words + nums))

    def retrieval_precision_recall(self, retrieved_docs: List[str], gold_evidence: List[str]) -> Dict[str, float]:
        if not gold_evidence:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        gt = [set(self._extract_tokens(g)) for g in gold_evidence if g]
        if not gt:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        covered = [False] * len(gt)
        relevant = 0
        for doc in retrieved_docs or []:
            dt = set(self._extract_tokens(doc))
            is_rel = False
            for i, g in enumerate(gt):
                overlap = len(dt & g) / max(1, len(g))
                if overlap >= 0.4:
                    covered[i] = True
                    is_rel = True
            if is_rel:
                relevant += 1

        precision = relevant / len(retrieved_docs) if retrieved_docs else 0.0
        recall = sum(covered) / len(gt)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate_batch(self, results: List[Dict[str, Any]], examples: List[Any] = None) -> Dict[str, float]:
        vals = []
        for i, r in enumerate(results):
            retrieved = [c.get("text", "") for c in r.get("retrieval", {}).get("table_contexts", [])]
            retrieved += [c.get("text", "") for c in r.get("retrieval", {}).get("text_contexts", [])]
            gold = examples[i].gold_evidence if examples and i < len(examples) else []
            vals.append(self.retrieval_precision_recall(retrieved, gold))
        return {
            "mean_precision": float(np.mean([v["precision"] for v in vals])) if vals else 0.0,
            "mean_recall": float(np.mean([v["recall"] for v in vals])) if vals else 0.0,
            "mean_f1": float(np.mean([v["f1"] for v in vals])) if vals else 0.0,
            "num_evaluated": len(vals),
        }


class CausalityDetectionMetrics:
    """Research-grade causal quality metrics."""

    @staticmethod
    def _edge_key(rel: Dict[str, Any]) -> Tuple[str, str]:
        return ((rel.get("cause") or "").lower().strip(), (rel.get("effect") or "").lower().strip())

    def graph_metrics(self, causal_info: Dict[str, Any]) -> Dict[str, float]:
        rels = causal_info.get("causal_relations", [])
        num_edges = len(rels)
        num_nodes = len({x for rel in rels for x in self._edge_key(rel) if x})
        density = num_edges / max(1, num_nodes * (num_nodes - 1))
        strengths = [float(r.get("confidence", 0.0)) for r in rels]
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "graph_density": density,
            "mean_edge_confidence": float(np.mean(strengths)) if strengths else 0.0,
        }

    def chain_quality(self, causal_info: Dict[str, Any]) -> Dict[str, float]:
        chains = causal_info.get("causal_chains", [])
        if not chains:
            return {"mean_chain_confidence": 0.0, "mean_chain_length": 0.0, "chain_coverage": 0.0}
        confs = [float(c.get("propagated_confidence", 0.0)) for c in chains if isinstance(c, dict)]
        lengths = [float(c.get("length", 0.0)) for c in chains if isinstance(c, dict)]
        return {
            "mean_chain_confidence": float(np.mean(confs)) if confs else 0.0,
            "mean_chain_length": float(np.mean(lengths)) if lengths else 0.0,
            "chain_coverage": min(1.0, len(chains) / 5.0),
        }

    def counterfactual_readiness(self, causal_info: Dict[str, Any]) -> Dict[str, float]:
        cfs = causal_info.get("counterfactuals", [])
        valid = sum(1 for c in cfs if c.get("counterfactual_question") and c.get("expected_direction"))
        return {"counterfactual_readiness": valid / max(1, len(cfs)) if cfs else 0.0}

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        detect, overlap = [], []
        graph_rows, chain_rows, cf_rows = [], [], []

        for r in results:
            causal = r.get("causal", {})
            is_causal = float(causal.get("is_causal", False))
            has_rel = 1.0 if causal.get("causal_relations") else 0.0
            if is_causal > 0:
                detect.append(has_rel)
            overlap.append(float(causal.get("temporal_causal_overlap", 0.0)))
            graph_rows.append(self.graph_metrics(causal))
            chain_rows.append(self.chain_quality(causal))
            cf_rows.append(self.counterfactual_readiness(causal))

        def mean_key(rows, key):
            return float(np.mean([r[key] for r in rows])) if rows else 0.0

        return {
            "detection_rate": float(np.mean(detect)) if detect else 0.0,
            "mean_temporal_causal_overlap": float(np.mean(overlap)) if overlap else 0.0,
            "mean_graph_density": mean_key(graph_rows, "graph_density"),
            "mean_edge_confidence": mean_key(graph_rows, "mean_edge_confidence"),
            "mean_chain_confidence": mean_key(chain_rows, "mean_chain_confidence"),
            "mean_chain_length": mean_key(chain_rows, "mean_chain_length"),
            "counterfactual_readiness": mean_key(cf_rows, "counterfactual_readiness"),
        }


class TemporalReasoningMetrics:
    """Research-grade temporal and temporal-causal metrics."""

    def temporal_entity_quality(self, temporal_info: Dict[str, Any]) -> Dict[str, float]:
        entities = temporal_info.get("temporal_entities", [])
        years = 0
        quarter_refs = 0
        for e in entities:
            text = str(e).lower()
            if re.search(r"\b(19\d{2}|20\d{2})\b", text):
                years += 1
            if "q1" in text or "q2" in text or "q3" in text or "q4" in text:
                quarter_refs += 1
        richness = min(1.0, (years + quarter_refs) / 4.0)
        return {"temporal_entity_richness": richness, "year_mentions": years, "quarter_mentions": quarter_refs}

    def trend_reasoning_quality(self, temporal_info: Dict[str, Any]) -> Dict[str, float]:
        trend = temporal_info.get("trend_analysis") or {}
        trend_val = trend.get("trend", "") if isinstance(trend, dict) else ""
        detected = trend_val not in {"", "insufficient_data", None}
        return {"trend_detected": float(detected), "trend_quality": 1.0 if detected else 0.0}

    def temporal_causal_alignment(self, result: Dict[str, Any]) -> Dict[str, float]:
        temporal = result.get("temporal", {})
        causal = result.get("causal", {})
        overlap = float(causal.get("temporal_causal_overlap", 0.0))
        joint_score = float(result.get("classification", {}).get("temporal_causal_joint", 0.0))
        alignment = min(1.0, 0.5 * joint_score + 0.5 * min(1.0, overlap / 2.0))
        return {"temporal_causal_alignment": alignment}

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        ent, tr, align = [], [], []
        for r in results:
            temporal = r.get("temporal", {})
            ent.append(self.temporal_entity_quality(temporal)["temporal_entity_richness"])
            tr.append(self.trend_reasoning_quality(temporal)["trend_quality"])
            align.append(self.temporal_causal_alignment(r)["temporal_causal_alignment"])
        richness = float(np.mean(ent)) if ent else 0.0
        trend_rate = float(np.mean(tr)) if tr else 0.0
        alignment = float(np.mean(align)) if align else 0.0
        # Composite temporal score for summary reporting
        mean_temporal_score = (richness + trend_rate + alignment) / 3.0
        return {
            "mean_temporal_entity_richness": richness,
            "trend_detection_rate": trend_rate,
            "mean_temporal_causal_alignment": alignment,
            "mean_temporal_score": mean_temporal_score,
        }


class ProgramInductionMetrics:
    """Metrics for evaluating program induction quality."""

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        total = len(results)
        if not total:
            return {"program_generation_rate": 0.0, "execution_success_rate": 0.0,
                    "self_refinement_improvement": 0.0, "mean_attempts": 0.0}

        generated, executed, refined_better = 0, 0, 0
        attempts_list = []
        for r in results:
            pot = r.get("pot_result", r.get("numerical", {}))
            if pot.get("program") or pot.get("generated_code"):
                generated += 1
            if pot.get("execution_success") or pot.get("success"):
                executed += 1
            att = pot.get("attempts", [])
            attempts_list.append(len(att) if att else 1)
            if pot.get("best_effort"):
                refined_better += 1

        return {
            "program_generation_rate": generated / total,
            "execution_success_rate": executed / max(1, generated),
            "self_refinement_improvement": refined_better / max(1, total),
            "mean_attempts": float(np.mean(attempts_list)) if attempts_list else 1.0,
        }


class ErrorAttributionMetrics:
    """Attribute errors to pipeline stages for targeted improvement."""

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        error_sources = defaultdict(int)
        total_errors = 0

        for r in results:
            pred, gold = r.get("predicted_answer", ""), r.get("gold_answer", "")
            if pred and gold and not answers_match(pred, gold):
                total_errors += 1
                pot = r.get("pot_result", r.get("numerical", {}))
                retrieval = r.get("retrieval", {})

                if not retrieval.get("table_contexts") and not retrieval.get("text_contexts"):
                    error_sources["retrieval_failure"] += 1
                elif not pot.get("program") and not pot.get("generated_code"):
                    error_sources["program_generation_failure"] += 1
                elif not (pot.get("execution_success") or pot.get("success")):
                    error_sources["execution_failure"] += 1
                else:
                    error_sources["reasoning_error"] += 1

        attribution = {k: v / max(1, total_errors) for k, v in error_sources.items()}
        attribution["total_errors"] = total_errors
        return attribution


class FinQAEvaluator:
    def __init__(self, tolerance: float = 0.01):
        self.numerical_metrics = NumericalReasoningMetrics(tolerance)
        self.context_metrics = ContextFilteringMetrics()
        self.causality_metrics = CausalityDetectionMetrics()
        self.temporal_metrics = TemporalReasoningMetrics()
        self.program_metrics = ProgramInductionMetrics()
        self.error_attribution = ErrorAttributionMetrics()

    def evaluate(self, results: List[Dict[str, Any]], examples: List[Any] = None) -> Dict[str, Any]:
        report = {
            "num_examples": len(results),
            "overall": {},
            "numerical_reasoning": self.numerical_metrics.evaluate_batch(results),
            "context_filtering": self.context_metrics.evaluate_batch(results, examples),
            "causality_detection": self.causality_metrics.evaluate_batch(results),
            "temporal_reasoning": self.temporal_metrics.evaluate_batch(results),
            "program_induction": self.program_metrics.evaluate_batch(results),
            "error_attribution": self.error_attribution.evaluate_batch(results),
        }

        total = 0
        correct = 0
        by_type = defaultdict(list)
        for r in results:
            pred, gold = r.get("predicted_answer", ""), r.get("gold_answer", "")
            if pred and gold:
                total += 1
                if answers_match(pred, gold):
                    correct += 1
            by_type[r.get("classification", {}).get("primary_type", "unknown")].append(r)

        report["overall"] = {"accuracy": correct / max(1, total), "correct": correct, "total": total}
        report["per_type_accuracy"] = {}
        for k, rows in by_type.items():
            c = sum(1 for r in rows if answers_match(r.get("predicted_answer", ""), r.get("gold_answer", "")))
            report["per_type_accuracy"][k] = {"accuracy": c / max(1, len(rows)), "count": len(rows), "correct": c}
        return report

    def print_report(self, report: Dict[str, Any]):
        print("\n" + "=" * 70)
        print("FINANCIAL QA SYSTEM - EVALUATION REPORT")
        print("=" * 70)
        print(f"\nExamples evaluated: {report['num_examples']}")
        print(f"\nOverall accuracy: {report['overall']['accuracy']:.4f}")
        print(f"Numerical execution accuracy: {report['numerical_reasoning'].get('execution_accuracy', 0):.4f}")
        print(f"Temporal-causal alignment: {report['temporal_reasoning'].get('mean_temporal_causal_alignment', 0):.4f}")
        print(f"Causal chain confidence: {report['causality_detection'].get('mean_chain_confidence', 0):.4f}")

        prog = report.get("program_induction", {})
        if prog:
            print(f"\n--- Program Induction ---")
            print(f"Generation rate: {prog.get('program_generation_rate', 0):.4f}")
            print(f"Execution success rate: {prog.get('execution_success_rate', 0):.4f}")
            print(f"Self-refinement improvement: {prog.get('self_refinement_improvement', 0):.4f}")
            print(f"Mean attempts per question: {prog.get('mean_attempts', 0):.2f}")

        err = report.get("error_attribution", {})
        if err and err.get("total_errors", 0) > 0:
            print(f"\n--- Error Attribution ({err['total_errors']} errors) ---")
            for k, v in err.items():
                if k != "total_errors":
                    print(f"  {k}: {v:.2%}")

        print("=" * 70)
