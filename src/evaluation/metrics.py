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

    def recursive_depth_metrics(self, causal_info: Dict[str, Any]) -> Dict[str, float]:
        """Metrics for recursive causal extraction depth and coverage."""
        nested = causal_info.get("nested_causal_relations", [])
        multi_hop = causal_info.get("multi_hop_relations", [])
        transitive = causal_info.get("transitive_relations", [])
        max_depth = causal_info.get("max_extraction_depth", 0)
        recursive_chains = causal_info.get("recursive_causal_chains", [])
        max_chain_len = max((c.get("length", 0) for c in recursive_chains), default=0) if recursive_chains else 0
        return {
            "nested_relation_count": len(nested),
            "multi_hop_relation_count": len(multi_hop),
            "transitive_relation_count": len(transitive),
            "max_extraction_depth": max_depth,
            "max_linked_chain_length": max_chain_len,
        }

    def scm_metrics(self, causal_info: Dict[str, Any]) -> Dict[str, float]:
        """Metrics for Structural Causal Model analysis quality."""
        scm_struct = causal_info.get("scm_structure", {})
        scm_paths = causal_info.get("scm_paths_ranked", [])
        scm_dsep = causal_info.get("scm_dseparation", [])
        scm_backdoor = causal_info.get("scm_backdoor_criterion", {})
        scm_frontdoor = causal_info.get("scm_frontdoor_criterion", {})
        scm_sensitivity = causal_info.get("scm_sensitivity", [])

        has_intervention = any(
            c.get("scm_propagation") or c.get("type") == "causal_effect_estimate"
            for c in causal_info.get("counterfactuals", [])
            if isinstance(c, dict)
        )

        path_support = [p.get("evidence_support", 0) for p in scm_paths]

        return {
            "scm_num_nodes": scm_struct.get("num_nodes", 0),
            "scm_num_equations": scm_struct.get("num_equations", 0),
            "scm_paths_found": len(scm_paths),
            "scm_mean_path_support": float(np.mean(path_support)) if path_support else 0.0,
            "scm_dsep_queries": len(scm_dsep),
            "scm_backdoor_valid": float(scm_backdoor.get("valid", False)),
            "scm_frontdoor_valid": float(scm_frontdoor.get("valid", False)),
            "scm_sensitivity_vars": len(scm_sensitivity),
            "scm_intervention_used": float(has_intervention),
        }

    def discourse_causality_quality(self, causal_info: Dict[str, Any]) -> Dict[str, float]:
        """Metrics for implicit discourse causality detection (PDTB-style)."""
        disc = causal_info.get("discourse_analysis", {})
        if not disc:
            return {
                "discourse_total": 0,
                "discourse_implicit_causal": 0,
                "discourse_explicit_causal": 0,
                "discourse_avg_confidence": 0.0,
                "discourse_has_features": 0.0,
            }
        rels = disc.get("relations", [])
        has_features = 1.0 if any(r.get("features") for r in rels) else 0.0
        return {
            "discourse_total": disc.get("total_discourse_relations", 0),
            "discourse_implicit_causal": disc.get("num_implicit_causal", 0),
            "discourse_explicit_causal": disc.get("num_explicit_causal", 0),
            "discourse_avg_confidence": float(disc.get("avg_confidence", 0.0)),
            "discourse_has_features": has_features,
        }

    def counterfactual_analysis_quality(self, causal_info: Dict[str, Any]) -> Dict[str, float]:
        """Metrics for counterfactual reasoning quality (Pearl, Ch 9)."""
        cf = causal_info.get("counterfactual_analysis", {})
        if not cf:
            return {
                "cf_query_parsed": 0.0,
                "cf_has_downstream_effects": 0.0,
                "cf_necessity_computed": 0.0,
                "cf_sufficiency_computed": 0.0,
                "cf_robustness_computed": 0.0,
                "cf_robust_conclusion": 0.0,
                "cf_explanation_length": 0,
                "cf_confidence": 0.0,
            }
        return {
            "cf_query_parsed": 1.0 if cf.get("treatment_var") else 0.0,
            "cf_has_downstream_effects": 1.0 if cf.get("downstream_effects") else 0.0,
            "cf_necessity_computed": 1.0 if cf.get("necessity_score") is not None else 0.0,
            "cf_sufficiency_computed": 1.0 if cf.get("sufficiency_score") is not None else 0.0,
            "cf_robustness_computed": 1.0 if cf.get("robustness") else 0.0,
            "cf_robust_conclusion": float(cf.get("robustness", {}).get("robust", False)) if cf.get("robustness") else 0.0,
            "cf_explanation_length": len(cf.get("explanation", "")),
            "cf_confidence": float(cf.get("confidence", 0)),
        }

    def evaluate_batch(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        detect, overlap = [], []
        graph_rows, chain_rows, cf_rows, depth_rows, scm_rows, cfa_rows, disc_rows = [], [], [], [], [], [], []

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
            depth_rows.append(self.recursive_depth_metrics(causal))
            scm_rows.append(self.scm_metrics(causal))
            cfa_rows.append(self.counterfactual_analysis_quality(causal))
            disc_rows.append(self.discourse_causality_quality(causal))

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
            "mean_nested_relations": mean_key(depth_rows, "nested_relation_count"),
            "mean_multi_hop_relations": mean_key(depth_rows, "multi_hop_relation_count"),
            "mean_transitive_relations": mean_key(depth_rows, "transitive_relation_count"),
            "mean_max_extraction_depth": mean_key(depth_rows, "max_extraction_depth"),
            "mean_max_chain_length": mean_key(depth_rows, "max_linked_chain_length"),
            "mean_scm_paths_found": mean_key(scm_rows, "scm_paths_found"),
            "mean_scm_path_support": mean_key(scm_rows, "scm_mean_path_support"),
            "scm_backdoor_rate": mean_key(scm_rows, "scm_backdoor_valid"),
            "scm_intervention_rate": mean_key(scm_rows, "scm_intervention_used"),
            "mean_scm_sensitivity_vars": mean_key(scm_rows, "scm_sensitivity_vars"),
            "cf_analysis_query_rate": mean_key(cfa_rows, "cf_query_parsed"),
            "cf_analysis_necessity_rate": mean_key(cfa_rows, "cf_necessity_computed"),
            "cf_analysis_sufficiency_rate": mean_key(cfa_rows, "cf_sufficiency_computed"),
            "cf_analysis_robustness_rate": mean_key(cfa_rows, "cf_robustness_computed"),
            "cf_analysis_mean_confidence": mean_key(cfa_rows, "cf_confidence"),
            "discourse_detection_rate": mean_key(disc_rows, "discourse_total"),
            "discourse_implicit_causal_rate": mean_key(disc_rows, "discourse_implicit_causal"),
            "discourse_explicit_causal_rate": mean_key(disc_rows, "discourse_explicit_causal"),
            "discourse_mean_confidence": mean_key(disc_rows, "discourse_avg_confidence"),
            "discourse_feature_rate": mean_key(disc_rows, "discourse_has_features"),
        }


class TemporalReasoningMetrics:
    """Research-grade temporal and temporal-causal metrics."""

    def temporal_entity_quality(self, temporal_info: Dict[str, Any]) -> Dict[str, float]:
        entities = temporal_info.get("temporal_entities", [])
        implicit_entities = temporal_info.get("implicit_temporal_entities", [])
        years = 0
        quarter_refs = 0
        for e in entities:
            text = str(e).lower()
            if re.search(r"\b(19\d{2}|20\d{2})\b", text):
                years += 1
            if "q1" in text or "q2" in text or "q3" in text or "q4" in text:
                quarter_refs += 1
        richness = min(1.0, (years + quarter_refs) / 4.0)
        deictic_count = len(implicit_entities)
        deictic_resolved = sum(
            1 for e in implicit_entities
            if isinstance(e, dict) and e.get("value") is not None
        )
        return {
            "temporal_entity_richness": richness,
            "year_mentions": years,
            "quarter_mentions": quarter_refs,
            "deictic_expressions_found": deictic_count,
            "deictic_expressions_resolved": deictic_resolved,
        }

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
        deictic_found, deictic_resolved = [], []
        for r in results:
            temporal = r.get("temporal", {})
            eq = self.temporal_entity_quality(temporal)
            ent.append(eq["temporal_entity_richness"])
            deictic_found.append(eq["deictic_expressions_found"])
            deictic_resolved.append(eq["deictic_expressions_resolved"])
            tr.append(self.trend_reasoning_quality(temporal)["trend_quality"])
            align.append(self.temporal_causal_alignment(r)["temporal_causal_alignment"])
        richness = float(np.mean(ent)) if ent else 0.0
        trend_rate = float(np.mean(tr)) if tr else 0.0
        alignment = float(np.mean(align)) if align else 0.0
        total_deictic = sum(deictic_found)
        total_resolved = sum(deictic_resolved)
        deictic_resolution_rate = total_resolved / max(1, total_deictic)
        mean_temporal_score = (richness + trend_rate + alignment) / 3.0
        return {
            "mean_temporal_entity_richness": richness,
            "trend_detection_rate": trend_rate,
            "mean_temporal_causal_alignment": alignment,
            "mean_temporal_score": mean_temporal_score,
            "deictic_expressions_found": total_deictic,
            "deictic_resolution_rate": deictic_resolution_rate,
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

        causal = report.get("causality_detection", {})
        if causal.get("mean_scm_paths_found", 0) > 0:
            print(f"\n--- Structural Causal Model ---")
            print(f"SCM paths found (mean): {causal.get('mean_scm_paths_found', 0):.2f}")
            print(f"SCM path evidence support: {causal.get('mean_scm_path_support', 0):.4f}")
            print(f"Backdoor identification rate: {causal.get('scm_backdoor_rate', 0):.4f}")
            print(f"Intervention usage rate: {causal.get('scm_intervention_rate', 0):.4f}")
            print(f"Sensitivity variables (mean): {causal.get('mean_scm_sensitivity_vars', 0):.2f}")

        if causal.get("cf_analysis_query_rate", 0) > 0:
            print(f"\n--- Counterfactual Analysis ---")
            print(f"Query parse rate: {causal.get('cf_analysis_query_rate', 0):.4f}")
            print(f"Necessity computation rate: {causal.get('cf_analysis_necessity_rate', 0):.4f}")
            print(f"Sufficiency computation rate: {causal.get('cf_analysis_sufficiency_rate', 0):.4f}")
            print(f"Robustness analysis rate: {causal.get('cf_analysis_robustness_rate', 0):.4f}")
            print(f"Mean confidence: {causal.get('cf_analysis_mean_confidence', 0):.4f}")

        if causal.get("discourse_detection_rate", 0) > 0:
            print(f"\n--- Implicit Discourse Causality ---")
            print(f"Detection rate (mean): {causal.get('discourse_detection_rate', 0):.2f}")
            print(f"Implicit causal (mean): {causal.get('discourse_implicit_causal_rate', 0):.2f}")
            print(f"Explicit causal (mean): {causal.get('discourse_explicit_causal_rate', 0):.2f}")
            print(f"Mean confidence: {causal.get('discourse_mean_confidence', 0):.4f}")
            print(f"Feature extraction rate: {causal.get('discourse_feature_rate', 0):.4f}")

        err = report.get("error_attribution", {})
        if err and err.get("total_errors", 0) > 0:
            print(f"\n--- Error Attribution ({err['total_errors']} errors) ---")
            for k, v in err.items():
                if k != "total_errors":
                    print(f"  {k}: {v:.2%}")

        print("=" * 70)
