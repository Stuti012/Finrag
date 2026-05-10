"""IRCoT: Interleaved Retrieval with Chain-of-Thought.

Implements the iterative retrieval-reasoning loop from Trivedi et al. (2023).
Each iteration: assess confidence -> reformulate query from reasoning traces ->
retrieve -> re-reason -> check termination.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .numerical_reasoner import NumericalReasoner


class ConfidenceAssessor:
    """Assesses reasoning confidence across modules to decide if more retrieval is needed."""

    NUMERICAL_SIGNALS = [
        ("success", 0.40),
        ("result_not_none", 0.30),
        ("program_generated", 0.15),
        ("plausible", 0.15),
    ]
    TEMPORAL_SIGNALS = [
        ("has_entities", 0.25),
        ("entity_count_ge2", 0.20),
        ("has_trend", 0.25),
        ("has_context", 0.15),
        ("has_constraints", 0.15),
    ]
    CAUSAL_SIGNALS = [
        ("has_relations", 0.30),
        ("relation_count_ge2", 0.15),
        ("has_context", 0.20),
        ("has_discourse", 0.15),
        ("has_granger", 0.20),
    ]

    def assess(self, result: Dict[str, Any], active_modules: List[str]) -> Dict[str, Any]:
        module_confidences = {}
        gaps = []

        if "numerical" in active_modules:
            conf, g = self._assess_numerical(result.get("numerical", {}))
            module_confidences["numerical"] = conf
            gaps.extend(g)

        if "temporal" in active_modules:
            conf, g = self._assess_temporal(result.get("temporal", {}))
            module_confidences["temporal"] = conf
            gaps.extend(g)

        if "causal" in active_modules:
            conf, g = self._assess_causal(result.get("causal", {}))
            module_confidences["causal"] = conf
            gaps.extend(g)

        if not module_confidences:
            module_confidences["baseline"] = 0.5

        overall = sum(module_confidences.values()) / len(module_confidences)

        return {
            "overall": overall,
            "module_confidences": module_confidences,
            "gaps": gaps,
            "primary_gap": gaps[0] if gaps else None,
        }

    def _assess_numerical(self, num: Dict[str, Any]) -> Tuple[float, List[Dict]]:
        score = 0.0
        gaps = []
        if num.get("success"):
            score += 0.40
        else:
            gaps.append({"module": "numerical", "type": "execution_failure",
                         "detail": num.get("error", "computation failed")})
        result_val = num.get("result")
        if result_val is not None:
            score += 0.30
        else:
            gaps.append({"module": "numerical", "type": "no_result",
                         "detail": "no numerical result produced"})
        if num.get("generated_code") or num.get("program"):
            score += 0.15
        if num.get("method") == "program_of_thought":
            score += 0.15
        if num.get("success") and result_val is not None:
            question = num.get("question", "")
            plausible, reason = NumericalReasoner.is_plausible_result(result_val, question)
            if not plausible:
                score -= 0.25
                gaps.append({"module": "numerical", "type": "implausible_result",
                             "detail": reason})
        return score, gaps

    def _assess_temporal(self, temp: Dict[str, Any]) -> Tuple[float, List[Dict]]:
        score = 0.0
        gaps = []
        entities = temp.get("temporal_entities", [])
        if entities:
            score += 0.25
        if len(entities) >= 2:
            score += 0.20
        else:
            gaps.append({"module": "temporal", "type": "insufficient_entities",
                         "detail": f"only {len(entities)} temporal entities found"})
        trend = temp.get("trend_analysis", {})
        if isinstance(trend, dict) and trend.get("trend") not in {None, "", "insufficient_data"}:
            score += 0.25
        else:
            gaps.append({"module": "temporal", "type": "no_trend",
                         "detail": "trend analysis incomplete"})
        if temp.get("temporal_context"):
            score += 0.15
        if temp.get("constraint_propagation"):
            score += 0.15
        return score, gaps

    def _assess_causal(self, causal: Dict[str, Any]) -> Tuple[float, List[Dict]]:
        score = 0.0
        gaps = []
        relations = causal.get("causal_relations", [])
        if relations:
            score += 0.30
        else:
            gaps.append({"module": "causal", "type": "no_relations",
                         "detail": "no causal relations detected"})
        if len(relations) >= 2:
            score += 0.15
        if causal.get("causal_context"):
            score += 0.20
        if causal.get("discourse_analysis"):
            score += 0.15
        if causal.get("granger_analysis"):
            score += 0.20
        return score, gaps


class QueryReformulator:
    """Reformulates retrieval queries using chain-of-thought reasoning traces."""

    GAP_EXPANSIONS = {
        "execution_failure": ["formula", "calculation", "compute", "values"],
        "no_result": ["table", "data", "values", "numbers"],
        "insufficient_entities": ["year", "quarter", "period", "date", "timeline"],
        "no_trend": ["increase", "decrease", "change", "growth", "trend", "over time"],
        "no_relations": ["because", "caused", "due to", "impact", "driver", "result"],
    }

    def reformulate(
        self,
        question: str,
        result: Dict[str, Any],
        assessment: Dict[str, Any],
        iteration: int,
    ) -> str:
        entities = self._extract_reasoning_entities(result)
        gap = assessment.get("primary_gap")
        gap_type = gap["type"] if gap else "general"
        expansion_terms = self.GAP_EXPANSIONS.get(gap_type, [])

        parts = [question]

        if entities:
            parts.append(" ".join(entities[:5]))

        if expansion_terms:
            subset = expansion_terms[:min(3, len(expansion_terms))]
            parts.append(" ".join(subset))

        trace_terms = self._extract_trace_terms(result, gap_type)
        if trace_terms:
            parts.append(" ".join(trace_terms[:3]))

        if iteration >= 2:
            parts.append(self._diversify_query(question, iteration))

        return " ".join(parts)

    def _extract_reasoning_entities(self, result: Dict[str, Any]) -> List[str]:
        entities: List[str] = []

        for ent in result.get("temporal", {}).get("temporal_entities", []):
            if isinstance(ent, dict):
                val = ent.get("label") or ent.get("value") or ent.get("text", "")
            else:
                val = str(ent)
            if val:
                entities.append(str(val))

        for rel in result.get("causal", {}).get("causal_relations", []):
            if isinstance(rel, dict):
                if rel.get("cause"):
                    entities.append(str(rel["cause"])[:50])
                if rel.get("effect"):
                    entities.append(str(rel["effect"])[:50])
            elif hasattr(rel, "cause") and rel.cause:
                entities.append(str(rel.cause)[:50])
            if hasattr(rel, "effect") and rel.effect:
                entities.append(str(rel.effect)[:50])

        seen: Set[str] = set()
        unique = []
        for e in entities:
            low = e.lower().strip()
            if low and low not in seen:
                seen.add(low)
                unique.append(e.strip())
        return unique

    def _extract_trace_terms(self, result: Dict[str, Any], gap_type: str) -> List[str]:
        terms = []
        for trace in result.get("reasoning_trace", []):
            if not isinstance(trace, str):
                continue
            words = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", trace)
            terms.extend(w for w in words if len(w) > 3)

        num = result.get("numerical", {})
        if gap_type in ("execution_failure", "no_result") and num.get("error"):
            error_nums = re.findall(r"\d[\d,.]+", str(num["error"]))
            terms.extend(error_nums[:2])

        return list(dict.fromkeys(terms))[:5]

    def _diversify_query(self, question: str, iteration: int) -> str:
        rewrites = [
            "what factors explain",
            "what evidence shows",
            "how does the data indicate",
        ]
        idx = (iteration - 2) % len(rewrites)
        core = re.sub(r"^(what|how|why|when|which)\b\s*", "", question.lower(), count=1)
        return rewrites[idx] + " " + core[:60]


class ContextMerger:
    """Merges and deduplicates retrieved contexts across iterations."""

    def __init__(self, max_length: int = 3500):
        self.max_length = max_length

    def merge(
        self,
        existing_context: str,
        new_results: Dict[str, Any],
        seen_doc_hashes: Set[int],
    ) -> Tuple[str, Set[int]]:
        new_texts = []
        for ctx in new_results.get("text_contexts", []):
            doc = ctx.get("document", "") if isinstance(ctx, dict) else str(ctx)
            doc_hash = hash(doc.strip().lower()[:200])
            if doc_hash not in seen_doc_hashes and doc.strip():
                seen_doc_hashes.add(doc_hash)
                new_texts.append(doc.strip())

        if not new_texts:
            return existing_context, seen_doc_hashes

        combined = existing_context
        for text in new_texts:
            if len(combined) + len(text) + 1 > self.max_length:
                remaining = self.max_length - len(combined) - 1
                if remaining > 50:
                    combined += " " + text[:remaining]
                break
            combined += " " + text

        return combined.strip(), seen_doc_hashes


class IRCoTController:
    """Manages the iterative retrieval-reasoning loop.

    Implements the core IRCoT pattern:
    1. Initial retrieval + reasoning
    2. Assess confidence across modules
    3. If below threshold: reformulate query from reasoning traces, retrieve more, re-reason
    4. Repeat until confident or max iterations reached
    """

    def __init__(
        self,
        max_iterations: int = 3,
        confidence_threshold: float = 0.7,
        min_improvement: float = 0.02,
        max_context_length: int = 3500,
    ):
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.min_improvement = min_improvement
        self.assessor = ConfidenceAssessor()
        self.reformulator = QueryReformulator()
        self.merger = ContextMerger(max_length=max_context_length)

    def run(
        self,
        question: str,
        example: Any,
        result: Dict[str, Any],
        context_text: str,
        active_modules: List[str],
        retrieve_fn: Callable,
        reason_fn: Callable,
    ) -> Dict[str, Any]:
        iterations = []
        seen_hashes: Set[int] = set()

        for chunk in context_text.split():
            if len(chunk) > 10:
                seen_hashes.add(hash(chunk.lower()[:200]))

        assessment = self.assessor.assess(result, active_modules)
        iterations.append({
            "iteration": 0,
            "type": "initial",
            "confidence": assessment["overall"],
            "gaps": [g["type"] for g in assessment["gaps"]],
            "query": None,
        })

        if assessment["overall"] >= self.confidence_threshold:
            return self._build_output(iterations, context_text, assessment, "threshold_met_initial")

        current_context = context_text
        prev_confidence = assessment["overall"]

        for i in range(1, self.max_iterations + 1):
            query = self.reformulator.reformulate(question, result, assessment, i)

            try:
                new_results = retrieve_fn(example, query)
            except Exception:
                new_results = {"text_contexts": [], "table_contexts": []}

            current_context, seen_hashes = self.merger.merge(
                current_context, new_results, seen_hashes
            )

            new_hits = len(new_results.get("text_contexts", []))
            temporal_signals = reason_fn(result, active_modules, example, current_context, f"ircot_pass{i}")

            assessment = self.assessor.assess(result, active_modules)
            improvement = assessment["overall"] - prev_confidence

            iterations.append({
                "iteration": i,
                "type": "ircot",
                "query": query,
                "new_hits": new_hits,
                "confidence": assessment["overall"],
                "improvement": improvement,
                "gaps": [g["type"] for g in assessment["gaps"]],
            })

            if assessment["overall"] >= self.confidence_threshold:
                return self._build_output(iterations, current_context, assessment, "threshold_met")

            if i >= 2 and improvement < self.min_improvement:
                return self._build_output(iterations, current_context, assessment, "plateau")

            prev_confidence = assessment["overall"]

        return self._build_output(iterations, current_context, assessment, "max_iterations")

    def _build_output(
        self,
        iterations: List[Dict],
        final_context: str,
        final_assessment: Dict[str, Any],
        termination_reason: str,
    ) -> Dict[str, Any]:
        total_iterations = len(iterations)
        confidences = [it["confidence"] for it in iterations]
        total_improvement = confidences[-1] - confidences[0] if len(confidences) >= 2 else 0.0
        converged = termination_reason in ("threshold_met", "threshold_met_initial")

        return {
            "iterations": iterations,
            "total_iterations": total_iterations,
            "final_confidence": final_assessment["overall"],
            "module_confidences": final_assessment["module_confidences"],
            "termination_reason": termination_reason,
            "converged": converged,
            "total_improvement": total_improvement,
            "final_context": final_context,
            "final_gaps": [g["type"] for g in final_assessment["gaps"]],
        }
