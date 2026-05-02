"""Tests for IRCoT interleaved retrieval with chain-of-thought (P3-4b).

Covers ConfidenceAssessor scoring, QueryReformulator entity/trace extraction,
ContextMerger deduplication, IRCoTController loop termination modes,
IRCoTMetrics evaluation, and pipeline integration."""

import pytest
from collections import defaultdict
from src.reasoning.ircot_controller import (
    ConfidenceAssessor,
    ContextMerger,
    IRCoTController,
    QueryReformulator,
)
from src.evaluation.metrics import IRCoTMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def assessor():
    return ConfidenceAssessor()


@pytest.fixture
def reformulator():
    return QueryReformulator()


@pytest.fixture
def merger():
    return ContextMerger(max_length=200)


@pytest.fixture
def controller():
    return IRCoTController(
        max_iterations=3,
        confidence_threshold=0.7,
        min_improvement=0.02,
        max_context_length=500,
    )


@pytest.fixture
def ircot_metrics():
    return IRCoTMetrics()


@pytest.fixture
def full_numerical_result():
    return {
        "numerical": {
            "success": True,
            "result": 42.0,
            "generated_code": "x = 42",
            "method": "program_of_thought",
        },
        "temporal": {},
        "causal": {},
    }


@pytest.fixture
def full_temporal_result():
    return {
        "numerical": {},
        "temporal": {
            "temporal_entities": [
                {"label": "2020"}, {"label": "2021"}, {"label": "Q3 2022"},
            ],
            "trend_analysis": {"trend": "increasing"},
            "temporal_context": "Revenue grew from 2020 to 2022",
            "constraint_propagation": {"resolved": True},
        },
        "causal": {},
    }


@pytest.fixture
def full_causal_result():
    return {
        "numerical": {},
        "temporal": {},
        "causal": {
            "causal_relations": [
                {"cause": "cost reduction", "effect": "margin increase", "confidence": 0.8},
                {"cause": "volume growth", "effect": "revenue increase", "confidence": 0.7},
            ],
            "causal_context": "Cost reduction drove margin improvement",
            "discourse_analysis": {"relations": [{"type": "causal"}]},
            "granger_analysis": {"num_tested": 2},
        },
    }


# ---------------------------------------------------------------------------
# 1. ConfidenceAssessor — numerical
# ---------------------------------------------------------------------------

class TestAssessNumerical:
    def test_full_success(self, assessor):
        num = {"success": True, "result": 42, "generated_code": "x=1", "method": "program_of_thought"}
        score, gaps = assessor._assess_numerical(num)
        assert score == pytest.approx(1.0)
        assert gaps == []

    def test_failure_no_result(self, assessor):
        num = {"success": False, "result": None, "error": "syntax error"}
        score, gaps = assessor._assess_numerical(num)
        assert score == pytest.approx(0.0)
        assert len(gaps) == 2
        assert gaps[0]["type"] == "execution_failure"
        assert gaps[1]["type"] == "no_result"

    def test_partial_success_no_code(self, assessor):
        num = {"success": True, "result": 10}
        score, gaps = assessor._assess_numerical(num)
        assert score == pytest.approx(0.70)

    def test_has_program_field(self, assessor):
        num = {"success": False, "result": None, "program": "def solve(): ..."}
        score, gaps = assessor._assess_numerical(num)
        assert score == pytest.approx(0.15)

    def test_empty_dict(self, assessor):
        score, gaps = assessor._assess_numerical({})
        assert score == pytest.approx(0.0)
        assert len(gaps) == 2


# ---------------------------------------------------------------------------
# 2. ConfidenceAssessor — temporal
# ---------------------------------------------------------------------------

class TestAssessTemporal:
    def test_full_temporal(self, assessor, full_temporal_result):
        score, gaps = assessor._assess_temporal(full_temporal_result["temporal"])
        assert score == pytest.approx(1.0)
        assert gaps == []

    def test_no_entities(self, assessor):
        score, gaps = assessor._assess_temporal({})
        assert score == pytest.approx(0.0)
        assert any(g["type"] == "insufficient_entities" for g in gaps)

    def test_one_entity_no_trend(self, assessor):
        temp = {"temporal_entities": [{"label": "2020"}]}
        score, gaps = assessor._assess_temporal(temp)
        assert score == pytest.approx(0.25)
        assert any(g["type"] == "insufficient_entities" for g in gaps)
        assert any(g["type"] == "no_trend" for g in gaps)

    def test_trend_insufficient_data(self, assessor):
        temp = {"temporal_entities": [{"label": "2020"}, {"label": "2021"}],
                "trend_analysis": {"trend": "insufficient_data"}}
        score, gaps = assessor._assess_temporal(temp)
        assert any(g["type"] == "no_trend" for g in gaps)

    def test_context_adds_score(self, assessor):
        temp = {"temporal_entities": [], "temporal_context": "some context"}
        score, _ = assessor._assess_temporal(temp)
        assert score == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# 3. ConfidenceAssessor — causal
# ---------------------------------------------------------------------------

class TestAssessCausal:
    def test_full_causal(self, assessor, full_causal_result):
        score, gaps = assessor._assess_causal(full_causal_result["causal"])
        assert score == pytest.approx(1.0)
        assert gaps == []

    def test_no_relations(self, assessor):
        score, gaps = assessor._assess_causal({})
        assert score == pytest.approx(0.0)
        assert gaps[0]["type"] == "no_relations"

    def test_one_relation_only(self, assessor):
        causal = {"causal_relations": [{"cause": "a", "effect": "b"}]}
        score, gaps = assessor._assess_causal(causal)
        assert score == pytest.approx(0.30)

    def test_two_relations_with_context(self, assessor):
        causal = {
            "causal_relations": [{"cause": "a"}, {"cause": "b"}],
            "causal_context": "because of A",
        }
        score, _ = assessor._assess_causal(causal)
        assert score == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# 4. ConfidenceAssessor — overall assess
# ---------------------------------------------------------------------------

class TestAssessOverall:
    def test_single_module_high(self, assessor, full_numerical_result):
        out = assessor.assess(full_numerical_result, ["numerical"])
        assert out["overall"] == pytest.approx(1.0)
        assert out["gaps"] == []
        assert "numerical" in out["module_confidences"]

    def test_multi_module_average(self, assessor, full_numerical_result):
        out = assessor.assess(full_numerical_result, ["numerical", "temporal"])
        num_conf = out["module_confidences"]["numerical"]
        temp_conf = out["module_confidences"]["temporal"]
        assert out["overall"] == pytest.approx((num_conf + temp_conf) / 2)

    def test_no_active_modules(self, assessor):
        out = assessor.assess({}, [])
        assert out["overall"] == pytest.approx(0.5)
        assert "baseline" in out["module_confidences"]

    def test_primary_gap_is_first(self, assessor):
        out = assessor.assess({"numerical": {"success": False}}, ["numerical"])
        assert out["primary_gap"] is not None
        assert out["primary_gap"]["type"] == out["gaps"][0]["type"]

    def test_no_gaps_when_successful(self, assessor, full_numerical_result):
        out = assessor.assess(full_numerical_result, ["numerical"])
        assert out["primary_gap"] is None


# ---------------------------------------------------------------------------
# 5. QueryReformulator — entity extraction
# ---------------------------------------------------------------------------

class TestExtractReasoningEntities:
    def test_temporal_dict_entities(self, reformulator):
        result = {
            "temporal": {
                "temporal_entities": [
                    {"label": "2020"}, {"value": "Q3"}, {"text": "December"},
                ]
            }
        }
        entities = reformulator._extract_reasoning_entities(result)
        assert "2020" in entities
        assert "Q3" in entities
        assert "December" in entities

    def test_temporal_string_entities(self, reformulator):
        result = {"temporal": {"temporal_entities": ["2019", "2020"]}}
        entities = reformulator._extract_reasoning_entities(result)
        assert "2019" in entities
        assert "2020" in entities

    def test_causal_dict_relations(self, reformulator):
        result = {
            "temporal": {},
            "causal": {
                "causal_relations": [
                    {"cause": "lower costs", "effect": "higher margins"},
                ]
            },
        }
        entities = reformulator._extract_reasoning_entities(result)
        assert "lower costs" in entities
        assert "higher margins" in entities

    def test_deduplication(self, reformulator):
        result = {
            "temporal": {"temporal_entities": ["2020", "2020", "2020"]},
        }
        entities = reformulator._extract_reasoning_entities(result)
        assert entities.count("2020") == 1

    def test_empty_result(self, reformulator):
        entities = reformulator._extract_reasoning_entities({})
        assert entities == []


# ---------------------------------------------------------------------------
# 6. QueryReformulator — trace terms
# ---------------------------------------------------------------------------

class TestExtractTraceTerms:
    def test_capitalized_words(self, reformulator):
        result = {"reasoning_trace": ["Revenue Growth was strong in North America"]}
        terms = reformulator._extract_trace_terms(result, "general")
        assert any("Revenue" in t for t in terms)

    def test_error_numbers_for_numerical_gap(self, reformulator):
        result = {
            "reasoning_trace": [],
            "numerical": {"error": "expected 1,234.56 but got 0"},
        }
        terms = reformulator._extract_trace_terms(result, "execution_failure")
        assert any("1,234.56" in t for t in terms)

    def test_no_traces(self, reformulator):
        result = {"reasoning_trace": []}
        terms = reformulator._extract_trace_terms(result, "general")
        assert terms == []

    def test_non_string_traces_skipped(self, reformulator):
        result = {"reasoning_trace": [42, None, {"foo": "bar"}]}
        terms = reformulator._extract_trace_terms(result, "general")
        assert terms == []


# ---------------------------------------------------------------------------
# 7. QueryReformulator — diversify query
# ---------------------------------------------------------------------------

class TestDiversifyQuery:
    def test_iteration_2(self, reformulator):
        q = "What caused the revenue increase?"
        d = reformulator._diversify_query(q, 2)
        assert d.startswith("what factors explain")

    def test_iteration_3(self, reformulator):
        d = reformulator._diversify_query("How did margins change?", 3)
        assert d.startswith("what evidence shows")

    def test_iteration_4(self, reformulator):
        d = reformulator._diversify_query("Why did revenue drop?", 4)
        assert d.startswith("how does the data indicate")

    def test_wraps_around(self, reformulator):
        d2 = reformulator._diversify_query("test question", 2)
        d5 = reformulator._diversify_query("test question", 5)
        assert d2 == d5


# ---------------------------------------------------------------------------
# 8. QueryReformulator — reformulate
# ---------------------------------------------------------------------------

class TestReformulate:
    def test_includes_question(self, reformulator):
        assessment = {"primary_gap": {"type": "no_result"}, "gaps": []}
        result = {"temporal": {}, "causal": {}, "reasoning_trace": []}
        q = "What is the net income?"
        out = reformulator.reformulate(q, result, assessment, 1)
        assert q in out

    def test_gap_expansion_terms(self, reformulator):
        assessment = {"primary_gap": {"type": "execution_failure"}, "gaps": []}
        result = {"temporal": {}, "causal": {}, "reasoning_trace": []}
        out = reformulator.reformulate("test", result, assessment, 1)
        assert "formula" in out or "calculation" in out

    def test_diversification_at_iteration_2(self, reformulator):
        assessment = {"primary_gap": None, "gaps": []}
        result = {"temporal": {}, "causal": {}, "reasoning_trace": []}
        out = reformulator.reformulate("What is revenue?", result, assessment, 2)
        assert "what factors explain" in out

    def test_entities_included(self, reformulator):
        assessment = {"primary_gap": None, "gaps": []}
        result = {
            "temporal": {"temporal_entities": [{"label": "FY2023"}]},
            "causal": {},
            "reasoning_trace": [],
        }
        out = reformulator.reformulate("test", result, assessment, 1)
        assert "FY2023" in out


# ---------------------------------------------------------------------------
# 9. ContextMerger — merge
# ---------------------------------------------------------------------------

class TestContextMerger:
    def test_adds_new_text(self, merger):
        new = {"text_contexts": [{"document": "New revenue data for Q4."}]}
        merged, hashes = merger.merge("Existing context.", new, set())
        assert "New revenue data" in merged

    def test_deduplicates(self, merger):
        doc = "Exact same document text here for testing."
        new = {"text_contexts": [{"document": doc}]}
        seen = {hash(doc.strip().lower()[:200])}
        merged, _ = merger.merge("start", new, seen)
        assert merged.count(doc) == 0

    def test_max_length_enforced(self):
        merger = ContextMerger(max_length=50)
        new = {"text_contexts": [{"document": "A" * 100}]}
        merged, _ = merger.merge("Base", new, set())
        assert len(merged) <= 50

    def test_empty_new_results(self, merger):
        merged, hashes = merger.merge("original", {"text_contexts": []}, set())
        assert merged == "original"

    def test_string_context_fallback(self, merger):
        new = {"text_contexts": ["plain string"]}
        merged, _ = merger.merge("base", new, set())
        assert "plain string" in merged

    def test_skips_empty_docs(self, merger):
        new = {"text_contexts": [{"document": ""}, {"document": "   "}]}
        merged, _ = merger.merge("base", new, set())
        assert merged == "base"

    def test_multiple_docs_appended(self):
        merger = ContextMerger(max_length=5000)
        new = {"text_contexts": [
            {"document": "Doc one."},
            {"document": "Doc two."},
            {"document": "Doc three."},
        ]}
        merged, hashes = merger.merge("", new, set())
        assert "Doc one" in merged
        assert "Doc two" in merged
        assert "Doc three" in merged
        assert len(hashes) == 3


# ---------------------------------------------------------------------------
# 10. IRCoTController — threshold met on initial assessment
# ---------------------------------------------------------------------------

class TestControllerThresholdInitial:
    def test_high_confidence_skips_loop(self, controller):
        result = {
            "numerical": {"success": True, "result": 42, "generated_code": "x=1", "method": "program_of_thought"},
            "temporal": {},
            "causal": {},
        }
        retrieve_fn = lambda ex, q: {"text_contexts": [], "table_contexts": []}
        reason_fn = lambda res, mods, ex, ctx, lbl: {}

        out = controller.run(
            question="What is revenue?",
            example=None,
            result=result,
            context_text="some context",
            active_modules=["numerical"],
            retrieve_fn=retrieve_fn,
            reason_fn=reason_fn,
        )
        assert out["termination_reason"] == "threshold_met_initial"
        assert out["converged"] is True
        assert out["total_iterations"] == 1
        assert len(out["iterations"]) == 1

    def test_output_keys(self, controller):
        result = {
            "numerical": {"success": True, "result": 1, "generated_code": "x", "method": "program_of_thought"},
        }
        out = controller.run(
            question="q", example=None, result=result,
            context_text="ctx", active_modules=["numerical"],
            retrieve_fn=lambda e, q: {"text_contexts": []},
            reason_fn=lambda *a: {},
        )
        for key in ["iterations", "total_iterations", "final_confidence",
                     "module_confidences", "termination_reason", "converged",
                     "total_improvement", "final_context", "final_gaps"]:
            assert key in out


# ---------------------------------------------------------------------------
# 11. IRCoTController — iterative loop
# ---------------------------------------------------------------------------

class TestControllerIterativeLoop:
    def test_max_iterations_reached(self):
        ctrl = IRCoTController(max_iterations=1, confidence_threshold=0.99, min_improvement=0.0)
        result = {"numerical": {"success": False}, "temporal": {}, "causal": {}}
        calls = {"count": 0}

        def retrieve_fn(ex, q):
            calls["count"] += 1
            return {"text_contexts": [{"document": f"doc {calls['count']}"}]}

        def reason_fn(res, mods, ex, ctx, lbl):
            return {}

        out = ctrl.run(
            question="test", example=None, result=result,
            context_text="base", active_modules=["numerical"],
            retrieve_fn=retrieve_fn, reason_fn=reason_fn,
        )
        assert out["termination_reason"] == "max_iterations"
        assert out["converged"] is False
        assert calls["count"] == 1

    def test_plateau_detection(self):
        ctrl = IRCoTController(max_iterations=5, confidence_threshold=0.99, min_improvement=0.05)
        result = {"numerical": {"success": False}, "temporal": {}, "causal": {}}

        out = ctrl.run(
            question="test", example=None, result=result,
            context_text="base", active_modules=["numerical"],
            retrieve_fn=lambda e, q: {"text_contexts": []},
            reason_fn=lambda *a: {},
        )
        assert out["termination_reason"] == "plateau"

    def test_retrieval_failure_handled(self):
        ctrl = IRCoTController(max_iterations=2, confidence_threshold=0.99)
        result = {"numerical": {"success": False}}

        def bad_retrieve(ex, q):
            raise RuntimeError("network error")

        out = ctrl.run(
            question="test", example=None, result=result,
            context_text="ctx", active_modules=["numerical"],
            retrieve_fn=bad_retrieve,
            reason_fn=lambda *a: {},
        )
        assert out["total_iterations"] >= 2

    def test_new_hits_tracked(self):
        ctrl = IRCoTController(max_iterations=1, confidence_threshold=0.99)
        result = {"numerical": {"success": False}}

        def retrieve_fn(ex, q):
            return {"text_contexts": [{"document": "a"}, {"document": "b"}]}

        out = ctrl.run(
            question="q", example=None, result=result,
            context_text="c", active_modules=["numerical"],
            retrieve_fn=retrieve_fn, reason_fn=lambda *a: {},
        )
        ircot_iters = [i for i in out["iterations"] if i["type"] == "ircot"]
        assert ircot_iters[0]["new_hits"] == 2


# ---------------------------------------------------------------------------
# 12. IRCoTController — build_output
# ---------------------------------------------------------------------------

class TestBuildOutput:
    def test_total_improvement_calculated(self, controller):
        iterations = [
            {"iteration": 0, "confidence": 0.3, "type": "initial", "gaps": []},
            {"iteration": 1, "confidence": 0.5, "type": "ircot", "gaps": []},
        ]
        assessment = {"overall": 0.5, "module_confidences": {"numerical": 0.5}, "gaps": []}
        out = controller._build_output(iterations, "ctx", assessment, "max_iterations")
        assert out["total_improvement"] == pytest.approx(0.2)

    def test_converged_flags(self, controller):
        iters = [{"iteration": 0, "confidence": 0.8, "type": "initial", "gaps": []}]
        assessment = {"overall": 0.8, "module_confidences": {}, "gaps": []}

        out1 = controller._build_output(iters, "c", assessment, "threshold_met")
        assert out1["converged"] is True

        out2 = controller._build_output(iters, "c", assessment, "plateau")
        assert out2["converged"] is False

        out3 = controller._build_output(iters, "c", assessment, "threshold_met_initial")
        assert out3["converged"] is True

    def test_single_iteration_zero_improvement(self, controller):
        iters = [{"iteration": 0, "confidence": 0.5, "type": "initial", "gaps": []}]
        assessment = {"overall": 0.5, "module_confidences": {}, "gaps": []}
        out = controller._build_output(iters, "c", assessment, "threshold_met_initial")
        assert out["total_improvement"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 13. IRCoTMetrics — ircot_quality
# ---------------------------------------------------------------------------

class TestIRCoTMetricsQuality:
    def test_valid_ircot_info(self, ircot_metrics):
        info = {
            "iterations": [{"iteration": 0}, {"iteration": 1}],
            "total_iterations": 2,
            "final_confidence": 0.85,
            "converged": True,
            "total_improvement": 0.15,
        }
        q = ircot_metrics.ircot_quality(info)
        assert q["ircot_iterations"] == 2
        assert q["ircot_final_confidence"] == pytest.approx(0.85)
        assert q["ircot_converged"] == pytest.approx(1.0)
        assert q["ircot_improvement"] == pytest.approx(0.15)
        assert q["ircot_has_retrieval"] == pytest.approx(1.0)

    def test_empty_info(self, ircot_metrics):
        q = ircot_metrics.ircot_quality({})
        assert q["ircot_iterations"] == 0
        assert q["ircot_converged"] == pytest.approx(0.0)

    def test_none_info(self, ircot_metrics):
        q = ircot_metrics.ircot_quality(None)
        assert q["ircot_iterations"] == 0

    def test_single_iteration_no_retrieval(self, ircot_metrics):
        info = {"iterations": [{"iteration": 0}], "total_iterations": 1,
                "final_confidence": 0.9, "converged": True, "total_improvement": 0.0}
        q = ircot_metrics.ircot_quality(info)
        assert q["ircot_has_retrieval"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 14. IRCoTMetrics — evaluate_batch
# ---------------------------------------------------------------------------

class TestIRCoTMetricsBatch:
    def test_batch_aggregation(self, ircot_metrics):
        results = [
            {"ircot": {"iterations": [{"i": 0}, {"i": 1}], "total_iterations": 2,
                       "final_confidence": 0.8, "converged": True,
                       "total_improvement": 0.1, "termination_reason": "threshold_met"}},
            {"ircot": {"iterations": [{"i": 0}], "total_iterations": 1,
                       "final_confidence": 0.9, "converged": True,
                       "total_improvement": 0.0, "termination_reason": "threshold_met_initial"}},
        ]
        batch = ircot_metrics.evaluate_batch(results)
        assert batch["mean_ircot_iterations"] == pytest.approx(1.5)
        assert batch["mean_ircot_confidence"] == pytest.approx(0.85)
        assert batch["ircot_convergence_rate"] == pytest.approx(1.0)
        assert "termination_reasons" in batch
        assert batch["termination_reasons"]["threshold_met"] == 1

    def test_empty_batch(self, ircot_metrics):
        batch = ircot_metrics.evaluate_batch([])
        assert batch["mean_ircot_iterations"] == pytest.approx(0.0)

    def test_no_ircot_key(self, ircot_metrics):
        results = [{"numerical": {}}, {"temporal": {}}]
        batch = ircot_metrics.evaluate_batch(results)
        assert batch["mean_ircot_iterations"] == pytest.approx(0.0)

    def test_termination_reasons_counted(self, ircot_metrics):
        results = [
            {"ircot": {"termination_reason": "plateau", "iterations": [{}],
                       "total_iterations": 1, "final_confidence": 0.4,
                       "converged": False, "total_improvement": 0.0}},
            {"ircot": {"termination_reason": "plateau", "iterations": [{}],
                       "total_iterations": 1, "final_confidence": 0.3,
                       "converged": False, "total_improvement": 0.0}},
            {"ircot": {"termination_reason": "max_iterations", "iterations": [{}],
                       "total_iterations": 1, "final_confidence": 0.2,
                       "converged": False, "total_improvement": 0.0}},
        ]
        batch = ircot_metrics.evaluate_batch(results)
        assert batch["termination_reasons"]["plateau"] == 2
        assert batch["termination_reasons"]["max_iterations"] == 1


# ---------------------------------------------------------------------------
# 15. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_assessor_unknown_module_ignored(self, assessor):
        out = assessor.assess({}, ["unknown_module"])
        assert out["overall"] == pytest.approx(0.5)

    def test_reformulator_no_gap(self, reformulator):
        assessment = {"primary_gap": None}
        result = {"temporal": {}, "causal": {}, "reasoning_trace": []}
        out = reformulator.reformulate("What is X?", result, assessment, 1)
        assert "What is X?" in out

    def test_merger_very_small_max_length(self):
        m = ContextMerger(max_length=10)
        new = {"text_contexts": [{"document": "A" * 100}]}
        merged, _ = m.merge("12345", new, set())
        assert len(merged) <= 10

    def test_controller_zero_max_iterations(self):
        ctrl = IRCoTController(max_iterations=0, confidence_threshold=0.01)
        result = {"numerical": {"success": False}}
        out = ctrl.run(
            question="q", example=None, result=result,
            context_text="c", active_modules=["numerical"],
            retrieve_fn=lambda e, q: {"text_contexts": []},
            reason_fn=lambda *a: {},
        )
        assert out["total_iterations"] == 1
        assert out["termination_reason"] in ("threshold_met_initial", "max_iterations")

    def test_causal_relation_with_hasattr(self, reformulator):
        class FakeRel:
            cause = "growth"
            effect = "profit"
        result = {"temporal": {}, "causal": {"causal_relations": [FakeRel()]}}
        entities = reformulator._extract_reasoning_entities(result)
        assert "growth" in entities
        assert "profit" in entities

    def test_context_merger_no_text_contexts_key(self, merger):
        merged, hashes = merger.merge("base", {}, set())
        assert merged == "base"

    def test_assessor_trend_empty_string(self, assessor):
        temp = {"temporal_entities": [], "trend_analysis": {"trend": ""}}
        score, gaps = assessor._assess_temporal(temp)
        assert any(g["type"] == "no_trend" for g in gaps)

    def test_assessor_trend_none(self, assessor):
        temp = {"temporal_entities": [], "trend_analysis": {"trend": None}}
        score, gaps = assessor._assess_temporal(temp)
        assert any(g["type"] == "no_trend" for g in gaps)
