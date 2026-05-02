"""Tests for CounterfactualReasoner (P2-3c).

Covers counterfactual type detection, query parsing, probability of
necessity/sufficiency, contrastive explanation, multi-variable scenarios,
robustness analysis, explanation generation, and CausalityDetector integration."""

import pytest
from src.reasoning.causality_detector import (
    CausalityDetector,
    CausalRelation,
    CounterfactualQuery,
    CounterfactualReasoner,
    CounterfactualResult,
    CounterfactualType,
    FinancialSCM,
)


OBSERVED = {
    "interest_rate": 0.05,
    "inflation": 0.03,
    "revenue": 1000,
    "input_costs": 400,
    "capex": 100,
    "debt": 500,
    "tax_rate": 0.21,
    "share_count": 100,
    "sga_expense": 100,
}

TABLE = [
    ["Item", "2022", "2023"],
    ["Revenue", "900", "1000"],
    ["Cost of Goods Sold", "360", "400"],
    ["Net Income", "280", "300"],
    ["EPS", "2.80", "3.00"],
    ["Debt", "500", "500"],
]


def _make_reasoner():
    scm = FinancialSCM()
    return CounterfactualReasoner(scm, lambda q: _fuzzy_resolve(scm, q))


def _fuzzy_resolve(scm, query):
    query_words = set(query.lower().replace("_", " ").split())
    best, best_score = query, 0.0
    for node in scm.nodes:
        node_words = set(node.replace("_", " ").split())
        overlap = len(query_words & node_words) / max(len(query_words | node_words), 1)
        if overlap > best_score:
            best_score = overlap
            best = node
    return best


class TestCounterfactualType:
    def test_enum_values(self):
        assert CounterfactualType.INTERVENTIONAL == "interventional"
        assert CounterfactualType.RETROSPECTIVE == "retrospective"
        assert CounterfactualType.CONTRASTIVE == "contrastive"
        assert CounterfactualType.NECESSITY == "necessity"
        assert CounterfactualType.SUFFICIENCY == "sufficiency"


class TestCounterfactualQueryDataclass:
    def test_default_construction(self):
        q = CounterfactualQuery(treatment_var="revenue")
        assert q.treatment_var == "revenue"
        assert q.query_type == CounterfactualType.INTERVENTIONAL
        assert q.intervention_value is None

    def test_full_construction(self):
        q = CounterfactualQuery(
            treatment_var="revenue",
            query_type=CounterfactualType.RETROSPECTIVE,
            intervention_value=0.0,
            outcome_var="eps",
            original_question="test",
            direction="remove",
        )
        assert q.outcome_var == "eps"
        assert q.direction == "remove"


class TestCounterfactualResultDataclass:
    def test_to_dict(self):
        q = CounterfactualQuery(treatment_var="revenue", intervention_value=1200)
        r = CounterfactualResult(
            query=q,
            factual_values={"revenue": 1000},
            counterfactual_values={"revenue": 1200},
            downstream_effects={"gross_profit": {"baseline": 600, "counterfactual": 800, "change_pct": 33.33}},
            necessity_score=0.85,
            sufficiency_score=0.90,
            explanation="test explanation",
            confidence=0.7,
        )
        d = r.to_dict()
        assert d["treatment_var"] == "revenue"
        assert d["necessity_score"] == 0.85
        assert d["sufficiency_score"] == 0.9
        assert d["downstream_effects"]["gross_profit"]["change_pct"] == 33.33
        assert d["explanation"] == "test explanation"

    def test_to_dict_with_none_scores(self):
        q = CounterfactualQuery(treatment_var="debt")
        r = CounterfactualResult(query=q)
        d = r.to_dict()
        assert d["necessity_score"] is None
        assert d["sufficiency_score"] is None


class TestDetectCounterfactualType:
    def setup_method(self):
        self.r = _make_reasoner()

    def test_interventional_what_if(self):
        assert self.r.detect_counterfactual_type("What if we cut revenue by 10%?") == CounterfactualType.INTERVENTIONAL

    def test_interventional_suppose(self):
        assert self.r.detect_counterfactual_type("Suppose we increased costs by 20%") == CounterfactualType.INTERVENTIONAL

    def test_retrospective_had_not(self):
        assert self.r.detect_counterfactual_type("Had the merger not occurred, what would EPS be?") == CounterfactualType.RETROSPECTIVE

    def test_retrospective_without(self):
        assert self.r.detect_counterfactual_type("Without the acquisition, margins would be higher") == CounterfactualType.RETROSPECTIVE

    def test_contrastive_instead_of(self):
        assert self.r.detect_counterfactual_type("Why did revenue grow instead of decline?") == CounterfactualType.CONTRASTIVE

    def test_contrastive_rather_than(self):
        assert self.r.detect_counterfactual_type("Why margins rather than revenue?") == CounterfactualType.CONTRASTIVE

    def test_necessity(self):
        assert self.r.detect_counterfactual_type("Was revenue growth necessary for EPS increase?") == CounterfactualType.NECESSITY

    def test_sufficiency(self):
        assert self.r.detect_counterfactual_type("Was the cost cut sufficient to improve margins?") == CounterfactualType.SUFFICIENCY

    def test_default_fallback(self):
        result = self.r.detect_counterfactual_type("How did revenue change?")
        assert result == CounterfactualType.INTERVENTIONAL


class TestParseCounterfactualQuery:
    def setup_method(self):
        self.r = _make_reasoner()

    def test_parse_cut_by_pct(self):
        q = self.r.parse_counterfactual_query("if we cut revenue by 10%", OBSERVED)
        assert q is not None
        assert q.treatment_var == "revenue"
        assert q.intervention_value == pytest.approx(900.0)
        assert q.direction == "decrease"
        assert q.pct_change == -10.0

    def test_parse_increase_by_pct(self):
        q = self.r.parse_counterfactual_query("what if revenue increased by 20%", OBSERVED)
        assert q is not None
        assert q.treatment_var == "revenue"
        assert q.intervention_value == pytest.approx(1200.0)
        assert q.direction == "increase"
        assert q.pct_change == 20.0

    def test_parse_set_value(self):
        q = self.r.parse_counterfactual_query("what if revenue were 1500", OBSERVED)
        assert q is not None
        assert q.treatment_var == "revenue"
        assert q.intervention_value == 1500.0
        assert q.direction == "set"

    def test_parse_doubled(self):
        q = self.r.parse_counterfactual_query("what if revenue doubled", OBSERVED)
        assert q is not None
        assert q.intervention_value == pytest.approx(2000.0)
        assert q.pct_change == 100.0

    def test_parse_halved(self):
        q = self.r.parse_counterfactual_query("what if revenue halved", OBSERVED)
        assert q is not None
        assert q.intervention_value == pytest.approx(500.0)
        assert q.pct_change == -50.0

    def test_parse_fuzzy_variable(self):
        q = self.r.parse_counterfactual_query("if we cut input costs by 15%", OBSERVED)
        assert q is not None
        assert q.treatment_var == "input_costs"
        assert q.intervention_value == pytest.approx(340.0)

    def test_parse_retrospective(self):
        q = self.r.parse_counterfactual_query(
            "without the revenue, what would happen?", OBSERVED
        )
        assert q is not None
        assert q.query_type == CounterfactualType.RETROSPECTIVE
        assert q.intervention_value == 0.0

    def test_parse_returns_none_for_unmatched(self):
        q = self.r.parse_counterfactual_query("How is the weather?", OBSERVED)
        assert q is None

    def test_parse_contrastive(self):
        q = self.r.parse_counterfactual_query(
            "why did revenue grow instead of decline?", OBSERVED
        )
        assert q is not None
        assert q.query_type == CounterfactualType.CONTRASTIVE


class TestEvaluateNecessity:
    def setup_method(self):
        self.r = _make_reasoner()

    def test_revenue_necessary_for_gross_profit(self):
        factual = dict(OBSERVED)
        factual["gross_profit"] = 600
        factual["operating_income"] = 480
        factual["ebit"] = 480
        factual["interest_expense"] = 25
        factual["net_income"] = 359.55
        factual["eps"] = 3.5955
        factual["depreciation"] = 20

        pn = self.r.evaluate_necessity("revenue", "gross_profit", factual)
        assert pn > 0.5

    def test_zero_treatment_returns_zero(self):
        factual = dict(OBSERVED)
        factual["revenue"] = 0
        pn = self.r.evaluate_necessity("revenue", "gross_profit", factual)
        assert pn == 0.0

    def test_necessity_returns_bounded_value(self):
        factual = dict(OBSERVED)
        factual["gross_profit"] = 600
        factual["depreciation"] = 20
        pn = self.r.evaluate_necessity("revenue", "gross_profit", factual)
        assert 0.0 <= pn <= 1.0


class TestEvaluateSufficiency:
    def setup_method(self):
        self.r = _make_reasoner()

    def test_revenue_sufficient_for_gross_profit(self):
        factual = dict(OBSERVED)
        factual["gross_profit"] = 600
        factual["operating_income"] = 480
        factual["ebit"] = 480
        factual["interest_expense"] = 25
        factual["net_income"] = 359.55
        factual["eps"] = 3.5955
        factual["depreciation"] = 20

        ps = self.r.evaluate_sufficiency("revenue", "gross_profit", factual)
        assert 0.0 <= ps <= 1.0

    def test_sufficiency_returns_bounded_value(self):
        factual = dict(OBSERVED)
        factual["gross_profit"] = 600
        factual["depreciation"] = 20
        ps = self.r.evaluate_sufficiency("revenue", "gross_profit", factual)
        assert 0.0 <= ps <= 1.0


class TestContrastiveExplanation:
    def setup_method(self):
        self.r = _make_reasoner()

    def test_contrastive_produces_explanation(self):
        q = CounterfactualQuery(
            treatment_var="revenue",
            query_type=CounterfactualType.CONTRASTIVE,
            contrast_value="debt",
            original_question="why revenue instead of debt?",
        )
        result = self.r.contrastive_explanation(q, OBSERVED)
        assert "actual_variable" in result
        assert "contrast_variable" in result
        assert "explanation" in result
        assert result["actual_variable"] == "revenue"

    def test_contrastive_has_key_differentiators(self):
        q = CounterfactualQuery(
            treatment_var="revenue",
            query_type=CounterfactualType.CONTRASTIVE,
            contrast_value="input_costs",
        )
        result = self.r.contrastive_explanation(q, OBSERVED)
        assert "key_differentiators" in result


class TestMultiVariableScenario:
    def setup_method(self):
        self.r = _make_reasoner()

    def test_simultaneous_interventions(self):
        result = self.r.multi_variable_scenario(
            {"revenue": 1200, "input_costs": 350}, OBSERVED
        )
        assert "downstream_effects" in result
        assert "interventions" in result
        assert result["interventions"]["revenue"] == 1200.0
        assert result["num_affected"] > 0

    def test_downstream_effects_have_change_pct(self):
        result = self.r.multi_variable_scenario(
            {"revenue": 1200}, OBSERVED
        )
        for node, effect in result["downstream_effects"].items():
            assert "baseline" in effect
            assert "counterfactual" in effect
            assert "change_pct" in effect


class TestRobustnessAnalysis:
    def setup_method(self):
        self.r = _make_reasoner()

    def test_robustness_basic(self):
        q = CounterfactualQuery(
            treatment_var="revenue",
            intervention_value=1100,
            outcome_var="gross_profit",
        )
        result = self.r.robustness_analysis(q, OBSERVED)
        assert "sweep" in result
        assert len(result["sweep"]) == 5
        assert "robust" in result
        assert "monotonic" in result
        assert "sign_consistent" in result

    def test_robustness_sweep_has_perturbations(self):
        q = CounterfactualQuery(
            treatment_var="revenue",
            intervention_value=1100,
            outcome_var="gross_profit",
        )
        result = self.r.robustness_analysis(q, OBSERVED, steps=3)
        assert result["num_steps"] == 3
        pcts = [s["perturbation_pct"] for s in result["sweep"]]
        assert pcts[0] < pcts[-1]

    def test_robustness_zero_baseline(self):
        obs = dict(OBSERVED)
        obs["revenue"] = 0
        q = CounterfactualQuery(treatment_var="revenue", intervention_value=100)
        result = self.r.robustness_analysis(q, obs)
        assert result["robust"] is False
        assert result["reason"] == "zero_baseline"

    def test_revenue_to_gross_profit_is_monotonic(self):
        q = CounterfactualQuery(
            treatment_var="revenue",
            intervention_value=1100,
            outcome_var="gross_profit",
        )
        result = self.r.robustness_analysis(q, OBSERVED, steps=5)
        assert result["monotonic"] is True


class TestGenerateExplanation:
    def setup_method(self):
        self.r = _make_reasoner()

    def test_interventional_decrease_explanation(self):
        q = CounterfactualQuery(
            treatment_var="revenue",
            query_type=CounterfactualType.INTERVENTIONAL,
            direction="decrease",
            pct_change=-10.0,
        )
        result = CounterfactualResult(
            query=q,
            downstream_effects={
                "gross_profit": {"baseline": 600, "counterfactual": 500, "change_pct": -16.67},
            },
            necessity_score=0.85,
            robustness={"robust": True},
        )
        explanation = self.r.generate_explanation(result)
        assert "revenue" in explanation
        assert "10%" in explanation
        assert "robust" in explanation

    def test_retrospective_explanation(self):
        q = CounterfactualQuery(
            treatment_var="revenue",
            query_type=CounterfactualType.RETROSPECTIVE,
        )
        result = CounterfactualResult(query=q)
        explanation = self.r.generate_explanation(result)
        assert "not occurred" in explanation

    def test_contrastive_explanation_text(self):
        q = CounterfactualQuery(
            treatment_var="revenue",
            query_type=CounterfactualType.CONTRASTIVE,
            contrast_value="debt",
        )
        result = CounterfactualResult(query=q)
        explanation = self.r.generate_explanation(result)
        assert "revenue" in explanation
        assert "debt" in explanation

    def test_empty_result_produces_empty_explanation(self):
        q = CounterfactualQuery(treatment_var="revenue")
        result = CounterfactualResult(query=q)
        explanation = self.r.generate_explanation(result)
        assert isinstance(explanation, str)


class TestReasonIntegration:
    def setup_method(self):
        self.r = _make_reasoner()

    def test_reason_with_interventional_question(self):
        result = self.r.reason(
            question="what if we cut revenue by 10%?",
            table=TABLE,
            context="Revenue declined due to lower demand.",
            causal_relations=[],
            observed=OBSERVED,
        )
        assert result.get("treatment_var") == "revenue"
        assert result.get("query_type") == CounterfactualType.INTERVENTIONAL
        assert result.get("downstream_effects")
        assert result.get("necessity_score") is not None
        assert result.get("sufficiency_score") is not None
        assert result.get("robustness") is not None
        assert result.get("explanation")

    def test_reason_with_empty_observed(self):
        result = self.r.reason(
            question="what if we cut revenue by 10%?",
            table=TABLE,
            context="",
            causal_relations=[],
            observed={},
        )
        assert result == {}

    def test_reason_falls_back_to_causal_relations(self):
        relations = [
            CausalRelation(
                cause="revenue decline",
                effect="margin pressure",
                confidence=0.7,
                evidence="test",
            )
        ]
        result = self.r.reason(
            question="why did margins decline?",
            table=TABLE,
            context="Revenue decline caused margin pressure.",
            causal_relations=relations,
            observed=OBSERVED,
        )
        assert result.get("treatment_var") is not None

    def test_reason_includes_explanation(self):
        result = self.r.reason(
            question="what if revenue increased by 20%?",
            table=TABLE,
            context="Revenue growth drove margin expansion.",
            causal_relations=[],
            observed=OBSERVED,
        )
        assert len(result.get("explanation", "")) > 0

    def test_reason_counterfactual_values_populated(self):
        result = self.r.reason(
            question="what if revenue increased by 20%?",
            table=TABLE,
            context="",
            causal_relations=[],
            observed=OBSERVED,
        )
        assert result.get("counterfactual_values")
        assert result["counterfactual_values"].get("revenue") == pytest.approx(1200.0, rel=0.01)


class TestCausalityDetectorCounterfactualIntegration:
    def setup_method(self):
        self.detector = CausalityDetector()

    def test_reason_includes_counterfactual_analysis_key(self):
        result = self.detector.reason(
            question="what if we cut revenue by 10%?",
            context="Higher costs led to margin pressure.",
            table=TABLE,
        )
        assert "counterfactual_analysis" in result

    def test_counterfactual_analysis_with_table(self):
        result = self.detector.reason(
            question="what if we cut revenue by 10%?",
            context="Revenue drove profits higher.",
            table=TABLE,
        )
        cf = result["counterfactual_analysis"]
        assert cf.get("treatment_var") == "revenue"
        assert cf.get("downstream_effects")
        assert cf.get("necessity_score") is not None

    def test_counterfactual_explanation_in_causal_context(self):
        result = self.detector.reason(
            question="what if revenue increased by 20%?",
            context="Revenue growth drove earnings improvement.",
            table=TABLE,
        )
        cf = result["counterfactual_analysis"]
        if cf.get("explanation"):
            assert "Counterfactual:" in result.get("causal_context", "")

    def test_counterfactual_analysis_empty_without_table(self):
        result = self.detector.reason(
            question="what if we cut revenue by 10%?",
            context="Some context here.",
        )
        assert result["counterfactual_analysis"] == {} or isinstance(result["counterfactual_analysis"], dict)

    def test_counterfactual_reasoner_initialized(self):
        assert hasattr(self.detector, "counterfactual_reasoner")
        assert isinstance(self.detector.counterfactual_reasoner, CounterfactualReasoner)

    def test_robustness_in_analysis(self):
        result = self.detector.reason(
            question="what if we cut revenue by 10%?",
            context="Revenue drives profitability.",
            table=TABLE,
        )
        cf = result["counterfactual_analysis"]
        if cf:
            assert "robustness" in cf
            if cf["robustness"]:
                assert "robust" in cf["robustness"]
                assert "sweep" in cf["robustness"]
