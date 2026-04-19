"""Tests for FinancialSCM: structural equations, d-separation, do-calculus,
backdoor/frontdoor criteria, and CausalityDetector SCM integration."""

import pytest
from src.reasoning.causality_detector import FinancialSCM, CausalityDetector


class TestFinancialSCMStructure:
    def setup_method(self):
        self.scm = FinancialSCM()

    def test_nodes_populated(self):
        assert len(self.scm.nodes) > 15
        for var in ("revenue", "net_income", "eps", "interest_rate", "debt"):
            assert var in self.scm.nodes

    def test_edges_correct(self):
        assert "revenue" in self.scm.parents["gross_profit"]
        assert "input_costs" in self.scm.parents["gross_profit"]
        assert "gross_profit" in self.scm.children["revenue"]

    def test_topological_order(self):
        order = self.scm.topological_order()
        assert len(order) == len(self.scm.nodes)
        idx = {n: i for i, n in enumerate(order)}
        for node in self.scm.nodes:
            for parent in self.scm.parents.get(node, []):
                assert idx[parent] < idx[node], f"{parent} should come before {node}"

    def test_ancestors(self):
        anc = self.scm.ancestors("eps")
        assert "net_income" in anc
        assert "revenue" in anc
        assert "interest_rate" in anc

    def test_descendants(self):
        desc = self.scm.descendants("revenue")
        assert "gross_profit" in desc
        assert "net_income" in desc
        assert "eps" in desc

    def test_find_all_paths(self):
        paths = self.scm.find_all_paths("revenue", "eps")
        assert len(paths) >= 1
        for p in paths:
            assert p[0] == "revenue"
            assert p[-1] == "eps"

    def test_exogenous_variables(self):
        summary = self.scm.get_structure_summary()
        exo = summary["exogenous_variables"]
        for var in exo:
            assert not self.scm.parents.get(var), f"{var} should have no parents"

    def test_structure_summary(self):
        s = self.scm.get_structure_summary()
        assert s["num_nodes"] > 0
        assert s["num_edges"] > 0
        assert s["num_equations"] > 0
        assert "net_income" in s["equation_specs"]


class TestDSeparation:
    def setup_method(self):
        self.scm = FinancialSCM()

    def test_connected_nodes_not_dseparated(self):
        assert not self.scm.d_separated("revenue", "eps")

    def test_unrelated_exogenous_dseparated(self):
        assert self.scm.d_separated("capex", "inflation")

    def test_conditioning_blocks_path(self):
        assert self.scm.d_separated("revenue", "eps", {"net_income"})

    def test_dsep_with_nonexistent_node(self):
        assert self.scm.d_separated("revenue", "nonexistent_node")

    def test_direct_parent_child_not_dseparated(self):
        assert not self.scm.d_separated("revenue", "gross_profit")

    def test_conditioning_on_mediator_blocks(self):
        assert self.scm.d_separated("capex", "operating_income", {"depreciation"})


class TestDoCalculus:
    def setup_method(self):
        self.scm = FinancialSCM()
        self.observed = {
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

    def test_do_intervention_propagates(self):
        result = self.scm.do_intervention({"revenue": 1200}, self.observed)
        assert result["revenue"] == 1200
        assert result["gross_profit"] == 1200 - 400

    def test_do_severs_incoming_edges(self):
        result = self.scm.do_intervention({"gross_profit": 999}, self.observed)
        assert result["gross_profit"] == 999
        assert result["revenue"] == self.observed["revenue"]

    def test_do_propagates_to_eps(self):
        base = self.scm.do_intervention({"revenue": 1000}, self.observed)
        shifted = self.scm.do_intervention({"revenue": 1100}, self.observed)
        assert shifted["eps"] > base["eps"]

    def test_do_debt_affects_net_income(self):
        base = self.scm.do_intervention({"debt": 500}, self.observed)
        high_debt = self.scm.do_intervention({"debt": 1000}, self.observed)
        assert high_debt["net_income"] < base["net_income"]


class TestBackdoorFrontdoor:
    def setup_method(self):
        self.scm = FinancialSCM()

    def test_backdoor_criterion_valid(self):
        result = self.scm.backdoor_criterion("revenue", "eps")
        assert result["valid"] is True
        assert result["treatment"] == "revenue"
        assert result["outcome"] == "eps"

    def test_backdoor_nonexistent_node(self):
        result = self.scm.backdoor_criterion("bogus", "eps")
        assert result["valid"] is False

    def test_frontdoor_criterion(self):
        result = self.scm.frontdoor_criterion("revenue", "eps")
        assert result["treatment"] == "revenue"
        assert result["outcome"] == "eps"
        if result["valid"]:
            assert "mediator_set" in result

    def test_frontdoor_no_path(self):
        result = self.scm.frontdoor_criterion("eps", "interest_rate")
        assert result["valid"] is False


class TestCounterfactual:
    def setup_method(self):
        self.scm = FinancialSCM()
        self.factual = {
            "interest_rate": 0.05,
            "inflation": 0.03,
            "revenue": 1000,
            "input_costs": 400,
            "capex": 100,
            "debt": 500,
            "tax_rate": 0.21,
            "share_count": 100,
            "sga_expense": 100,
            "gross_profit": 600,
            "depreciation": 20,
            "operating_income": 480,
            "ebit": 480,
            "interest_expense": 25,
            "net_income": 359.55,
            "eps": 3.5955,
        }

    def test_counterfactual_returns_structure(self):
        result = self.scm.counterfactual(self.factual, {"revenue": 1200})
        assert "factual" in result
        assert "intervention" in result
        assert "counterfactual_values" in result
        assert "abducted_residuals" in result

    def test_counterfactual_revenue_increase(self):
        result = self.scm.counterfactual(self.factual, {"revenue": 1200})
        cf = result["counterfactual_values"]
        assert cf["revenue"] == 1200
        assert cf["gross_profit"] > self.factual["gross_profit"]

    def test_counterfactual_preserves_residuals(self):
        result = self.scm.counterfactual(self.factual, {"revenue": 1200})
        assert isinstance(result["abducted_residuals"], dict)


class TestCausalEffectAndSensitivity:
    def setup_method(self):
        self.scm = FinancialSCM()
        self.observed = {
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

    def test_causal_effect_estimate(self):
        result = self.scm.causal_effect_estimate("revenue", "eps", self.observed)
        assert result["treatment"] == "revenue"
        assert result["outcome"] == "eps"
        assert result["marginal_effect"] != 0
        assert result["num_causal_paths"] >= 1

    def test_sensitivity_analysis(self):
        results = self.scm.sensitivity_analysis("eps", self.observed)
        assert len(results) > 0
        assert all("elasticity" in r for r in results)
        assert abs(results[0]["elasticity"]) >= abs(results[-1]["elasticity"])


class TestCausalityDetectorSCMIntegration:
    def setup_method(self):
        self.detector = CausalityDetector()

    def test_financial_scm_is_scm_class(self):
        assert isinstance(self.detector.financial_scm, FinancialSCM)

    def test_scm_paths_uses_new_scm(self):
        paths = self.detector._scm_paths("interest_rate", "eps")
        assert len(paths) >= 1
        for p in paths:
            assert p[0] == "interest_rate"
            assert p[-1] == "eps"

    def test_reason_includes_scm_structure(self):
        result = self.detector.reason(
            question="why did eps decline?",
            context="Higher interest rates led to increased borrowing costs.",
        )
        assert "scm_structure" in result
        assert result["scm_structure"]["num_nodes"] > 0
        assert "scm_dseparation" in result
        assert "scm_backdoor_criterion" in result
        assert "scm_frontdoor_criterion" in result

    def test_reason_scm_sensitivity_with_table(self):
        table = [
            ["Item", "2022", "2023"],
            ["Revenue", "1000", "1100"],
            ["Cost of Goods Sold", "400", "450"],
            ["Net Income", "300", "320"],
            ["EPS", "3.00", "3.20"],
        ]
        result = self.detector.reason(
            question="why did eps decline?",
            context="Rising costs led to lower margins.",
            table=table,
        )
        assert "scm_sensitivity" in result

    def test_interventional_counterfactual_cut(self):
        table = [
            ["Item", "2023"],
            ["Revenue", "1000"],
            ["Cost of Goods Sold", "400"],
            ["Net Income", "300"],
        ]
        result = self.detector._interventional_counterfactual(
            "if we cut revenue by 10%", table
        )
        assert result.get("scm_propagation") is True
        assert result["assumption"] == "structural_equations"
        assert "downstream_effects" in result

    def test_interventional_counterfactual_increase(self):
        table = [
            ["Item", "2023"],
            ["Revenue", "1000"],
            ["Cost of Goods Sold", "400"],
        ]
        result = self.detector._interventional_counterfactual(
            "what if revenue increased by 20%", table
        )
        assert result.get("scm_propagation") is True
        assert result["predicted"] == 1200.0

    def test_extract_observed_from_table(self):
        table = [
            ["Item", "2023"],
            ["Revenue", "1,000"],
            ["Cost of Goods Sold", "400"],
            ["Net Income", "300"],
            ["Debt", "500"],
        ]
        observed = self.detector._extract_observed_from_table(table)
        assert observed["revenue"] == 1000.0
        assert observed["input_costs"] == 400.0
        assert observed["net_income"] == 300.0
        assert observed["debt"] == 500.0
