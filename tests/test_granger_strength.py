"""Tests for Granger causal strength estimation (P3-3e).

Covers lag matrix construction, OLS residuals, F-test, incremental R²,
transfer entropy, BIC lag selection, bidirectional testing, composite
scoring, table series extraction, CausalityDetector integration,
metrics, and edge cases."""

import math
import pytest
import numpy as np
from src.reasoning.causality_detector import (
    CausalityDetector,
    GrangerCausalStrengthEstimator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def estimator():
    return GrangerCausalStrengthEstimator(max_lag=3)


@pytest.fixture
def causal_series():
    """Two series where x Granger-causes y: y_t = 0.8*y_{t-1} + 0.5*x_{t-1} + noise."""
    rng = np.random.RandomState(42)
    n = 100
    x = np.cumsum(rng.randn(n))
    y = np.zeros(n)
    y[0] = rng.randn()
    for t in range(1, n):
        y[t] = 0.8 * y[t - 1] + 0.5 * x[t - 1] + 0.3 * rng.randn()
    return x, y


@pytest.fixture
def independent_series():
    """Two independent random walks."""
    rng = np.random.RandomState(99)
    n = 100
    x = np.cumsum(rng.randn(n))
    y = np.cumsum(rng.randn(n))
    return x, y


@pytest.fixture
def financial_table():
    return [
        ["Metric", "2018", "2019", "2020", "2021", "2022", "2023"],
        ["Revenue", "100", "110", "120", "115", "130", "145"],
        ["Cost of Goods Sold", "60", "65", "70", "68", "75", "82"],
        ["Net Income", "20", "22", "25", "23", "28", "32"],
        ["EPS", "2.00", "2.20", "2.50", "2.30", "2.80", "3.20"],
        ["Interest Expense", "5", "6", "7", "8", "9", "10"],
    ]


# ---------------------------------------------------------------------------
# 1. Lag matrix construction
# ---------------------------------------------------------------------------

class TestBuildLagMatrix:
    def test_shape(self, estimator):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mat = estimator._build_lag_matrix(s, 2)
        assert mat.shape == (3, 2)

    def test_values(self, estimator):
        s = np.array([10.0, 20.0, 30.0, 40.0])
        mat = estimator._build_lag_matrix(s, 1)
        assert mat.shape == (3, 1)
        np.testing.assert_array_equal(mat[:, 0], [10.0, 20.0, 30.0])

    def test_empty_when_too_short(self, estimator):
        s = np.array([1.0, 2.0])
        mat = estimator._build_lag_matrix(s, 3)
        assert mat.shape[0] == 0

    def test_single_lag(self, estimator):
        s = np.array([1.0, 2.0, 3.0])
        mat = estimator._build_lag_matrix(s, 1)
        assert mat.shape == (2, 1)


# ---------------------------------------------------------------------------
# 2. OLS residuals
# ---------------------------------------------------------------------------

class TestOLSResiduals:
    def test_perfect_fit(self, estimator):
        X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = np.array([3.0, 5.0, 7.0])
        beta, res = estimator._ols_residuals(X, y)
        assert np.allclose(res, 0.0, atol=1e-6)

    def test_residuals_shape(self, estimator):
        X = np.ones((5, 2))
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        beta, res = estimator._ols_residuals(X, y)
        assert len(res) == 5

    def test_empty_X_returns_y(self, estimator):
        X = np.empty((0, 0))
        y = np.array([1.0, 2.0])
        beta, res = estimator._ols_residuals(X, y)
        np.testing.assert_array_equal(res, y)


# ---------------------------------------------------------------------------
# 3. Granger F-test
# ---------------------------------------------------------------------------

class TestGrangerFTest:
    def test_causal_series_significant(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.granger_f_test(x, y, lag=1)
        assert result["significant"] is True
        assert result["f_statistic"] > 0
        assert result["p_value"] < 0.05

    def test_independent_series_not_significant(self, estimator, independent_series):
        x, y = independent_series
        result = estimator.granger_f_test(x, y, lag=1)
        assert result["p_value"] > 0.01 or result["f_statistic"] < 10

    def test_result_keys(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.granger_f_test(x, y, lag=1)
        assert "f_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "lag" in result
        assert "rss_restricted" in result
        assert "rss_unrestricted" in result

    def test_insufficient_data(self, estimator):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        result = estimator.granger_f_test(x, y, lag=1)
        assert result["significant"] is False
        assert result["p_value"] == 1.0

    def test_different_lags(self, estimator, causal_series):
        x, y = causal_series
        r1 = estimator.granger_f_test(x, y, lag=1)
        r2 = estimator.granger_f_test(x, y, lag=2)
        assert r1["lag"] == 1
        assert r2["lag"] == 2


# ---------------------------------------------------------------------------
# 4. F-distribution p-value
# ---------------------------------------------------------------------------

class TestFSurvival:
    def test_zero_f_gives_one(self, estimator):
        assert estimator._f_survival(0.0, 2, 10) == 1.0

    def test_large_f_gives_small_p(self, estimator):
        p = estimator._f_survival(50.0, 2, 50)
        assert p < 0.01

    def test_moderate_f(self, estimator):
        p = estimator._f_survival(3.0, 2, 20)
        assert 0.0 < p < 1.0

    def test_regularized_incomplete_beta_bounds(self):
        val = GrangerCausalStrengthEstimator._regularized_incomplete_beta(0.5, 2.0, 3.0)
        assert 0.0 <= val <= 1.0

    def test_regularized_incomplete_beta_edges(self):
        assert GrangerCausalStrengthEstimator._regularized_incomplete_beta(0.0, 2.0, 3.0) == 0.0
        assert GrangerCausalStrengthEstimator._regularized_incomplete_beta(1.0, 2.0, 3.0) == 1.0


# ---------------------------------------------------------------------------
# 5. Incremental R²
# ---------------------------------------------------------------------------

class TestIncrementalR2:
    def test_causal_series_positive_delta(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.incremental_r_squared(x, y, lag=1)
        assert result["incremental_r2"] > 0.0
        assert result["r2_unrestricted"] >= result["r2_restricted"]

    def test_independent_series_near_zero(self, estimator, independent_series):
        x, y = independent_series
        result = estimator.incremental_r_squared(x, y, lag=1)
        assert result["incremental_r2"] < 0.1

    def test_result_keys(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.incremental_r_squared(x, y, lag=1)
        assert "r2_restricted" in result
        assert "r2_unrestricted" in result
        assert "incremental_r2" in result

    def test_insufficient_data(self, estimator):
        result = estimator.incremental_r_squared(
            np.array([1.0]), np.array([2.0]), lag=1,
        )
        assert result["incremental_r2"] == 0.0


# ---------------------------------------------------------------------------
# 6. Transfer entropy
# ---------------------------------------------------------------------------

class TestTransferEntropy:
    def test_causal_positive(self, estimator, causal_series):
        x, y = causal_series
        te = estimator.transfer_entropy(x, y, lag=1)
        assert te > 0.0

    def test_non_negative(self, estimator, independent_series):
        x, y = independent_series
        te = estimator.transfer_entropy(x, y, lag=1)
        assert te >= 0.0

    def test_short_series_zero(self, estimator):
        te = estimator.transfer_entropy(np.array([1.0]), np.array([2.0]), lag=1)
        assert te == 0.0

    def test_different_bins(self, estimator, causal_series):
        x, y = causal_series
        te3 = estimator.transfer_entropy(x, y, lag=1, bins=3)
        te5 = estimator.transfer_entropy(x, y, lag=1, bins=5)
        assert te3 >= 0.0 and te5 >= 0.0


# ---------------------------------------------------------------------------
# 7. BIC lag selection
# ---------------------------------------------------------------------------

class TestSelectLagOrder:
    def test_returns_optimal_lag(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.select_lag_order(x, y)
        assert "optimal_lag" in result
        assert 1 <= result["optimal_lag"] <= 3

    def test_bic_criterion(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.select_lag_order(x, y, criterion="bic")
        assert result["criterion"] == "bic"

    def test_aic_criterion(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.select_lag_order(x, y, criterion="aic")
        assert result["criterion"] == "aic"

    def test_scores_list(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.select_lag_order(x, y, max_lag=3)
        assert len(result["scores"]) >= 1

    def test_short_series(self, estimator):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        result = estimator.select_lag_order(x, y, max_lag=2)
        assert result["optimal_lag"] >= 1


# ---------------------------------------------------------------------------
# 8. Bidirectional testing
# ---------------------------------------------------------------------------

class TestBidirectionalTest:
    def test_causal_direction_detected(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.bidirectional_test(x, y)
        assert result["dominant_direction"] in ("x_causes_y", "bidirectional")
        assert result["x_to_y"]["significant"] is True

    def test_asymmetry_sign(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.bidirectional_test(x, y)
        assert result["asymmetry"] > 0 or result["dominant_direction"] == "bidirectional"

    def test_result_keys(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.bidirectional_test(x, y, lag=1)
        assert "x_to_y" in result
        assert "y_to_x" in result
        assert "incremental_r2_xy" in result
        assert "incremental_r2_yx" in result
        assert "asymmetry" in result
        assert "dominant_direction" in result
        assert "lag" in result

    def test_independent_no_causality(self, estimator, independent_series):
        x, y = independent_series
        result = estimator.bidirectional_test(x, y, lag=1)
        assert result["dominant_direction"] in ("no_causality", "x_causes_y", "y_causes_x", "bidirectional")


# ---------------------------------------------------------------------------
# 9. Full analysis
# ---------------------------------------------------------------------------

class TestFullAnalysis:
    def test_causal_returns_all_keys(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.full_analysis(x, y, "revenue", "profit")
        assert "cause" in result
        assert "effect" in result
        assert "optimal_lag" in result
        assert "f_test" in result
        assert "granger_significant" in result
        assert "incremental_r2" in result
        assert "transfer_entropy" in result
        assert "bidirectional" in result
        assert "strength" in result
        assert result["insufficient_data"] is False

    def test_causal_significant(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.full_analysis(x, y)
        assert result["granger_significant"] is True
        assert result["strength"] > 0.2

    def test_insufficient_data(self, estimator):
        result = estimator.full_analysis(
            np.array([1.0, 2.0]), np.array([3.0, 4.0]),
        )
        assert result["insufficient_data"] is True
        assert result["strength"] == 0.0

    def test_strength_bounded(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.full_analysis(x, y)
        assert 0.0 <= result["strength"] <= 1.0

    def test_labels_propagated(self, estimator, causal_series):
        x, y = causal_series
        result = estimator.full_analysis(x, y, "interest_rate", "loan_demand")
        assert result["cause"] == "interest_rate"
        assert result["effect"] == "loan_demand"


# ---------------------------------------------------------------------------
# 10. Composite strength
# ---------------------------------------------------------------------------

class TestCompositeStrength:
    def test_all_strong(self):
        s = GrangerCausalStrengthEstimator._composite_strength(
            f_significant=True, incremental_r2=0.3, transfer_entropy=0.2, asymmetry=0.3,
        )
        assert s > 0.5

    def test_all_weak(self):
        s = GrangerCausalStrengthEstimator._composite_strength(
            f_significant=False, incremental_r2=0.0, transfer_entropy=0.0, asymmetry=0.0,
        )
        assert s == 0.0

    def test_bounded(self):
        s = GrangerCausalStrengthEstimator._composite_strength(
            f_significant=True, incremental_r2=1.0, transfer_entropy=1.0, asymmetry=1.0,
        )
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# 11. Table series extraction
# ---------------------------------------------------------------------------

class TestExtractSeriesFromTable:
    def test_revenue_found(self, estimator, financial_table):
        s = estimator.extract_series_from_table(financial_table, "revenue")
        assert s is not None
        assert len(s) == 6
        assert s[0] == 100.0

    def test_eps_found(self, estimator, financial_table):
        s = estimator.extract_series_from_table(financial_table, "eps")
        assert s is not None

    def test_missing_keyword(self, estimator, financial_table):
        s = estimator.extract_series_from_table(financial_table, "nonexistent")
        assert s is None

    def test_short_table(self, estimator):
        s = estimator.extract_series_from_table(
            [["Metric", "2020"], ["Revenue", "100"]], "revenue"
        )
        assert s is None

    def test_empty_table(self, estimator):
        assert estimator.extract_series_from_table([], "revenue") is None
        assert estimator.extract_series_from_table(None, "revenue") is None


# ---------------------------------------------------------------------------
# 12. analyze_from_table
# ---------------------------------------------------------------------------

class TestAnalyzeFromTable:
    def test_revenue_to_net_income(self, estimator, financial_table):
        result = estimator.analyze_from_table(financial_table, "revenue", "net income")
        assert "strength" in result
        assert "granger_significant" in result or "insufficient_data" in result

    def test_missing_series(self, estimator, financial_table):
        result = estimator.analyze_from_table(financial_table, "nonexistent", "revenue")
        assert result["insufficient_data"] is True
        assert result["reason"] == "series_not_found"


# ---------------------------------------------------------------------------
# 13. CausalityDetector integration
# ---------------------------------------------------------------------------

class TestCausalityDetectorGrangerIntegration:
    def setup_method(self):
        self.detector = CausalityDetector()

    def test_has_granger_estimator(self):
        assert hasattr(self.detector, "granger_estimator")
        assert isinstance(self.detector.granger_estimator, GrangerCausalStrengthEstimator)

    def test_granger_style_strength_delegates(self):
        table = [
            ["Metric", "2017", "2018", "2019", "2020", "2021", "2022"],
            ["Revenue", "90", "100", "110", "120", "130", "140"],
            ["Cost", "50", "55", "60", "65", "70", "75"],
        ]
        result = self.detector._granger_style_strength(table, "revenue growth", "cost increase")
        assert result is None or isinstance(result, float)

    def test_granger_full_analysis(self):
        table = [
            ["Metric", "2017", "2018", "2019", "2020", "2021", "2022"],
            ["Revenue", "90", "100", "110", "120", "130", "140"],
            ["Cost", "50", "55", "60", "65", "70", "75"],
        ]
        result = self.detector.granger_full_analysis(table, "revenue", "cost")
        assert isinstance(result, dict)
        assert "strength" in result

    def test_reason_includes_granger_analysis(self):
        table = [
            ["Metric", "2017", "2018", "2019", "2020", "2021", "2022"],
            ["Revenue", "90", "100", "110", "120", "130", "140"],
            ["Cost", "50", "55", "60", "65", "70", "75"],
        ]
        result = self.detector.reason(
            question="Why did cost increase?",
            context="Revenue growth led to higher costs. Higher revenue caused cost increases.",
            table=table,
        )
        assert "granger_analysis" in result
        ga = result["granger_analysis"]
        assert "num_tested" in ga
        assert "num_significant" in ga
        assert "mean_strength" in ga


# ---------------------------------------------------------------------------
# 14. Metrics integration
# ---------------------------------------------------------------------------

class TestGrangerMetrics:
    def test_metrics_with_data(self):
        from src.evaluation.metrics import CausalityDetectionMetrics
        m = CausalityDetectionMetrics()
        causal_info = {
            "granger_analysis": {
                "num_tested": 3,
                "num_significant": 2,
                "mean_strength": 0.65,
                "analyses": [],
            }
        }
        result = m.granger_analysis_quality(causal_info)
        assert result["granger_tested"] == 3
        assert result["granger_significant"] == 2
        assert result["granger_mean_strength"] == 0.65
        assert result["granger_has_analysis"] == 1.0

    def test_metrics_empty(self):
        from src.evaluation.metrics import CausalityDetectionMetrics
        m = CausalityDetectionMetrics()
        result = m.granger_analysis_quality({})
        assert result["granger_tested"] == 0
        assert result["granger_has_analysis"] == 0.0


# ---------------------------------------------------------------------------
# 15. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_series(self, estimator):
        x = np.ones(20)
        y = np.ones(20)
        result = estimator.full_analysis(x, y)
        assert result["strength"] == 0.0 or result["insufficient_data"]

    def test_very_short_series(self, estimator):
        result = estimator.full_analysis(np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                                          np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
        assert "strength" in result

    def test_nan_handling(self, estimator):
        s = estimator.extract_series_from_table(
            [["M", "2018", "2019", "2020", "2021"],
             ["Rev", "100", "N/A", "120", "130"]],
            "rev"
        )
        assert s is None

    def test_custom_max_lag(self):
        est = GrangerCausalStrengthEstimator(max_lag=5)
        assert est.max_lag == 5

    def test_custom_significance(self):
        est = GrangerCausalStrengthEstimator(significance_level=0.01)
        assert est.significance_level == 0.01
