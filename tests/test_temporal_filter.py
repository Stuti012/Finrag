"""Tests for TemporalCompatibilityFilter (FinTAG-RAG §3.3).

Covers period extraction T(·), query-constraint construction T(q) with table
fallback, the compatibility score T(d,q), and the τ-gate behaviour including the
recall-protecting refinements (neutral retention, min_keep guard).
"""

import pytest

from src.retrieval.temporal_filter import (
    FiscalPeriod,
    TemporalCompatibilityFilter,
)


class TestPeriodExtraction:
    def setup_method(self):
        self.f = TemporalCompatibilityFilter()

    def test_extract_plain_years(self):
        years = {p.year for p in self.f.extract_periods("revenue rose from 2019 to 2020")}
        assert years == {2019, 2020}

    def test_extract_fy_shorthand(self):
        years = {p.year for p in self.f.extract_periods("FY19 vs fiscal 2020 and FY'21")}
        assert years == {2019, 2020, 2021}

    def test_fy_two_digit_pivot(self):
        # <50 → 20xx, >=50 → 19xx
        assert {p.year for p in self.f.extract_periods("FY05")} == {2005}
        assert {p.year for p in self.f.extract_periods("FY98")} == {1998}

    def test_year_with_quarter(self):
        periods = self.f.extract_periods("Q3 2020 results")
        assert FiscalPeriod(2020, 3) in periods

    def test_bare_quarter_ignored(self):
        # A quarter with no anchoring year is too ambiguous to keep.
        assert self.f.extract_periods("the third quarter was strong") == set()

    def test_empty_text(self):
        assert self.f.extract_periods("") == set()


class TestQueryPeriods:
    def setup_method(self):
        self.f = TemporalCompatibilityFilter()

    def test_explicit_question_years(self):
        years = {p.year for p in self.f.query_periods("change in revenue 2019 to 2020?")}
        assert years == {2019, 2020}

    def test_table_fallback(self):
        table = [["Item", "2021", "2020"], ["Revenue", "5829", "5735"]]
        years = {p.year for p in self.f.query_periods("what was the revenue?", table)}
        assert years == {2020, 2021}

    def test_question_takes_priority_over_table(self):
        table = [["Item", "2021", "2020"]]
        years = {p.year for p in self.f.query_periods("revenue in 2018?", table)}
        assert years == {2018}

    def test_no_constraint(self):
        assert self.f.query_periods("what is the revenue?") == set()


class TestCompatibilityScore:
    def setup_method(self):
        self.f = TemporalCompatibilityFilter()

    def test_full_overlap(self):
        q = self.f.query_periods("revenue 2019 and 2020?")
        score, neutral = self.f.compatibility("for 2019 and 2020 revenue grew", q)
        assert score == 1.0 and not neutral

    def test_partial_overlap(self):
        q = self.f.query_periods("revenue 2019 and 2020?")
        score, neutral = self.f.compatibility("2020 revenue was 5829", q)
        assert score == 0.5 and not neutral

    def test_off_period(self):
        q = self.f.query_periods("revenue 2019 and 2020?")
        score, neutral = self.f.compatibility("2015 revenue was 3100", q)
        assert score == 0.0 and not neutral

    def test_neutral_passage(self):
        q = self.f.query_periods("revenue 2019 and 2020?")
        score, neutral = self.f.compatibility("revenue is recognised on delivery", q)
        assert score == 0.0 and neutral

    def test_no_constraint_compatible(self):
        score, neutral = self.f.compatibility("anything", set())
        assert score == 1.0


class TestFilterGate:
    def setup_method(self):
        self.f = TemporalCompatibilityFilter(tau=0.5)
        self.passages = [
            {"text": "Net revenue in 2020 was $5,829 million.", "score": 0.9},
            {"text": "In 2015 revenue was $3,100 million.", "score": 0.8},
            {"text": "Revenue is recognized when control transfers.", "score": 0.7},
            {"text": "For 2019 and 2020 costs rose.", "score": 0.6},
        ]

    def test_discards_off_period(self):
        kept, diag = self.f.filter(self.passages, "pct change in revenue 2019 to 2020?")
        texts = " ".join(p["text"] for p in kept)
        assert "2015" not in texts
        assert diag.discarded == 1
        assert diag.discarded_periods == [2015]

    def test_keeps_neutral_by_default(self):
        kept, diag = self.f.filter(self.passages, "pct change 2019 to 2020?")
        assert diag.neutral_kept == 1
        assert any(p["temporal_neutral"] for p in kept)

    def test_strict_drops_neutral(self):
        strict = TemporalCompatibilityFilter(tau=0.5, keep_neutral=False, min_keep=0)
        kept, diag = strict.filter(self.passages, "pct change 2019 to 2020?")
        assert all(not p["temporal_neutral"] for p in kept)

    def test_noop_when_no_constraint(self):
        kept, diag = self.f.filter(self.passages, "what is the revenue?")
        assert not diag.applied
        assert len(kept) == len(self.passages)

    def test_min_keep_guard(self):
        # All passages off-period → guard salvages the top min_keep.
        off = [
            {"text": "2010 figures", "score": 0.9},
            {"text": "2011 figures", "score": 0.8},
        ]
        guarded = TemporalCompatibilityFilter(tau=0.5, keep_neutral=False, min_keep=1)
        kept, diag = guarded.filter(off, "revenue in 2020?")
        assert len(kept) >= 1

    def test_annotations_written(self):
        kept, _ = self.f.filter(self.passages, "pct change 2019 to 2020?")
        for p in kept:
            assert "temporal_compatibility" in p
            assert "temporal_neutral" in p

    def test_preserves_order(self):
        kept, _ = self.f.filter(self.passages, "pct change 2019 to 2020?")
        scores = [p["score"] for p in kept]
        assert scores == sorted(scores, reverse=True)

    def test_empty_passages(self):
        kept, diag = self.f.filter([], "revenue 2020?")
        assert kept == []


class TestFiscalPeriod:
    def test_annual_subsumes_quarter(self):
        assert FiscalPeriod(2020).compatible_with(FiscalPeriod(2020, 3))
        assert FiscalPeriod(2020, 3).compatible_with(FiscalPeriod(2020))

    def test_quarter_conflict(self):
        assert not FiscalPeriod(2020, 1).compatible_with(FiscalPeriod(2020, 2))

    def test_year_conflict(self):
        assert not FiscalPeriod(2019).compatible_with(FiscalPeriod(2020))


def test_invalid_tau():
    with pytest.raises(ValueError):
        TemporalCompatibilityFilter(tau=1.5)
