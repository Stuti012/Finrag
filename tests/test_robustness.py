"""Tests for the robustness/perturbation harness (src/evaluation/robustness.py)."""

import pytest

from src.data.finqa_loader import FinQAExample
from src.evaluation.robustness import (
    DistractorInjector,
    SynonymParaphraser,
    summarize_flip_rate,
)


def make_example(eid, question, pre_text=None, post_text=None, table=None):
    return FinQAExample(
        id=eid,
        question=question,
        table=table or [["Item", "2020", "2019"], ["Revenue", "5829", "5735"]],
        pre_text=pre_text or [],
        post_text=post_text or [],
        program=["subtract(5829, 5735)"],
        answer="94",
    )


class TestDistractorInjector:
    def test_injects_off_period_sentence(self):
        pool = [
            make_example("pool-1", "what was revenue in 2015?", post_text=["In 2015 revenue was $3,100 million."]),
        ]
        target = make_example("target-1", "what was the change in revenue in 2019 and 2020?")
        injector = DistractorInjector(pool)
        perturbed, sentence, year = injector.inject(target)

        assert sentence is not None
        assert year == 2015
        assert sentence in perturbed.post_text
        # Original example object is untouched.
        assert sentence not in target.post_text

    def test_skips_when_no_query_periods(self):
        pool = [make_example("pool-1", "x", post_text=["In 2015 revenue was $3,100 million."])]
        target = make_example("target-1", "what is the revenue?", table=[["Item", "Value"]])
        injector = DistractorInjector(pool)
        perturbed, sentence, year = injector.inject(target)
        assert sentence is None and year is None
        assert perturbed is target

    def test_skips_when_no_disjoint_distractor_available(self):
        # Pool only has distractors that overlap the target's own years.
        pool = [make_example("pool-1", "x", post_text=["In 2019 revenue was $3,100 million."])]
        target = make_example("target-1", "what was the change in revenue in 2019 and 2020?")
        injector = DistractorInjector(pool)
        perturbed, sentence, year = injector.inject(target)
        assert sentence is None and year is None

    def test_deterministic_with_seed(self):
        pool = [
            make_example("pool-1", "x", post_text=["In 2015 revenue was $3,100 million."]),
            make_example("pool-2", "x", post_text=["In 2010 costs rose to $900 million."]),
        ]
        target = make_example("target-1", "what was the change in revenue in 2019 and 2020?")
        a = DistractorInjector(pool, seed=7).inject(target)
        b = DistractorInjector(pool, seed=7).inject(target)
        assert a[1] == b[1]


class TestSynonymParaphraser:
    def setup_method(self):
        self.p = SynonymParaphraser()

    def test_substitutes_known_term(self):
        out = self.p.paraphrase("what was the revenue in 2020?")
        assert out is not None
        assert "revenue" not in out.lower()

    def test_no_match_returns_none(self):
        assert self.p.paraphrase("what color is the sky?") is None

    def test_custom_ontology(self):
        p = SynonymParaphraser(ontology={"foo": ["bar"]})
        assert p.paraphrase("what is foo?") == "what is bar?"


class TestSummarizeFlipRate:
    def test_empty(self):
        assert summarize_flip_rate([]) == {
            "n": 0, "base_accuracy": 0.0, "perturbed_accuracy": 0.0, "flip_rate": 0.0
        }

    def test_flip_rate_computation(self):
        records = [
            {"base_correct": True, "perturbed_correct": True},
            {"base_correct": True, "perturbed_correct": False},
            {"base_correct": False, "perturbed_correct": False},
        ]
        summary = summarize_flip_rate(records)
        assert summary["n"] == 3
        assert summary["base_accuracy"] == pytest.approx(2 / 3)
        assert summary["perturbed_accuracy"] == pytest.approx(1 / 3)
        assert summary["flip_rate"] == pytest.approx(0.5)  # 1 of 2 base-correct flipped
