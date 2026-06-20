"""Tests for QueryExpander (FinTAG-RAG §3.1): q' = q ∪ E(q)."""

import pytest

from src.retrieval.query_expansion import QueryExpander


class TestExpandTerms:
    def setup_method(self):
        self.e = QueryExpander()

    def test_colloquial_triggers_formal_synonym(self):
        terms = self.e.expand_terms("what were sales in 2020?")
        assert "revenue" in [t.lower() for t in terms]

    def test_abbreviation_triggers_full_form(self):
        terms = self.e.expand_terms("what was the cogs?")
        assert any("cost of" in t.lower() for t in terms)

    def test_no_match_returns_empty(self):
        assert self.e.expand_terms("what color is the sky?") == []

    def test_empty_query(self):
        assert self.e.expand_terms("") == []

    def test_does_not_duplicate_present_terms(self):
        terms = self.e.expand_terms("what was the revenue and net sales?")
        assert "revenue" not in [t.lower() for t in terms]
        assert "net sales" not in [t.lower() for t in terms]

    def test_max_terms_cap(self):
        e = QueryExpander(max_terms=2)
        terms = e.expand_terms("revenue, net income, cogs, capex, eps all changed")
        assert len(terms) <= 2

    def test_word_boundary_no_partial_match(self):
        # "eps" should not falsely trigger on a word containing it as substring.
        terms = self.e.expand_terms("the steps taken were minor")
        assert "eps" not in [t.lower() for t in terms]


class TestExpand:
    def setup_method(self):
        self.e = QueryExpander()

    def test_appends_terms_to_query(self):
        expanded = self.e.expand("what were sales in 2020?")
        assert expanded.startswith("what were sales in 2020?")
        assert "revenue" in expanded.lower()

    def test_noop_when_no_ontology_hit(self):
        q = "what color is the sky?"
        assert self.e.expand(q) == q


class TestCustomOntology:
    def test_custom_ontology_used(self):
        e = QueryExpander(ontology={"foo": ["bar", "baz"]})
        terms = e.expand_terms("what is foo?")
        assert set(t.lower() for t in terms) == {"bar", "baz"}
