from src.data import Chunk, Fact
from src.ontology import QueryProcessor
from src.retrieval import RetrievedChunk
from src.symbolic import SymbolicReasoner, resolve_comparison_years


def _make_chunk(facts):
    return Chunk(chunk_id="c1", doc_id="doc1", text="synthetic", kind="table", fiscal_years=[f.year for f in facts], facts=facts)


def test_percentage_change():
    facts = [Fact(metric="net income", year=2019, value=100.0), Fact(metric="net income", year=2020, value=150.0)]
    chunk = RetrievedChunk(chunk=_make_chunk(facts), score=0.9)
    eq = QueryProcessor().process("What was the percentage increase in net income from 2019 to 2020?")
    result = SymbolicReasoner().reason(eq, [chunk])
    assert result.success
    assert result.operation == "percentage_change"
    assert abs(result.value - 50.0) < 1e-6
    assert len(result.operands) == 2


def test_difference():
    facts = [Fact(metric="revenue", year=2019, value=1000.0), Fact(metric="revenue", year=2020, value=1200.0)]
    chunk = RetrievedChunk(chunk=_make_chunk(facts), score=0.9)
    eq = QueryProcessor().process("What was the difference in revenue between 2019 and 2020?")
    result = SymbolicReasoner().reason(eq, [chunk])
    assert result.success
    assert result.value == 200.0


def test_division_by_zero_reported_not_crashed():
    facts = [Fact(metric="net income", year=2019, value=0.0), Fact(metric="net income", year=2020, value=150.0)]
    chunk = RetrievedChunk(chunk=_make_chunk(facts), score=0.9)
    eq = QueryProcessor().process("What was the percentage increase in net income from 2019 to 2020?")
    result = SymbolicReasoner().reason(eq, [chunk])
    assert result.success is False
    assert result.error == "division_by_zero"


def test_missing_operand_reports_insufficient_data():
    facts = [Fact(metric="revenue", year=2019, value=1000.0)]
    chunk = RetrievedChunk(chunk=_make_chunk(facts), score=0.9)
    eq = QueryProcessor().process("What was the percentage increase in revenue from 2019 to 2020?")
    result = SymbolicReasoner().reason(eq, [chunk])
    assert result.success is False
    assert result.error == "insufficient_data"


def test_lookup_single_value():
    facts = [Fact(metric="net income", year=2020, value=150.0)]
    chunk = RetrievedChunk(chunk=_make_chunk(facts), score=0.9)
    eq = QueryProcessor().process("What was net income in 2020?")
    result = SymbolicReasoner().reason(eq, [chunk])
    assert result.success
    assert result.operation == "lookup"
    assert result.value == 150.0


def test_resolve_comparison_years_heuristic():
    eq = QueryProcessor().process("What was the change in revenue from last year to 2020?")
    years, heuristic = resolve_comparison_years(eq)
    assert heuristic is True
    assert years == [2019, 2020]


def test_no_facts_reports_insufficient_data():
    eq = QueryProcessor().process("What was net income in 2020?")
    result = SymbolicReasoner().reason(eq, [])
    assert result.success is False
    assert result.error == "insufficient_data"


def test_reason_with_fallback_tries_wider_tiers_when_narrow_tier_is_empty():
    facts = [Fact(metric="net income", year=2019, value=100.0), Fact(metric="net income", year=2020, value=150.0)]
    chunk = RetrievedChunk(chunk=_make_chunk(facts), score=0.9)
    eq = QueryProcessor().process("What was the percentage increase in net income from 2019 to 2020?")

    result = SymbolicReasoner().reason_with_fallback(
        eq, [("temporally_filtered", []), ("context_filtered", []), ("retrieved", [chunk])]
    )
    assert result.success
    assert result.tier_used == "retrieved"
    assert abs(result.value - 50.0) < 1e-6


def test_reason_with_fallback_reports_insufficient_data_when_all_tiers_empty():
    eq = QueryProcessor().process("What was net income in 2020?")
    result = SymbolicReasoner().reason_with_fallback(eq, [("temporally_filtered", []), ("retrieved", [])])
    assert result.success is False
    assert result.error == "insufficient_data"


def test_default_prior_year_heuristic_when_no_implicit_phrase_present():
    # "during 2015" alone (no "last year"/"previous year" phrase) still falls
    # back to comparing against the prior year as a last resort.
    eq = QueryProcessor().process("what was the change in net income during 2015?")
    years, heuristic = resolve_comparison_years(eq)
    assert years == [2014, 2015]
    assert heuristic is True


def test_prefers_directly_stated_change_over_recomputing_from_endpoints():
    # Regression test for a real FinQA case: the narrative states the delta
    # directly ("net revenue ... increased $94 million"), so the reasoner
    # should use that stated value rather than failing to find two endpoint
    # years to subtract (the question only names one explicit year, 2015).
    facts = [Fact(metric="for entergy corporation for 2015 increased", year=None, value=94.0)]
    chunk = RetrievedChunk(chunk=_make_chunk(facts), score=0.9)
    eq = QueryProcessor().process("what is the net change in net revenue during 2015 for entergy corporation?")
    result = SymbolicReasoner().reason(eq, [chunk])
    assert result.success
    assert result.operation == "stated_value"
    assert result.value == 94.0
