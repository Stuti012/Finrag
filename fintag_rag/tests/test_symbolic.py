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
