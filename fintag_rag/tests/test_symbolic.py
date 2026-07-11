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


def test_stated_value_used_only_when_no_clean_table_endpoints_exist():
    # When NO table facts exist for the relevant years, a directly-stated
    # narrative value is used as a fallback (the question only names one
    # explicit year, 2015 -- the reasoner must default to comparing it
    # against 2014).
    facts = [Fact(metric="for entergy corporation for 2015 increased", year=None, value=94.0, kind="text")]
    chunk = RetrievedChunk(chunk=_make_chunk(facts), score=0.9)
    eq = QueryProcessor().process("what is the net change in net revenue during 2015 for entergy corporation?")
    result = SymbolicReasoner().reason(eq, [chunk])
    assert result.success
    assert result.operation == "stated_value"
    assert result.value == 94.0


def test_part_whole_ratio_question_end_to_end():
    # Regression test from a real FinQA example: "what percentage of total
    # facilities as measured in square feet are leased?" -> divide(8.1, 56.0).
    facts = [
        Fact(metric="total facilities", year=None, value=56.0, kind="table"),
        Fact(metric="leased facilities", year=None, value=8.1, kind="table"),
    ]
    chunk = RetrievedChunk(chunk=_make_chunk(facts), score=0.9)
    eq = QueryProcessor().process("what percentage of total facilities as measured in square feet are leased?")
    assert eq.ratio_phrase is not None
    result = SymbolicReasoner().reason(eq, [chunk])
    assert result.success
    assert result.operation == "ratio"
    assert abs(result.value - 8.1 / 56.0) < 1e-6


def test_clean_table_endpoints_preferred_over_noisy_stated_value():
    # Regression test built directly from a real failure: ETR/2016/page_23.pdf-2
    # ("what is the net change in net revenue during 2015 for entergy
    # corporation?", gold=94.0 via subtract(5829, 5735)). The retrieved
    # evidence also contains an unrelated, noisier "increase" mention from a
    # different subsidiary's disclosure -- the reasoner must prefer the
    # clean, verifiable two-endpoint table subtraction over that distractor,
    # not the other way around.
    table_facts = [
        Fact(metric="2014 net revenue", year=2014, value=5735.0, kind="table"),
        Fact(metric="2015 net revenue", year=2015, value=5829.0, kind="table"),
    ]
    distractor_facts = [
        Fact(metric="increase at entergy mississippi of", year=None, value=16.0, kind="text"),
    ]
    chunks = [
        RetrievedChunk(chunk=_make_chunk(table_facts), score=0.9),
        RetrievedChunk(chunk=_make_chunk(distractor_facts), score=0.9),
    ]
    eq = QueryProcessor().process("what is the net change in net revenue during 2015 for entergy corporation?")
    result = SymbolicReasoner().reason(eq, chunks)
    assert result.success
    assert result.operation == "difference"
    assert result.value == 94.0
