from src.data import (
    Fact,
    _extract_table_facts,
    build_corpus,
    extract_fiscal_years,
    FinQAExample,
    parse_financial_number,
)


def test_parse_financial_number_basic():
    assert parse_financial_number("1,234.5") == 1234.5
    assert parse_financial_number("$1,234") == 1234.0
    assert parse_financial_number("(123)") == -123.0
    assert parse_financial_number("25%") == 0.25
    assert parse_financial_number("n/a") is None
    assert parse_financial_number("") is None


def test_extract_fiscal_years():
    assert extract_fiscal_years("What was revenue in fiscal year 2021 vs 2019?") == [2019, 2021]
    assert extract_fiscal_years("FY2022 results") == [2022]
    assert extract_fiscal_years("no years here") == []


def test_extract_table_facts():
    table = [
        ["", "2019", "2020"],
        ["net income", "100", "150"],
        ["revenue", "1,000", "1,200"],
    ]
    text, facts = _extract_table_facts(table)
    assert len(facts) == 4
    assert any(f.metric == "net income" and f.year == 2019 and f.value == 100.0 for f in facts)
    assert any(f.metric == "revenue" and f.year == 2020 and f.value == 1200.0 for f in facts)
    assert "net income" in text


def test_build_corpus_produces_table_and_text_chunks():
    ex = FinQAExample(
        id="doc1-0",
        question="What was net income in 2020?",
        table=[["", "2019", "2020"], ["net income", "100", "150"]],
        pre_text=["The company reported strong performance in fiscal 2020."],
        post_text=["Management expects continued growth."],
        program=[],
        answer="150",
    )
    chunks = build_corpus([ex], chunk_size_tokens=50, chunk_overlap_tokens=5)
    kinds = {c.kind for c in chunks}
    assert "table" in kinds
    assert "text" in kinds
    table_chunk = next(c for c in chunks if c.kind == "table")
    assert len(table_chunk.facts) == 2
    # documents are deduplicated by doc_id
    chunks2 = build_corpus([ex, ex], chunk_size_tokens=50, chunk_overlap_tokens=5)
    assert len(chunks2) == len(chunks)
