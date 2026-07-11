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


def test_extract_table_facts_falls_back_to_row_label_year():
    # Regression test from a real FinQA example (ETR/2016/page_23.pdf-2):
    # the column header carries no year at all ("amount ( in millions )"),
    # so the year must be recovered from the row label itself
    # ("2014 net revenue" / "2015 net revenue"). Without this fallback every
    # fact in this (very common "bridge table") FinQA pattern gets year=None
    # and can never be matched against a query asking about a specific year.
    table = [
        ["", "amount ( in millions )"],
        ["2014 net revenue", "$ 5735"],
        ["retail electric price", "187"],
        ["2015 net revenue", "$ 5829"],
    ]
    _, facts = _extract_table_facts(table)
    by_metric = {f.metric: f for f in facts}
    assert by_metric["2014 net revenue"].year == 2014
    assert by_metric["2014 net revenue"].value == 5735.0
    assert by_metric["2015 net revenue"].year == 2015
    assert by_metric["2015 net revenue"].value == 5829.0
    # a row with no year in its label and no year in the header stays year=None
    assert by_metric["retail electric price"].year is None


def test_extract_table_facts_prefers_header_year_over_row_label_year():
    # When the header DOES carry the year, it should win even if the row
    # label happens to also contain a 4-digit number.
    table = [["", "2019", "2020"], ["2015 vintage reserve", "10", "12"]]
    _, facts = _extract_table_facts(table)
    years = sorted(f.year for f in facts)
    assert years == [2019, 2020]
