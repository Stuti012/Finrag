from src.ontology import QueryProcessor, extract_entity_phrase, extract_ratio_phrase


def test_normalize():
    qp = QueryProcessor()
    assert qp.normalize("  What WAS   Revenue?  ") == "what was revenue?"


def test_ontology_expansion():
    qp = QueryProcessor()
    eq = qp.process("What was the EPS in 2021?")
    assert "earnings per share" in eq.expanded_terms
    assert eq.explicit_years == [2021]


def test_synonym_expansion():
    qp = QueryProcessor()
    eq = qp.process("What was net income in 2020?")
    assert any("net earnings" in t or "profit after tax" in t for t in eq.expanded_terms)


def test_operation_detection():
    qp = QueryProcessor()
    eq = qp.process("What was the percentage increase in revenue from 2019 to 2020?")
    assert eq.operation_hint == "percentage_change"
    assert eq.explicit_years == [2019, 2020]


def test_implicit_temporal_detection():
    qp = QueryProcessor()
    eq = qp.process("What was the change in revenue from last year?")
    assert "last year" in eq.implicit_temporal_refs
    assert eq.has_unresolved_temporal_reference is True


def test_entity_phrase_disambiguates_subsidiary_from_parent():
    a = extract_entity_phrase("what is the net change in net revenue during 2015 for entergy corporation?")
    b = extract_entity_phrase("what is the roa for entergy new orleans , inc . in 2015?")
    assert a == "entergy corporation"
    assert b == "entergy new orleans , inc"
    assert a != b


def test_entity_phrase_handles_of_pattern():
    assert extract_entity_phrase("what was the percentage increase in net income of apple inc in 2020?") == "apple inc"


def test_entity_phrase_none_when_no_entity_mentioned():
    assert extract_entity_phrase("what was revenue in 2020?") is None


# The following are regression tests built directly from real FinQA test-set
# questions (verified against https://github.com/czyssrs/FinQA) that
# previously fell through to a naive single-value lookup instead of being
# recognized as part/whole ratio questions.


def test_ratio_phrase_percentage_of_x_are_y():
    assert extract_ratio_phrase("what percentage of total facilities as measured in square feet are leased?") == (
        "total facilities as measured in square feet",
        "leased",
    )


def test_ratio_phrase_percent_of_x_to_y():
    assert extract_ratio_phrase(
        "in 2010 what was the percent of the income tax benefit to the stock based compensation cost"
    ) == ("stock based compensation cost", "income tax benefit")


def test_ratio_phrase_of_x_what_percentage_is_y():
    assert extract_ratio_phrase(
        "of the total contractual obligations and off-balance sheet arrangements contractual obligations "
        "what percentage is due to capital lease obligations?"
    ) == ("total contractual obligations and off-balance sheet arrangements contractual obligations", "capital lease obligations")


def test_ratio_phrase_x_as_a_percentage_of_y():
    assert extract_ratio_phrase("what is net income as a percentage of total revenue?") == ("total revenue", "net income")
