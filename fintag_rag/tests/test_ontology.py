from src.ontology import QueryProcessor


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
