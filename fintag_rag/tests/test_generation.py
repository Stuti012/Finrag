from src.generation import extract_final_number


def test_extracts_from_answer_line():
    text = "Revenue grew from 100 to 125, a 25% increase.\nANSWER: 25"
    assert extract_final_number(text) == 25.0


def test_extracts_negative_and_currency():
    text = "The company lost money.\nANSWER: -$1,234.5"
    assert extract_final_number(text) == -1234.5


def test_no_answer_line_returns_none_even_with_numbers_present():
    # Regression test: the model rambling through unrelated evidence numbers
    # (e.g. a $9,018,834 balance-sheet figure) must NOT be picked up as the
    # predicted answer just because it's the last number in the text.
    text = "Total assets were $9,018,834 thousand in 2020 and grew from 2019."
    assert extract_final_number(text) is None


def test_empty_text_returns_none():
    assert extract_final_number("") is None
