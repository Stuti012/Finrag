from src.causal import CausalExtractor, evaluate_causal_accuracy, is_causal_question, split_sentences


def test_split_sentences():
    text = "Revenue grew. Costs fell due to efficiency gains. Margins improved!"
    sentences = split_sentences(text)
    assert len(sentences) == 3


def test_effect_first_extraction():
    rel = CausalExtractor().extract_from_sentence("Operating margin declined due to rising raw material costs.")
    assert rel is not None
    assert "rising raw material costs" in rel.cause
    assert "operating margin declined" in rel.effect.lower()
    assert rel.cue_phrase == "due to"


def test_cause_first_extraction():
    rel = CausalExtractor().extract_from_sentence("Higher interest rates contributed to a decline in mortgage originations.")
    assert rel is not None
    assert "higher interest rates" in rel.cause.lower()
    assert "decline in mortgage originations" in rel.effect.lower()


def test_no_cue_returns_none():
    assert CausalExtractor().extract_from_sentence("Revenue was $150 million in fiscal 2020.") is None


def test_is_causal_question():
    assert is_causal_question("Why did operating margin decline in 2020?")
    assert not is_causal_question("What was net income in 2020?")


def test_evaluate_causal_accuracy_runs_and_is_bounded():
    report = evaluate_causal_accuracy()
    assert report["n_examples"] > 0
    assert 0.0 <= report["detection_rate"] <= 1.0
    assert 0.0 <= report["mean_span_f1"] <= 1.0
    # This module's extractor should detect the large majority of the
    # hand-labeled cue-phrase sentences (they were written to contain a cue).
    assert report["detection_rate"] >= 0.8
