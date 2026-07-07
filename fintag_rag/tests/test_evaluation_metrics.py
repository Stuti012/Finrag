from src.evaluation import EvalRecord, aggregate_metrics


def _rec(predicted, gold, correct):
    return EvalRecord(example_id="x", mode="full", question="q", predicted_numeric=predicted, gold_numeric=gold, is_correct=correct)


def test_unconditional_accuracy_reflects_answer_rate():
    records = [
        _rec(50.0, 50.0, True),
        _rec(None, 50.0, None),  # no answer produced
        _rec(None, 50.0, None),
        _rec(None, 50.0, None),
    ]
    m = aggregate_metrics(records)
    assert m["numerical_accuracy"] == 1.0  # 1/1 answered questions correct
    assert m["numerical_accuracy_unconditional"] == 0.25  # 1/4 of all questions
    assert m["answer_rate"] == 0.25


def test_median_ae_is_robust_to_outliers():
    records = [
        _rec(101.0, 100.0, False),
        _rec(102.0, 100.0, False),
        _rec(9_000_000.0, 100.0, False),  # one wild outlier
    ]
    m = aggregate_metrics(records)
    assert m["median_ae"] == 2.0  # errors are [1, 2, 8999900] -> median 2
    assert m["mae"] > 1000  # mean is dominated by the outlier
