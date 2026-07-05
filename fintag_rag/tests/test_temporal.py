from src.temporal import temporal_compatibility


def test_full_overlap():
    assert temporal_compatibility([2019, 2020], [2019, 2020, 2021]) == 1.0


def test_partial_overlap():
    assert temporal_compatibility([2019, 2020], [2020]) == 0.5


def test_no_overlap():
    assert temporal_compatibility([2019, 2020], [2021]) == 0.0


def test_no_query_constraint_is_permissive():
    # |T(q)| = 0 -> nothing to validate against, so the passage is not penalized.
    assert temporal_compatibility([], [2021]) == 1.0
