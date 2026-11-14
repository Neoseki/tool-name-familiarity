import pytest

from src.decide import ModelResult, decide


def m(b, c, intrusions=0, tasks=0, orig=0):
    return ModelResult(b, c, intrusions, tasks, orig)


@pytest.mark.parametrize("b,c,p", [(5, 0, 1 / 32), (6, 1, 8 / 128), (7, 1, 9 / 256)])  # README: 0.031, 0.063, 0.035
def test_reference_p_values(b, c, p):
    assert m(b, c).p == pytest.approx(p)


def test_h3_needs_count_tasks_and_excess_over_original():
    assert m(0, 0, 3, 2, 2).h3
    assert not m(0, 0, 3, 1, 0).h3  # one task only
    assert not m(0, 0, 3, 2, 3).h3  # not above the original-condition baseline


@pytest.mark.parametrize("models,row", [
    ({"a": m(5, 0), "b": m(3, 0)}, "strong"),
    ({"a": m(5, 0), "b": m(2, 0)}, "inconclusive"),  # net >= 3 in only one model
    ({"a": m(4, 1), "b": m(3, 0)}, "suggestive"),  # p = 0.19 and 0.125
    ({"a": m(0, 3), "b": m(1, 4)}, "reversed"),
    ({"a": m(2, 0), "b": m(1, 0), "c": m(1, 0, 3, 2)}, "behaviour_without_score_change"),  # used to match two rows
    ({"a": m(1, 0), "b": m(0, 1)}, "null"),
    ({"a": m(0, 0), "b": m(2, 0), "c": m(1, 0)}, "inconclusive"),  # used to match no row
    ({"a": m(0, 2), "b": m(0, 0), "c": m(1, 0)}, "inconclusive"),  # used to match no row
    ({"a": m(5, 0)}, "inconclusive"),  # one model passing the gates: rows 1-3 can't match
])
def test_every_outcome_maps_to_one_row(models, row):
    assert decide(models) == row
