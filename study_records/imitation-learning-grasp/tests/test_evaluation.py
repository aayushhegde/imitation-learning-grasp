import pytest

from ilgrasp.evaluation import wilson_interval


def test_wilson_interval_contains_empirical_rate():
    low, high = wilson_interval(70, 100)
    assert low < 0.70 < high
    assert low == pytest.approx(0.604, abs=0.002)
    assert high == pytest.approx(0.781, abs=0.002)


def test_wilson_interval_validates_inputs():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
