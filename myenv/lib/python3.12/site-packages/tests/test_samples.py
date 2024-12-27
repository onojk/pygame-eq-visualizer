import pandas as pd
import pytest

import audmath


@pytest.mark.parametrize(
    "duration, sampling_rate, expected",
    [
        (1, 16000, 16000),
        (1.14, 16000, 18240),
        (pd.Timedelta("0 days 00:00:01.140000").total_seconds(), 16000, 18240),
        (0.5, 10, 5),
        (-0.5, 10, -5),
        (0.55, 10, 6),
        (-0.55, 10, -6),
    ],
)
def test_samples(duration, sampling_rate, expected):
    samples = audmath.samples(duration, sampling_rate)
    assert isinstance(samples, int)
    assert samples == expected
