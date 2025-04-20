import numpy as np
import pytest
import scipy.special
import scipy.stats

import audmath


@pytest.mark.parametrize(
    "y, expected_x",
    [
        (0, -np.inf),
        (1, np.inf),
        ([0, 1], np.array([-np.inf, np.inf])),
        (np.array([0, 1]), np.array([-np.inf, np.inf])),
    ],
)
def test_inverse_normal_distribution(y, expected_x):
    x = audmath.inverse_normal_distribution(y)
    np.testing.assert_allclose(x, expected_x)
    if isinstance(x, np.ndarray):
        assert np.issubdtype(x.dtype, np.floating)
    else:
        np.issubdtype(type(x), np.floating)


@pytest.mark.parametrize(
    "y",
    [
        0,
        np.exp(-32),
        0.1,
        0.2,
        0.3,
        1,
        -1,
        10,
        np.linspace(0, 1, 50),
    ],
)
def test_inverse_normal_distribution_scipy(y):
    x = audmath.inverse_normal_distribution(y)
    np.testing.assert_allclose(x, scipy.special.ndtri(y))
    np.testing.assert_allclose(x, scipy.stats.norm.ppf(y))
