import numpy as np
import pytest

import audmath


@pytest.mark.parametrize(
    "x, bottom, expected_y",
    [
        (2, -120, 6.020599913279624),
        (0, None, -np.inf),
        (0, -120, -120),
        (-1, None, -np.inf),
        (-1, -120, -120),
        (0.0, None, -np.inf),
        (0.0, -120, -120),
        (-1.0, None, -np.inf),
        (-1.0, -120, -120),
        ([], None, np.array([])),
        ([], -120, np.array([])),
        (np.array([]), None, np.array([])),
        (np.array([]), -120, np.array([])),
        ([[]], None, np.array([[]])),
        ([[]], -120, np.array([[]])),
        (np.array([[]]), None, np.array([[]])),
        (np.array([[]]), -120, np.array([[]])),
        ([0, 1], None, np.array([-np.inf, 0.0])),
        ([0, 1], -120, np.array([-120, 0.0])),
        ([0.0, 1.0], None, np.array([-np.inf, 0.0])),
        ([0.0, 1.0], -120, np.array([-120, 0.0])),
        (np.array([0, 1]), None, np.array([-np.inf, 0.0])),
        (np.array([0, 1]), -120, np.array([-120, 0.0])),
        (np.array([0.0, 1.0]), None, np.array([-np.inf, 0.0])),
        (np.array([0.0, 1.0]), -120, np.array([-120, 0.0])),
        (np.array([[0], [1]]), None, np.array([[-np.inf], [0.0]])),
        (np.array([[0], [1]]), -120, np.array([[-120], [0.0]])),
        (np.array([[0.0], [1.0]]), None, np.array([[-np.inf], [0.0]])),
        (np.array([[0.0], [1.0]]), -120, np.array([[-120], [0.0]])),
    ],
)
def test_db(x, bottom, expected_y):
    y = audmath.db(x, bottom=bottom)
    np.testing.assert_allclose(y, expected_y)
    if isinstance(y, np.ndarray):
        assert np.issubdtype(y.dtype, np.floating)
    else:
        np.issubdtype(type(y), np.floating)
