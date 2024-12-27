import numpy as np
import pytest

import audmath


@pytest.mark.parametrize(
    "y, bottom, expected_x",
    [
        (0, None, 1.0),
        (0, -120, 1.0),
        (0.0, None, 1.0),
        (0.0, -120, 1.0),
        (-np.inf, None, 0.0),
        (-np.inf, -120, 0.0),
        (-160, None, 1e-08),
        (-160, -120, 0.0),
        (-160.0, None, 1e-08),
        (-160.0, -120, 0.0),
        (-120, None, 1e-06),
        (-120, -120, 0.0),
        (-120.0, None, 1e-06),
        (-120.0, -120, 0.0),
        (-1, None, 0.8912509381337456),
        (-1, -120, 0.8912509381337456),
        (-1.0, None, 0.8912509381337456),
        (-1.0, -120, 0.8912509381337456),
        ([-np.inf, -120], None, np.array([0.0, 1e-06])),
        ([-np.inf, -120], -120, np.array([0.0, 0.0])),
        ([], None, np.array([])),
        ([], -120, np.array([])),
        (np.array([]), None, np.array([])),
        (np.array([]), -120, np.array([])),
        ([[]], None, np.array([[]])),
        ([[]], -120, np.array([[]])),
        (np.array([[]]), None, np.array([[]])),
        (np.array([[]]), -120, np.array([[]])),
        ([0, -1], None, np.array([1.0, 0.8912509381337456])),
        ([0, -1], -120, np.array([1.0, 0.8912509381337456])),
        ([0.0, -1.0], None, np.array([1.0, 0.8912509381337456])),
        ([0.0, -1.0], -120, np.array([1.0, 0.8912509381337456])),
        (np.array([-np.inf, -120]), None, np.array([0.0, 1e-06])),
        (np.array([-np.inf, -120]), -120, np.array([0.0, 0.0])),
        (np.array([0, -1]), None, np.array([1.0, 0.8912509381337456])),
        (np.array([0, -1]), -120, np.array([1.0, 0.8912509381337456])),
        (np.array([0.0, -1.0]), None, np.array([1.0, 0.8912509381337456])),
        (np.array([0.0, -1.0]), -120, np.array([1.0, 0.8912509381337456])),
        (np.array([[-np.inf], [-120]]), None, np.array([[0.0], [1e-06]])),
        (np.array([[-np.inf], [-120]]), -120, np.array([[0.0], [0.0]])),
        (np.array([[0], [-1]]), None, np.array([[1.0], [0.8912509381337456]])),
        (np.array([[0], [-1]]), -120, np.array([[1.0], [0.8912509381337456]])),
        (
            np.array([[0.0], [-1.0]]),
            None,
            np.array([[1.0], [0.8912509381337456]]),
        ),
        (
            np.array([[0.0], [-1.0]]),
            -120,
            np.array([[1.0], [0.8912509381337456]]),
        ),
    ],
)
def test_inverse_db(y, bottom, expected_x):
    x = audmath.inverse_db(y, bottom=bottom)
    np.testing.assert_allclose(x, expected_x)
    if isinstance(x, np.ndarray):
        assert np.issubdtype(x.dtype, np.floating)
    else:
        np.issubdtype(type(x), np.floating)
