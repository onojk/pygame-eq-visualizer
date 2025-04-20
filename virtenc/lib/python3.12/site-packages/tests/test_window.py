import numpy as np
import pytest

import audmath


@pytest.mark.parametrize(
    "shape",
    [
        "linear",
        "kaiser",
        "tukey",
        "exponential",
        "logarithmic",
    ],
)
@pytest.mark.parametrize(
    "samples, half, expected",
    [
        (-1, "left", np.array([])),
        (0, "left", np.array([])),
        (1, "left", np.array([0])),
        (2, "left", np.array([0, 1])),
        (-1, "right", np.array([])),
        (0, "right", np.array([])),
        (1, "right", np.array([0])),
        (2, "right", np.array([1, 0])),
        (-1, None, np.array([])),
        (0, None, np.array([])),
        (1, None, np.array([0])),
        (2, None, np.array([0, 0])),
        (3, None, np.array([0, 1, 0])),
    ],
)
def test_window_level(shape, samples, half, expected):
    win = audmath.window(samples, shape=shape, half=half)
    np.testing.assert_allclose(win, expected)
    assert np.issubdtype(win.dtype, np.floating)


@pytest.mark.parametrize(
    "samples, shape, half, expected",
    [
        (3, "linear", "left", np.array([0, 0.5, 1])),
        (3, "kaiser", "left", np.array([0, 4.6272e-01, 1])),
        (3, "tukey", "left", np.array([0, 0.5, 1])),
        (3, "exponential", "left", np.array([0, 0.26894142, 1])),
        (3, "logarithmic", "left", np.array([0, 0.63092975, 1])),
        (3, "linear", "right", np.array([1, 0.5, 0])),
        (3, "kaiser", "right", np.array([1, 4.6272e-01, 0])),
        (3, "tukey", "right", np.array([1, 0.5, 0])),
        (3, "exponential", "right", np.array([1, 0.26894142, 0])),
        (3, "logarithmic", "right", np.array([1, 0.63092975, 0])),
        (5, "linear", None, np.array([0, 0.5, 1, 0.5, 0])),
        (5, "kaiser", None, np.array([0, 4.6272e-01, 1, 4.6272e-01, 0])),
        (5, "tukey", None, np.array([0, 0.5, 1, 0.5, 0])),
        (5, "exponential", None, np.array([0, 0.26894142, 1, 0.26894142, 0])),
        (5, "logarithmic", None, np.array([0, 0.63092975, 1, 0.63092975, 0])),
        (4, "linear", None, np.array([0, 0.5, 0.5, 0])),
        (4, "kaiser", None, np.array([0, 4.6272e-01, 4.6272e-01, 0])),
        (4, "tukey", None, np.array([0, 0.5, 0.5, 0])),
        (4, "exponential", None, np.array([0, 0.26894142, 0.26894142, 0])),
        (4, "logarithmic", None, np.array([0, 0.63092975, 0.63092975, 0])),
    ],
)
def test_window_shape(samples, shape, half, expected):
    win = audmath.window(samples, shape=shape, half=half)
    np.testing.assert_allclose(win, expected, rtol=1e-05)
    assert np.issubdtype(win.dtype, np.floating)


@pytest.mark.parametrize(
    "shape, half, error, error_msg",
    [
        (
            "unknown",
            None,
            ValueError,
            (
                "shape has to be one of the following: "
                f"{(', ').join(audmath.core.api.WINDOW_SHAPES)},"
                f"not 'unknown'."
            ),
        ),
        (
            "linear",
            "center",
            ValueError,
            ("half has to be 'left' or 'right' " "not 'center'."),
        ),
    ],
)
def test_window_error(shape, half, error, error_msg):
    with pytest.raises(error, match=error_msg):
        audmath.window(3, shape=shape, half=half)
