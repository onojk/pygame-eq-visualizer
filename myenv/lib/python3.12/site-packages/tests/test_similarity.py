import numpy as np
import pandas as pd
import pytest
import scipy.spatial

import audmath


@pytest.mark.parametrize(
    "u, v, expected",
    [
        (
            [1],
            [0.5],
            1.0,
        ),
        (
            [1, 0],
            [0, 1],
            0.0,
        ),
        (
            [1, 0],
            [1, 0],
            1.0,
        ),
        (
            [1, 0],
            [-1, 0],
            -1.0,
        ),
        (
            [[1, 0]],
            [0, 1],
            np.array([0]),
        ),
        (
            [1, 0],
            [[0, 1]],
            np.array([0]),
        ),
        (
            [[1, 0]],
            [[0, 1]],
            np.array([[0]]),
        ),
        (
            [[1, 0], [0, 1]],
            [0, 1],
            np.array([0, 1]),
        ),
        (
            [1, 0],
            [[0, 1], [1, 0]],
            np.array([0, 1]),
        ),
        (
            [[1, 0], [0, 1]],
            [[0, 1]],
            np.array([[0], [1]]),
        ),
        (
            [[1, 0]],
            [[0, 1], [1, 0]],
            np.array([[0, 1]]),
        ),
        (
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            np.array([[0, 1], [1, 0]]),
        ),
        (
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1], [-1, 0]],
            np.array([[1, 0, -1], [0, 1, 0]]),
        ),
        (
            [[1, 0], [0, 1]],
            [[1, 1], [1, 1]],
            np.array(
                [
                    [1 / np.sqrt(2), 1 / np.sqrt(2)],
                    [1 / np.sqrt(2), 1 / np.sqrt(2)],
                ]
            ),
        ),
        (
            [0.23, 0.58],
            [0.12, 0.36],
            1
            - scipy.spatial.distance.cosine(
                [0.23, 0.58],
                [0.12, 0.36],
            ),
        ),
    ],
)
def test_similarity(u, v, expected):
    similarity = audmath.similarity(u, v)
    np.testing.assert_array_equal(similarity, expected)
    if isinstance(expected, np.ndarray):
        assert similarity.shape == expected.shape


@pytest.mark.parametrize(
    "u, v, expected",
    [
        (
            [1],
            [1],
            1.0,
        ),
        (
            [[1], [1]],
            [1],
            np.array([1, 1]),
        ),
        (
            [1],
            [[1], [1]],
            np.array([1, 1]),
        ),
        (
            [[1], [1]],
            [[1], [1]],
            np.array([[1, 1], [1, 1]]),
        ),
        (
            [[1, 0], [1, 0]],
            [[1, 0], [1, 0]],
            np.array([[1, 1], [1, 1]]),
        ),
        (
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            np.array([[0, 1], [1, 0]]),
        ),
    ],
)
def test_distance_shapes(u, v, expected):
    for u in [u, np.array(u), to_pandas(u)]:
        for v in [v, np.array(v), to_pandas(v)]:
            similarity = audmath.similarity(u, v)
            np.testing.assert_array_equal(similarity, expected)
            if isinstance(expected, np.ndarray):
                assert similarity.shape == expected.shape


def to_pandas(x):
    x = np.array(x)
    if x.ndim < 2:
        x_pandas = pd.Series(x, name="0")
    else:
        x_pandas = pd.DataFrame(
            data=x,
            columns=[str(n) for n in range(x.shape[-1])],
        )
    return x_pandas
