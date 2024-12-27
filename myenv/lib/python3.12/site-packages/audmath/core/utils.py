import typing


def polyval(
    x: float,
    coefs: typing.Sequence,
) -> float:
    r"""Evaluation of polynomial.

    Args:
        x: input value
        coefs: polynomial coefficients

    Returns:
        evaluated polynomial

    """
    answer = 0
    power = len(coefs) - 1
    for coef in coefs:
        try:
            answer += coef * x**power
        except OverflowError:  # pragma: nocover
            pass
        power -= 1
    return answer
