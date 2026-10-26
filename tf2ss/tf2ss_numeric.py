"""Numerical (floating-point) MIMO transfer function to state-space conversion.

Counterpart of :func:`tf2ss.base.tf2ss` that does not use sympy. For every
input column, each distinct denominator gets one controller-canonical block,
shared by the outputs that use it. There is no root finding or pole matching,
which spreads repeated roots and drops tiny coefficients. Redundant states are
removed afterwards by :func:`tf2ss.minreal.minimal_realization`.
"""

import numpy as np
from control import TransferFunction, tf
from numpy.typing import NDArray

from .minreal import minimal_realization


def _trim(p: NDArray) -> NDArray:
    """Drop leading zeros (keep a single zero for the zero polynomial)."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    nz = np.flatnonzero(p)
    return p[nz[0] :] if nz.size else np.zeros(1)


def _companion(
    num: NDArray, den: NDArray
) -> tuple[NDArray, NDArray, NDArray, float]:
    """Controller form of num/den: A with first row -a[1:], B = e1, C, D."""
    num, den = _trim(num) / _trim(den)[0], _trim(den) / _trim(den)[0]
    n = den.size - 1
    if num.size > n + 1:
        raise ValueError("Transfer function is improper")
    b = np.zeros(n + 1)
    b[n + 1 - num.size :] = num
    A = np.eye(n, k=-1)
    A[:1, :] = -den[1:]
    B = np.zeros((n, 1))
    B[:1, 0] = 1.0
    return A, B, (b[1:] - b[0] * den[1:])[None, :], b[0]


def tf2ss(
    *args: TransferFunction | list[list[list[int | float]]],
    minreal: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Convert a MIMO transfer function to state space numerically.

    Parameters:
        *args: Either a single TransferFunction, or two nested lists
            ``numerators[i][j]`` and ``denominators[i][j]`` that hold the
            coefficients of output ``i`` / input ``j`` in descending powers.
        minreal: Reduce to a minimal realisation (orthogonal staircase).

    Returns:
        A, B, C, D of the realisation.

    Examples:
        >>> A, B, C, D = tf2ss([[[1]]], [[[1, 2]]])
        >>> A, B, C, D
        (array([[-2.]]), array([[1.]]), array([[1.]]), array([[0.]]))

        >>> A, B, C, D = tf2ss([[[1], [2]]], [[[1, 1], [1, 1]]], minreal=False)
        >>> A.shape                   # one block per input
        (2, 2)
        >>> tf2ss([[[1], [2]]], [[[1, 1], [1, 1]]])[0].shape  # minimal
        (1, 1)
    """
    if len(args) == 1 and isinstance(args[0], TransferFunction):
        sys = args[0]
    elif len(args) == 2:
        sys = tf(*args)
    else:
        raise ValueError(
            "Provide a TransferFunction or numerators and denominators."
        )
    p, m = sys.noutputs, sys.ninputs

    blocks = []  # (input j, A_k, B_k, C_k) per distinct denominator of column j
    D = np.zeros((p, m))
    for j in range(m):
        distinct: list[NDArray] = []
        for i in range(p):
            den = _trim(sys.den[i][j])
            den = den / den[0]
            k = next(
                (
                    q
                    for q, e in enumerate(distinct)
                    if e.size == den.size and np.array_equal(e, den)
                ),
                None,
            )
            if k is None:
                distinct.append(den)
            _, _, _, D[i, j] = _companion(sys.num[i][j], sys.den[i][j])
        for den in distinct:
            if den.size == 1:
                continue  # static gain: D only
            A_k, B_k, _, _ = _companion(np.ones(1), den)
            C_k = np.zeros((p, den.size - 1))
            for i in range(p):
                den_i = _trim(sys.den[i][j])
                if den_i.size == den.size and np.array_equal(
                    den_i / den_i[0], den
                ):
                    C_k[i] = _companion(sys.num[i][j], sys.den[i][j])[2][0]
            blocks.append((j, A_k, B_k, C_k))

    n = sum(A_k.shape[0] for _, A_k, _, _ in blocks)
    A, B, C = np.zeros((n, n)), np.zeros((n, m)), np.zeros((p, n))
    k = 0
    for j, A_k, B_k, C_k in blocks:
        r = A_k.shape[0]
        A[k : k + r, k : k + r] = A_k
        B[k : k + r, j : j + 1] = B_k
        C[:, k : k + r] = C_k
        k += r
    if minreal:
        return (*minimal_realization(A, B, C), D)
    return A, B, C, D
