"""Numerical (floating-point) MIMO transfer function to state-space conversion.

Counterpart of :func:`tf2ss.base.tf2ss` that does not use sympy. Every input
column gets its own controller-canonical block, built from that column's
common denominator. The common denominators come from python-control's
``TransferFunction._common_den``, which is the same pole-matching step that
python-control runs before SLICOT ``td04ad('C')``.

The realisation is controllable by construction. It is not necessarily
observable, so it can have more states than the McMillan degree. For
example, ``[[1/(s+1), 2/(s+1)]]`` gets one state per input.
"""

import numpy as np
from control import TransferFunction, tf
from numpy.typing import NDArray
from scipy.linalg import block_diag


def tf2ss(
    *args: TransferFunction | list[list[list[int | float]]],
    minreal: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Convert a MIMO transfer function to state space numerically.

    Parameters:
        *args: Either a single TransferFunction, or two nested lists
            ``numerators[i][j]`` and ``denominators[i][j]`` that hold the
            coefficients of output ``i`` / input ``j`` in descending powers.
        minreal: Cancel pole-zero pairs per entry before the conversion.

    Returns:
        A, B, C, D: One controller-canonical block per input. The block for
        input ``j`` is ordered like ``scipy.signal.tf2ss`` orders it.

    Examples:
        >>> A, B, C, D = tf2ss([[[1]]], [[[1, 2]]])
        >>> A, B, C, D
        (array([[-2.]]), array([[1.]]), array([[1.]]), array([[0.]]))

        >>> A, B, C, D = tf2ss([[[1], [2]]], [[[1, 1], [1, 1]]])
        >>> A
        array([[-1.,  0.],
               [ 0., -1.]])
        >>> C
        array([[1., 2.]])
    """
    if len(args) == 1 and isinstance(args[0], TransferFunction):
        sys = args[0]
    elif len(args) == 2:
        sys = tf(*args)
    else:
        raise ValueError(
            "Provide a TransferFunction or numerators and denominators."
        )
    if minreal:
        sys = sys.minreal(tol=1e-8)

    # num has shape (k, k, kd), where k = max(noutputs, ninputs). den has
    # shape (ninputs, kd). For input j, the first denorder[j] + 1
    # coefficients are aligned and in descending powers of s.
    num, den, denorder = sys._common_den()
    n_out, n_in = sys.noutputs, sys.ninputs

    A_blocks, B_blocks, C_blocks = [], [], []
    D = np.zeros((n_out, n_in))
    for j in range(n_in):
        n = int(denorder[j])
        a = den[j, : n + 1]
        b = num[:n_out, j, : n + 1] / a[0]
        a = a / a[0]
        A_j = np.eye(n, k=-1)
        A_j[:1, :] = -a[1:]
        B_j = np.zeros((n, 1))
        B_j[:1, 0] = 1.0
        A_blocks.append(A_j)
        B_blocks.append(B_j)
        C_blocks.append(b[:, 1:] - np.outer(b[:, 0], a[1:]))
        D[:, j] = b[:, 0]

    A = block_diag(*A_blocks)
    B = block_diag(*B_blocks)
    C = np.hstack(C_blocks)
    return A, B, C, D
