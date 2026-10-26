"""Minimal realisation by the orthogonal staircase (Kalman decomposition).

Pure numpy/scipy counterpart of SLICOT TB01PD: remove the uncontrollable part,
then the unobservable part, with SVD rank decisions (Van Dooren 1981, IEEE TAC
26:111; Varga 1981). Never forms the Krylov matrix, which is unreliable (Paige
1981). Rank decisions use singular values relative to the norm of the data, so
they do not depend on the overall scale of the system.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import matrix_balance, svd

EPS = np.finfo(float).eps


def _nearest_pow2(x: NDArray) -> NDArray:
    """Nearest power of 2 of each positive entry (1 for zeros): exact scaling."""
    safe = np.where(x > 0, x, 1.0)
    return np.where(x > 0, 2.0 ** np.round(np.log2(safe)), 1.0)


def _controllable_part(
    A: NDArray, B: NDArray, C: NDArray, tol: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Keep the controllable part: compress B, then each sub-diagonal block of A."""
    A, B, C = A.copy(), B.copy(), C.copy()
    n, nc, prev = A.shape[0], 0, None
    while nc < n:
        block = B[nc:, :] if prev is None else A[nc:, prev[0] : prev[1]]
        if block.size == 0:
            break
        U, s, _ = svd(block)
        r = int(np.sum(s > tol))
        if r == 0:
            break
        # SVD signs are arbitrary: make each compressed row's largest entry
        # positive, so already-minimal canonical forms come back unchanged.
        top = U[:, :r].T @ block
        sign = np.sign(top[np.arange(r), np.argmax(np.abs(top), axis=1)])
        U[:, :r] *= np.where(sign == 0, 1.0, sign)
        A[nc:, :] = U.T @ A[nc:, :]
        A[:, nc:] = A[:, nc:] @ U
        B[nc:, :] = U.T @ B[nc:, :]
        C[:, nc:] = C[:, nc:] @ U
        prev = (nc, nc + r)
        nc += r
    return A[:nc, :nc], B[:nc, :], C[:, :nc]


def minimal_realization(
    A: NDArray, B: NDArray, C: NDArray, tol: float | None = None
) -> tuple[NDArray, NDArray, NDArray]:
    """Remove uncontrollable and unobservable states (D is unchanged).

    Parameters:
        A, B, C: State-space matrices.
        tol: Absolute singular-value threshold for the rank decisions. Default
            ``n**2 * eps * max(||A||, ||B||, ||C||)`` after balancing, the SLICOT
            default: the result is exactly minimal for a realisation perturbed
            by about ``tol``. Larger values merge nearby poles and cancel near
            pole-zero pairs.

    Returns:
        A, B, C of the minimal realisation.

    Examples:
        >>> A = np.diag([-1.0, -1.0])        # 1/(s+1) and 2/(s+1) as two states
        >>> B = np.eye(2)
        >>> C = np.array([[1.0, 2.0]])
        >>> Am, Bm, Cm = minimal_realization(A, B, C)
        >>> Am.shape
        (1, 1)
        >>> float(Am[0, 0])
        -1.0
    """
    n = A.shape[0]
    if n == 0:
        return A, B, C
    # Parlett-Reinsch balancing and per-channel scaling, both by powers of 2:
    # exact in floating point, and they do not change which states are minimal.
    A, T = matrix_balance(A, permute=True)
    B, C = np.linalg.solve(T, B), C @ T
    si = _nearest_pow2(np.linalg.norm(B, axis=0))
    so = _nearest_pow2(np.linalg.norm(C, axis=1))
    B, C = B / si, C / so[:, None]
    if tol is None:
        scale = max(
            float(np.linalg.norm(A, 1)),
            float(np.linalg.norm(B, 1)),
            float(np.linalg.norm(C, np.inf)),
        )
        tol = n * n * EPS * scale
    A, B, C = _controllable_part(A, B, C, tol)
    At, Ct, Bt = _controllable_part(A.T, C.T, B.T, tol)  # observability = dual
    return At.T, Bt.T * si, Ct.T * so[:, None]
