"""Tests of tf2ss on various sample systems and consistency with SLYCOT."""

import control as ctrl
import numpy as np
import pytest
from control.exception import slycot_check

from tf2ss import _get_lcm_norm_coeffs, tf2ss
from tf2ss.tf2ss_numeric import tf2ss as tf2ss_numeric

systems = [
    ([[[0.5]]], [[[32]]]),
    ([[[1]]], [[[1, 2]]]),  # H(s) = 1 / (s+2)
    ([[[1, 1]]], [[[1, 3, 2]]]),  # H(s) = (s+1) / (s^2 + 3s + 2)
    (
        [[[1], [0.5]], [[0], [1]]],
        [[[1, 2], [1, 2]], [[0, 1], [1, 2]]],
    ),  # MIMO case - different denominators
    (
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[3.0, -1.0, 1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]],
        ],
        [
            [[1.0, 0.4, 3.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.4, 3.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.4, 3.0], [1.0, 1.0, 0.0]],
        ],
    ),  # MIMO case - different denominators
    (
        [
            [[1.0, 6.0, 12.0, 7.0], [0.0, 1.0, 4.0, 3.0]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 8.0, 20.0, 15.0]],
        ],
        [
            [[1.0, 6.0, 11.0, 6.0], [1.0, 6.0, 11.0, 6.0]],
            [[1.0, 6.0, 11.0, 6.0], [1.0, 6.0, 11.0, 6.0]],
        ],
    ),
]


# Extra cases for the transfer-function round-trip checks.
roundtrip_systems = [
    *systems,
    ([[[1, 3]]], [[[1, 3, 2]]]),  # (s+3)/(s^2+3s+2): non-palindromic num
    ([[[2, 6]]], [[[2, 2]]]),  # (2s+6)/(2s+2): biproper, non-monic den
    ([[[1]], [[1]]], [[[1, 1]], [[1, 2]]]),  # SIMO
    ([[[1], [2]]], [[[1, 1], [1, 1]]]),  # MISO with a shared pole
    ([[[1], [1]]], [[[1, 1], [1, 2]]]),  # MISO with distinct poles
]

# Points in the complex plane where G(s) is compared (away from all poles).
S_POINTS = (0.37 + 1.1j, -0.71 + 2.3j, 1.9 - 0.4j, 0.05 + 0.5j, 3.3 + 3.0j)


def tf_response(num, den, s):
    """Evaluate the transfer matrix given as nested coefficient lists."""
    return np.array(
        [
            [np.polyval(n, s) / np.polyval(d, s) for n, d in zip(nrow, drow)]
            for nrow, drow in zip(num, den)
        ]
    )


def ss_response(A, B, C, D, s):
    """Evaluate C (sI - A)^-1 B + D."""
    n = A.shape[0]
    return C @ np.linalg.solve(s * np.eye(n) - A, B) + D


def assert_same_transfer_function(num, den, A, B, C, D, rtol=1e-8):
    """Assert that (A, B, C, D) realises the transfer matrix num/den."""
    for s in S_POINTS:
        np.testing.assert_allclose(
            ss_response(A, B, C, D, s),
            tf_response(num, den, s),
            rtol=rtol,
            atol=rtol,
        )


# Known failures of the symbolic path in tf2ss.base (v1.2.0): C is filled in
# reversed order for the first-row companion A from scipy, D takes the
# constant numerator term instead of the leading one, and for several inputs
# the C row is overwritten per input while B is tiled.
_SYMBOLIC_KNOWN_BAD = {0, 3, 4, 5, 6, 7, 8, 9, 10}


def perturb(coeffs, rng, eps=1e-8):
    """Perturb nonzero coefficients by eps-scaled noise; zeros stay zero.

    Perturbing a structural zero (e.g. the leading 0 in [0, 1]) would add a
    pole or zero near infinity and change the system's structure.
    """
    a = np.asarray(coeffs, dtype=float)
    return list(a + eps * rng.standard_normal(a.shape) * (a != 0))


@pytest.mark.parametrize("perturbed", [False, True])
@pytest.mark.parametrize(
    "impl, num, den",
    [
        *(
            pytest.param(
                tf2ss,
                *case,
                marks=pytest.mark.xfail(
                    strict=True, reason="tf2ss.base realisation bug"
                ),
            )
            if i in _SYMBOLIC_KNOWN_BAD
            else (tf2ss, *case)
            for i, case in enumerate(systems)
        ),
        *((tf2ss_numeric, *case) for case in systems),
    ],
)
def test_vs_slycot(impl, num, den, perturbed):
    if not slycot_check():
        pytest.skip("Slycot not available, skipping test")
    if perturbed:
        # Near-cancelling pole-zero pairs must not break the conversion.
        rng = np.random.default_rng(0)
        num = [[perturb(n, rng) for n in row] for row in num]
        den = [[perturb(d, rng) for d in row] for row in den]
    # Reference: SLICOT td04ad realisation of the same MIMO system.
    sys_slycot = ctrl.tf2ss(ctrl.tf(num, den))
    assert_same_transfer_function(num, den, *ctrl.ssdata(sys_slycot))

    A, B, C, D = impl(num, den, minreal=False)

    assert A.shape[0] >= sys_slycot.nstates
    assert_same_transfer_function(num, den, A, B, C, D)


@pytest.mark.parametrize("num, den", systems)
def test_tf2ss_runability(num, den):
    A, B, C, D = tf2ss(num, den)


@pytest.mark.parametrize(
    "den",
    [
        [
            [
                [1.0, 0.5],  # (s + 0.5)
                [1.0, 0.5, 0.25],  # (s^2 + 0.5s + 0.25)
            ]
        ],  # [1.0, 1.0, 0.75, 0.125]
        [[[1, 7, 6], [1, -5, -6]]],
        [[[3, -6, -9, 0], [7, 21, 14, 0, 0]]],
        [
            [[3, -6, -9, 0], [1, 0]],
            [[7, 21, 14, 0, 0], [1, 0]],
        ],
        [[[3, -6, -9, 0], [7, 21, 14, 0, 0]], [[1, 7, 6], [1, -5, -6]]],
    ],
)
def test_coeffs_retrieval(den):
    den_our = _get_lcm_norm_coeffs(den, mode="local")
    sys = ctrl.tf(
        [[[1] * len(den) for den in row] for row in den],
        den,
    )
    if isinstance(sys, ctrl.TransferFunction):
        _, den_ctrl, _ = sys._common_den()
        _, den_min, _ = sys.minreal()._common_den()
        assert np.allclose(den_our, den_ctrl, atol=1e-5)
        # TODO: create test evaluating minimum realization when implemented
        # assert np.allclose(den_our, den_min, atol=1e-5)


@pytest.mark.parametrize(
    "num, den",
    [
        pytest.param(
            *case,
            marks=pytest.mark.xfail(
                strict=True, reason="tf2ss.base realisation bug"
            ),
        )
        if i in _SYMBOLIC_KNOWN_BAD
        else case
        for i, case in enumerate(roundtrip_systems)
    ],
)
def test_tf2ss_realisation_reproduces_transfer_function(num, den):
    A, B, C, D = tf2ss(num, den, minreal=False)
    assert_same_transfer_function(num, den, A, B, C, D)


@pytest.mark.parametrize(
    "num, den",
    [
        ([[[1e-9]]], [[[1, 1]]]),  # small gain
        ([[[1]]], [[[1, 1e-9]]]),  # slow pole
    ],
)
def test_tf2ss_small_coefficients_are_not_rounded_away(num, den):
    A, B, C, D = tf2ss(num, den, minreal=False)

    # Relative tolerance only: a lost 1e-9 gain is below any sensible atol.
    for s in S_POINTS:
        np.testing.assert_allclose(
            ss_response(A, B, C, D, s), tf_response(num, den, s), rtol=1e-8
        )
    np.testing.assert_allclose(
        np.linalg.eigvals(A), np.roots(den[0][0]), rtol=1e-6
    )
