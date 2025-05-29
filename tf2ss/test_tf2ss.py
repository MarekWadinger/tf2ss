"""Tests of tf2ss on various sample systems and consistency with SLYCOT."""

import control as ctrl
import numpy as np
import pytest
from control.exception import slycot_check
from scipy import signal

from tf2ss import _get_lcm_norm_coeffs, tf2ss

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


def compare_poles(poles_1, poles_2):
    # Check if all poles are stable (real part < 0)
    assert np.all(poles_1 <= 0), "Unstable poles in 1"
    assert np.all(poles_2 <= 0), "Unstable poles in 2"

    # Sort poles by their real parts (most dominant first)
    poles_1 = sorted(poles_1, key=lambda x: -np.real(x))
    poles_2 = sorted(poles_2, key=lambda x: -np.real(x))

    # Find the number of dominant poles to compare
    min_length = min(len(poles_1), len(poles_2))

    # Log information about the different dimensions if they exist
    if len(poles_1) != len(poles_2):
        print(
            f"Note: Different number of poles detected - A_1: {len(poles_1)}, A_2: {len(poles_2)}"
        )

    # Compare the dominant poles
    for i in range(min_length):
        assert np.isclose(poles_1[i], poles_2[i], rtol=1e-5, atol=1e-5), (
            f"Dominant pole {i} differs: {poles_1[i]} vs {poles_2[i]}"
        )


def compare_systems(sys_1: signal.StateSpace, sys_2: signal.StateSpace):
    # Additional check: compare system responses
    # This ensures that even with different numbers of poles, the systems behave similarly
    t = np.linspace(0, 10, 1000)  # Time vector for simulation

    # Generate step responses
    # TODO: fix error - ValueError: System does not define that many inputs.
    _, y_1 = signal.step(sys_1, T=t)
    _, y_2 = signal.step(sys_2, T=t)

    # Compare step responses
    assert np.allclose(y_1, y_2, rtol=1e-4, atol=1e-4), (
        "Step responses differ between original and minimal"
    )

    assert np.allclose(sys_1.A, sys_2.A, atol=1e-6), "Mismatch in A matrix"
    assert np.allclose(sys_1.B, sys_2.B, atol=1e-6), "Mismatch in B matrix"
    assert np.allclose(sys_1.C, sys_2.C, atol=1e-6), "Mismatch in C matrix"
    assert np.allclose(sys_1.D, sys_2.D, atol=1e-6), "Mismatch in D matrix"


@pytest.mark.parametrize("num, den", systems)
def test_vs_slycot(num, den):
    if not slycot_check():
        pytest.skip("Slycot not available, skipping test")
    A, B, C, D = tf2ss(num, den, minreal=False)

    # Convert to control.TransferFunction format
    n_outputs = len(num)
    n_inputs = len(num[0])
    tf_mimo = [
        [ctrl.TransferFunction(num[i][j], den[i][j]) for j in range(n_inputs)]
        for i in range(n_outputs)
    ]

    # Convert using control.tf2ss (uses slycot if available)
    A_slycot, B_slycot, C_slycot, D_slycot = ctrl.ssdata(
        ctrl.append(*[ctrl.append(*row) for row in tf_mimo])
    )
    poles = np.linalg.eigvals(A)
    poles_slycot = np.linalg.eigvals(A_slycot)
    import control

    sys1 = control.StateSpace(A, B, C, D)
    sys2 = control.StateSpace(A_slycot, B_slycot, C_slycot, D_slycot)

    # Get transmission zeros for both systems
    zeros1 = control.zeros(sys1)
    zeros2 = control.zeros(sys2)

    # Sort zeros for consistent comparison
    zeros1 = np.sort_complex(zeros1)
    zeros2 = np.sort_complex(zeros2)
    print("zero found by us", zeros1)
    print("zeros found by slycot", zeros2)

    # Ensure same number of zeros
    assert len(zeros1) == len(zeros2), (
        f"Number of zeros differs: {len(zeros1)} vs {len(zeros2)}"
    )

    # Compare each zero
    for i in range(len(zeros1)):
        assert np.isclose(zeros1[i], zeros2[i], rtol=1e-5, atol=1e-5), (
            f"Zero {i} differs: {zeros1[i]} vs {zeros2[i]}"
        )

    print("poles found by us", poles)
    print("poles found by slycot", poles_slycot)
    compare_poles(poles, poles_slycot)

    # compare_systems(
    #     signal.StateSpace(A, B, C, D),
    #     signal.StateSpace(A_slycot, B_slycot, C_slycot, D_slycot),
    # )


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
