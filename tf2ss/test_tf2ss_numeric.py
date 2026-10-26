"""Tests of the numerical tf2ss path against the transfer function and SLYCOT."""

import control as ctrl
import numpy as np
import pytest
from control.exception import slycot_check

from tf2ss.test_tf2ss import assert_same_transfer_function, roundtrip_systems
from tf2ss.tf2ss_numeric import tf2ss


def random_stable_mimo(rng, n_outputs, n_inputs, order):
    """Return random biproper MIMO coefficients with stable, distinct poles."""

    def den():
        k = order // 2
        re = -rng.uniform(0.5, 2.0, k)
        im = rng.uniform(0.1, 1.0, k)
        real = -rng.uniform(0.5, 2.0, order - 2 * k)
        roots = np.r_[re + 1j * im, re - 1j * im, real]
        return list(np.real(np.poly(roots)))

    num = [
        [list(rng.standard_normal(order + 1)) for _ in range(n_inputs)]
        for _ in range(n_outputs)
    ]
    dens = [[den() for _ in range(n_inputs)] for _ in range(n_outputs)]
    return num, dens


@pytest.mark.parametrize("minreal", [True, False])
@pytest.mark.parametrize("num, den", roundtrip_systems)
def test_numeric_tf2ss_sample_systems_reproduce_transfer_function(
    num, den, minreal
):
    A, B, C, D = tf2ss(num, den, minreal=minreal)
    assert_same_transfer_function(num, den, A, B, C, D)


@pytest.mark.parametrize(
    "n_outputs, n_inputs", [(1, 1), (1, 3), (3, 1), (2, 2), (3, 4), (5, 5)]
)
def test_numeric_tf2ss_random_mimo_reproduces_transfer_function(
    n_outputs, n_inputs
):
    rng = np.random.default_rng(n_outputs * 10 + n_inputs)
    num, den = random_stable_mimo(rng, n_outputs, n_inputs, order=4)

    A, B, C, D = tf2ss(num, den)

    assert A.shape == (4 * n_inputs * n_outputs,) * 2
    assert_same_transfer_function(num, den, A, B, C, D)


def test_numeric_tf2ss_transfer_function_object_matches_lists():
    num, den = roundtrip_systems[5]

    from_lists = tf2ss(num, den)
    from_object = tf2ss(ctrl.tf(num, den))

    for m_lists, m_object in zip(from_lists, from_object):
        np.testing.assert_array_equal(m_lists, m_object)


def test_numeric_tf2ss_static_gain_has_no_states():
    A, B, C, D = tf2ss([[[0.5]]], [[[32]]])

    assert A.shape == (0, 0)
    assert B.shape == (0, 1)
    assert C.shape == (1, 0)
    np.testing.assert_allclose(D, [[0.5 / 32]])


def test_numeric_tf2ss_invalid_arguments_raise():
    with pytest.raises(ValueError):
        tf2ss([[[1]]])


@pytest.mark.parametrize("num, den", roundtrip_systems)
def test_numeric_tf2ss_vs_slycot(num, den):
    if not slycot_check():
        pytest.skip("Slycot not available, skipping test")
    sys_slycot = ctrl.tf2ss(ctrl.tf(num, den))

    A, B, C, D = tf2ss(num, den)

    # td04ad returns a minimal realisation. Per-input blocks are
    # controllable but may keep unobservable modes.
    assert A.shape[0] >= sys_slycot.nstates
    assert_same_transfer_function(num, den, *ctrl.ssdata(sys_slycot))
    assert_same_transfer_function(num, den, A, B, C, D)
