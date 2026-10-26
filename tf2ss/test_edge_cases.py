"""Edge cases that show where the symbolic and numeric tf2ss paths differ.

Each case checks the round-trip error at points chosen for the case and the
number of states. Known weaknesses are strict xfails, so the suite documents
them and fails loudly if a path changes behaviour.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tf2ss import base, tf2ss_numeric

AWAY = (
    0.37 + 1.1j,
    -0.71 + 2.3j,
    1.9 - 0.4j,
    3.3 + 3.0j,
)  # |s| ~ 1, away from poles
P = lambda *roots: list(np.poly(roots))  # noqa: E731

METHODS: dict[str, Callable[..., Any]] = {
    "symbolic": base.tf2ss,
    "numeric": tf2ss_numeric.tf2ss,
}


def roundtrip_error(num, den, A, B, C, D, points):
    """Largest relative error of C (sI-A)^-1 B + D against num/den."""
    worst = 0.0
    for s in points:
        G = np.array(
            [
                [np.polyval(n, s) / np.polyval(d, s) for n, d in zip(nr, dr)]
                for nr, dr in zip(num, den)
            ]
        )
        H = C @ np.linalg.solve(s * np.eye(A.shape[0]) - A, B) + D
        worst = max(worst, np.max(np.abs(H - G)) / np.max(np.abs(G)))
    return worst


# (id, num, den, points, minreal, max error, expected states per method, marks per method)
CASES = [
    (
        "exact repeated poles (s+1)^4, (s+1)^5 in one column",
        [[[1]], [[1]]],
        [[P(*[-1] * 4)], [P(*[-1] * 5)]],
        AWAY,
        True,
        1e-12,
        {"symbolic": 5, "numeric": 5},
        {},
    ),
    (
        "float repeated poles (s+1/3)^4, (s+1/3)^5",
        [[[1]], [[1]]],
        [[P(*[-1 / 3] * 4)], [P(*[-1 / 3] * 5)]],
        AWAY,
        True,
        1e-12,
        # Neither exact LCM nor root matching merges float-inconsistent
        # repeated poles (9 states); the staircase rank decision does.
        {"symbolic": 5, "numeric": 5},
        {},
    ),
    (
        "poles 1e-12 apart in one column",
        [[[1]], [[1]]],
        [[P(-1)], [P(-1 - 1e-12)]],
        AWAY,
        True,
        1e-10,
        # 1e-12 is above the default n^2*eps rank tolerance: both states kept.
        {"symbolic": 2, "numeric": 2},
        {},
    ),
    (
        "near pole-zero cancellation (s+1+1e-10)/((s+1)(s+2))",
        [[P(-1 - 1e-10)]],
        [[P(-1, -2)]],
        AWAY,
        True,
        1e-9,
        # Not cancelled at the default n^2*eps rank tolerance.
        {"symbolic": 2, "numeric": 2},
        {},
    ),
    (
        "slow pole 1e-15, evaluated near DC",
        [[[1]]],
        [[P(-1e-15)]],
        (1e-14 + 0j,),
        True,
        1e-9,
        # Relative rationalisation keeps the pole (absolute rounding made it an
        # integrator, 10% off at s=1e-14).
        {"symbolic": 1, "numeric": 1},
        {},
    ),
    (
        "tiny gain 1e-16/(s+1)",
        [[[1e-16]]],
        [[P(-1)]],
        AWAY,
        False,
        1e-9,
        {"symbolic": 1, "numeric": 1},
        {},
    ),
    (
        "non-terminating decimals (1/3, 0.1)",
        [[[1 / 3]]],
        [[[1, 0.1, 1 / 3]]],
        AWAY,
        True,
        1e-15,
        # 0.333... is recovered as exactly 1/3.
        {"symbolic": 2, "numeric": 2},
        {},
    ),
    (
        "stiff poles 1e-6 and 1e6",
        [[[1]]],
        [[P(-1e-6, -1e6)]],
        AWAY,
        True,
        1e-12,
        {"symbolic": 2, "numeric": 2},
        {},
    ),
    (
        "lightly damped, zeta 1e-4",
        [[[1]]],
        [[[1, 2e-4, 1]]],
        AWAY,
        True,
        1e-12,
        {"symbolic": 2, "numeric": 2},
        {},
    ),
    (
        "MISO, same pole on both inputs",
        [[[1], [2]]],
        [[P(-1), P(-1)]],
        AWAY,
        True,
        1e-12,
        # One block per input has 2 states; the staircase removes the
        # unobservable one (McMillan degree 1).
        {"symbolic": 1, "numeric": 1},
        {},
    ),
    (
        "redundant along rows and columns [[1, 2], [3, 6]]/(s+1)",
        [[[1], [2]], [[3], [6]]],
        [[P(-1), P(-1)], [P(-1), P(-1)]],
        AWAY,
        True,
        1e-12,
        # Rank-1 residue: per-input and per-output blocks both give 2 states.
        {"symbolic": 1, "numeric": 1},
        {},
    ),
    (
        "SIMO, same pole on both outputs",
        [[[1]], [[2]]],
        [[P(-1)], [P(-1)]],
        AWAY,
        True,
        1e-12,
        {"symbolic": 1, "numeric": 1},
        {},
    ),
]


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c[0]) for c in CASES],
)
def test_edge_case(case, method, request):
    name, num, den, points, minreal, tol, states, marks = case
    if method in marks:
        request.applymarker(marks[method])
    A, B, C, D = METHODS[method](num, den, minreal=minreal)
    assert roundtrip_error(num, den, A, B, C, D, points) <= tol
    assert A.shape[0] == states[method]


@pytest.mark.parametrize("method", list(METHODS))
def test_tol_merges_nearby_poles(method):
    """A larger tol trades exactness for order: poles 1e-12 apart become one."""
    num, den = [[[1]], [[1]]], [[P(-1)], [P(-1 - 1e-12)]]
    assert METHODS[method](num, den)[0].shape[0] == 2
    A, B, C, D = METHODS[method](num, den, tol=1e-8)
    assert A.shape[0] == 1
    assert roundtrip_error(num, den, A, B, C, D, AWAY) <= 1e-10
