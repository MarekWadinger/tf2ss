"""Helper functions for tf2ss conversion."""

from math import gcd
from typing import Any, Literal, overload

import numpy as np
import sympy as sp
from control import TransferFunction, tf
from numpy.typing import NDArray
from scipy.signal import tf2ss as tf2ss_siso

from .minreal import minimal_realization


def _pad_numerators(
    numerators: list[list[list[int | float]]],
) -> list[list[list[int | float]]]:
    """Pad the numerator matrix of polynomials with zeros to have all equal length, padding from the left.

    Parameters:
        numerators: List of lists of numerators for each (output, input) transfer function.

    Returns:
        Padded numerator list with all numerators having the same length.

    Examples:
        >>> _pad_numerators([[[1], [1, 2]], [[1, 2, 3], [1]]])
        [[[0, 0, 1], [0, 1, 2]], [[1, 2, 3], [0, 0, 1]]]
    """
    max_len = max(len(num) for row in numerators for num in row)
    pad_with = 0 if isinstance(numerators[0][0][0], int) else 0.0
    padded_numerators = [
        [[pad_with] * (max_len - len(num)) + num for num in row]
        for row in numerators
    ]
    return padded_numerators


def list_to_poly(coefs: list[Any], s: sp.Symbol = sp.Symbol("s")) -> sp.Poly:
    """Convert a list of coefficients (in descending order) into a sympy Poly.

    Parameters:
      coefs: List of coefficients (descending order).
             For example, [1, 3, 2] represents 1*s**2 + 3*s + 2.
      s: sympy symbol (default is s).

    Returns:
      A sympy Poly object.

    Examples:
        >>> p = list_to_poly([1, 3, 2])
        >>> sp.expand(p.as_expr())
        s**2 + 3*s + 2
        >>> sp.expand(p.as_expr()) == sp.expand(sp.Symbol('s')**2 + 3*sp.Symbol('s') + 2)
        True
    """
    poly_expr = sum(
        coef * s ** (len(coefs) - i - 1) for i, coef in enumerate(coefs)
    )
    return sp.Poly(poly_expr, s)


def compute_lcd_from_denominators(
    denominators: list[list[list[float]]], s: sp.Symbol = sp.Symbol("s")
) -> sp.Poly:
    """Compute the least common denominator (LCD) of a MIMO system's denominators.

    Parameters:
      denominators: A list of lists of denominators. Each denominator is a list of coefficients
                in descending order.
      s: sympy symbol (default is s).

    Returns:
      A sympy Poly representing the LCD.

    Examples:
        >>> lcd = compute_lcd_from_denominators([[[1, 2]], [[1, 2]]])
        >>> sp.expand(lcd.as_expr())
        s + 2
    """
    # Start with the first denominator in the list
    first_poly = list_to_poly(denominators[0][0], s)
    lcd = first_poly
    for row in denominators:
        for den in row:
            poly = list_to_poly(den, s)
            lcd = sp.lcm(lcd, poly)
    return lcd


def compute_adjusted_num(
    numerator: list[Any],
    lcd: sp.Poly,
    denominator: list[Any],
    s: sp.Symbol = sp.Symbol("s"),
) -> list[Any]:
    """Compute the adjusted numerator polynomial coefficients given the numerator and denominator of a transfer function and the common LCD.

    This function multiplies the original numerator by the LCD and divides by the original
    denominator. The resulting quotient (assumed to be exact) gives the adjusted numerator.

    Parameters:
      numerator: List of numerator coefficients (in descending order).
      lcd: A sympy Poly representing the least common denominator.
      denominator: List of denominator coefficients (in descending order).
      s: sympy symbol (default is s).

    Returns:
      A numpy array of adjusted numerator coefficients (in descending order).

    Examples:
        >>> compute_adjusted_num([1, 1], list_to_poly([1, 3, 2]), [1, 3, 2])
        [1, 1]
    """
    num_poly = list_to_poly(numerator, s)
    den_poly = list_to_poly(denominator, s)
    # Multiply numerator polynomial by LCD
    new_expr = sp.expand(num_poly.as_expr() * lcd.as_expr())
    new_poly = sp.Poly(new_expr, s)
    quotient, remainder = sp.div(new_poly, den_poly)
    if remainder.as_expr() != 0:
        raise ValueError("Adjusted numerator division has non-zero remainder")
    return quotient.all_coeffs()


def transpose(matrix: list[list[Any]]) -> list[list[Any]]:
    """Transpose a list of lists (matrix).

    Parameters:
        matrix: List of lists to be transposed.

    Returns:
        Transposed list of lists.

    Examples:
        >>> transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
    """
    return [list(row) for row in zip(*matrix)]


def state_space_from_poly(poly: sp.Poly) -> tuple[NDArray, NDArray]:
    """Compute the state-space representation (A, B) from the denominator polynomial using tf2ss.

    Parameters:
      poly: A sympy Poly representing the denominator.

    Returns:
      A, B: Matrices from the tf2ss representation of the system with transfer function 1/poly.

    Examples:
        >>> A, B = state_space_from_poly(list_to_poly([1, 2]))
        >>> A.shape
        (1, 1)
        >>> B.shape
        (1, 1)
    """
    lcd_coeffs = np.array(poly.all_coeffs(), dtype=np.float64)
    A, B, _, _ = tf2ss_siso([1], lcd_coeffs)
    return A, B


@overload
def _get_lcm_norm_coeffs(
    denominators: list[list[list[float]]],
    mode: Literal["global"],
) -> list[float]: ...


@overload
def _get_lcm_norm_coeffs(
    denominators: list[list[list[float]]],
    mode: Literal["local"],
) -> list[list[float]]: ...


def _get_lcm_norm_coeffs(
    denominators: list[list[list[float]]],
    mode: Literal["global", "local"] = "global",
) -> list[float] | list[list[float]]:
    """Compute the least common multiple (LCM) of a list of floating-point polynomials.

    Parameters:
      denominators: A list of lists of lists of floating-point coefficients representing the denominators.
      mode: A string indicating the mode of LCM computation. Can be "global" or "local".
            "global" computes a single LCM for all denominators.
            "local" computes LCMs for each input of denominators.

    Returns:
      If mode is "global", returns a list of floating-point coefficients representing the LCM.
      If mode is "local", returns a list of lists of floating-point coefficients representing the LCMs for each column.

    Examples:
        >>> denominators = [[[1.0, 2.0], [1.0, 1.0]], [[1.0, 2.0], [1.0, 3.0]]]
        >>> _get_lcm_norm_coeffs(denominators, mode="global")
        [1.0, 6.0, 11.0, 6.0]

        >>> denominators = [[[1.0, 2.0], [1.0, 1.0]], [[1.0, 2.0], [1.0, 3.0]]]
        >>> _get_lcm_norm_coeffs(denominators, mode="local")
        [[1.0, 2.0, 0.0], [1.0, 4.0, 3.0]]
    """
    if mode == "local":
        # TransferFunction.common_den() right-pads the denominators with zeros
        normalized_coeffs_list = [
            _get_lcm_norm_coeffs([col], "global")
            for col in transpose(denominators)
        ]
        max_len = max(len(coeffs) for coeffs in normalized_coeffs_list)
        return [
            coeffs + [0.0] * (max_len - len(coeffs))
            for coeffs in normalized_coeffs_list
        ]
    else:
        lcm_poly = compute_lcd_from_denominators(denominators)
        # Extract coefficients as floats
        lcd_coeffs = [float(c) for c in lcm_poly.all_coeffs()]

        # Normalize by the greatest common divisor of integer coefficients
        coeff_gcd = gcd(*(int(c) for c in lcd_coeffs if c != 0))
        normalized_coeffs = [c / coeff_gcd for c in lcd_coeffs]
    return normalized_coeffs


def rjust(list_: list[int | float], width) -> list[int | float]:
    """Right-justify a list by padding with zeros or truncating to specified width.

    Parameters:
        list_: List of numeric values to justify
        width: Target width of the resulting list

    Returns:
        Right-justified list, either padded with zeros or truncated to width

    Examples:
        >>> rjust([1, 2, 3], 4)
        [1, 2, 3, 0]
        >>> rjust([1, 2, 3, 4, 5], 4)
        [1, 2, 3, 4]
    """
    pad_with = 0 if isinstance(list_[0], int) else 0.0

    return list(list_[:width]) + [pad_with] * max(width - len(list_), 0)


def controllable_canonical_form(
    denominator: list[float] | sp.Poly,
) -> tuple[NDArray, NDArray]:
    """Compute the controllable canonical form (A, B) matrices for a given common denominator polynomial.

    Parameters:
        denominator (list): Coefficients of the denominator polynomial [a_n, ..., a_1, a_0],
                            where the highest-degree term is first.

    Returns:
        tuple: (A, B) state-space representation in controllable canonical form.

    Examples:
        >>> controllable_canonical_form([1, 2, 3])
        (array([[ 0.,  1.],
            [-2., -3.]]), array([[0.],
            [1.]]))

    """
    if isinstance(denominator, np.ndarray):
        denominator = list(denominator)
    elif isinstance(denominator, sp.Poly):
        denominator = list(
            np.array(denominator.all_coeffs(), dtype=np.float64)
        )

    n = len(denominator) - 1  # System order
    if n < 1:
        return np.array([0.0]), np.array([0.0])
    # Construct the A matrix in controllable canonical form
    A = np.zeros((n, n))
    A[:-1, 1:] = np.eye(n - 1)  # Upper diagonal ones
    A[-1, :] = -np.array(
        denominator[1:]
    )  # Last row is negative denominator coefficients

    # Construct the B matrix (last column is 1)
    B = np.zeros((n, 1))
    B[-1, 0] = 1

    return A, B


# Underscores required for positional only arguments (https://github.com/python/mypy/issues/6187)
@overload
def tf2ss(
    __sys: TransferFunction,
    *,
    minreal: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]: ...


@overload
def tf2ss(
    __numerators: list[list[list[int | float]]],
    __denominators: list[list[list[int | float]]],
    *,
    minreal: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]: ...


def tf2ss(
    *args: TransferFunction | list[list[list[int | float]]],
    minreal: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Convert a MIMO transfer function to a minimal state-space representation.

    Parameters:
      *args: Transfer function specification, which can be: (1) A single TransferFunction object, or (2) Two lists containing numerators and denominators, where each list contains coefficient sequences in descending degree order. The denominator polynomials must be at least as long as their corresponding numerator polynomials.
      minreal: Cancel pole-zero pairs exactly per entry, then remove uncontrollable and
               unobservable states with the orthogonal staircase (default is True).

    Returns:
      A, B, C, D: Minimal state space representation of the system (one controller-canonical
                  block per input when minreal=False).

    Examples:
        >>> sys = TransferFunction([1], [1, 2])
        >>> A, B, C, D = tf2ss(sys)
        >>> A, B, C, D
        (array([[-2.]]), array([[1.]]), array([[1.]]), array([[0.]]))

        >>> numerators = [[[1]], [[1]]]
        >>> denominators = [[[1, 2]], [[1, 2]]]
        >>> A, B, C, D = tf2ss(numerators, denominators)
        >>> A, B, C, D
        (array([[-2.]]), array([[1.]]), array([[1.],
           [1.]]), array([[0.],
           [0.]]))
    """
    if len(args) == 1:
        sys = args[0]
        if not isinstance(sys, TransferFunction):
            raise ValueError(
                "Single argument must be a TransferFunction object."
            )
    elif len(args) == 2:
        numerators, denominators = args
        if not (
            isinstance(numerators, list) and isinstance(denominators, list)
        ):
            raise ValueError(
                "Two arguments must be lists of numerators and denominators."
            )
        sys = tf(numerators, denominators)
    else:
        raise ValueError(
            "Invalid number of arguments. Provide either a TransferFunction or numerators and denominators."
        )

    if sys is None:
        raise ValueError("Invalid transfer function")

    n_out, n_in = sys.noutputs, sys.ninputs
    s = sp.Symbol("s")
    # Exact arithmetic on the user's own coefficients. Floats are rounded to
    # 14 decimals first; that is the highest precision that survives the
    # float -> decimal string -> Rational round trip.
    num = [
        [_exact_poly(sys.num[i][j], s) for j in range(n_in)]
        for i in range(n_out)
    ]
    den = [
        [_exact_poly(sys.den[i][j], s) for j in range(n_in)]
        for i in range(n_out)
    ]
    if minreal:  # exact pole-zero cancellation per entry
        for i in range(n_out):
            for j in range(n_in):
                if not num[i][j].is_zero:
                    g = sp.gcd(num[i][j], den[i][j])
                    num[i][j], den[i][j] = (
                        sp.div(num[i][j], g)[0],
                        sp.div(den[i][j], g)[0],
                    )

    # One controller-canonical block per input j, built on the exact LCM of
    # column j's denominators (same block layout as scipy.signal.tf2ss).
    blocks = []
    D = np.zeros((n_out, n_in))
    for j in range(n_in):
        lcd = sp.Poly(1, s)
        for i in range(n_out):
            lcd = sp.lcm(lcd, den[i][j])
        lcd = lcd.monic()
        n = lcd.degree()
        a = lcd.all_coeffs()  # a[0] == 1
        b = np.zeros((n_out, n + 1))
        for i in range(n_out):
            q, r = sp.div(num[i][j] * lcd, den[i][j])
            if not r.is_zero:
                raise ValueError(
                    "Adjusted numerator division has non-zero remainder"
                )
            coeffs = q.all_coeffs()
            if len(coeffs) > n + 1:
                raise ValueError(f"Transfer function ({i}, {j}) is improper")
            b[i, n + 1 - len(coeffs) :] = [float(c) for c in coeffs]
        a_f = np.array([float(c) for c in a])
        A_j = np.eye(n, k=-1)
        A_j[:1, :] = -a_f[1:]
        B_j = np.zeros((n, 1))
        B_j[:1, 0] = 1.0
        blocks.append((A_j, B_j, b[:, 1:] - np.outer(b[:, 0], a_f[1:])))
        D[:, j] = b[:, 0]

    n_states = sum(A_j.shape[0] for A_j, _, _ in blocks)
    A = np.zeros((n_states, n_states))
    B = np.zeros((n_states, n_in))
    C = np.zeros((n_out, n_states))
    k = 0
    for j, (A_j, B_j, C_j) in enumerate(blocks):
        n = A_j.shape[0]
        A[k : k + n, k : k + n] = A_j
        B[k : k + n, j : j + 1] = B_j
        C[:, k : k + n] = C_j
        k += n
    if minreal:
        A, B, C = minimal_realization(A, B, C)
    return A, B, C, D


def _exact_poly(coefs: Any, s: sp.Symbol) -> sp.Poly:
    """Polynomial with exact rational coefficients (floats rounded to 14 decimals).

    Examples:
        >>> _exact_poly([0.5, 1], sp.Symbol("s"))
        Poly(1/2*s + 1, s, domain='QQ')
    """
    return sp.Poly(
        [sp.Rational(str(round(float(c), 14))) for c in np.atleast_1d(coefs)],
        s,
        domain="QQ",
    )
