"""Helper functions for tf2ss conversion."""

from math import gcd
from typing import Any, Literal, overload

import numpy as np
import sympy as sp
from control import TransferFunction, tf
from numpy.typing import NDArray
from scipy.signal import tf2ss as tf2ss_siso


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
    # type_ = type(numerators[0][0][0])
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
      minreal: Whether to perform minimal realization (default is True).

    Returns:
      A, B, C, D: Minimal state space representation of the system, in controller canonical form.

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

    if minreal:
        sys = sys.minreal(tol=1e-8)

    # TODO: Apparently, the common_den() method changes the number of outputs of the system (tested for case where noutputs < ninputs)
    numerators_, denominators_, _ = sys._common_den()

    denominators_ = np.expand_dims(denominators_, axis=0)
    denominators_ = np.tile(denominators_, (numerators_.shape[0], 1, 1))

    # Remove the last row of numerators if all elements are zero
    # Seems to be a bug in control.common_den() that adds an extra row
    def trim_zeros_along_axis(arr, axis, trim="b"):
        """Trip zeros along the given axis, preserving at last one array."""
        slices = [slice(None)] * arr.ndim
        for idx in range(arr.shape[axis]):
            # Type annotation to fix the error in line 361
            # Use explicit typing for the list index
            slices[axis] = (
                slice(idx, idx + 1) if trim == "f" else slice(-idx - 1, -idx)
            )
            if not np.all(arr[tuple(slices)] == 0):
                break
        start = idx if trim == "f" else 0
        end = arr.shape[axis] - idx if trim == "b" else arr.shape[axis]
        slices[axis] = slice(start, end)
        return arr[tuple(slices)]

    numerators_ = trim_zeros_along_axis(numerators_, axis=1, trim="b")
    numerators_ = np.vectorize(lambda x: sp.Rational(str(x)))(numerators_)
    denominators_ = np.vectorize(lambda x: sp.Rational(str(x)))(denominators_)
    n_outputs = len(numerators_)
    n_inputs = len(numerators_[0])
    s = sp.Symbol("s")
    # Step 1: Compute the LCD of all denominators.
    lcd = compute_lcd_from_denominators(denominators_, s)
    # Step 2: Get state-space representation (A, B) from the LCD.
    # A, B_scalar = controllable_canonical_form(lcd)
    A, B_scalar = state_space_from_poly(lcd)
    n_states = A.shape[0]
    C = np.zeros((n_outputs, n_states))
    D = np.zeros((n_outputs, n_inputs))
    # Step 3: For each transfer function, compute the adjusted numerator and fill C and D.
    for i_out in range(n_outputs):
        for j_in in range(n_inputs):
            den = (
                denominators_[i_out][j_in]
                if denominators_.ndim == 3
                else denominators_[j_in]
            )
            adjusted_num_coeffs = compute_adjusted_num(
                numerators_[i_out][j_in], lcd, den, s
            )
            # In the controllable canonical (companion) form, the output matrix uses the reversed order.
            C[i_out, :] = rjust(adjusted_num_coeffs[::-1], n_states)
            # If the degree is less than n_states, assign the constant term to D.
            if len(adjusted_num_coeffs) < n_states:
                D[i_out, j_in] = adjusted_num_coeffs[-1]
            else:
                D[i_out, j_in] = 0
    B = np.tile(B_scalar, (1, n_inputs))
    return A, B, C, D
