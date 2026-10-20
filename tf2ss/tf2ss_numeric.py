"""Helper functions for tf2ss conversion (numerical version).

NOTE: This module is not used in the main tf2ss package. It is kept here for reference and testing purposes. If you are willing to contribute, consider verifying this module.
"""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


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


def poly_gcd(a: list[float], b: list[float], tol: float = 1e-6) -> list[float]:
    """Compute the greatest common divisor (GCD) of two polynomials numerically.

    Parameters:
        a: Coefficients of the first polynomial in descending order.
        b: Coefficients of the second polynomial in descending order.
        tol: Tolerance for considering remainder as zero.

    Returns:
        Coefficients of the GCD polynomial in descending order.

    Examples:
        >>> poly_gcd([1, 2], [1, 2])
        [1.0, 2.0]
    """
    a_: np.ndarray = np.trim_zeros(np.array(a, dtype=float), "b")
    b_: np.ndarray = np.trim_zeros(np.array(b, dtype=float), "b")
    while len(b_) > 0 and not np.allclose(b_, 0.0, atol=tol):
        _, remainder = np.polydiv(a_, b_)
        remainder = np.trim_zeros(remainder, "b")
        a_, b_ = b_, remainder
    if len(a_) == 0 or np.allclose(a_, 0.0, atol=tol):
        return [1.0]
    a_ = a_ / a_[0]  # Make monic
    return a_.tolist()


def poly_lcm(a: list[float], b: list[float], tol: float = 1e-6) -> list[float]:
    """Compute the least common multiple (LCM) of two polynomials numerically.

    Parameters:
        a: Coefficients of the first polynomial in descending order.
        b: Coefficients of the second polynomial in descending order.
        tol: Tolerance for considering remainder as zero.

    Returns:
        Coefficients of the LCM polynomial in descending order.
    """
    gcd_ab = poly_gcd(a, b, tol)
    product = np.convolve(a, b)
    lcm, remainder = np.polydiv(product, gcd_ab)
    if not np.allclose(remainder, 0.0, atol=tol):
        raise ValueError("Non-zero remainder in LCM computation.")
    return lcm.tolist()


def compute_lcd_from_denominators(
    denominators: list[list[list[float]]],
) -> list[float]:
    """Compute the least common denominator (LCD) of a MIMO system's denominators numerically.

    Parameters:
        denominators: A list of lists of denominators. Each denominator is a list of coefficients.

    Returns:
        Coefficients of the LCD polynomial in descending order.

    Examples:
        >>> compute_lcd_from_denominators([[[1, 2]], [[1, 2]]])
        [1.0, 2.0]
    """
    all_den = [den_row for row in denominators for den_row in row]

    current_lcd = all_den[0]
    for d in all_den[1:]:
        current_lcd = poly_lcm(current_lcd, d)
    return current_lcd


def compute_adjusted_num(
    numerator: list[float],
    lcd: list[float],
    denominator: list[float],
    tol: float = 1e-4,
) -> list[float]:
    """Compute the adjusted numerator coefficients after aligning to the LCD.

    Parameters:
        numerator: Coefficients of the numerator polynomial in descending order.
        lcd: Coefficients of the LCD polynomial in descending order.
        denominator: Coefficients of the denominator polynomial in descending order.
        tol: Tolerance for remainder check.

    Returns:
        Adjusted numerator coefficients in descending order.

    Examples:
        >>> compute_adjusted_num([1, 1], [1, 3, 2], [1, 3, 2])
        [1.0, 1.0]
    """
    product = np.convolve(numerator, lcd)
    quotient, remainder = np.polydiv(product, denominator)
    if not np.allclose(remainder, 0.0, atol=tol):
        raise ValueError(
            f"Adjusted numerator division has non-zero remainder {remainder}. Check your system."
        )
    return quotient.tolist()


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


def controllable_canonical_form(
    denominator: list[float] | np.ndarray,
) -> tuple[NDArray, NDArray]:
    """Compute the controllable canonical form (A, B) matrices from the denominator coefficients.

    Parameters:
        denominator: Coefficients of the denominator polynomial in descending order.

    Returns:
        A, B matrices of the state-space representation.

    Examples:
        >>> controllable_canonical_form([1, 2, 3])
        (array([[ 0.,  1.],
            [-2., -3.]]), array([[0.],
            [1.]]))
    """
    denominator = np.array(denominator)
    denominator = np.trim_zeros(denominator, trim="b")
    if len(denominator) == 0:
        return np.zeros((0, 0)), np.zeros((0, 1))
    n = len(denominator) - 1
    if n < 1:
        return np.zeros((0, 0)), np.zeros((0, 1))
    A = np.eye(n, k=1)
    A[-1, :] = -np.array(denominator[1:])
    B = np.zeros((n, 1))
    if n > 0:
        B[-1, 0] = 1.0
    return A, B


def _get_lcm_norm_coeffs(
    denominators: list[list[list[float]]],
    mode: Literal["global", "local"] = "global",
) -> list[float] | list[list[float]]:
    """Compute the LCM of denominators either globally or per input column.

    Parameters:
        denominators: List of lists of denominator coefficients.
        mode: 'global' for single LCM, 'local' for per-column LCM.

    Returns:
        LCM coefficients as specified by the mode.

    Examples:
        >>> denominators = [[[1.0, 2.0], [1.0, 1.0]], [[1.0, 2.0], [1.0, 3.0]]]
        >>> _get_lcm_norm_coeffs(denominators, mode="global")
        [1.0, 6.0, 11.0, 6.0]

        >>> denominators = [[[1.0, 2.0], [1.0, 1.0]], [[1.0, 2.0], [1.0, 3.0]]]
        >>> _get_lcm_norm_coeffs(denominators, mode="local")
        [[1.0, 2.0, 0.0], [1.0, 4.0, 3.0]]
    """
    if mode == "local":
        transposed = transpose(denominators)
        lcds = []
        for col in transposed:
            all_den = [row for row in col]
            if not all_den:
                lcd = []
            else:
                lcd = all_den[0]
                for d in all_den[1:]:
                    lcd = poly_lcm(lcd, d)
            lcds.append(lcd)
        max_len = max(len(lcd) for lcd in lcds)
        return [lcd + [0.0] * (max_len - len(lcd)) for lcd in lcds]
    else:
        all_den = [den_row for row in denominators for den_row in row]
        if not all_den:
            return []
        lcd = all_den[0]
        for d in all_den[1:]:
            lcd = poly_lcm(lcd, d)
        return lcd


def rjust(list_: list[int | float], width: int) -> list[int | float]:
    """Right-justify the list to the specified width, padding with zeros on the left.

    Parameters:
        list_: The list to pad.
        width: The target width.

    Returns:
        The padded list.

    Examples:
        >>> rjust([1, 2, 3], 4)
        [1, 2, 3, 0]
        >>> rjust([1, 2, 3, 4, 5], 4)
        [1, 2, 3, 4]
    """
    pad_with = 0 if isinstance(list_[0], int) else 0.0
    return list_[:width] + [pad_with] * (width - len(list_))


def tf2ss(
    numerators: list[list[list[float]]],
    denominators: list[list[list[float]]],
    minreal: bool = False,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Convert a MIMO transfer function to state-space representation (numerical).

    Parameters:
        numerators: List of lists of numerator coefficients.
        denominators: List of lists of denominator coefficients.
        minreal: Flag for minimal realization (not implemented).

    Returns:
        A, B, C, D state-space matrices.

    Examples:
        >>> numerators = [[[1]], [[1]]]
        >>> denominators = [[[1, 2]], [[1, 2]]]
        >>> A, B, C, D = tf2ss(numerators, denominators)
        >>> A.shape[0] > 0
        True
    """
    n_outputs = len(numerators)
    n_inputs = len(numerators[0]) if n_outputs > 0 else 0

    # Compute least common denominator
    lcd_coeffs = compute_lcd_from_denominators(denominators)

    # Compute state-space matrices from LCD
    A, B_scalar = controllable_canonical_form(lcd_coeffs)
    n_states = A.shape[0]

    # Initialize C and D matrices
    C = np.zeros((n_outputs, n_states))
    D = np.zeros((n_outputs, n_inputs))

    # Process each output-input pair
    for i_out in range(n_outputs):
        for j_in in range(n_inputs):
            num = numerators[i_out][j_in]
            den = denominators[i_out][j_in]
            adjusted_num = compute_adjusted_num(num, lcd_coeffs, den)
            reversed_coeffs = adjusted_num[::-1]
            padded_coeffs = rjust(reversed_coeffs, n_states)
            C[i_out, :] = padded_coeffs
            D[i_out, j_in] = (
                adjusted_num[-1]
                if len(adjusted_num) <= len(lcd_coeffs)
                else 0.0
            )

    # Expand B matrix for MIMO
    B = np.tile(B_scalar, (1, n_inputs))

    return A, B, C, D
