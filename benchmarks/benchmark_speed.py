import time

import numpy as np
from control import tf2ss as ctrl_tf2ss

from tf2ss import tf2ss


def random_stable_poly(order):
    """Generate coefficients of a stable real-coefficient polynomial of given order."""
    # Place roots in left half-plane (real part < 0)
    real_parts = -np.random.uniform(0.5, 2.0, size=order)
    imag_parts = np.random.uniform(-1.0, 1.0, size=order)
    roots = real_parts + 1j * imag_parts
    coeffs = np.poly(roots)
    coeffs = np.real_if_close(coeffs)
    return coeffs


def random_mimo_tf(n_outputs, n_inputs, order):
    """Generate random MIMO transfer function coefficients (num, den) with stable poles."""
    den = [
        [random_stable_poly(order) for _ in range(n_inputs)]
        for _ in range(n_outputs)
    ]
    # Numerators: random coefficients, degree <= denominator
    num = []
    for i in range(n_outputs):
        row = []
        for j in range(n_inputs):
            deg = np.random.randint(1, order + 1)
            c = np.random.randn(deg)
            # Pad to match denominator length
            c = np.pad(c, (len(den[i][j]) - len(c), 0), "constant")
            row.append(c)
        num.append(row)
    return num, den


def benchmark_tf2ss(n_outputs, n_inputs, order, n_trials=3):
    """Benchmark both tf2ss implementations for a random MIMO system."""
    num, den = random_mimo_tf(n_outputs, n_inputs, order)
    # tf2ss expects 3D lists
    num_3d = [[list(map(float, n)) for n in row] for row in num]
    den_3d = [[list(map(float, d)) for d in row] for row in den]

    # tf2ss (ours)
    t_ours = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        tf2ss(num_3d, den_3d)
        t_ours.append(time.perf_counter() - t0)
    t_ours = min(t_ours)

    # control.tf2ss (reference)
    t_ctrl = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        ctrl_tf2ss(num_3d, den_3d)
        t_ctrl.append(time.perf_counter() - t0)
    t_ctrl = min(t_ctrl)

    return t_ours, t_ctrl


def print_benchmark_table(sizes, order):
    """Print a table comparing tf2ss and control.tf2ss speed for MIMO systems."""
    print("Benchmark: tf2ss vs control.tf2ss for random stable MIMO systems")
    print(f"System order: {order}")
    print()
    header = "{:>8} {:>8} {:>14} {:>14} {:>10}".format(
        "Outputs", "Inputs", "tf2ss [s]", "ctrl_tf2ss [s]", "Slowdown"
    )
    print(header)
    print("-" * len(header))
    for n_outputs, n_inputs in sizes:
        t_ours, t_ctrl = benchmark_tf2ss(n_outputs, n_inputs, order)
        slowdown = t_ours / t_ctrl if t_ctrl > 0 else float("inf")
        print(
            f"{n_outputs:8d} {n_inputs:8d} {t_ours:14.6f} {t_ctrl:14.6f} {slowdown:10.2f}"
        )


if __name__ == "__main__":
    # List of (outputs, inputs) to benchmark
    sizes = [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (8, 8),
        (10, 10),
    ]
    order = 4  # System order (denominator degree)
    print_benchmark_table(sizes, order)
