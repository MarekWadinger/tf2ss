import time

import numpy as np
from control import tf2ss as ctrl_tf2ss

from tf2ss import tf2ss
from tf2ss.tf2ss_numeric import tf2ss as tf2ss_numeric


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


def _min_time(fn, *args, n_trials=3):
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def benchmark_tf2ss(n_outputs, n_inputs, order, n_trials=3):
    """Benchmark the tf2ss implementations for a random MIMO system."""
    num, den = random_mimo_tf(n_outputs, n_inputs, order)
    # tf2ss expects 3D lists
    num_3d = [[list(map(float, n)) for n in row] for row in num]
    den_3d = [[list(map(float, d)) for d in row] for row in den]

    t_ours = _min_time(tf2ss, num_3d, den_3d, n_trials=n_trials)
    t_num = _min_time(tf2ss_numeric, num_3d, den_3d, n_trials=n_trials)
    t_ctrl = _min_time(ctrl_tf2ss, num_3d, den_3d, n_trials=n_trials)
    return t_ours, t_num, t_ctrl


def print_benchmark_table(sizes, order):
    """Print a table comparing symbolic, numeric and control.tf2ss speed."""
    print("Benchmark: tf2ss vs control.tf2ss for random stable MIMO systems")
    print(f"System order: {order}")
    print()
    header = "{:>8} {:>8} {:>14} {:>14} {:>14} {:>10}".format(
        "Outputs",
        "Inputs",
        "tf2ss [s]",
        "numeric [s]",
        "ctrl_tf2ss [s]",
        "Speedup",
    )
    print(header)
    print("-" * len(header))
    for n_outputs, n_inputs in sizes:
        t_ours, t_num, t_ctrl = benchmark_tf2ss(n_outputs, n_inputs, order)
        speedup = t_ours / t_num if t_num > 0 else float("inf")
        print(
            f"{n_outputs:8d} {n_inputs:8d} {t_ours:14.6f} {t_num:14.6f}"
            f" {t_ctrl:14.6f} {speedup:9.1f}x"
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
