# tf2ss - MIMO Transfer Function to State-Space Conversion

🚀 **Easy-to-install alternative to SLYCOT** | 🎯 **MATLAB-consistent results** | 🔧 **No Fortran compiler required**

[![PyPI Package latest release](https://img.shields.io/pypi/v/tf2ss.svg?style=)](https://pypi.org/project/tf2ss/)
[![Supported versions](https://img.shields.io/pypi/pyversions/tf2ss.svg?style=)](https://pypi.org/project/tf2ss/)
[![Quality and Tests](https://github.com/MarekWadinger/tf2ss/actions/workflows/ci.yml/badge.svg)](https://github.com/MarekWadinger/tf2ss/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MarekWadinger/tf2ss/branch/main/graph/badge.svg)](https://codecov.io/gh/MarekWadinger/tf2ss)

`tf2ss` is a Python library for converting transfer function representations to state-space form, supporting **Multi-Input Multi-Output (MIMO) systems**, compilation-less!

## 🤔 Why tf2ss?

**Struggling with SLYCOT installation?** [control](https://github.com/python-control/python-control) with [SLYCOT](https://github.com/python-control/Slycot) requires Fortran compilers and BLAS libraries, making it difficult to install across different platforms.

**Need MIMO support in Python?** SciPy's `tf2ss` only handles SISO systems, leaving MIMO users without options.

**Want MATLAB-consistent results?** Our implementation produces identical results to MATLAB's `tf2ss` function.

**tf2ss solves all these problems** with a Python implementation that's easy to install and works everywhere Python does—no compilers, no manual builds, just `pip install tf2ss`!

| Library | Installation Complexity | Dependencies        | MATLAB Consistency | MIMO Support |
|---------|-------------------------|---------------------|--------------------|--------------|
| tf2ss   | Simple (`pip install`)  | None                | ✅                 | ✅           |
| SLYCOT  | Complex (requires Fortran) | Fortran, BLAS    | ❌                 | ✅           |
| scipy   | Simple (`pip install`)  | None                | ✅                 | ❌           |

## ⚡️ Quickstart

Convert a MIMO system from transfer function to state-space, just like you do in scipy, control, MATLAB even:

```python
from tf2ss import tf2ss
import numpy as np

# 2x2 MIMO system
numerators = [
    [[1, 0], [0.5, 1]],  # First row: [s/(s+2), (0.5s+1)/(s+2)]
    [[2], [1, 1]],  # Second row: [2/(s+2), (s+1)/(s+2)]
]
denominators = [[[1, 2], [1, 2]], [[1, 2], [1, 2]]]  # Common denominator s+2

A, B, C, D = tf2ss(numerators, denominators)
```

You can also work directly with `control` library objects:

```python
import control as ctrl
from tf2ss import tf2ss

# Create transfer function using control library
sys_tf = ctrl.tf([1, 1], [1, 3, 2])
A, B, C, D = tf2ss(sys_tf)

# Create equivalent state-space system
sys_ss = ctrl.StateSpace(A, B, C, D)
```

## 🔧 Features

### Core Functionality

- **MIMO Support**: Full support for Multi-Input Multi-Output systems
- **Minimal Realization**: Optional minimal realization to reduce system order
- **Numerical Stability**: Robust algorithms for numerical computation
- **Control Integration**: Seamless integration with Python Control library

### Advanced Features

- **Common Denominator**: Automatic computation of least common multiple for denominators
- **Pole Preservation**: Maintains system poles accurately
- **Zero Preservation**: Preserves transmission zeros when possible
- **Flexible Input Formats**: Accepts both coefficient lists and control library objects

### Time Response Analysis

The package includes `forced_response` function for time-domain analysis:

```python
from tf2ss import tf2ss, forced_response
import numpy as np

# Convert to state-space
A, B, C, D = tf2ss(numerators, denominators)

# Generate time response
t = np.linspace(0, 10, 1000)
u = np.ones((1, len(t)))  # Step input
result = forced_response((A, B, C, D), t, u)

# Plot results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(result.time, result.outputs[0])
plt.xlabel("Time [s]")
plt.ylabel("Output")
plt.title("Step Response")
plt.grid(True)
plt.show()
```

## 🛠 Installation

Requires Python 3.11 or higher.

### From PyPI (recommended)

```bash
pip install tf2ss
```

### From Source with UV

```bash
git clone https://github.com/MarekWadinger/tf2ss.git
cd tf2ss
uv sync
```

### Development Installation

```bash
git clone https://github.com/MarekWadinger/tf2ss.git
cd tf2ss
uv sync --all-extras
```

Alternatively, you can use pip for development:

```bash
git clone https://github.com/MarekWadinger/tf2ss.git
cd tf2ss
pip install -e .[dev]
```

## 📖 Documentation

### Mathematical Background

The conversion from transfer function to state-space follows these key steps:

1. **Common Denominator Computation**: For each input, compute the exact least common multiple (LCM) of that column's denominators
2. **Numerator Adjustment**: Adjust numerators based on the common denominator
3. **Controllable Canonical Form**: Construct one controller-canonical block per input
4. **Minimal Realization**: With `minreal=True` (default), remove uncontrollable and unobservable states with the orthogonal staircase (Kalman decomposition, as SLICOT TB01PD), using a tolerance relative to the system's scale

### API Reference

#### `tf2ss(numerators, denominators, *, minreal=True, tol=None)`

Convert transfer function to state-space representation.

**Parameters:**

- `numerators`: List of lists of lists containing numerator coefficients
- `denominators`: List of lists of lists containing denominator coefficients
- `minreal`: Boolean, whether to compute minimal realization (default: True)
- `tol`: Rank tolerance of the minimal realization (default: `n²·eps` relative to the system's norm). A larger value merges nearby poles and cancels near pole-zero pairs, trading exactness for a smaller state dimension.

`tf2ss.tf2ss_numeric.tf2ss` has the same signature and uses floating point instead of sympy; it is several times faster.

**Returns:**

- `A, B, C, D`: State-space matrices as numpy arrays

#### `forced_response(system, T, U, X0=0.0, **kwargs)`

Compute forced response of state-space system.

**Parameters:**

- `system`: State-space system (A, B, C, D) or TransferFunction
- `T`: Time vector
- `U`: Input array
- `X0`: Initial conditions (default: 0)

**Returns:**

- Time response data object with time, outputs, states, and inputs

## 🧪 Testing

Run the full test suite:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=tf2ss --cov-report=html
```

Run specific test categories:

```bash
uv run pytest -m "not slow"  # Skip slow tests
```

The test suite includes:

- Comparison with SLYCOT implementations
- Verification against MATLAB results
- MIMO system validation
- Numerical accuracy tests

## 🔬 Algorithm Validation

This report compares the cases where different implementations produce different poles of state-space realizations:

1. **tf2ss**: Our implementation based on Controllable Canonical Form
2. **SLYCOT**: Industry-standard control library
3. **MATLAB Control System Toolbox**: Reference implementation

For a detailed comparison of poles and transmission zeros between our implementation, SLYCOT, and MATLAB, see [tf2ss_report.md](tf2ss/tf2ss_report.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`uv sync --all-extras`)
4. Make your changes
5. Run tests (`uv run pytest`)
6. Run linting (`uv run ruff check .`)
7. Commit your changes (`git commit -m 'Add some amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [Python Control](https://github.com/python-control/python-control): Control systems library for Python
- [SLYCOT](https://github.com/python-control/Slycot): Python wrapper for SLICOT control library
- [SIPPY](https://github.com/CPCLAB-UNIPI/SIPPY): Systems Identification Package for Python

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{tf2ss,
  title = {tf2ss: Transfer Function to State-Space Conversion for MIMO Systems},
  author = {Wadinger, Marek},
  url = {https://github.com/MarekWadinger/tf2ss},
  year = {2025}
}
```

## 📞 Support

If you encounter any issues or have questions:

- Open an [issue](https://github.com/MarekWadinger/tf2ss/issues) on GitHub
- Check the existing documentation and examples
- Review the test cases for usage patterns
