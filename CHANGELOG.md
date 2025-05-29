# Changelog

All notable changes to this project will be documented in this file.

The CHANGELOG is powered by [Commitizen](https://commitizen-tools.github.io/commitizen/).

## [Unreleased]

### Added

- Initial release preparation

## [1.0.0] - 2025-05-29

### Added

- feat: initial release of tf2ss package
- feat: transfer function to state-space conversion for MIMO systems
- feat: support for both coefficient lists and control library objects
- feat: minimal realization option
- feat: time response analysis with `forced_response` function
- test: comprehensive test suite with SLYCOT and MATLAB validation
- docs: documentation and examples
- ci: GitHub Actions CI/CD pipeline
- chore: pre-commit hooks for code quality

### Features

- feat: full support for Multi-Input Multi-Output systems
- feat: robust algorithms for accurate conversion
- feat: seamless integration with Python Control library
- feat: automatic LCM computation for denominators
- feat: system poles and zeros preservation

### Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- sympy >= 1.8
- control >= 0.9.0
- Python >= 3.10

[Unreleased]: https://github.com/mwadinger/tf2ss/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/mwadinger/tf2ss/releases/tag/v1.0.0
