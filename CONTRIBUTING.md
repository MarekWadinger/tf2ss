# Contributing to tf2ss

Thank you for your interest in contributing to tf2ss! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the [Issues](https://github.com/MarekWadinger/tf2ss/issues)
2. If not, create a new issue using the bug report template
3. Provide a clear description and minimal reproducible example
4. Include relevant system information (OS, Python version, package versions)

### Suggesting Features

1. Check if the feature has already been suggested in the [Issues](https://github.com/MarekWadinger/tf2ss/issues)
2. If not, create a new issue using the feature request template
3. Clearly describe the proposed feature and its use case
4. Discuss the feature with maintainers before implementing

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/yourusername/tf2ss.git
   cd tf2ss
   ```

3. Install [UV](https://docs.astral.sh/uv/) (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or on Windows:
   # powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

4. Install dependencies:

   ```bash
   uv sync --all-extras
   ```

5. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

### Making Changes

1. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards below
3. Add or update tests as appropriate
4. Update documentation if necessary
5. Run the test suite:

   ```bash
   uv run pytest
   ```

6. Run code quality checks:

   ```bash
   uv run ruff check .
   uv run mypy tf2ss/
   ```

7. Commit your changes with a descriptive commit message
8. Push to your fork and create a pull request

### Coding Standards

#### Python Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 88)
- Use type hints for function signatures
- Use Google-style docstrings
- First line of docstring should be in imperative mood

#### Code Quality

- Maintain test coverage above 90%
- Write clear, self-documenting code
- Add comments for complex algorithms
- Use meaningful variable and function names

#### Testing

- Write tests for all new functionality
- Use pytest for testing framework
- Include both unit tests and integration tests
- Test edge cases and error conditions

#### Documentation

- Update README.md for significant changes
- Add docstrings to all public functions and classes
- Include examples in docstrings where helpful
- Update CHANGELOG.md for all changes

### Pull Request Process

1. Ensure your code passes all tests and quality checks
2. Update documentation as needed
3. Add an entry to CHANGELOG.md
4. Create a pull request using the provided template
5. Address any feedback from code review
6. Once approved, your changes will be merged

### Testing

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

### Release Process

1. Update version with commitizen: `cz bump --increment patch|minor|major`
   This will automatically:
   - Update version number
   - Update CHANGELOG.md
   - Create a git tag
   - Commit the changes with a standardized message
2. Push changes and tag: `git push origin main --tags`
3. GitHub Actions will automatically build and publish to PyPI

## Development Guidelines

### Algorithm Implementation

- Prioritize numerical stability and accuracy
- Include references to academic papers or textbooks
- Validate against established implementations (MATLAB, SLYCOT)
- Consider computational efficiency for large systems

### Error Handling

- Use appropriate exception types
- Provide clear error messages
- Include context about what operation failed
- Validate input parameters early

### Performance

- Profile critical code paths
- Consider memory usage for large systems
- Use NumPy operations where possible
- Document computational complexity

## Questions?

If you have questions about contributing, please:

1. Check the existing documentation
2. Search through existing issues
3. Create a new issue with the "question" label
4. Contact the maintainers directly

Thank you for contributing to tf2ss!
