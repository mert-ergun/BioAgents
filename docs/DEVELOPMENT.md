# Development Guide

## Setup

This project uses `uv` for dependency management and `pre-commit` for code quality checks.

### Initial Setup

```bash
# Install dependencies
uv sync --all-groups

# Install pre-commit hooks
uv run pre-commit install
```

## Development Workflow

### Using Make Commands

```bash
make help              # Show all available commands
make dev-install       # Install with dev dependencies
make format            # Format code with ruff
make lint              # Lint code with ruff
make type-check        # Run type checking with mypy
make security          # Run security checks with bandit
make test              # Run tests
make pre-commit-run    # Run all pre-commit hooks
make all               # Run all checks (format, lint, type-check, security, test)
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. They include:

- **File checks**: trailing whitespace, end-of-file, YAML/TOML/JSON validation
- **Ruff**: Fast Python linting and formatting
- **Bandit**: Security vulnerability scanning
- **Mypy**: Static type checking
- **Pytest**: Run test suite
- **UV sync check**: Ensure dependencies are in sync

To run hooks manually on all files:
```bash
make pre-commit-run
# or
uv run pre-commit run --all-files
```

To skip hooks temporarily (not recommended):
```bash
git commit --no-verify
```

### Code Quality Standards

- **Line length**: 100 characters (enforced by ruff)
- **Python version**: 3.12+
- **Type hints**: Encouraged but not strictly enforced
- **Docstrings**: Required for public functions and classes
- **Tests**: Required for new features

### Running Tests

```bash
# Run all tests
make test

# Run with verbose output
make test-verbose

# Run with coverage report
make test-coverage

# Run specific test file
uv run pytest tests/test_proteomics_tools.py

# Run tests matching pattern
uv run pytest -k "test_fetch"
```

### Updating Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Update all dependencies
uv sync --upgrade

# Update pre-commit hooks
make pre-commit-update
```

## Configuration Files

- `pyproject.toml`: Project metadata, dependencies, and tool configurations
- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `Makefile`: Common development commands
- `.python-version`: Python version specification for uv

## Troubleshooting

### Pre-commit hooks failing

If hooks fail, they usually auto-fix issues. Review changes and commit again:
```bash
git add -u
git commit
```

### Import errors in tests

Make sure you've installed the package in development mode:
```bash
uv sync --all-groups
```

### Type checking errors

Mypy is configured to ignore missing imports for third-party libraries. If you encounter issues:
```bash
make type-check
```
