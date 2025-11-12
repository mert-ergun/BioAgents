.PHONY: help install dev-install format lint test clean pre-commit-install pre-commit-run

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

dev-install:  ## Install with dev dependencies
	uv sync --all-groups

format:  ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint code with ruff
	uv run ruff check .

type-check:  ## Run type checking with mypy
	uv run mypy bioagents

security:  ## Run security checks with bandit
	uv run bandit -r bioagents -c pyproject.toml

test:  ## Run tests
	uv run pytest

test-verbose:  ## Run tests with verbose output
	uv run pytest -vv

test-coverage:  ## Run tests with coverage
	uv run pytest --cov=bioagents --cov-report=html --cov-report=term

pre-commit-install:  ## Install pre-commit hooks
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

pre-commit-run:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hooks
	uv run pre-commit autoupdate

clean:  ## Clean up cache and build files
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: format lint type-check security test  ## Run all checks
