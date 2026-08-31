.PHONY: audit build check format test

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
	uv run pytest --cov=lazy_grounding --cov-report=term-missing

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

build:
	uv build

audit:
	uv run vulture src tests --min-confidence 80
	uv run bandit -q -r src
	uv run pip-audit --local --skip-editable --progress-spinner off
