.PHONY: init check clean test lint format typecheck benchmark experiment

init:
	@command -v uv >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv sync --all-extras
	uv run pre-commit install 2>/dev/null || true

check: format lint typecheck test

format:
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/ --fix

typecheck:
	uv run mypy src/brain_sim/

test:
	uv run pytest tests/ -v

benchmark:
	uv run python -m benchmarks.evaluate $(APPROACH)

experiment:
	uv run python -m experiments.causal_dictionaries.run $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info
