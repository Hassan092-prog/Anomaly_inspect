SHELL := /bin/bash
PATH := $(HOME)/.local/bin:$(PATH)

VENV := .venv
ifeq ($(wildcard $(VENV)/bin/python),)
PYTHON := python3
RUFF := ruff
PYTEST := pytest
MYPY := mypy
else
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest
MYPY := $(VENV)/bin/mypy
endif

.PHONY: help setup setup-research setup-backend lint format test test-all bench report up clean

help:
	@echo "Targets:"
	@echo "  setup           Create .venv (Python 3.11) and install dev tools"
	@echo "  setup-research  Install pinned research deps (CPU by default; GPU=1 for CUDA)"
	@echo "  setup-backend   Install pinned backend API deps"
	@echo "  lint            Run ruff check, ruff format --check, and mypy"
	@echo "  format          Auto-fix lint issues and format code with ruff"
	@echo "  test            Run fast pytest suite (-m 'not slow')"
	@echo "  test-all        Run full pytest suite including slow/GPU tests"
	@echo "  bench           Run benchmark runner (pass ARGS='...')"
	@echo "  report          Regenerate tables and figures from results/"
	@echo "  up              Start platform services with docker compose"
	@echo "  clean           Remove Python and tool caches"

setup:
	@if command -v uv >/dev/null 2>&1; then \
		[ -d $(VENV) ] || uv venv --python 3.11 $(VENV); \
		uv pip install --python $(VENV)/bin/python -e ".[dev]"; \
	else \
		[ -d $(VENV) ] || python3 -m venv $(VENV); \
		$(VENV)/bin/pip install --upgrade pip && \
		$(VENV)/bin/pip install -e ".[dev]"; \
	fi

setup-research: setup
	@REQ=requirements-research-cpu.txt; \
	if [ "$(GPU)" = "1" ]; then REQ=requirements-research.txt; fi; \
	if command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(VENV)/bin/python --index-strategy unsafe-best-match -r $$REQ -e ".[dev]"; \
	else \
		$(VENV)/bin/pip install -r $$REQ -e ".[dev]"; \
	fi

setup-backend: setup
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(VENV)/bin/python -r requirements-backend.txt -e ".[dev]"; \
	else \
		$(VENV)/bin/pip install -r requirements-backend.txt -e ".[dev]"; \
	fi

lint:
	$(RUFF) check .
	$(RUFF) format --check .
	$(MYPY) research/metrics research/fusion backend/app

format:
	$(RUFF) check --fix .
	$(RUFF) format .

test:
	$(PYTEST) -m "not slow"

test-all:
	$(PYTEST)

bench:
	$(PYTHON) -m research.benchmarks.run $(ARGS)

report:
	$(PYTHON) -m research.analysis.report

up:
	docker compose up --build

clean:
	find . -path ./$(VENV) -prune -o -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
