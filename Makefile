PYTHON ?= python3

.PHONY: help install run test test-app test-research conformance check demo research notebook

help:
	@printf '%s\n' 'install       Install development dependencies in your active environment' 'run           Start the Strata application' 'test          Run all regression tests' 'test-app      Run application and API tests' 'test-research Run research pipeline tests' 'conformance   Verify every registered strategy' 'check         Run tests and strategy conformance' 'demo          Build the self-contained offline demo' 'research      Regenerate research outputs from the frozen data' 'notebook      Build the research notebook after running research'

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-app.txt

run:
	$(PYTHON) -m app.server

test:
	$(PYTHON) -m pytest -q

test-app:
	$(PYTHON) -m pytest -q tests/application

test-research:
	$(PYTHON) -m pytest -q tests/research

conformance:
	$(PYTHON) -m app.conformance

check: test conformance

demo:
	$(PYTHON) tools/build_demo.py

research:
	$(PYTHON) -m src.pipeline --source cached

notebook:
	$(PYTHON) scripts/build_notebook.py
