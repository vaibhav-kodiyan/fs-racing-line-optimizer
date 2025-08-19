.PHONY: venv install lint test check run headless

PY?=python3
VENV=venv

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(VENV)/bin/pip install -e .[dev]

lint:
	$(VENV)/bin/flake8 fmsim main.py scripts tests

test:
	$(VENV)/bin/pytest -q

check: lint test

run:
	$(VENV)/bin/python main.py --cones data/sample_cones.json

headless:
	$(VENV)/bin/python scripts/sim_headless.py --out artifacts
