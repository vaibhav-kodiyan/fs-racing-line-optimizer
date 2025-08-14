.PHONY: venv install lint test check run headless

PY?=python3
PIP?=pip
VENV=.venv
ACT=. $(VENV)/bin/activate;

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(ACT) $(PIP) install -r requirements.txt

lint:
	$(ACT) flake8 .

test:
	$(ACT) pytest -q

check: lint test

run:
	$(ACT) $(PY) main.py --cones data/sample_cones.json

headless:
	$(ACT) $(PY) scripts/sim_headless.py --out artifacts
