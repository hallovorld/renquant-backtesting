PYTHON ?= $(shell if [ -x ../RenQuant/.venv/bin/python ]; then printf '%s' ../RenQuant/.venv/bin/python; else printf '%s' python3; fi)
COMMON_SRC ?= ../renquant-common/src
BASE_DATA_SRC ?= ../renquant-base-data/src
ARTIFACTS_SRC ?= ../renquant-artifacts/src
PIPELINE_SRC ?= ../renquant-pipeline/src
export PYTHONPATH := $(COMMON_SRC):$(BASE_DATA_SRC):$(ARTIFACTS_SRC):$(PIPELINE_SRC):src:$(PYTHONPATH)

.PHONY: test doctor latest-report

test:
	$(PYTHON) -m pytest -q

doctor:
	$(PYTHON) -c "from renquant_backtesting import BacktestPipeline; from renquant_common import Pipeline; print('renquant-backtesting ok')"

latest-report:
	$(PYTHON) -m renquant_backtesting.reporting.latest_run_docs
