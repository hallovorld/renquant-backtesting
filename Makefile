PYTHON ?= python3
COMMON_SRC ?= ../renquant-common/src
export PYTHONPATH := $(COMMON_SRC):src:$(PYTHONPATH)

.PHONY: test doctor

test:
	$(PYTHON) -m pytest -q

doctor:
	$(PYTHON) -c "from renquant_backtesting import BacktestPipeline; from renquant_common import Pipeline; print('renquant-backtesting ok')"
