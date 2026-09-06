.PHONY: help install data data-discharge data-bulletins panels baselines \
        backtest storage figures calibration test all clean

PY ?= python
SRC := PYTHONPATH=src $(PY)

help:
	@echo "tn-reservoir-forecast"
	@echo ""
	@echo "  make install          install dependencies"
	@echo "  make data             download all raw sources (~720 MB)"
	@echo "  make data-discharge   NWDP river discharge only (~80 MB, target 1)"
	@echo "  make data-bulletins   CWC weekly bulletin PDFs (~640 MB, target 2)"
	@echo ""
	@echo "  make panels           raw -> tidy weekly panels"
	@echo "  make baselines        score baselines only (the bar, no model)"
	@echo "  make backtest         full discharge backtest incl. TimesFM"
	@echo "  make storage          full storage backtest incl. TimesFM"
	@echo "  make calibration      coverage / PIT checks, both targets"
	@echo "  make figures          fan charts + skill figures"
	@echo "  make test             API regression tests"
	@echo ""
	@echo "  make all              panels -> backtests -> calibration -> figures"

install:
	pip install -e .

data:
	$(PY) scripts/fetch_data.py --all
data-discharge:
	$(PY) scripts/fetch_data.py --discharge
data-bulletins:
	$(PY) scripts/fetch_data.py --bulletins

panels:
	$(SRC) src/build_panel.py
	$(SRC) src/build_storage_panel.py

# Baselines are deliberately runnable on their own: the study design requires
# the bar to be established before the model is scored against it.
baselines:
	$(SRC) src/backtest.py --no-timesfm --every 2

backtest:
	$(SRC) src/backtest.py --every 2 --transform log1p

storage:
	$(SRC) src/backtest_storage.py --every 2

calibration:
	$(SRC) src/calibration.py

figures:
	$(SRC) src/plots.py

test:
	$(PY) -m pytest tests/ -q

all: panels backtest storage calibration figures

clean:
	rm -rf src/__pycache__ .pytest_cache data/processed/*.parquet
	@echo "raw data left intact — delete data/raw manually if you mean it"
