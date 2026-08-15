PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup data train train-fast app deck dashboard notebook clean all

help:
	@echo "make setup       - create venv and install pinned dependencies"
	@echo "make data        - download and verify the UCI dataset"
	@echo "make train       - full reproducible training pipeline (~19 min)"
	@echo "make train-fast  - skip tuning and ablations (~2 min)"
	@echo "make notebook    - rebuild and execute the notebook end to end"
	@echo "make app         - launch the Streamlit demo"
	@echo "make deck        - build the PPTX and HTML presentations"
	@echo "make dashboard   - build the self-contained HTML results dashboard"
	@echo "make all         - data + train + deck + dashboard"

setup:
	/opt/homebrew/bin/python3.13 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PY) -m ipykernel install --user --name ml-for-good --display-name "ML for Good (3.13)"

data:
	$(PY) -m src.data_loader

train:
	$(PY) -m src.train

train-fast:
	$(PY) -m src.train --fast

notebook:
	$(PY) tools/build_notebook.py
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=2400 \
		--ExecutePreprocessor.kernel_name=ml-for-good \
		notebooks/01_ml_for_social_good_readmission.ipynb

app:
	.venv/bin/streamlit run app/streamlit_app.py

deck:
	$(PY) presentation/build_deck.py

dashboard:
	$(PY) dashboard/build_dashboard.py

all: data train deck dashboard

clean:
	rm -rf results/figures/* results/metrics/* results/explainability/* models/*.joblib
	find . -name '__pycache__' -type d -exec rm -rf {} +
