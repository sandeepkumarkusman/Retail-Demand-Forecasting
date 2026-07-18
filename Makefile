PYTHON ?= python
VENV   ?= venv

.PHONY: install run train test serve report clean

install:
	$(PYTHON) -m pip install -r requirements.txt

run: train
	@echo "Pipeline complete. Submission at outputs/submission.csv"

train:
	$(PYTHON) -m src.pipeline

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

serve:
	streamlit run demo/app.py

report:
	@echo "Opening Final Report..."
	@cat reports/Final_Report.md

clean:
	rm -rf models/*.joblib models/*.json outputs/predictions.parquet outputs/submission.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
