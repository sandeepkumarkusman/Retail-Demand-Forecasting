PYTHON ?= python

.PHONY: run test demo

run:
	$(PYTHON) -m src.pipeline

test:
	$(PYTHON) -m unittest discover -s tests -v

demo:
	$(PYTHON) demo/app.py
