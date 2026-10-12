PY ?= python
CONFIG ?= config/default.yaml

.PHONY: data train eval app test

data:
	$(PY) -m qpgen.data.build_dataset --config $(CONFIG)

train:
	$(PY) -m qpgen.train --config $(CONFIG)

eval:
	$(PY) -m qpgen.evaluate --config $(CONFIG)

app:
	streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q
