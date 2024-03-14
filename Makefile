# Makefile
SHELL = /bin/bash

# Default
default: splitter

# Install Dev Tools Needed for Make
.PHONY: install
install: install-dev install-pipeline

.PHONY: install-gpu
install-gpu:
	pip uninstall faiss-cpu || 0
	pip install faiss-gpu vllm

.PHONY: install-dev
install-dev:
	@echo "installing build pip modules ..."
	pip install flake8 pyupgrade isort pytest runpod
	pip install git+https://github.com/psf/black

.PHONY: install-pipeline
install-pipeline:
	@echo "installing for pdf parser ..."
	pip install pydantic pypdf2 pycryptodome pandas faiss-cpu umap-learn rich pyarrow fastparquet plotly kaleido
	@echo "installing for google ..."
	pip install --upgrade google-cloud-bigquery google-cloud-documentai google-cloud-storage
	@echo "installing for chunker ..."
	pip install spacy torch nltk sentence_transformers scikit-learn pillow ydata_profiling
	python -m spacy download en_core_web_sm
	pushd code/libdocs && pip install --upgrade -e .
	mkdir -p ./data/processing/pdfs/chunks
	mkdir -p ./data/processing/pdfs/split

# Test
.PHONY: test
test:
	@echo "Running tests..."
	@pytest -v

# Styling
.PHONY: style
style:
	@echo "Styling up folder..."
	@black --exclude=data/cache .
	@flake8 --max-line-length 160 --exclude=experiments,build --extend-exclude 'prompt.py __init__.py' --extend-ignore F811 --extend-ignore E203
	@python -m isort --known-local-folder data/cache .
	@pyupgrade

# Cleaning
.PHONY: clean
clean: style
	@echo "Cleaning up extra files..."
	@find . -type f -name "*.DS_Store" -ls -delete
	@find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	@find . | grep -E ".pytest_cache" | xargs rm -rf
	@find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	@rm -rf .coverage*

# Generation
.PHONY: splitter
splitter:
	python code/splitter.py
