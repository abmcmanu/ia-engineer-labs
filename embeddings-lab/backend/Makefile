VENV := .env
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

.PHONY: venv install run dev freeze clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) main.py

dev:
	$(VENV)/bin/uvicorn main:app --reload

freeze:
	$(PIP) freeze > requirements.txt

clean:
	rm -rf $(VENV) __pycache__

docker-build:
	docker build -t embeddings-lab-backend .

docker-run:
	docker run --rm -p 5050:5050 embeddings-lab-backend
