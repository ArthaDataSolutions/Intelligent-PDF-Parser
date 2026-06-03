.PHONY: help install dev test samples build up down logs clean

help:
	@echo "install   - pip install requirements (+ dev)"
	@echo "dev       - run API locally with reload"
	@echo "test      - run pytest (offline, mocked providers)"
	@echo "samples   - generate sample handwritten test PDFs into ./samples"
	@echo "build     - docker compose build"
	@echo "up        - docker compose up (API only)"
	@echo "up-local  - docker compose up with self-hosted Ollama"
	@echo "down      - docker compose down"

install:
	pip install -r requirements.txt

dev:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

samples:
	python scripts/generate_samples.py

build:
	docker compose build

up:
	docker compose up --build

up-local:
	docker compose --profile local-llm up --build

down:
	docker compose down

logs:
	docker compose logs -f api

clean:
	rm -rf __pycache__ .pytest_cache **/__pycache__
