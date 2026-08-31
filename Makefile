.PHONY: help up down migrate seed bench replay-verify tunnel test lint

help:
	@echo "Paydger development commands:"
	@echo "  make up             Start all services with Docker Compose"
	@echo "  make down           Stop all services"
	@echo "  make migrate        Run database migrations"
	@echo "  make seed           Generate synthetic population & seed data"
	@echo "  make bench          Run the randomized holdout benchmark"
	@echo "  make replay-verify  Rebuild projections, re-decide cases, diff against recorded decisions"
	@echo "  make tunnel         Expose /webhooks/razorpay via local tunnel"
	@echo "  make test           Run test suite with pytest"
	@echo "  make lint           Run code linter"

up:
	docker compose up -d

down:
	docker compose down

migrate:
	python -m api.migrate

seed:
	python -m bench.generator

bench:
	python -m bench.evaluation

replay-verify:
	python -m api.replay

tunnel:
	ngrok http 8000

test:
	pytest tests/ -v

lint:
	ruff check .
