# Gap Trading System - Makefile

.PHONY: help install dev test lint format type-check security clean build docker run-dashboard run-cli docs

# Default target
help: ## Show this help message
	@echo "Gap Trading System - Available commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install dependencies with Poetry
	poetry install

install-dev: ## Install with development dependencies
	poetry install --with dev
	poetry run pre-commit install

# Development
dev: install-dev ## Setup development environment
	@echo "Development environment ready!"
	@echo "Activate with: poetry shell"

update: ## Update dependencies
	poetry update
	poetry run pre-commit autoupdate

# Testing
test: ## Run all tests
	poetry run pytest tests/ -v

test-cov: ## Run tests with coverage report
	poetry run pytest tests/ -v --cov=gappers --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests only (skip slow integration tests)
	poetry run pytest tests/ -v -m "not slow"

test-watch: ## Run tests in watch mode
	poetry run ptw tests/ gappers/ --runner "python -m pytest tests/ -v"

# Code Quality
lint: ## Run linting checks
	poetry run ruff check gappers/ tests/
	poetry run ruff format --check gappers/ tests/

format: ## Format code with black and ruff
	poetry run ruff format gappers/ tests/
	poetry run ruff check --fix gappers/ tests/

type-check: ## Run type checking with mypy
	poetry run mypy gappers/

security: ## Run security checks with bandit
	poetry run bandit -r gappers/ -ll

quality: lint type-check security ## Run all code quality checks

pre-commit: ## Run pre-commit hooks on all files
	poetry run pre-commit run --all-files

# Application
run-dashboard: ## Start Streamlit dashboard
	poetry run streamlit run app.py

run-cli: ## Show CLI help
	poetry run python -m gappers.cli --help

backtest: ## Run sample backtest
	poetry run python -m gappers.cli backtest --start-date 2023-01-01 --end-date 2023-12-31

scan: ## Scan for today's gaps
	poetry run python -m gappers.cli scan --min-gap 0.02 --top-k 10

live-demo: ## Run live trading demo (dry run)
	poetry run python -m gappers.cli live --scan-only --dry-run

# Building
build: ## Build package with Poetry
	poetry build

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Docker
docker-build: ## Build Docker image
	docker build -t gappers-trader:latest .

docker-build-dev: ## Build development Docker image
	docker build --target development -t gappers-trader:dev .

docker-run: ## Run Docker container
	docker run -p 8501:8501 gappers-trader:latest

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## View logs from all services
	docker-compose logs -f

# Documentation
docs-serve: ## Serve documentation locally
	poetry run mkdocs serve

docs-build: ## Build documentation
	poetry run mkdocs build

docs-deploy: ## Deploy documentation to GitHub Pages
	poetry run mkdocs gh-deploy

# Database and Data
setup-data: ## Create data directories
	mkdir -p data/ohlcv data/universe data/cache logs

download-sample: ## Download sample data for testing
	poetry run python -m gappers.cli download --symbols AAPL MSFT GOOGL --start-date 2023-01-01 --end-date 2023-12-31

clear-cache: ## Clear data cache
	poetry run python -m gappers.cli status --clear-cache

# Performance and Monitoring
benchmark: ## Run performance benchmarks
	poetry run python -c "
	from gappers import Backtester, GapParams
	from datetime import datetime
	import time
	
	print('Running performance benchmark...')
	start_time = time.time()
	
	backtester = Backtester()
	params = GapParams()
	results = backtester.run_backtest(
		datetime(2020, 1, 1), 
		datetime(2023, 12, 31), 
		params
	)
	
	duration = time.time() - start_time
	print(f'Backtest completed in {duration:.2f} seconds')
	print(f'Total trades: {len(results[\"trades\"])}')
	"

profile: ## Profile application performance
	poetry run python -m cProfile -o profile.stats -m gappers.cli backtest --start-date 2023-01-01 --end-date 2023-06-30
	@echo "Profile saved to profile.stats"
	@echo "View with: python -c 'import pstats; pstats.Stats(\"profile.stats\").sort_stats(\"cumulative\").print_stats(20)'"

memory-check: ## Check for memory leaks
	poetry run python -m memory_profiler -m gappers.cli backtest --start-date 2023-01-01 --end-date 2023-03-31

# CI/CD
ci: quality test ## Run CI pipeline locally
	@echo "✅ CI pipeline completed successfully"

check-dependencies: ## Check for dependency vulnerabilities
	poetry run safety check
	poetry run pip-audit

# Release
version-patch: ## Bump patch version
	poetry version patch
	@echo "Version bumped to: $$(poetry version -s)"

version-minor: ## Bump minor version
	poetry version minor
	@echo "Version bumped to: $$(poetry version -s)"

version-major: ## Bump major version
	poetry version major
	@echo "Version bumped to: $$(poetry version -s)"

changelog: ## Generate changelog
	@echo "# Changelog" > CHANGELOG.md
	@echo "" >> CHANGELOG.md
	@git log --pretty=format:"- %s" --since="2024-01-01" >> CHANGELOG.md

# Environment
env-create: ## Create .env file from template
	cp .env.example .env
	@echo ".env file created from template"
	@echo "Please edit .env with your configuration"

env-check: ## Check environment configuration
	poetry run python -m gappers.cli status --check-config

# Development shortcuts
quick-test: ## Quick test and quality check for development
	poetry run pytest tests/test_signals.py -v
	poetry run ruff check gappers/signals.py

jupyter: ## Start Jupyter notebook for analysis
	poetry run jupyter notebook

# Production deployment helpers
prod-check: ## Check production readiness
	@echo "Checking production readiness..."
	@poetry run python -c "
	from gappers.config import config
	print('✅ Configuration loaded')
	
	from gappers import DataFeed
	feed = DataFeed()
	print('✅ DataFeed initialized')
	
	if config.has_alpaca_credentials:
		print('✅ Alpaca credentials configured')
	else:
		print('⚠️  Alpaca credentials not configured')
	
	print('Production readiness check complete')
	"

deploy-check: ci build ## Full deployment check
	@echo "✅ Deployment check completed successfully"

# Monitoring
health-check: ## Check system health
	poetry run python -m gappers.cli status --test-connection

logs: ## View application logs
	tail -f logs/*.log

# Example workflows
example-backtest: ## Run example backtest workflow
	poetry run python -c "
	from datetime import datetime
	from gappers import Backtester, GapParams, PerformanceAnalyzer
	
	print('Running example backtest...')
	backtester = Backtester()
	params = GapParams(profit_target=0.05, stop_loss=0.02, top_k=10)
	
	results = backtester.run_backtest(
		datetime(2023, 1, 1),
		datetime(2023, 6, 30), 
		params
	)
	
	analyzer = PerformanceAnalyzer()
	analysis = analyzer.analyze_backtest_results(results)
	
	print(f'Total trades: {len(results[\"trades\"])}')
	if analysis.get('performance_metrics'):
		perf = analysis['performance_metrics']
		print(f'Total return: {perf.get(\"total_return_pct\", 0):.2f}%')
		print(f'Sharpe ratio: {perf.get(\"sharpe_ratio\", 0):.2f}')
	"

example-scan: ## Run example gap scan
	poetry run python -c "
	from datetime import datetime
	from gappers import SignalGenerator, DataFeed, UniverseBuilder
	
	print('Running example gap scan...')
	data_feed = DataFeed()
	universe = UniverseBuilder(data_feed)
	signals = SignalGenerator(data_feed, universe)
	
	try:
		gaps = signals.calculate_gaps(datetime.now(), min_gap_pct=0.02)
		print(f'Found {len(gaps)} gaps today')
		if not gaps.empty:
			top_gaps = signals.rank_gaps(gaps, top_k=5)
			print('Top 5 gaps:')
			for _, gap in top_gaps.iterrows():
				print(f'  {gap[\"symbol\"]}: {gap[\"gap_pct\"]*100:.2f}%')
	except Exception as e:
		print(f'Scan failed (expected without market data): {e}')
	"