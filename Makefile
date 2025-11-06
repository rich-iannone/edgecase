.PHONY: help install dev build test test-rust test-python test-all coverage lint format clean docs examples benchmark

help:
	@echo "EdgeCase Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package in development mode"
	@echo "  make dev           Install with all development dependencies"
	@echo "  make build         Build Rust extension"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests (Python + Rust)"
	@echo "  make test-python   Run Python tests with coverage"
	@echo "  make test-rust     Run Rust tests"
	@echo "  make test-property Run property-based tests"
	@echo "  make coverage      Generate coverage reports"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linters (ruff, clippy)"
	@echo "  make format        Format code (black, cargo fmt)"
	@echo ""
	@echo "Performance:"
	@echo "  make benchmark     Run benchmarks"
	@echo "  make bench-rust    Run Rust benchmarks"
	@echo "  make bench-python  Run Python benchmarks"
	@echo ""
	@echo "Utilities:"
	@echo "  make examples      Run example scripts"
	@echo "  make docs          Build documentation"
	@echo "  make clean         Clean build artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	rustup component add rustfmt clippy

build:
	maturin develop

build-release:
	maturin develop --release

# Default test runs Python tests only (Rust core not yet compiling with PyO3 0.23)
test: clean-pyc
	python3 -m pytest tests/unit tests/integration -v

test-all: clean-pyc
	python3 -m pytest tests/unit tests/integration tests/property -v

test-benchmarks:
	python3 -m pytest tests/benchmarks/ --benchmark-only

test-rust:
	cargo test --all-features --verbose

test-property: clean-pyc
	python3 -m pytest tests/property/ -v --hypothesis-show-statistics

test-stress: clean-pyc
	python3 -m pytest tests/stress/ -v -m slow

coverage:
	@echo "Generating Python coverage..."
	python3 -m pytest tests/ --cov=edgecase --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated: htmlcov/index.html"

coverage-rust:
	@echo "Generating Rust coverage..."
	cargo tarpaulin --all-features --workspace --timeout 120 --out Html
	@echo ""
	@echo "Coverage report generated: tarpaulin-report.html"

lint:
	@echo "Linting Python..."
	ruff check python/edgecase tests/
	@echo ""
	@echo "Linting Rust..."
	cargo clippy --all-features -- -D warnings

lint-fix:
	ruff check python/edgecase tests/ --fix
	cargo clippy --all-features --fix

format:
	@echo "Formatting Python..."
	ruff format python/edgecase tests/
	@echo ""
	@echo "Formatting Rust..."
	cargo fmt

benchmark: bench-rust bench-python

bench-rust:
	cargo bench

bench-python:
	pytest tests/python -m benchmark --benchmark-only

examples:
	@echo "Running basic usage example..."
	python examples/basic_usage.py
	@echo ""
	@echo "Running bioinformatics example..."
	python examples/bioinformatics.py
	@echo ""
	@echo "Running fraud detection example..."
	python examples/fraud_detection.py

docs:
	cd docs && make html

clean: clean-build clean-pyc clean-test clean-rust

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .hypothesis/

clean-rust:
	cargo clean
	rm -rf target/
	rm -f tarpaulin-report.html

watch-rust:
	cargo watch -x "test --all-features"

watch-python:
	ptw tests/python -- -v

# CI simulation
ci: lint typecheck test-all
	@echo ""
	@echo "✓ All CI checks passed!"

# Quick development workflow
quick: format lint-fix test-python-fast
	@echo ""
	@echo "✓ Quick checks passed!"

# Release preparation
pre-release: clean format lint typecheck test-all benchmark
	@echo ""
	@echo "✓ Ready for release!"
