# NasForge Makefile — local CI shortcuts (venv is assumed to already exist)

setup:
	pip install -e ".[dev]"

lint:
	ruff check nasforge

test:
	python -m pytest -q

demo:
	python -m nasforge.cli demo

# gating: scan for committed secrets, then verify all .py files are valid UTF-8 (no BOM/NUL).
# Chinese characters (e.g. author name 晨星) are explicitly allowed.
gate:
	python scripts/secret_scan.py
	python -c "import pathlib,sys; bad=[str(p) for p in pathlib.Path('nasforge').rglob('*.py') if (lambda b:b[:3]==b'\xef\xbb\xbf' or b'\x00' in b)(p.read_bytes())]; print('[FAIL] bad UTF-8/BOM/NUL:', bad) if bad else None; sys.exit(1 if bad else 0)"

# run the full local CI pipeline in sequence
ci: lint test demo gate

build:
	python -m pip install --upgrade build && python -m build

docker:
	docker build -t nasforge:0.1.0 .

push:
	python scripts/gh_push.py

.PHONY: setup lint test demo gate ci build docker push
