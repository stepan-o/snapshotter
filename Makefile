# Variables
PYTHON ?= python3
UV ?= uv
IMAGE ?= snapshotter:0.1.0
PACKAGE ?= snapshotter

.PHONY: init fmt lint type test run docker-build docker-run

init:
	$(UV) sync --all-groups

fmt:
	$(UV) run ruff format

lint:
	$(UV) run ruff check

type:
	$(UV) run mypy -p $(PACKAGE)

test:
	$(UV) run pytest

# Example: JOB_FILE=job.json make run
run:
	@if [ -n "$$JOB_FILE" ]; then \
		$(UV) run $(PYTHON) -m $(PACKAGE).cli --job-file "$$JOB_FILE" ; \
	else \
		$(UV) run $(PYTHON) -m $(PACKAGE).cli ; \
	fi

docker-build:
	$(UV) lock --all-groups
	docker build -t $(IMAGE) .

# Example: docker run --rm -e SNAPSHOTTER_JOB_JSON='{}' snapshotter:0.1.0
# Or: mount local job file: -v "$$(pwd)":/app -w /app
docker-run:
	@if [ -n "$$SNAPSHOTTER_JOB_JSON" ]; then \
		docker run --rm -e SNAPSHOTTER_JOB_JSON --name snapshotter $(IMAGE) ; \
	elif [ -n "$$JOB_FILE" ]; then \
		docker run --rm -v "$$(pwd)":/app -w /app --name snapshotter $(IMAGE) python -m $(PACKAGE).cli --job-file "$$JOB_FILE" ; \
	else \
		echo "Provide SNAPSHOTTER_JOB_JSON or JOB_FILE to docker-run" && exit 1 ; \
	fi
