set shell := ["zsh", "-cu"]

# format by ruff
fmt:
	ruff format .

# lint by ruff
lint:
	ruff check --fix .

# check types by mypy
type-check:
	mypy .

# run pytest in tests directory
test:
	pytest

# run fmt lint type-check test
all: fmt lint type-check test

# start up all the services in detached mode
compose-up:
	export UID=$(id -u)
	export GID=$(id -g)
	docker compose up -d
