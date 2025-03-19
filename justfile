set shell := ["zsh", "-cu"]
# format by ruff
fmt:
    uv run ruff format ${PWD}

ci-fmt:
    uv run ruff format --check ${PWD}

# lint by ruff
lint:
    uv run ruff check --fix ${PWD}

ci-lint:
    uv run ruff check ${PWD}

# check types by mypy
type-check:
    uv run mypy ${PWD}

# run pytest in tests directory
test:
    uv run pytest

# run fmt lint type-check test
all: fmt lint type-check test

# start up all the services in detached mode
compose-up:
    export UID=$(id -u)
    export GID=$(id -g)
    docker compose up -d

# compose down
compose-down:
    docker compose down

# lint Dockerfile by hadolint
hadolint:
    hadolint Dockerfile

# create a new project from template
new project_name:
    mkdir -p {{project_name}}
    cp -r template/* {{project_name}}/ 2>/dev/null || true
    find template -name ".*" -type f -exec cp {} {{project_name}}/ \; 2>/dev/null || true
    sed -i 's/name = "template"/name = "{{project_name}}"/g' {{project_name}}/pyproject.toml
    echo "Project {{project_name}} created from template"
