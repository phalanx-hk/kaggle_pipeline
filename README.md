# kaggle_pipeline

## Overview

This is a template for a Kaggle pipeline with GPU instance. The repository is structured as a monorepo to facilitate code sharing and reuse across multiple Kaggle projects. The following features are included for accelerating the development:

- 📁 <b>Directory structure</b> : Organized as a monorepo to facilitate code sharing and reuse across multiple Kaggle projects. This structure makes it easier for AI tools like Cline to reference existing code when implementing new features.
- :package: <b>Container</b> : Docker is used to create a container for the pipeline.
- 📥 <b>Package installer</b> : [uv](https://github.com/astral-sh/uv) is used to speedup package installation.
- 📥 <b>Setup tool for develop environment</b> : [mise](https://mise.jdx.dev/) is used to manage CLI tools.
- :chart_with_upwards_trend: <b>ML Experiment manager</b>: [wandb](https://github.com/wandb/wandb) is used, but anything(e.g., MLflow and Comet) would be fine.
- :white_check_mark: <b>Code lint/format</b> : [ruff](https://github.com/astral-sh/ruff)
- :white_check_mark: <b>Type check</b> : [mypy](https://github.com/python/mypy)
- :pencil: <b>Test</b> : [pytest]()

### Why monorepo?

The main reason for adopting a monorepo structure is to efficiently implement new code while referencing existing code. Particularly when utilizing AI tools (Cline) to partially automate new code implementation, having the existing codebase within the same repository makes it easier for AI to understand the context. This provides the following benefits:

- Increased reusability of existing code
- Maintained consistency across codebase
- AI tools can learn from existing implementation patterns and make more appropriate suggestions
- Easier management of project-wide dependencies

The monorepo structure creates an environment where both developers and AI tools can efficiently implement new features while referencing existing code.

## Directory Structure

Only the main file and directory structure is described

```
kaggle_pipeline/
├── .mise.toml                # mise configuration for CLI tools
└── projects/                 # Directory containing all projects
    └── template/             # Template directory for new projects
        ├── .clineignore      # Cline ignore configuration
        ├── .clinerules       # Cline rules configuration
        ├── data/             # Put the data for the project here
        ├── outputs/          # Put the output of the experiment (e.g., logs, model weights)
        ├── src/              # Put the code that is commonly used in each experiment here (e.g., src/dataset.py, src/model.py)
        |── exp/              # Put the code for each experiment here (e.g., exp001/train.py, exp001/config.yaml)
        └── tests/            # Tests directory

```

## Prerequirements

- Docker >= 20.10.13 (for using composeV2)
- [NVIDIA GPU Driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

## USAGE

### Install CLI tools

```bash
# If you don't have `mise`, install it by running the following command:
curl https://mise.run | sh

# Install the CLI tool
mise install
```

The development environment is assumed to be either a local environment or a devcontainer. Please proceed with the environment that is easy for you to develop.

### For local development

#### Create a new project

`mise run new <project_name>` command creates a new project directory in the `projects` directory. The new project directory is created based on the template directory.

```bash
PROJECT_NAME=<new_project_name>
mise run new ${PROJECT_NAME}

# Check whethe the project directory is created
ls projects
>>> template/ ${PROJECT_NAME}/
```

#### Install dependencies && activate virtualenv

```bash
cd projects/${PROJECT_NAME}
uv sync && source .venv/bin/activate

# Change the working directory to the project directory
code ./ -r
```

### For devcontainer

#### Create a new project

`mise run new <project_name>` command creates a new project directory in the `projects` directory. The new project directory is created based on the template directory.

```bash
PROJECT_NAME=<new_project_name>
mise run new ${PROJECT_NAME}
```

#### Create, start a docker container

`mise run compose-up <project_name>` command creates a new docker container and starts it. The argument `<project_name>` is the name of the project directory.

```bash
mise run compose-up ${PROJRCT_NAME}
```

#### Start devcontainer

`Cmd + Shift + P` -> `Dev Containers: Reopen in Container`
