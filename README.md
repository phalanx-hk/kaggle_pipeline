# kaggle_pipeline

## Overview

This is a template for a Kaggle pipeline for GPU instance. The following features are included for accelerating the development:

- :package: <b>Container</b> : Docker is used to create a container for the pipeline. To optimize training on NVIDIA GPUs, it is based on the [PyTorch NGC Container]().
- :package: <b>devcontainer</b> : By using devcontainers, it is possible to ensure reproducibility and develop without polluting the local environment.
 - 📥 <b>Package installer</b> : pip
 > [!CAUTION]
 > I have been using uv, but due to [this problem](https://github.com/astral-sh/uv/issues/2560), the dependency cannot be resolved properly, so I will switch to using pip.
 > Developer of uv plan to fix this issue, So I will switch to using uv in the future.

- :chart_with_upwards_trend: <b>ML Experiment manager</b>: [wandb](https://github.com/wandb/wandb) is used, but anything(e.g., MLflow and Comet) would be fine.
- :white_check_mark: <b>Code lint/format</b> : [ruff](https://github.com/astral-sh/ruff) is used for both lint and format.
- :white_check_mark: <b>Type check</b> : [mypy](https://github.com/python/mypy)
- :pencil: <b>Test</b> : [pytest]()

## Prerequirements

- Docker >= 20.10.13 (for using composeV2)
- [NVIDIA GPU Driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

## USAGE

### install just

```bash
INSTALL_DIR=~/.local/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to $INSTALL_DIR
export PATH="$PATH:$INSTALL_DIR"
# check the command which is used in development.
just --list
```

### Build and run the container in detached mode

```bash
just devcontainer-up
```

### Attach container to the vscode

Attach the container to the vscode in Docker extension.
`Docker extension` -> `CONTAINERS` -> `kaggle_pipeline.kaggle_pipeline-kaggle` -> `Attach Visual Studio Code`

<img src="./imgs/attach_container_to_the_vscode.jpg" width="70%" />
