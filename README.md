# kaggle_pipeline_template

## Overview

This is a template for a Kaggle pipeline for GPU instance. The following features are included for accelerating the development:

- <b>Container</b>: Docker is used to create a container for the pipeline. To optimize training on NVIDIA GPUs, it is based on the [PyTorch NGC Container]().
- <b>Package management</b>: [rye](https://github.com/astral-sh/rye) is used to manage the packages. We use [uv](https://github.com/astral-sh/uv) as backend for rye.
- <b>code lint/format</b>: [ruff](https://github.com/astral-sh/ruff) is used to lint and format the code.
- <b>type check</b>: [mypy](https://github.com/python/mypy) is used to check the type of the code.
- <b>test</b>: [pytest]() is used to test the code.

## Prerequirements

- Docker >= 20.10.13 (for using composeV2)
- pre-commit
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
