# kaggle_pipeline_template

## Overview
This is a template for a Kaggle pipeline for CPU and GPU instance. The following features are included for accelerating the development:
- <b>Container</b>: Docker is used to create a container for the pipeline. To minimize the differences with the Kaggle enviroment, it is based on the [Kaggle Docker image](https://github.com/Kaggle/docker-python).
- <b>Package management</b>: [rye](https://github.com/astral-sh/rye) is used to manage the packages. We use [uv](https://github.com/astral-sh/uv) as backend for rye.
- <b>code lint/format</b>: [ruff](https://github.com/astral-sh/ruff) is used to lint and format the code.
- <b>type check</b>: [mypy](https://github.com/python/mypy) is used to check the type of the code.
- <b>test</b>: [pytest]() is used to test the code.

## Requirements
- Python 3.11
- Docker >= 20.10.13
- rye ([how to install](https://rye-up.com/guide/installation/))
- uv ([how to install](https://github.com/astral-sh/uv?tab=readme-ov-file#getting-started))


