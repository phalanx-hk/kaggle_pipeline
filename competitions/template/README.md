# template

## What

Write a brief description of the competition here.
competition link: [link](https://www.kaggle.com/c/competition-name)

## Requirements

Write the requirements here.

## Usage

### Download competition dataset

How to download the dataset.

```bash
DATA_DIR=./data/competition-name
$ kaggle competitions download -c competition-name -p ${DATA_DIR}
$ unzip ${DATA_DIR}/competition-name.zip -d ${DATA_DIR}
$ rm ${DATA_DIR}/competition-name.zip
```

### Setup development environment

If you want to develop in a Docker container, run the following command.

```bash
$ compose up -d
$ compose exec template bash
```

If you want to develop on your local machine, run the following command.

```bash
$ uv sync && source .venv/bin/activate
```

### How to Train/Eval
