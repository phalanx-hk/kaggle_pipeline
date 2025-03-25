# template

## How to use this template
- テンプレートは直接使わず、ルートディレクトリのREADME.mdを参考に、このディレクトリをコピーして利用してください
- 開発はclineを前提としており、`.clinerules`にclineに守らせたいルールを、`.clineignore`にclineに参照させたくないファイルを記載しています。テンプレートとして仮で記載しているので、開発者の思考に合わせて適宜変更してください。
  - clineの詳しい使い方は公式ドキュメント（[https://docs.cline.bot/](https://docs.cline.bot/)）を参照
- コンペティションの詳細、再現実装の手順などは以下のように記述してください

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

Refer to [README.md](../README.md) for setting up the development environment.

### How to Train/Eval

Write how to train and evaluate the model.
