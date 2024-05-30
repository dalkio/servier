# Servier technical test: simple molecule classification project

## Prerequisites

- Python `>=3.9` already setup
- Docker already installed and running

## Project architecture overview

```
│── Dockerfile
│── LICENSE
│── README.md
│── build
│── conf
│   └── config.yaml
│── data
│   └── dataset_multi.csv
│   └── dataset_single.csv
│── doc
│   └── description.pdf
│── models
│   └── <TRAINED MODELS HERE>
│── poetry.lock
│── pyproject.toml
│── servier
│   └── api
│       └── app.py
│   └── cli.py
│   └── config.py
│   └── constants.py
│   └── main.py
│   └── model
│       └── common.py
│       └── evaluate.py
│       └── models.py
│       └── predict.py
│       └── train.py
│   └── schemas.py
│   └── utils
│       └── feature_extractor.py
│       └── data_preprocessing.py
│── setup.py
```

## Setup local environment

- `curl -sSL https://install.python-poetry.org | python -`
- `poetry shell`
- `poetry install --with dev --no-root`

## Package installation

Use `poetry2setup` to automatically generate a `setup.py` installer from the `pyproject.toml`:
- `poetry2setup > setup.py`

Afterwards, you can install the package `servier` as follows:
- `python setup.py install`

## Configuration

The [config.yaml](/conf/config.yaml) contains several preprocessing and training hyperparameters, and can be updated accordingly.

## Models description

Two different models are proposed here:
- The first one, called `simple_nn`, takes the extracted features of a molecule as input and predict the P1 property.
This first approach uses a molecule fingerprint preprocessing step with the help of the library `rdkit`. <br>
The architecture network is essentially a multilayer perceptron with a sigmoid function as final activation layer to perform binary classification. Dropout layers are used to mitigate the overfitting issue.
- The second one, called `chem_berta`, leverages transfer learning with the following BERT pretrained model: https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1. <br>
This model directly process raw chemical SMILES strings and performs binary classification on it.
We finetune the n last layers of the network, namely the classification head, on our own dataset.

## Run locally

### Train a model

- `python -m servier.cli train --data-path data/dataset_single.csv --model-name {simple_nn, chem_berta}`

### Evaluate a model of a seeded test dataset split

- `python -m servier.cli evaluate --data-path data/dataset_single.csv --model-name {simple_nn, chem_berta}`

### Make a prediction using a trained model

For instance, for a molecule having the SMILES `Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C`:

- `python -m servier.cli predict --model-name {simple_nn, chem_berta} --input Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C`

## Run locally using the web server

### Development server

- `python -m servier.api.app`

### Gunicorn server

- `gunicorn --workers 4 --timeout 300 --bind 0.0.0.0:5000 servier.api.app:app`

### Request example using curl for chem_berta model

- `curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"input": "Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C", "model_name": "chem_berta"}'`


## Run locally using Docker

### Rebuild the image locally

- `docker build --platform=linux/amd64 -t servier:latest .`
- `docker run -p 8000:8000 servier`

### Use an already built and pushed image

Use the docker image hosted here: https://hub.docker.com/r/neuronest/servier and run it:

- `docker run -p 8000:8000 neuronest/servier`
