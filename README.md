# Servier technical test: simple molecule classification project

## Prerequisites

- Python `>=3.9` already set up
- Docker already installed and running

## Project architecture overview

```
│── Dockerfile
│── LICENSE
│── MANIFEST.in
│── README.md
│── build
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
│   └── config.yaml
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

Synchronize the dependencies versions from `pyproject.toml` to `setup.py` if the former has changed.

Afterward, you can install the package `servier` as follows:
- `python setup.py install`

## Configuration

The [config.yaml](/servier/config.yaml) file contains several preprocessing and training hyperparameters and can be updated accordingly.

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

- `gunicorn --workers 4 --timeout 300 --bind 0.0.0.0:8000 servier.api.app:app`

### Request example using curl for chem_berta model

- `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"input": "Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C", "model_name": "chem_berta"}'`


## Run locally using Docker

### Rebuild the image locally

- `docker build --platform=linux/amd64 -t servier:latest .`
- `docker run -p 8000:8000 servier`

### Use an already built and pushed image

Use the docker image hosted here: https://hub.docker.com/r/neuronest/servier and run it:

- `docker run -p 8000:8000 neuronest/servier`

## Evaluation performances

### Model: simple_nn

Reproducible results with:
- `python -m servier.cli evaluate --data-path data/dataset_single.csv --model-name simple_nn` <br>
or
- `servier evaluate --data-path data/dataset_single.csv --model-name simple_nn` <br>
or
- `docker run -v $(pwd)/data:/app/data neuronest/servier python -m servier.main evaluate --data-path data/dataset_single.csv --model-name simple_nn`

|           | precision | recall | f1-score | support       |
|-----------|-----------|--------|----------|---------------| 
| 0         | 0.26      | 0.41   | 0.31     | 81            |
| 1         | 0.87      | 0.77   | 0.82     | 419           |
| accuracy  |           |        | 0.71     | 500           | 
| macro avg | 0.56      | 0.59   | 0.57     | 500           | 
| accuracy  | 0.77      | 0.71   | 0.74     | 500           |

### Model: chem_berta

Can take a while if running on CPU.

Reproducible results with:
- `python -m servier.cli evaluate --data-path data/dataset_single.csv --model-name chem_berta` <br>
or
- `servier evaluate --data-path data/dataset_single.csv --model-name chem_berta` <br>
or
- `docker run -v $(pwd)/data:/app/data neuronest/servier python -m servier.main evaluate --data-path data/dataset_single.csv --model-name chem_berta`

|           | precision | recall | f1-score | support       |
|-----------|-----------|--------|----------|---------------| 
| 0         | 0.24      | 0.51   | 0.33     | 81            |
| 1         | 0.88      | 0.69   | 0.77     | 419           |
| accuracy  |           |        | 0.66     | 500           | 
| macro avg | 0.56      | 0.60   | 0.55     | 500           | 
| accuracy  | 0.77      | 0.66   | 0.70     | 500           |

## Discussions and disclaimers

- The dataset is quite unbalanced (~82% of 1s and ~18% of 0s). For this reason, a weighted random sampler (from `torch.utils.data.WeightedRandomSampler`) has been used to counterbalance the label apparition during training. <br>
In addition, the f1-score should be preferred, especially the macro avg f1-score.
- Early stopping has been used with some fixed patience as soon as the validation loss no longer decreases.
- The scores seem to be quite low: not much time was allocated to optimizing model architectures as well as hyperparameters. <br>
It is possible that the number of samples (5000) is insufficient for the model to generalize well. <br>
It would have been interesting to implement simpler models like logistic regression or KNN and thus establish a baseline to iterate upon.
- A factory pattern refactoring should be helpful, especially if other models are added.
- Finally, some components are missing, like unit tests, CI/CD, and proper model serving, but it was outside the exercise's scope.
