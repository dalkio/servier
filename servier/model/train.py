import os.path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from servier.model.predict import do_inference
from servier.schemas import TrainArgs
from servier.utils.data_preprocessing import prepare_datasets


class SimpleNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        first_hidden_size: int,
        second_hidden_size: int,
        dropout: float,
    ):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, first_hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(first_hidden_size, second_hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(second_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out


def make_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    class_sample_counts = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
    )
    weights = 1.0 / class_sample_counts
    samples_weights = weights[y_train]
    sampler = WeightedRandomSampler(
        weights=samples_weights, num_samples=len(samples_weights), replacement=True
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=sampler
    )

    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, validation_loader


def run_epochs(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_loader, validation_loader = make_data_loaders(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    best_validation_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    best_model_state = model.state_dict()

    for epoch in range(max_epochs):
        if early_stop is True:
            logger.info("Early stopping triggered")
            break

        model.train()

        running_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(X_train)

        model.eval()
        running_validation_loss = 0.0
        with torch.no_grad():
            for validation_inputs, validation_labels in validation_loader:
                validation_outputs = model(validation_inputs)
                validation_loss = criterion(validation_outputs, validation_labels)
                running_validation_loss += (
                    validation_loss.item() * validation_inputs.size(0)
                )

        validation_loss = running_validation_loss / len(X_validation)

        logger.info(
            f"Epoch {epoch+1}/{max_epochs}, "
            f"Training loss: {train_loss}, "
            f"Validation loss: {validation_loss}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
                logger.info("Restoring best model weights")
                model.load_state_dict(best_model_state)


def train(config: DictConfig, args: TrainArgs):
    (train_features, train_target, validation_features, validation_target, _, _) = (
        prepare_datasets(
            data_path=args.data_path,
            config_data_preprocessing=config.data_preprocessing,
            config_feature_extraction=config.feature_extraction,
            apply_fingerprint=True,
        )
    )

    model = SimpleNN(
        input_size=train_features.shape[1],
        first_hidden_size=config.training.first_hidden_size,
        second_hidden_size=config.training.second_hidden_size,
        dropout=config.training.dropout,
    )
    run_epochs(
        model=model,
        X_train=train_features,
        y_train=train_target,
        X_validation=validation_features,
        y_validation=validation_target,
        max_epochs=config.training.max_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        patience=config.training.patience,
    )

    os.makedirs(os.path.dirname(config.paths.model), exist_ok=True)
    torch.save(model, config.paths.model)

    _, predicted_classes = do_inference(model=model, features=validation_features)
    class_report = classification_report(validation_target, predicted_classes)

    logger.info(class_report)
