import os.path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import (AutoModelForSequenceClassification, PreTrainedModel,
                          PreTrainedTokenizerBase)

from servier.model.common import chem_berta_tokenize, make_tokenizer
from servier.model.models import SimpleNN
from servier.model.predict import do_inference
from servier.schemas import ModelName, TrainArgs
from servier.utils.data_preprocessing import prepare_datasets


def count_trainable_parameters(model: torch.nn) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def get_config_training(model_name: ModelName, config: DictConfig) -> DictConfig:
    if model_name in (ModelName.SIMPLE_NN, ModelName.CHEM_BERTA):
        return getattr(config.training, model_name.value)

    raise NotImplementedError


def make_model(
    model_name: ModelName, input_size: int, config_training: DictConfig
) -> Union[SimpleNN, PreTrainedModel]:
    if model_name == ModelName.SIMPLE_NN:
        model = SimpleNN(
            input_size=input_size,
            first_hidden_size=config_training.first_hidden_size,
            second_hidden_size=config_training.second_hidden_size,
            dropout=config_training.dropout,
        )

    elif model_name == ModelName.CHEM_BERTA:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name.to_huggingface_model_name(), num_labels=2
        )

        for param in [parameter for parameter in model.parameters()][
            : -config_training.trainable_layers
        ]:
            param.requires_grad = False

    else:
        raise NotImplementedError

    logger.info(f"Number of trainable parameters: {count_trainable_parameters(model)}")

    return model


def make_criterion(model_name: ModelName) -> Union[nn.BCELoss, nn.CrossEntropyLoss]:
    if model_name == ModelName.SIMPLE_NN:
        return nn.BCELoss()

    if model_name == ModelName.CHEM_BERTA:
        return nn.CrossEntropyLoss()

    raise NotImplementedError


def simple_nn_forward(model: torch.nn, batch: List[torch.Tensor]) -> torch.Tensor:
    # batch: [inputs, labels]
    inputs, _ = batch

    return model(inputs)


def chem_berta_forward(model: torch.nn, batch: List[torch.Tensor]) -> torch.Tensor:
    # batch: [inputs, masks, labels]
    inputs, masks, _ = batch

    return model(input_ids=inputs, attention_mask=masks).logits


def make_forward_caller(
    model_name: ModelName,
) -> Callable[[torch.nn, List[Any]], torch.Tensor]:
    if model_name == ModelName.SIMPLE_NN:
        return simple_nn_forward

    if model_name == ModelName.CHEM_BERTA:
        return chem_berta_forward

    raise NotImplementedError


def _make_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    class_sample_counts = np.array(
        [len(np.where(y_train == y_train_i)[0]) for y_train_i in np.unique(y_train)]
    )
    weights = 1.0 / class_sample_counts
    samples_weights = weights[y_train]

    return WeightedRandomSampler(
        weights=samples_weights, num_samples=len(samples_weights), replacement=True
    )


def make_simple_nn_data_loaders(
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

    sampler = _make_sampler(y_train=y_train)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=sampler
    )

    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, validation_loader


def make_chem_berta_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    train_input_ids, train_attention_mask = chem_berta_tokenize(
        tokenizer=tokenizer, samples=X_train
    )
    validation_input_ids, validation_attention_mask = chem_berta_tokenize(
        tokenizer=tokenizer, samples=X_validation
    )

    train_dataset = TensorDataset(
        train_input_ids, train_attention_mask, torch.tensor(y_train)
    )
    validation_dataset = TensorDataset(
        validation_input_ids, validation_attention_mask, torch.tensor(y_validation)
    )

    sampler = _make_sampler(y_train=y_train)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=sampler
    )

    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, validation_loader


def make_data_loaders(
    model_name: ModelName,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    batch_size: int,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    if model_name == ModelName.SIMPLE_NN:
        return make_simple_nn_data_loaders(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            batch_size=batch_size,
        )

    if model_name == ModelName.CHEM_BERTA:
        return make_chem_berta_data_loaders(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )


def run_epochs(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    forward_caller: Callable[[torch.nn, List[Any]], torch.Tensor],
    criterion: Union[nn.BCELoss, nn.CrossEntropyLoss],
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
):
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
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
        processed_samples = 0
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            inputs, labels = batch[0], batch[-1]
            outputs = forward_caller(model, batch)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            processed_samples += train_loader.batch_size
            logger.debug(
                f"Epoch {epoch + 1}/{max_epochs}, "
                f"processed samples: {processed_samples}"
            )

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        running_validation_loss = 0.0
        with torch.no_grad():
            for validation_batch in validation_loader:
                validation_batch = [item.to(device) for item in validation_batch]
                validation_inputs, validation_labels = (
                    validation_batch[0],
                    validation_batch[-1],
                )
                validation_outputs = forward_caller(model, validation_batch)
                validation_loss = criterion(validation_outputs, validation_labels)
                running_validation_loss += (
                    validation_loss.item() * validation_inputs.size(0)
                )

        validation_loss = running_validation_loss / len(validation_loader.dataset)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_training = get_config_training(model_name=args.model_name, config=config)

    (train_features, train_target, validation_features, validation_target, _, _) = (
        prepare_datasets(
            data_path=args.data_path,
            config_data_preprocessing=config.data_preprocessing,
            config_feature_extraction=config.feature_extraction,
            apply_fingerprint=args.model_name == ModelName.SIMPLE_NN,
        )
    )

    model = make_model(
        model_name=args.model_name,
        input_size=config.feature_extraction.size,
        config_training=config_training,
    ).to(device)
    tokenizer = make_tokenizer(model_name=args.model_name)
    train_loader, validation_loader = make_data_loaders(
        model_name=args.model_name,
        X_train=train_features,
        y_train=train_target,
        X_validation=validation_features,
        y_validation=validation_target,
        batch_size=config_training.batch_size,
        tokenizer=tokenizer,
    )

    run_epochs(
        model=model,
        device=device,
        train_loader=train_loader,
        validation_loader=validation_loader,
        forward_caller=make_forward_caller(model_name=args.model_name),
        criterion=make_criterion(model_name=args.model_name),
        max_epochs=config_training.max_epochs,
        learning_rate=config_training.learning_rate,
        weight_decay=config_training.weight_decay,
        patience=config_training.patience,
    )

    model_path = config.paths.model.format(model_name=args.model_name)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)

    _, predicted_classes = do_inference(
        model_name=args.model_name,
        device=device,
        model=model,
        features=validation_features,
        tokenizer=tokenizer,
    )
    class_report = classification_report(validation_target, predicted_classes)

    logger.info(class_report)
