from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from servier.schemas import PredictArgs
from servier.utils.feature_extractor import fingerprint_features


def do_inference(
    model: torch.nn,
    features: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32)
        outputs = model(inputs)
        predictions = outputs.squeeze().numpy()
        predicted_classes = (predictions >= threshold).astype(int)

    return predictions, predicted_classes


def prepare_features_and_predict(
    model: torch.nn, config: DictConfig, sample: str
) -> Tuple[np.ndarray, np.ndarray]:
    features = np.array(
        fingerprint_features(
            smile_string=sample,
            radius=config.feature_extraction.radius,
            size=config.feature_extraction.size,
        ).ToList()
    )

    predictions, predicted_classes = do_inference(model=model, features=features)

    logger.info(
        f"Predicted class: {predicted_classes}, "
        f"with probability={predictions * 100:.2f}%"
    )

    return predictions, predicted_classes


def predict(config: DictConfig, args: PredictArgs) -> Tuple[np.ndarray, np.ndarray]:
    model = torch.load(config.paths.model)

    return prepare_features_and_predict(model=model, config=config, sample=args.input)
