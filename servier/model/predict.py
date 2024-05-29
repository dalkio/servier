from typing import Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from servier.model.common import do_inference, make_tokenizer
from servier.schemas import ModelName, PredictArgs
from servier.utils.feature_extractor import fingerprint_features


def prepare_features_and_predict(
    model_name: ModelName,
    model: torch.nn,
    device: torch.device,
    config: DictConfig,
    sample: str,
    apply_fingerprint: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if apply_fingerprint is True:
        features = np.array(
            fingerprint_features(
                smile_string=sample,
                radius=config.feature_extraction.radius,
                size=config.feature_extraction.size,
            ).ToList()
        )
    else:
        features = np.array(sample)

    tokenizer = make_tokenizer(model_name=model_name)

    predictions, predicted_classes = do_inference(
        model_name=model_name,
        device=device,
        model=model,
        features=features,
        tokenizer=tokenizer,
    )

    logger.info(
        f"Predicted class: {predicted_classes.item()}, "
        f"with probability={predictions.item() * 100:.2f}%"
    )

    return predictions, predicted_classes


def predict(config: DictConfig, args: PredictArgs) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = config.paths.model.format(model_name=args.model_name)
    model = torch.load(model_path, map_location=device)

    return prepare_features_and_predict(
        model_name=args.model_name,
        model=model,
        device=device,
        config=config,
        sample=args.input,
        apply_fingerprint=args.model_name == ModelName.SIMPLE_NN,
    )
