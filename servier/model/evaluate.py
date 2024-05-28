import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import classification_report

from servier.model.predict import do_inference
from servier.schemas import EvaluateArgs
from servier.utils.data_preprocessing import prepare_datasets


def evaluate(config: DictConfig, args: EvaluateArgs):
    (
        _,
        _,
        _,
        _,
        test_features,
        test_target,
    ) = prepare_datasets(
        data_path=args.data_path,
        config_data_preprocessing=config.data_preprocessing,
        config_feature_extraction=config.feature_extraction,
        apply_fingerprint=True,
    )

    model = torch.load(config.paths.model)

    _, predicted_classes = do_inference(model=model, features=test_features)
    class_report = classification_report(test_target, predicted_classes)

    logger.info(class_report)
