import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import classification_report

from servier.model.common import do_inference, make_tokenizer
from servier.schemas import EvaluateArgs, ModelName
from servier.utils.data_preprocessing import prepare_datasets


def evaluate(config: DictConfig, args: EvaluateArgs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        apply_fingerprint=args.model_name == ModelName.SIMPLE_NN,
    )

    model_path = config.paths.model.format(model_name=args.model_name)
    model = torch.load(model_path, map_location=device)

    tokenizer = make_tokenizer(model_name=args.model_name)

    predicted_classes = []
    for test_features_chunk in np.array_split(
        test_features, len(test_features) // args.batch_size
    ):
        _, predicted_classes_chunk = do_inference(
            model_name=args.model_name,
            device=device,
            model=model,
            features=test_features_chunk,
            tokenizer=tokenizer,
        )
        predicted_classes.append(predicted_classes_chunk)

    predicted_classes = np.concatenate(predicted_classes)

    class_report = classification_report(test_target, predicted_classes)

    logger.info(class_report)
