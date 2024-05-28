from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from servier.constants import SEED
from servier.utils.feature_extractor import fingerprint_features


def load_data(data_path: str, sep: str = ",") -> pd.DataFrame:
    return pd.read_csv(data_path, sep=sep)


def split_train_validation_test(
    data: pd.DataFrame, config_data_preprocessing: DictConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert (
        config_data_preprocessing.train_proportion
        + config_data_preprocessing.validation_proportion
        + config_data_preprocessing.test_proportion
        == 1.0
    )

    train_data, validation_test_data = train_test_split(
        data, train_size=config_data_preprocessing.train_proportion, random_state=SEED
    )
    validation_data, test_data = train_test_split(
        validation_test_data,
        train_size=config_data_preprocessing.validation_proportion
        / (
            config_data_preprocessing.validation_proportion
            + config_data_preprocessing.test_proportion
        ),
        random_state=SEED,
    )

    assert set(train_data.index).intersection(validation_data.index) == set()
    assert set(train_data.index).intersection(test_data.index) == set()
    assert set(validation_data.index).intersection(test_data.index) == set()

    assert len(train_data) + len(validation_data) + len(test_data) == len(data)

    return train_data, validation_data, test_data


def split_features_target(
    data: pd.DataFrame,
    config_feature_extraction: DictConfig,
    apply_fingerprint: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if apply_fingerprint is True:
        features = np.stack(
            data["smiles"].map(
                lambda smile: fingerprint_features(
                    smile_string=smile,
                    radius=config_feature_extraction.radius,
                    size=config_feature_extraction.size,
                ).ToList()
            )
        )
    else:
        features = data["smiles"].values

    return features, data["P1"].values


def prepare_datasets(
    data_path: str,
    config_data_preprocessing: DictConfig,
    config_feature_extraction: DictConfig,
    apply_fingerprint: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_data(data_path=data_path)

    train_data, validation_data, test_data = split_train_validation_test(
        data=data, config_data_preprocessing=config_data_preprocessing
    )

    train_features, train_target = split_features_target(
        train_data,
        config_feature_extraction=config_feature_extraction,
        apply_fingerprint=apply_fingerprint,
    )
    validation_features, validation_target = split_features_target(
        validation_data,
        config_feature_extraction=config_feature_extraction,
        apply_fingerprint=apply_fingerprint,
    )
    test_features, test_target = split_features_target(
        test_data,
        config_feature_extraction=config_feature_extraction,
        apply_fingerprint=apply_fingerprint,
    )

    return (
        train_features,
        train_target,
        validation_features,
        validation_target,
        test_features,
        test_target,
    )
