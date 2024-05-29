import os
from enum import Enum

from pydantic import BaseModel, field_validator


class ModelName(str, Enum):
    SIMPLE_NN = "simple_nn"
    CHEM_BERTA = "chem_berta"

    def to_huggingface_model_name(self) -> str:
        if self == self.CHEM_BERTA:
            return "seyonec/ChemBERTa-zinc-base-v1"

        raise NotImplementedError


class TrainArgs(BaseModel):
    data_path: str
    model_name: ModelName

    @field_validator("data_path")
    def validate_data_path(cls, data_path: str) -> str:
        if not os.path.exists(data_path):
            raise ValueError(f"The provided data_path doesn't exist: '{data_path}'")

        return data_path


class EvaluateArgs(BaseModel):
    data_path: str
    model_name: ModelName
    batch_size: int = 16


class PredictArgs(BaseModel):
    input: str
    model_name: ModelName


class PredictRequest(BaseModel):
    input: str
    model_name: ModelName


class PredictResponse(BaseModel):
    prediction: float
    predicted_class: int
