import os

from pydantic import BaseModel, field_validator


class TrainArgs(BaseModel):
    data_path: str

    @field_validator("data_path")
    def validate_data_path(cls, data_path: str) -> str:
        if not os.path.exists(data_path):
            raise ValueError(f"The provided data_path doesn't exist: '{data_path}'")

        return data_path


class EvaluateArgs(BaseModel):
    data_path: str


class PredictArgs(BaseModel):
    input: str


class PredictRequest(BaseModel):
    input: str


class PredictResponse(BaseModel):
    prediction: float
    predicted_class: int
