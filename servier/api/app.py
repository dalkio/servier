from functools import lru_cache
from typing import Tuple

import torch
from flask import Flask, Response, jsonify, request
from pydantic import ValidationError

from servier.config import config
from servier.model.predict import prepare_features_and_predict
from servier.schemas import ModelName, PredictRequest, PredictResponse

app = Flask(__name__)


@lru_cache
def load_model(model_name: ModelName, device: torch.device) -> torch.nn:
    model_path = config.paths.model.format(model_name=model_name)

    return torch.load(model_path, map_location=device)


@app.route("/predict", methods=["POST"])
def predict() -> Tuple[Response, int]:
    data = request.json

    try:
        predict_request = PredictRequest.parse_obj(data)
    except ValidationError as exception:
        return jsonify(exception.errors()), 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_name=predict_request.model_name, device=device)
    predictions, predicted_classes = prepare_features_and_predict(
        model_name=predict_request.model_name,
        model=model,
        device=device,
        config=config,
        sample=predict_request.input,
        apply_fingerprint=predict_request.model_name == ModelName.SIMPLE_NN,
    )

    prediction = predictions.item()
    predicted_class = predicted_classes.item()

    response = PredictResponse(prediction=prediction, predicted_class=predicted_class)

    return jsonify(response.dict()), 200


if __name__ == "__main__":
    app.run(debug=True)
