from functools import lru_cache
from typing import Tuple

import torch
from flask import Flask, Response, jsonify, request
from pydantic import ValidationError

from servier.config import config
from servier.model.predict import prepare_features_and_predict
from servier.schemas import PredictRequest, PredictResponse

app = Flask(__name__)


@lru_cache
def load_model() -> torch.nn:
    return torch.load(config.paths.model)


@app.route("/predict", methods=["POST"])
def predict() -> Tuple[Response, int]:
    data = request.json

    try:
        predict_request = PredictRequest.parse_obj(data)
    except ValidationError as exception:
        return jsonify(exception.errors()), 400

    model = load_model()
    predictions, predicted_classes = prepare_features_and_predict(
        model=model, config=config, sample=predict_request.input
    )

    prediction = predictions.item()
    predicted_class = predicted_classes.item()

    response = PredictResponse(prediction=prediction, predicted_class=predicted_class)

    return jsonify(response.dict()), 200


if __name__ == "__main__":
    app.run(debug=True)
