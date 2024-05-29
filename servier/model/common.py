from typing import Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from servier.schemas import ModelName


def chem_berta_tokenize(
    tokenizer: PreTrainedTokenizerBase, samples: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenized_samples = tokenizer(
        samples.tolist(), padding="max_length", truncation=True, return_tensors="pt"
    )

    return tokenized_samples["input_ids"], tokenized_samples["attention_mask"]


def make_tokenizer(model_name: ModelName) -> Optional[PreTrainedTokenizerBase]:
    if model_name == ModelName.SIMPLE_NN:
        return None

    if model_name == ModelName.CHEM_BERTA:
        return AutoTokenizer.from_pretrained(model_name.to_huggingface_model_name())

    raise NotImplementedError


def do_simple_nn_inference(
    model: torch.nn,
    device: torch.device,
    features: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32).to(device)
        outputs = model(inputs)
        predictions = outputs.squeeze().cpu().numpy()
        predicted_classes = (predictions >= threshold).astype(int)

    return predictions, predicted_classes


def do_chem_berta_inference(
    model: torch.nn,
    device: torch.device,
    features: np.ndarray,
    tokenizer: Optional[PreTrainedTokenizerBase],
):
    if tokenizer is None:
        raise RuntimeError("A tokenizer should be specified")

    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = chem_berta_tokenize(
            tokenizer=tokenizer, samples=features
        )
        raw_predictions = (
            model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
            )
            .logits.cpu()
            .numpy()
        )

    predictions, predicted_classes = np.max(raw_predictions, axis=1), np.argmax(
        raw_predictions, axis=1
    )

    return predictions, predicted_classes


def do_inference(
    model_name: ModelName,
    device: torch.device,
    model: torch.nn,
    features: np.ndarray,
    tokenizer: Optional[PreTrainedTokenizerBase],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if model_name == ModelName.SIMPLE_NN:
        return do_simple_nn_inference(
            model=model, device=device, features=features, threshold=threshold
        )

    if model_name == ModelName.CHEM_BERTA:
        return do_chem_berta_inference(
            model=model, device=device, features=features, tokenizer=tokenizer
        )

    raise NotImplementedError
