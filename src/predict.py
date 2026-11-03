"""Single-review and bulk sentiment inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.dataset_loader import LABEL_NAMES
from src.preprocessing import clean_text


def prediction_column_names(columns: Any) -> tuple[str, str]:
    """Choose output names that do not overwrite any uploaded CSV columns."""
    used = set(columns)

    def available_name(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_prediction" if suffix == 2 else f"{base}_prediction_{suffix - 1}"
        used.add(candidate)
        return candidate

    return available_name("sentiment"), available_name("sentiment_confidence")


def predict_text(text: Any, model: Any, vectorizer: Any) -> dict[str, Any]:
    """Predict one non-empty review and return label and optional probabilities."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Enter a non-empty review to analyze.")

    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(features)[0]
        probabilities = {
            LABEL_NAMES[int(label)]: float(probability)
            for label, probability in zip(model.classes_, raw_probabilities)
        }

    return {
        "sentiment": LABEL_NAMES.get(prediction, str(prediction)),
        "confidence": max(probabilities.values()) if probabilities else None,
        "probabilities": probabilities,
    }


def analyze_dataframe(
    frame: pd.DataFrame,
    text_column: str,
    model: Any,
    vectorizer: Any,
) -> pd.DataFrame:
    """Preserve all input columns and append sentiment and confidence results."""
    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    if text_column not in frame.columns:
        raise ValueError(f"Text column {text_column!r} was not found.")

    result = frame.copy()
    sentiment_column, confidence_column = prediction_column_names(result.columns)
    sentiments: list[str] = []
    confidences: list[float] = []
    for text in result[text_column]:
        if not clean_text(text):
            sentiments.append("Unclassified")
            confidences.append(float("nan"))
            continue
        prediction = predict_text(text, model, vectorizer)
        sentiments.append(prediction["sentiment"])
        confidence = prediction["confidence"]
        confidences.append(float(confidence) if confidence is not None else float("nan"))

    result[sentiment_column] = sentiments
    result[confidence_column] = np.asarray(confidences, dtype=float)
    return result
