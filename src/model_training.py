"""Train, compare, evaluate, and persist IMDb sentiment classifiers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.dataset_loader import LABEL_NAMES, DatasetReport, load_imdb_dataset
from src.evaluation import evaluate_predictions

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
EVALUATION_PATH = MODEL_DIR / "evaluation.json"


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary_path)
    os.replace(temporary_path, path)


def train_and_save(
    *,
    sample_size: int | None = 10_000,
    test_size: float = 0.2,
    random_state: int = 42,
    model_dir: Path | str = MODEL_DIR,
) -> dict[str, Any]:
    """Train Logistic Regression and Multinomial NB with train-only TF-IDF fit."""
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5.")
    if sample_size is not None:
        minimum_sample = max(
            4,
            math.ceil(2 / test_size),
            math.ceil(2 / (1 - test_size)),
        )
        if sample_size < minimum_sample:
            raise ValueError(
                f"sample_size must be at least {minimum_sample} for this test_size."
            )

    frame, data_report = load_imdb_dataset(sample_size, random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        frame["cleaned_review"],
        frame["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=frame["label"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1_000,
            random_state=random_state,
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
    }
    comparison: dict[str, dict[str, Any]] = {}
    trained_models = {}
    for name, estimator in candidates.items():
        estimator.fit(x_train_features, y_train)
        predictions = estimator.predict(x_test_features)
        comparison[name] = evaluate_predictions(y_test, predictions)
        trained_models[name] = estimator

    selected_name = max(
        comparison,
        key=lambda name: (comparison[name]["f1"], comparison[name]["accuracy"]),
    )
    selected_model = trained_models[selected_name]
    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / MODEL_PATH.name
    vectorizer_path = target_dir / VECTORIZER_PATH.name
    evaluation_path = target_dir / EVALUATION_PATH.name

    _atomic_joblib_dump(selected_model, model_path)
    _atomic_joblib_dump(vectorizer, vectorizer_path)
    evaluation = {
        "dataset": "stanfordnlp/imdb (labeled train split)",
        "label_names": LABEL_NAMES,
        "neutral_supported": False,
        "sample_size": len(frame),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "random_state": random_state,
        "selected_model": selected_name,
        "model_comparison": comparison,
        "data_quality": data_report.__dict__,
        "feature_count": len(vectorizer.get_feature_names_out()),
    }
    temporary_evaluation = evaluation_path.with_suffix(".json.tmp")
    temporary_evaluation.write_text(
        json.dumps(evaluation, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary_evaluation, evaluation_path)

    return {
        "evaluation": evaluation,
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "evaluation_path": evaluation_path,
    }


def load_artifacts(model_dir: Path | str = MODEL_DIR):
    """Load the selected estimator, TF-IDF vectorizer, and evaluation metadata."""
    target_dir = Path(model_dir)
    required_paths = [
        target_dir / MODEL_PATH.name,
        target_dir / VECTORIZER_PATH.name,
        target_dir / EVALUATION_PATH.name,
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Trained artifacts are missing: "
            + ", ".join(path.name for path in missing)
            + ". Run `python train.py` first."
        )
    model = joblib.load(required_paths[0])
    vectorizer = joblib.load(required_paths[1])
    evaluation = json.loads(required_paths[2].read_text(encoding="utf-8"))
    return model, vectorizer, evaluation
