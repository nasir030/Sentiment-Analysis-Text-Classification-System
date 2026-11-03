"""Loading, validating, and sampling the IMDb dataset from Hugging Face."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

LABEL_NAMES = {0: "Negative", 1: "Positive"}


@dataclass(frozen=True)
class DatasetReport:
    """Data-quality counts recorded before dropping invalid rows."""

    source_rows: int
    missing_reviews: int
    missing_labels: int
    duplicate_reviews: int
    rows_after_cleaning: int


def load_imdb_dataset(
    sample_size: int | None = 10_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, DatasetReport]:
    """Load IMDb's labeled training split and return review, label, and cleaned text.

    IMDb labels only positive and negative reviews. The unlabeled split and
    synthetic neutral labels are intentionally excluded.
    """
    try:
        dataset = load_dataset("stanfordnlp/imdb", split="train")
    except Exception as exc:
        raise RuntimeError(
            "Could not load stanfordnlp/imdb from Hugging Face. Check the "
            "internet connection and dataset access, then try again."
        ) from exc

    frame = dataset.to_pandas()
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"IMDb dataset must contain columns {sorted(required)}.")

    frame = frame.rename(columns={"text": "review"})[["review", "label"]]
    source_rows = len(frame)
    missing_reviews = int(frame["review"].isna().sum())
    missing_labels = int(frame["label"].isna().sum())
    frame = frame.dropna(subset=["review", "label"]).copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["label"]).copy()
    frame["label"] = frame["label"].astype(int)
    if not frame["label"].isin(LABEL_NAMES).all():
        raise ValueError("IMDb training split contains labels outside {0, 1}.")

    from src.preprocessing import clean_series

    frame["review"] = frame["review"].astype(str)
    frame["cleaned_review"] = clean_series(frame["review"])
    duplicate_mask = frame["cleaned_review"].duplicated(keep="first")
    duplicate_reviews = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask & frame["cleaned_review"].ne("")].reset_index(drop=True)
    report = DatasetReport(
        source_rows=source_rows,
        missing_reviews=missing_reviews,
        missing_labels=missing_labels,
        duplicate_reviews=duplicate_reviews,
        rows_after_cleaning=len(frame),
    )

    if frame["label"].nunique() != 2:
        raise ValueError("Training requires both positive and negative IMDb labels.")
    if sample_size is not None:
        if sample_size < 4:
            raise ValueError("sample_size must be at least 4.")
        if sample_size > len(frame):
            raise ValueError(
                f"sample_size ({sample_size}) exceeds available rows ({len(frame)})."
            )
        if sample_size < 2 * int(frame["label"].nunique()):
            raise ValueError("sample_size must leave at least two rows per class.")
        if sample_size < len(frame):
            frame, _ = train_test_split(
                frame,
                train_size=sample_size,
                random_state=random_state,
                stratify=frame["label"],
            )
            frame = frame.sort_index().reset_index(drop=True)

    return frame, report
