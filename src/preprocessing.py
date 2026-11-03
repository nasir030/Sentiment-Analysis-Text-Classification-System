"""Text normalization helpers shared by training and inference."""

from __future__ import annotations

import re
from typing import Any, Iterable

import pandas as pd
from nltk.tokenize import TreebankWordTokenizer

_HTML_TAG = re.compile(r"<[^>]*>")
_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_NON_WORD = re.compile(r"[^a-z0-9\s']", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_TOKENIZER = TreebankWordTokenizer()
_NEGATIONS = {"no", "not", "nor", "never"}


def clean_text(text: Any) -> str:
    """Return normalized text while preserving sentiment-bearing negations."""
    if text is None or pd.isna(text):
        return ""
    cleaned = str(text).lower()
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = _URL.sub(" ", cleaned)
    cleaned = _NON_WORD.sub(" ", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def tokenize_text(
    text: Any,
    *,
    remove_stopwords: bool = False,
    stopwords: Iterable[str] | None = None,
) -> list[str]:
    """Tokenize normalized text, optionally excluding stopwords but not negations."""
    tokens = _TOKENIZER.tokenize(clean_text(text))
    if not remove_stopwords:
        return tokens

    excluded = {word.lower() for word in (stopwords or ())} - _NEGATIONS
    return [token for token in tokens if token not in excluded]


def clean_series(texts: pd.Series) -> pd.Series:
    """Clean a pandas text column without changing its index."""
    return texts.map(clean_text)
