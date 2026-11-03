"""Matplotlib and Seaborn visualizations for dataset exploration."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.dataset_loader import LABEL_NAMES
from src.preprocessing import tokenize_text

sns.set_theme(style="whitegrid")


def sentiment_distribution(frame: pd.DataFrame):
    """Plot labeled review counts."""
    counts = frame["label"].map(LABEL_NAMES).value_counts().reindex(
        ["Negative", "Positive"], fill_value=0
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=axis)
    axis.set(title="IMDb Review Sentiment Distribution", xlabel="Sentiment", ylabel="Reviews")
    return figure


def review_length_distribution(frame: pd.DataFrame):
    """Plot word-count distribution by ground-truth sentiment."""
    data = frame.assign(
        review_length=frame["cleaned_review"].str.split().str.len(),
        sentiment=frame["label"].map(LABEL_NAMES),
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=data,
        x="review_length",
        hue="sentiment",
        bins=40,
        element="step",
        stat="count",
        common_norm=False,
        ax=axis,
    )
    axis.set(title="Review Length by Sentiment", xlabel="Words per review", ylabel="Reviews")
    return figure


def frequent_words(frame: pd.DataFrame, label: int, limit: int = 20) -> pd.DataFrame:
    """Return most frequent tokens for a label without removing negations."""
    words = Counter()
    for text in frame.loc[frame["label"].eq(label), "cleaned_review"]:
        words.update(tokenize_text(text))
    return pd.DataFrame(words.most_common(limit), columns=["word", "count"])


def frequent_words_plot(frame: pd.DataFrame, label: int, limit: int = 20):
    """Plot frequent tokens for positive or negative reviews."""
    words = frequent_words(frame, label, limit)
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=words, x="count", y="word", color="#3b82f6", ax=axis)
    axis.set(
        title=f"Most Frequent Words — {LABEL_NAMES[label]} Reviews",
        xlabel="Occurrences",
        ylabel="Word",
    )
    return figure


def wordcloud_plot(frame: pd.DataFrame, label: int):
    """Create a reproducible word cloud for one sentiment."""
    text = " ".join(frame.loc[frame["label"].eq(label), "cleaned_review"])
    if not text.strip():
        raise ValueError(f"No {LABEL_NAMES[label].lower()} reviews are available.")
    cloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        random_state=42,
        collocations=False,
    ).generate(text)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.imshow(cloud, interpolation="bilinear")
    axis.axis("off")
    axis.set_title(f"{LABEL_NAMES[label]} Review Word Cloud")
    return figure


def missing_values_plot(missing_values: dict[str, int]):
    """Plot missing-value counts captured by the dataset loader."""
    data = pd.DataFrame(
        {"column": list(missing_values), "missing values": list(missing_values.values())}
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.barplot(data=data, x="column", y="missing values", color="#f59e0b", ax=axis)
    axis.set(title="Missing Values in Source Dataset", xlabel="Field", ylabel="Rows")
    return figure
