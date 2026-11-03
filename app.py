"""Streamlit interface for IMDb-trained customer review sentiment analysis."""

from __future__ import annotations

import hashlib
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.dataset_loader import LABEL_NAMES
from src.model_training import MODEL_DIR, load_artifacts
from src.predict import analyze_dataframe, prediction_column_names, predict_text

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

st.set_page_config(
    page_title="Review Sentiment Studio",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_model_artifacts():
    return load_artifacts()


def _record_sentiment(sentiment: str) -> None:
    st.session_state.setdefault("analysis_history", [])
    st.session_state["analysis_history"].append(sentiment)


def _load_csv(uploaded_file) -> pd.DataFrame:
    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("Upload a CSV file.")
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("The CSV exceeds the 20 MB upload limit.")
    try:
        frame = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Could not read the CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    if len(frame.columns) == 0:
        raise ValueError("The uploaded CSV has no columns.")
    return frame


def _show_metric(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


st.title("💬 Review Sentiment Studio")
st.caption("Explore, evaluate, and classify movie reviews with a locally trained ML model.")
st.session_state.setdefault("analysis_history", [])

try:
    model, vectorizer, evaluation = get_model_artifacts()
except FileNotFoundError as exc:
    model = vectorizer = evaluation = None
    st.warning(str(exc))
except (OSError, ValueError, KeyError) as exc:
    model = vectorizer = evaluation = None
    st.error(f"Could not load trained artifacts: {exc}")

with st.sidebar:
    st.header("Project status")
    if evaluation:
        st.success(f"Ready · {evaluation['selected_model']}")
        st.caption("Binary IMDb sentiment model: Positive / Negative")
    else:
        st.info("Train the model to enable prediction and performance views.")
        st.code("python train.py", language="powershell")
    st.divider()
    st.caption("IMDb provides positive and negative labels only. Neutral is not supported.")

dashboard_tab, predict_tab, bulk_tab, performance_tab, about_tab = st.tabs(
    ["Dashboard", "Single prediction", "Bulk CSV", "Model performance", "About"]
)

with dashboard_tab:
    st.subheader("Analysis dashboard")
    history = st.session_state.get("analysis_history", [])
    positives = history.count("Positive")
    negatives = history.count("Negative")
    total = positives + negatives
    metric_columns = st.columns(4)
    with metric_columns[0]:
        _show_metric("Reviews analyzed", str(total))
    with metric_columns[1]:
        _show_metric("Positive", str(positives))
    with metric_columns[2]:
        _show_metric("Negative", str(negatives))
    with metric_columns[3]:
        _show_metric("Neutral", "Not supported", "IMDb training data has no neutral class.")

    if total:
        chart_frame = pd.DataFrame(
            {"Sentiment": ["Negative", "Positive"], "Reviews": [negatives, positives]}
        )
        st.bar_chart(chart_frame, x="Sentiment", y="Reviews", color="#3b82f6")
    else:
        st.info("Classify one or more reviews to see the live sentiment distribution here.")

    if evaluation:
        st.caption(
            f"Model trained on {evaluation['sample_size']:,} IMDb reviews "
            f"({evaluation['train_size']:,} train / {evaluation['test_size']:,} test). "
            f"Random seed: {evaluation['random_state']}."
        )

with predict_tab:
    st.subheader("Analyze one review")
    review_text = st.text_area(
        "Review text",
        placeholder="I really love this product. Its quality is amazing.",
        height=160,
        disabled=model is None,
    )
    if st.button("Analyze review", type="primary", disabled=model is None):
        if not review_text.strip():
            st.error("Enter a review before analyzing.")
        else:
            try:
                prediction = predict_text(review_text, model, vectorizer)
                _record_sentiment(prediction["sentiment"])
                st.success(f"Predicted sentiment: **{prediction['sentiment']}**")
                if prediction["probabilities"]:
                    st.caption(
                        "These are model probabilities, not guaranteed certainty."
                    )
                    st.bar_chart(
                        pd.DataFrame(
                            {
                                "Sentiment": list(prediction["probabilities"]),
                                "Model probability": list(prediction["probabilities"].values()),
                            }
                        ),
                        x="Sentiment",
                        y="Model probability",
                        color="#10b981",
                    )
            except ValueError as exc:
                st.error(str(exc))

with bulk_tab:
    st.subheader("Classify reviews from CSV")
    st.write("The original columns are preserved. Blank review cells are marked Unclassified.")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        disabled=model is None,
        help="Maximum file size: 20 MB.",
    )
    if uploaded_file is not None:
        try:
            input_frame = _load_csv(uploaded_file)
            if not input_frame.columns.is_unique:
                raise ValueError("CSV column names must be unique.")
            st.write("Preview")
            st.dataframe(input_frame.head(), width="stretch")
            text_column = st.selectbox("Select the review/text column", input_frame.columns)
            upload_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            result_key = (upload_key, text_column)
            if st.button("Analyze CSV", type="primary", disabled=model is None):
                result = analyze_dataframe(input_frame, text_column, model, vectorizer)
                st.session_state["bulk_result"] = (result_key, result)
                sentiment_column, _ = prediction_column_names(input_frame.columns)
                st.session_state["analysis_history"].extend(
                    result.loc[
                        result[sentiment_column].isin(LABEL_NAMES.values()),
                        sentiment_column,
                    ].tolist()
                )
            stored_result = st.session_state.get("bulk_result")
            if stored_result is not None and stored_result[0] == result_key:
                result = stored_result[1]
                sentiment_column, _ = prediction_column_names(input_frame.columns)
                st.dataframe(result, width="stretch")
                counts = result[sentiment_column].value_counts()
                count_columns = st.columns(3)
                count_columns[0].metric("Positive", int(counts.get("Positive", 0)))
                count_columns[1].metric("Negative", int(counts.get("Negative", 0)))
                count_columns[2].metric("Unclassified", int(counts.get("Unclassified", 0)))
                st.download_button(
                    "Download classified CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )
        except (ValueError, OSError) as exc:
            st.error(str(exc))

with performance_tab:
    st.subheader("Model performance")
    if not evaluation:
        st.info("Train the model first to view measured evaluation results.")
    else:
        selected_metrics = evaluation["model_comparison"][evaluation["selected_model"]]
        metric_columns = st.columns(4)
        for column, metric_name in zip(
            metric_columns, ["accuracy", "precision", "recall", "f1"]
        ):
            with column:
                _show_metric(metric_name.title(), f"{selected_metrics[metric_name]:.3f}")

        matrix = selected_metrics["confusion_matrix"]
        figure, axis = plt.subplots(figsize=(5, 4))
        import seaborn as sns

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=axis,
        )
        axis.set(title="Confusion Matrix", xlabel="Predicted label", ylabel="True label")
        st.pyplot(figure)
        plt.close(figure)

        st.markdown("**Classification report**")
        report_frame = pd.DataFrame(selected_metrics["classification_report"]).transpose()
        st.dataframe(report_frame, width="stretch")
        comparison_rows = [
            {
                "Model": name,
                **{metric: values[metric] for metric in ("accuracy", "precision", "recall", "f1")},
            }
            for name, values in evaluation["model_comparison"].items()
        ]
        st.markdown("**Model comparison** (same split and training-only TF-IDF)")
        st.dataframe(
            pd.DataFrame(comparison_rows).set_index("Model"),
            width="stretch",
        )

with about_tab:
    st.subheader("About this project")
    st.markdown(
        """
        **Review Sentiment Studio** is a local, traditional machine-learning
        application for exploring and classifying review text.

        - **Dataset:** [IMDb on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb),
          using its labeled training split.
        - **Labels:** IMDb contains positive and negative labels only. This
          application does not create neutral examples; neutral classification
          requires a separately labeled three-class dataset and retraining.
        - **NLP:** lowercase normalization, HTML/URL removal, conservative
          character cleanup, tokenization, and training-only TF-IDF features.
          Negation terms such as “not”, “never”, and “no” are retained.
        - **Models:** Logistic Regression and Multinomial Naive Bayes; the
          higher test-set F1 model is selected.
        - **Stack:** Python, Pandas, NumPy, scikit-learn, NLTK, Matplotlib,
          Seaborn, Joblib, Hugging Face Datasets, and Streamlit.

        Probability scores describe the classifier's estimates, not guaranteed
        certainty. The IMDb domain is movie reviews, so results on other
        products or services may not generalize.
        """
    )
    st.code("python train.py\nstreamlit run app.py", language="powershell")
