# Review Sentiment Studio

A modular sentiment-analysis application for dataset exploration, model
evaluation, single-review predictions, and bulk CSV classification. It uses
traditional machine learning and runs locally without any paid API.

## Features

- Load the labeled IMDb training split programmatically from Hugging Face.
- Report source rows, missing values, duplicate cleaned reviews, and label
  distribution; retain original and cleaned review text as separate fields.
- Explore sentiment counts, review lengths, frequent terms, word clouds, and
  missing-value counts.
- Compare TF-IDF + Logistic Regression with TF-IDF + Multinomial Naive Bayes.
- Fit TF-IDF on training data only, use a reproducible stratified split, and
  save the selected classifier, vectorizer, and measured evaluation results.
- Classify one review or a CSV while preserving its original columns.
- Provide probabilities as model estimates, not guarantees.

## Dataset and label scope

The project uses [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb),
specifically its labeled `train` split. The default is a reproducible,
stratified 10,000-review sample; the source split contains 25,000 reviews.
Training all available labeled reviews is supported. IMDb has only **positive**
and **negative** labels. The app does not infer or invent neutral labels;
neutral support requires real, separately labeled three-class training data.
Movie-review results may not generalize to other product domains.

## Project structure

```text
app.py                    Streamlit dashboard and user workflows
train.py                  Reproducible model-training CLI
predict.py                Single-review CLI
src/
  dataset_loader.py       Hugging Face loading and data-quality checks
  preprocessing.py       Reusable conservative text cleaning/tokenization
  eda.py                 Matplotlib/Seaborn/WordCloud visualizations
  model_training.py      Model comparison, persistence, artifact loading
  evaluation.py          Metrics, confusion matrix, classification report
  predict.py             Single and bulk inference
tests/                    Focused preprocessing and inference tests
data/README.md             Dataset and labeling notes
models/                    Generated model artifacts (not committed)
notebooks/                 Optional notebook-based workflow
```

## Installation

Python 3.11 or newer is supported. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first training run needs internet access to download the dataset. If the
`py -3.11` launcher is unavailable, create the environment with another
supported Python interpreter (the current project environment may be 3.13).

## Train the model

```powershell
python train.py
```

Options:

```powershell
python train.py --sample-size 5000 --test-size 0.2 --random-state 42
python train.py --sample-size 0
```

The training run reports actual metrics and writes these local files:

- `models/sentiment_model.joblib`
- `models/tfidf_vectorizer.joblib`
- `models/evaluation.json`

The artifacts are ignored by Git because they are reproducible and can be
size-limited for deployment. No performance numbers are published here before
a real training run. `evaluation.json` stores accuracy, precision, recall,
F1, confusion matrix, classification report, and both model comparison
results from the held-out test data.

## Run the application

```powershell
streamlit run app.py
```

The Streamlit app is usable for prediction after model training. Without
artifacts, it gives the explicit command to train rather than showing fake
metrics or placeholder predictions.

Optional command-line prediction:

```powershell
python predict.py "I really love this movie. Its quality is amazing."
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover preprocessing, negation preservation, empty input handling,
probability output, and preservation of original columns during CSV inference.

## Deployment preparation

Streamlit Community Cloud can deploy this repository with `app.py` as the
entry point and `requirements.txt` as its dependency file. Since model files
are intentionally not committed, add a build step in the deployment workflow
or generate/download the trained artifacts before the app starts. For small
Cloud builds, train on the default 10,000-row sample and verify the resulting
artifact size against the current platform limits. Do not treat deployment
as complete until the hosted app has been built and tested.

The dataset is downloaded at training time, not bundled in the app. Avoid
committing datasets, secrets, or large generated model files.

## Screenshots

Screenshots can be added here after running the application locally.

## Future improvements

- Add a genuinely labeled three-class dataset before claiming neutral support.
- Validate performance on customer-review domains, not only IMDb movies.
- Add threshold calibration, language handling, and drift monitoring.
- Optionally compare a locally hosted transformer model without making the
  application dependent on a paid API.
