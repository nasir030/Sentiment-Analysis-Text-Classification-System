"""Quick command-line prediction using locally trained artifacts."""

from __future__ import annotations

import argparse

from src.model_training import load_artifacts
from src.predict import predict_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", help="Review text to classify.")
    args = parser.parse_args()
    model, vectorizer, _ = load_artifacts()
    result = predict_text(args.review, model, vectorizer)
    print(f"Sentiment: {result['sentiment']}")
    if result["confidence"] is not None:
        print(f"Model probability (not certainty): {result['confidence']:.1%}")
        for label, probability in result["probabilities"].items():
            print(f"  {label}: {probability:.1%}")


if __name__ == "__main__":
    main()
