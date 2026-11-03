"""Command-line entry point for reproducible model training."""

from __future__ import annotations

import argparse

from src.model_training import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Number of labeled reviews to use (default: 10000; use 0 for all).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    sample_size = None if args.sample_size == 0 else args.sample_size
    result = train_and_save(
        sample_size=sample_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    evaluation = result["evaluation"]
    print(f"Selected model: {evaluation['selected_model']}")
    print(f"Training rows: {evaluation['train_size']}")
    print(f"Test rows: {evaluation['test_size']}")
    print(f"Saved artifacts to: {result['model_path'].parent.resolve()}")
    for name, metrics in evaluation["model_comparison"].items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
