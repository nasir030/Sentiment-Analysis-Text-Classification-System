import unittest

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.predict import analyze_dataframe, predict_text


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts = [
            "wonderful brilliant movie",
            "excellent and enjoyable",
            "terrible boring movie",
            "awful and disappointing",
        ]
        labels = [1, 1, 0, 0]
        cls.vectorizer = TfidfVectorizer()
        features = cls.vectorizer.fit_transform(texts)
        cls.model = LogisticRegression(random_state=42).fit(features, labels)

    def test_prediction_includes_probabilities(self):
        result = predict_text("wonderful enjoyable", self.model, self.vectorizer)
        self.assertIn(result["sentiment"], {"Positive", "Negative"})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)

    def test_empty_single_text_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_text("  ", self.model, self.vectorizer)

    def test_bulk_preserves_columns_and_marks_blank_rows(self):
        source = pd.DataFrame({"review": ["excellent", None], "id": [1, 2]})
        result = analyze_dataframe(source, "review", self.model, self.vectorizer)
        self.assertEqual(result["id"].tolist(), [1, 2])
        self.assertEqual(result.loc[1, "sentiment"], "Unclassified")
        self.assertTrue(pd.isna(result.loc[1, "sentiment_confidence"]))

    def test_bulk_does_not_overwrite_existing_prediction_columns(self):
        source = pd.DataFrame(
            {
                "review": ["excellent"],
                "sentiment": ["customer-provided"],
                "sentiment_confidence": [0.25],
            }
        )
        result = analyze_dataframe(source, "review", self.model, self.vectorizer)
        self.assertEqual(result.loc[0, "sentiment"], "customer-provided")
        self.assertEqual(result.loc[0, "sentiment_confidence"], 0.25)
        self.assertIn("sentiment_prediction", result.columns)
        self.assertIn("sentiment_confidence_prediction", result.columns)


if __name__ == "__main__":
    unittest.main()
