import unittest
from unittest.mock import patch

import pandas as pd

from src.dataset_loader import load_imdb_dataset


class DatasetLoaderTests(unittest.TestCase):
    @patch("src.dataset_loader.load_dataset")
    def test_quality_report_and_duplicate_removal(self, load_dataset):
        source = pd.DataFrame(
            {
                "text": ["<b>Great</b>", "great", "Not good", None, "   "],
                "label": [1, 1, 0, 0, 1],
            }
        )
        load_dataset.return_value.to_pandas.return_value = source

        frame, report = load_imdb_dataset(sample_size=None)

        self.assertEqual(report.source_rows, 5)
        self.assertEqual(report.missing_reviews, 1)
        self.assertEqual(report.duplicate_reviews, 1)
        self.assertEqual(report.rows_after_cleaning, 2)
        self.assertEqual(set(frame["label"]), {0, 1})
        self.assertIn("review", frame.columns)
        self.assertIn("cleaned_review", frame.columns)


if __name__ == "__main__":
    unittest.main()
