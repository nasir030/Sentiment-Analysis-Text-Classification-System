import unittest

from src.preprocessing import clean_text, tokenize_text


class PreprocessingTests(unittest.TestCase):
    def test_removes_markup_and_url_and_preserves_negation(self):
        cleaned = clean_text("<b>NOT good!</b> Visit https://example.com now.")
        self.assertEqual(cleaned, "not good visit now")

    def test_missing_text_becomes_empty(self):
        self.assertEqual(clean_text(None), "")

    def test_optional_stopword_removal_keeps_negations(self):
        tokens = tokenize_text(
            "not a good movie",
            remove_stopwords=True,
            stopwords={"not", "a"},
        )
        self.assertIn("not", tokens)
        self.assertNotIn("a", tokens)


if __name__ == "__main__":
    unittest.main()
