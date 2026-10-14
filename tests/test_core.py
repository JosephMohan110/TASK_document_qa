import os
import unittest

from generator import generate_answer, validate_citations
from utils import ConfigurationError, NO_ANSWER, load_config


class CoreBehaviorTests(unittest.TestCase):
    def test_citations_must_be_known_and_present(self):
        self.assertEqual(validate_citations("Claim [S1].", {"S1"}), (True, []))
        self.assertFalse(validate_citations("Claim without evidence.", {"S1"})[0])
        self.assertFalse(validate_citations("Claim [S9].", {"S1"})[0])
        self.assertEqual(validate_citations(NO_ANSWER, set()), (True, []))

    def test_invalid_configuration_is_rejected(self):
        original_top_k = os.environ.get("TOP_K")
        try:
            os.environ["TOP_K"] = "zero"
            with self.assertRaises(ConfigurationError):
                load_config()
        finally:
            if original_top_k is None:
                os.environ.pop("TOP_K", None)
            else:
                os.environ["TOP_K"] = original_top_k

    def test_local_answer_has_retrieved_citation_without_api_key(self):
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result = generate_answer(
                "What is the budget?",
                [{"citation_id": "S1", "text": "The project budget is 50000 dollars.", "source": "notes.txt", "page": None}],
            )
            self.assertTrue(result["valid"])
            self.assertEqual(result["mode"], "local")
            self.assertIn("[S1]", result["answer"])
        finally:
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key


if __name__ == "__main__":
    unittest.main()