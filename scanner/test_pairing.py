"""Meaningful style and metadata checks for the offline sommelier."""

import unittest

from .pairing import recommend_pairings


class PairingTests(unittest.TestCase):
    def test_red_meat_uses_documented_tannins(self):
        card = {
            "slug": "red",
            "name": "Каберне сухое",
            "category": "Красное",
            "description": "Выраженные танины и ягоды.",
        }
        result = recommend_pairings(card, "red_meat")
        self.assertEqual(result["selected"]["fit"], "хорошо подходит")
        self.assertIn("танин", result["selected"]["reason"])

    def test_sweetness_comes_from_name_not_tasting_notes(self):
        white = {
            "slug": "white",
            "name": "Шардоне сухое",
            "category": "Белое",
            "description": "Аромат сладких яблок, высокая кислотность.",
        }
        result = recommend_pairings(white, "fish")
        self.assertEqual(result["suggestions"][0]["dish"], "fish")
        self.assertIn("кислотност", result["selected"]["reason"])
        sweet = {**white, "slug": "sweet", "name": "Шардоне полусладкое"}
        self.assertEqual(recommend_pairings(sweet)["suggestions"][0]["dish"], "spicy")

    def test_unlisted_pairing_is_cautious(self):
        card = {"slug": "red", "name": "Каберне сухое", "category": "Красное"}
        result = recommend_pairings(card, "fish")
        self.assertEqual(result["selected"]["fit"], "сочетание зависит от блюда")
        self.assertIn("нельзя уверенно оценить", result["selected"]["reason"])

    def test_unknown_dish_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown dish"):
            recommend_pairings({"slug": "red", "category": "Красное"}, "unknown")


if __name__ == "__main__":
    unittest.main()
