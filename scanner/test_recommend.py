"""Analogues and sommelier picks on a small synthetic catalog."""

import unittest

from .recommend import Recommender, is_sparkling, sweetness_level


def card(slug, name, winery, category="Белое", grape="Рислинг", region="Кубань", description=""):
    return {"slug": slug, "name": name, "winery": winery, "category": category, "grape": grape,
            "region": region, "description": description, "image_name": f"{slug}.webp",
            "reference_available": "True"}


FILLER = [card(f"filler-{i}", f"Вино {i}", f"Винодельня {i}", "Красное", "Саперави", "Крым",
               "Ноты сливы и чернослива, табака.") for i in range(12)]
CATALOG = {item["slug"]: item for item in [
    card("riesling-a", "Рислинг сухое", "A", description="Ноты зелёного яблока и лайма. Подать к рыбе."),
    card("riesling-a2", "Рислинг Резерв", "A", description="Ноты зелёного яблока и лайма."),
    card("riesling-b", "Рислинг", "B", description="Зелёное яблоко, лайм, минеральность. Хорош к рыбе."),
    card("chardonnay-c", "Шардоне", "C", grape="Шардоне", region="Крым", description="Дыня, ваниль, сливочное масло."),
    card("rose-d", "Розе", "D", category="Розовое", grape="Рислинг", description="Клубника и лайм."),
    card("brut-e", "Рислинг брют", "E", description="Яблоко, бриошь. К устрицам."),
    card("sweet-f", "Мускат сладкое", "F", grape="Мускат", description="Абрикос и мёд. К десерту."),
    *FILLER,
]}


class RecommendTests(unittest.TestCase):
    def setUp(self):
        self.recommender = Recommender(CATALOG)

    def test_card_attributes(self):
        self.assertEqual(sweetness_level(CATALOG["riesling-a"]), "сухое")
        self.assertEqual(sweetness_level(CATALOG["sweet-f"]), "сладкое")
        self.assertIsNone(sweetness_level(CATALOG["chardonnay-c"]))
        self.assertTrue(is_sparkling(CATALOG["brut-e"]))
        self.assertTrue(is_sparkling(card("x", "Лесная просека", "G", description="Лёгкое игристое брют.")))
        self.assertFalse(is_sparkling(CATALOG["riesling-b"]))

    def test_analogues_come_from_other_wineries_of_the_same_style(self):
        found = self.recommender.analogues("riesling-a")
        slugs = [item["slug"] for item in found]
        self.assertEqual(slugs[0], "riesling-b")
        self.assertNotIn("riesling-a2", slugs)          # same winery
        self.assertNotIn("rose-d", slugs)               # other colour
        self.assertNotIn("brut-e", slugs)               # sparkling vs still
        self.assertIn("тот же сорт: рислинг", found[0]["reasons"])

    def test_sommelier_respects_answers_and_explains(self):
        picks = self.recommender.sommelier(dish="fish", color="white", sweetness="dry")["picks"]
        self.assertTrue(picks)
        self.assertEqual({item["slug"] for item in picks} & {"rose-d", "brut-e", "sweet-f"}, set())
        self.assertTrue(any("рыба" in reason for reason in picks[0]["reasons"]))
        sweet = self.recommender.sommelier(dish="dessert", sweetness="sweet")["picks"]
        self.assertEqual([item["slug"] for item in sweet], ["sweet-f"])
        sparkling = self.recommender.sommelier(color="sparkling")["picks"]
        self.assertEqual([item["slug"] for item in sparkling], ["brut-e"])

    def test_sommelier_never_returns_empty_when_style_exists(self):
        # No rosé in the catalog is recommended for seafood: rank by style and say so.
        result = self.recommender.sommelier(dish="seafood", color="rose")
        self.assertEqual([item["slug"] for item in result["picks"]], ["rose-d"])
        self.assertIn("Прямых рекомендаций", result["note"])
        # No sweet sparkling wine: drop the sweetness filter and explain.
        relaxed = self.recommender.sommelier(color="sparkling", sweetness="sweet")
        self.assertEqual([item["slug"] for item in relaxed["picks"]], ["brut-e"])
        self.assertIn("без учёта сладости", relaxed["note"])

    def test_sommelier_rejects_unknown_answers(self):
        with self.assertRaises(ValueError):
            self.recommender.sommelier(dish="pizza")
        with self.assertRaises(ValueError):
            self.recommender.sommelier(like="missing")


if __name__ == "__main__":
    unittest.main()
