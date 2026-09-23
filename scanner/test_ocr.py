"""OCR may resolve a close label variant without overturning clear visual evidence."""

import unittest

from .ocr import OCRLine, OCRReranker
from .vision import Candidate


class OCRRankingTests(unittest.TestCase):
    def setUp(self):
        self.reranker = OCRReranker({
            "velvet-riesling": {"slug": "velvet-riesling", "name": "Velvet Season",
                                "winery": "Фанагория", "grape": "Рислинг",
                                "reference_available": True},
            "velvet-muscat": {"slug": "velvet-muscat", "name": "Velvet Season",
                              "winery": "Фанагория", "grape": "Мускат Оттонель",
                              "reference_available": True},
        })
        self.lines = [OCRLine("VELVET", 0.99), OCRLine("MUS CAT", 0.99)]

    def test_distinctive_grape_resolves_close_series(self):
        candidates = [Candidate("velvet-riesling", 0.751, 0.7, 0.8, "r.webp"),
                      Candidate("velvet-muscat", 0.742, 0.7, 0.8, "m.webp")]
        self.assertEqual(self.reranker.rerank(candidates, self.lines)[0].slug,
                         "velvet-muscat")

    def test_clear_visual_winner_does_not_change(self):
        candidates = [Candidate("velvet-riesling", 0.84, 0.8, 0.9, "r.webp"),
                      Candidate("velvet-muscat", 0.76, 0.8, 0.9, "m.webp")]
        self.assertEqual(self.reranker.rerank(candidates, self.lines), candidates)


if __name__ == "__main__":
    unittest.main()
