"""Verifier fusion and card text, without loading the language model."""

import unittest

import numpy as np

from .verifier import CANDIDATES, VERIFIER_WEIGHT, CandidateVerifier, sweetness
from .vision import Candidate


def card(slug, name, image_name="x.webp", description=""):
    return {"slug": slug, "name": name, "image_name": image_name,
            "description": description, "winery": "W", "category": "Розовое", "grape": "G"}


class FakeVerifier(CandidateVerifier):
    def __init__(self, log_probs):
        self.log_probs = np.log(np.asarray(log_probs, dtype=np.float32))

    def log_probabilities(self, image, candidates, references):
        return self.log_probs[:len(candidates)]


class VerifierTests(unittest.TestCase):
    def test_sweetness_comes_from_file_name_when_card_name_is_identical(self):
        self.assertEqual(sweetness(card("pino-1", "Пино Нуар", "Pino_Muskat_p_suh_1.webp")), "полусухое")
        self.assertEqual(sweetness(card("pino-2", "Пино Нуар", "Pino_Muskat_p_sl_2.webp")), "полусладкое")
        self.assertEqual(sweetness(card("x", "Вино")), "не указана")

    def test_confident_text_evidence_overrides_small_visual_gap(self):
        ranked = [Candidate("semi-sweet", 0.80, "a.webp"), Candidate("semi-dry", 0.79, "b.webp")]
        reordered, fused = FakeVerifier([0.1, 0.9]).rerank(None, ranked, lambda item: None)
        self.assertEqual([item.slug for item in reordered], ["semi-dry", "semi-sweet"])
        self.assertAlmostEqual(fused[0], 0.79 + VERIFIER_WEIGHT * np.log(0.9), places=5)

    def test_weak_text_evidence_keeps_clear_visual_winner(self):
        ranked = [Candidate("a", 0.85, "a.webp"), Candidate("b", 0.78, "b.webp")]
        reordered, _ = FakeVerifier([0.4, 0.6]).rerank(None, ranked, lambda item: None)
        self.assertEqual(reordered[0].slug, "a")

    def test_only_the_first_candidates_are_reranked(self):
        ranked = [Candidate(str(i), 0.9 - i / 100, f"{i}.webp") for i in range(CANDIDATES + 2)]
        reordered, fused = FakeVerifier([0.2] * CANDIDATES).rerank(None, ranked, lambda item: None)
        self.assertEqual(len(fused), CANDIDATES)
        self.assertEqual([item.slug for item in reordered[CANDIDATES:]],
                         [str(i) for i in range(CANDIDATES, CANDIDATES + 2)])


if __name__ == "__main__":
    unittest.main()
