"""Saved predictions can be rescored without changing source images."""

import json
from pathlib import Path
import tempfile
import unittest

from .score import score_predictions


class ScoreTests(unittest.TestCase):
    def test_scores_only_verified_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.tsv"
            predictions = root / "predictions.jsonl"
            labels.write_text(
                "image_path\timage_sha256\tstatus\tslug\tevidence\n"
                f"a.jpg\t{'a' * 64}\tverified\texpected\tLabel is readable\n"
                f"b.jpg\t{'b' * 64}\tunknown\t\tNo exact catalog entry\n",
                encoding="utf-8",
            )
            rows = [
                {"image_path": "a.jpg", "image_sha256": "a" * 64,
                 "predicted_slug": "other", "top5_slugs": ["other", "expected"]},
                {"image_path": "b.jpg", "image_sha256": "b" * 64,
                 "predicted_slug": "anything", "top5_slugs": ["anything"]},
            ]
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                   encoding="utf-8")
            result = score_predictions(predictions, labels)
            self.assertEqual(result["verified_labels"], 1)
            self.assertEqual(result["unknown_labels"], 1)
            self.assertEqual(result["top1_correct"], 0)
            self.assertEqual(result["top5_correct"], 1)
            rows[0]["image_sha256"] = "c" * 64
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                   encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Photo hash differs"):
                score_predictions(predictions, labels)


if __name__ == "__main__":
    unittest.main()
