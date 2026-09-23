"""Verified labels must be explicit and tied to immutable photo bytes."""

from pathlib import Path
import tempfile
import unittest

from .labels import read_labels


class ManualLabelTests(unittest.TestCase):
    def test_verified_and_unresolved_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.tsv"
            path.write_text(
                "image_path\timage_sha256\tstatus\tslug\tevidence\n"
                f"known.webp\t{'a' * 64}\tverified\texact-slug\tVisible product name\n"
                f"many.webp\t{'b' * 64}\tambiguous\t\tSeveral wines in frame\n",
                encoding="utf-8")
            labels = read_labels(path)
            self.assertEqual(labels["known.webp"].slug, "exact-slug")
            self.assertEqual(labels["many.webp"].status, "ambiguous")

    def test_unknown_cannot_claim_ground_truth_slug(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.tsv"
            path.write_text(
                "image_path\timage_sha256\tstatus\tslug\tevidence\n"
                f"unknown.webp\t{'a' * 64}\tunknown\tguessed\tMissing from catalog\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
                read_labels(path)


if __name__ == "__main__":
    unittest.main()
