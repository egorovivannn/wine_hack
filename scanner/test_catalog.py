"""Catalog identity and reference-integrity checks."""

import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from .catalog import REQUIRED_COLUMNS, build_manifest, load_cards


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.csv_path = root / "catalog.csv"
        self.images_dir = root / "images"
        self.images_dir.mkdir()
        Image.new("RGB", (20, 30), "red").save(self.images_dir / "shared.png")

    def write_rows(self, rows):
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def row(slug, filename="shared.png", valid="True", name="Wine"):
        return dict(vine_name=name, category="Белое", color="Соломенный", region="Крым",
                    grape_type="Алиготе", desc="Описание", winery="Винодельня",
                    slug=slug, fname=filename, is_valid_img=valid)

    def test_deduplicates_rows_but_keeps_distinct_slugs_sharing_one_image(self):
        first = self.row("wine-2024")
        self.write_rows([first, first, self.row("wine-2025"),
                         self.row("unknown", "missing.png", "False")])
        manifest = build_manifest(self.csv_path, self.images_dir)
        self.assertEqual(manifest["counts"], dict(unique_cards=3, indexed_cards=2,
                         reference_images=1, shared_reference_images=1, excluded_cards=1))
        self.assertEqual(manifest["images"][0]["slugs"], ["wine-2024", "wine-2025"])
        self.assertEqual(manifest["excluded_slugs"], ["unknown"])
        self.assertEqual(manifest["source_rows"], 4)

    def test_conflicting_duplicate_slug_is_rejected(self):
        self.write_rows([self.row("same"), self.row("same", name="Other")])
        with self.assertRaisesRegex(ValueError, "Conflicting rows"):
            load_cards(self.csv_path)

    def test_missing_valid_reference_is_rejected(self):
        self.write_rows([self.row("wine", "missing.png")])
        with self.assertRaises(FileNotFoundError):
            build_manifest(self.csv_path, self.images_dir)

    def test_image_path_cannot_escape_reference_directory(self):
        self.write_rows([self.row("wine", "../escape.png")])
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            load_cards(self.csv_path)


if __name__ == "__main__":
    unittest.main()
