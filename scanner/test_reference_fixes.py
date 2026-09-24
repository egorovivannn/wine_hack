"""Check that catalog repairs are explicit and pinned to reviewed source bytes."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from .catalog import sha256_file, write_manifest
from .reference_fixes import ARCHIVE_PREFIX, build_corrected_manifest


class ReferenceFixTests(unittest.TestCase):
    def test_reassigns_only_reviewed_slug_and_keeps_baseline_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (20, 20), "red").save(images / "wrong.png")
            Image.new("RGB", (20, 20), "blue").save(images / "right.png")
            baseline = root / "baseline.json"
            source = {
                "schema_version": 1,
                "cards": [
                    {
                        "slug": "fix",
                        "image_name": "wrong.png",
                        "reference_available": True,
                    },
                    {
                        "slug": "keep",
                        "image_name": "wrong.png",
                        "reference_available": True,
                    },
                ],
                "images": [
                    {
                        "filename": "wrong.png",
                        "sha256": sha256_file(images / "wrong.png"),
                        "slugs": ["fix", "keep"],
                    }
                ],
                "excluded_slugs": [],
            }
            write_manifest(source, baseline)
            corrections = root / "fixes.json"
            spec = {
                "baseline_manifest_sha256": sha256_file(baseline),
                "fixes": [
                    {
                        "slug": "fix",
                        "from_image": "wrong.png",
                        "target_image": "right.png",
                        "target_sha256": sha256_file(images / "right.png"),
                        "archive_member": ARCHIVE_PREFIX + "right.png",
                    },
                ],
            }
            corrections.write_text(json.dumps(spec), encoding="utf-8")
            corrected = build_corrected_manifest(
                baseline, corrections, images, root / "unused.rar"
            )
            self.assertEqual(
                [card["image_name"] for card in corrected["cards"]],
                ["right.png", "wrong.png"],
            )
            self.assertEqual(corrected["counts"]["reference_images"], 2)
            self.assertEqual(json.loads(baseline.read_text()), source)
            spec["fixes"][0]["target_sha256"] = "0" * 64
            corrections.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_corrected_manifest(
                    baseline, corrections, images, root / "unused.rar"
                )


if __name__ == "__main__":
    unittest.main()
