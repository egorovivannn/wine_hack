"""Checks for external-data identity joins, splits, crops and retrieval scores."""

from __future__ import annotations

import json
import unittest
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from .clean_winesensed import clean, near_pairs
from .external_views import box_crop
from .prepare_norwegian import assign_scene_splits, exact_joins, stable_split
from .train_external_adapter import retrieval
from .train_external_reranker import (
    current_hybrid_proxy,
    lexical_score,
    metrics,
    tokens,
)


class ExternalDataTests(unittest.TestCase):
    def test_exact_join_rejects_ambiguous_and_missing_reference(self):
        categories = [
            {"id": 0, "name": "Unique"},
            {"id": 1, "name": "Same"},
            {"id": 2, "name": "Absent"},
        ]
        products = [
            {"product_name": "Unique", "product_code": "a", "has_images": True},
            {"product_name": "Same", "product_code": "b", "has_images": True},
            {"product_name": "Same", "product_code": "c", "has_images": True},
            {"product_name": "Absent", "product_code": "d", "has_images": False},
        ]
        self.assertEqual({0: products[0]}, exact_joins(categories, products))
        self.assertEqual(stable_split("product-a"), stable_split("product-a"))

    def test_scene_components_are_never_split(self):
        components = Counter({"same-shelf": 70, "other": 20, "val-a": 5, "test-a": 5})
        assignments, sizes = assign_scene_splits(components)
        self.assertEqual("train", assignments["same-shelf"])
        self.assertEqual(set(assignments.values()), {"train", "val", "test"})
        self.assertEqual(sum(sizes.values()), sum(components.values()))
        self.assertEqual(assignments, assign_scene_splits(components)[0])

    def test_near_duplicate_classes_stay_together(self):
        frame = pd.DataFrame(
            [
                dict(
                    vintage_id=v,
                    filename=f"{v}-{i}",
                    sha256=f"{v}-{i}",
                    winery_id="",
                    path=f"{v}-{i}",
                    split="train",
                )
                for v in ("a", "b", "c")
                for i in range(2)
            ]
        )
        hashes = [
            1,
            0xF0F0F0F0F0F0F0F0,
            1,
            0x0F0F0F0F0F0F0F0F,
            0xFFFFFFFFFFFFFFFF,
            0xAAAAAAAAAAAAAAAA,
        ]
        pairs = list(near_pairs(np.array(hashes, dtype=np.uint64), threshold=0))
        self.assertIn((0, 2, 0), pairs)
        cleaned, report = clean(frame, hashes)
        self.assertEqual(
            cleaned[cleaned.vintage_id == "a"].split.iloc[0],
            cleaned[cleaned.vintage_id == "b"].split.iloc[0],
        )
        self.assertGreaterEqual(report["near_duplicate_pairs_cross_class"], 1)
        forced, forced_report = clean(frame, hashes, {"c"})
        self.assertEqual("train", forced[forced.vintage_id == "c"].split.iloc[0])
        self.assertEqual(1, forced_report["previously_evaluated_vintages_forced_train"])

    def test_oracle_crop_retains_product_box(self):
        image = Image.new("RGB", (100, 100), "white")
        image.paste("red", (35, 20, 65, 80))
        crop = box_crop(image, [35, 20, 30, 60])
        self.assertEqual((384, 384), crop.size)
        self.assertEqual((255, 0, 0), crop.getpixel((192, 192)))

    def test_retrieval_uses_eligible_gallery_and_exact_identity(self):
        rows = [
            dict(source="norwegian", role="reference", identity=key, split=split)
            for key, split in (("a", "train"), ("b", "val"), ("c", "test"))
        ]
        rows.append(dict(source="norwegian", role="query", identity="b", split="val"))
        vectors = np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        result = retrieval(rows, vectors, "norwegian", "val")
        self.assertEqual(2, result["gallery"])
        self.assertEqual(1, result["top1"])

    def test_reranker_scores_truth_and_ocr(self):
        example = dict(
            view="a",
            identity="a",
            truth=1,
            candidates=["b", "a"],
            values=np.array(
                [[0.4, 0.5, 0, 0, 0, 0], [0.3, 0.6, 0, 0, 0, 1]], dtype=np.float32
            ),
        )
        self.assertEqual(0, metrics([example], lambda values: values[:, 0])["top1"])
        self.assertEqual(1, metrics([example], lambda values: values[:, 5])["top1"])
        json.dumps(metrics([example], lambda values: values[:, 5]))
        missing = dict(example, truth=-1, identity="unseen")
        measured = metrics([example, missing], lambda values: values[:, 5])
        self.assertEqual(
            (1, 1, 2), (measured["top1"], measured["recall20"], measured["queries"])
        )
        self.assertGreater(
            lexical_score(
                [{"text": "Evergood", "confidence": 0.9}],
                "EVERGOOD CLASSIC",
                {"evergood": 1},
                100,
            ),
            0,
        )
        self.assertIn("evergood", tokens("EVERGOOD CLASSIC"))

    def test_current_hybrid_proxy_uses_same_ocr_trigger(self):
        rows = [
            dict(
                source="norwegian",
                role="reference",
                identity="a",
                split="train",
                product_name="Riesling",
                view="a.jpg",
            ),
            dict(
                source="norwegian",
                role="reference",
                identity="b",
                split="val",
                product_name="Muscat",
                view="b.jpg",
            ),
            dict(
                source="norwegian",
                role="query",
                identity="b",
                split="val",
                product_name="Muscat",
                view="query.jpg",
            ),
        ]
        vectors = np.array([[0.8, 0], [0.79, 0], [1, 0]], dtype=np.float32)
        result = current_hybrid_proxy(
            rows,
            vectors,
            {"query.jpg": [{"text": "Muscat", "confidence": 0.99}]},
            "val",
        )
        self.assertEqual(1, result["ocr_triggered"])
        self.assertEqual(1, result["top1"])


if __name__ == "__main__":
    unittest.main()
