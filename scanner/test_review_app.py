"""HTTP contract and media allowlist for the manual photo browser."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from .labels import ManualLabel
from .review_app import ReviewData, create_app


class FakeReviewData:
    def __init__(self, root: Path) -> None:
        self.labels = {"one.webp": object()}
        self.counts = {"verified": 1, "unknown": 0, "ambiguous": 0}
        self.provenance = {"labels_sha256": "a" * 64}
        self.photos_dir = root
        self.references_dir = root
        self.allowed_references = {"ref.webp"}
        (root / "one.webp").write_bytes(b"photo")
        (root / "ref.webp").write_bytes(b"reference")
        (root / "thumb.jpg").write_bytes(b"thumb")

    def summary(self, name: str) -> dict:
        return {"name": name, "status": "verified", "slug": "slug-one", "evidence": "label",
                "top1": {"name": "wine", "winery": "winery", "slug": "slug-one", "score": 0.9}}

    def detail(self, name: str) -> dict:
        if name not in self.labels:
            raise KeyError(name)
        return {"name": name, "status": "verified", "candidates": [{"rank": 1}]}

    def thumbnail(self, name: str) -> Path:
        if name not in self.labels:
            raise KeyError(name)
        return self.photos_dir / "thumb.jpg"


class ReviewAppTests(unittest.TestCase):
    def test_list_detail_and_media_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(FakeReviewData(Path(directory)))) as client:
                self.assertEqual(client.get("/health").json()["photos"], 1)
                self.assertEqual(client.get("/api/summary").json()["counts"]["verified"], 1)
                self.assertEqual(client.get("/api/photos").json()["total"], 1)
                self.assertEqual(client.get("/api/photos", params={"q": "wine"}).json()["total"], 1)
                self.assertEqual(client.get("/api/photos", params={"q": "missing"}).json()["total"], 0)
                self.assertEqual(client.get("/api/photos", params={"status": "bad"}).status_code, 400)
                self.assertEqual(client.get("/api/photos/one.webp").json()["name"], "one.webp")
                self.assertEqual(client.get("/api/photos/missing.webp").status_code, 404)
                self.assertEqual(client.get("/media/photos/one.webp").content, b"photo")
                self.assertEqual(client.get("/media/references/ref.webp").content, b"reference")
                self.assertEqual(client.get("/media/thumbs/one.webp").content, b"thumb")
                self.assertEqual(client.get("/media/photos/missing.webp").status_code, 404)
                self.assertEqual(client.get("/media/references/missing.webp").status_code, 404)
                self.assertEqual(client.get("/media/thumbs/missing.webp").status_code, 404)
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.get("/review.js").status_code, 200)
                self.assertEqual(client.get("/review.css").status_code, 200)


    def test_score_gaps_keep_raw_scores_and_verified_rank(self) -> None:
        data = ReviewData.__new__(ReviewData)
        data.labels = {"one.webp": ManualLabel("a" * 64, "verified", "s2", "visible label")}
        candidates = [{"rank": i + 1, "slug": f"s{i}", "filename": f"f{i}.webp",
                       "score": round(0.84 - i * 0.001, 4), "name": f"wine {i}",
                       "winery": "winery", "grape": "grape"} for i in range(20)]
        data.visual = {"one.webp": {"candidates": candidates}}
        data.ocr = {"one.webp": {"lines": []}}
        data.cards = {f"s{i}": {"name": f"wine {i}", "winery": "winery",
                       "category": "red", "color": "red", "region": "region",
                       "grape": "grape", "description": "description", "image_name": f"f{i}.webp"}
                      for i in range(20)}
        data.reference_hashes = {f"f{i}.webp": "b" * 64 for i in range(20)}
        data.allowed_references = set(data.reference_hashes)
        summary = data.summary("one.webp")
        self.assertEqual(summary["top1"]["score"], 0.84)
        self.assertEqual(summary["gap_first_second"], 0.001)
        self.assertEqual(summary["spread_top20"], 0.019)
        self.assertEqual(summary["verified_rank"], 3)
        self.assertEqual(data.detail("one.webp")["candidates"][2]["gap_to_first"], 0.002)


if __name__ == "__main__":
    unittest.main()
