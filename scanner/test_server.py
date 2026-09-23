"""The evaluator must receive one exact slug in a flat JSON response."""

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from .image import decode_image
from .server import create_app


class FakeEngine:
    cards = {"verified-slug": {"slug": "verified-slug", "name": "Проверочное вино",
                                "image_name": "reference.webp", "reference_available": True}}

    def __init__(self, image_path):
        self.image_path = image_path

    def predict(self, content):
        if decode_image(content).size != (20, 30):
            raise ValueError("Unexpected image")
        return {"slug": "verified-slug", "card": self.cards["verified-slug"],
                "visual_similarity": .9, "margin_to_second": .2,
                "shared_reference": False, "alternatives": []}

    def get_card(self, slug):
        return self.cards.get(slug)

    def get_image_path(self, slug):
        return self.image_path if slug in self.cards else None


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.image_path = Path(self.temporary.name) / "reference.webp"
        Image.new("RGB", (20, 30), "blue").save(self.image_path)

    def test_evaluator_contract_accepts_webp_bytes_with_jpg_filename(self):
        with TestClient(create_app(FakeEngine(self.image_path))) as client:
            response = client.post("/v1/eval/predict",
                                   files={"image": ("photo.jpg", self.image_path.read_bytes(), "image/jpeg")})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"slug": "verified-slug"})
            self.assertEqual(client.get("/v1/wines/verified-slug").status_code, 200)
            self.assertEqual(client.get("/v1/wines/verified-slug/image").status_code, 200)

    def test_bad_or_missing_image_returns_clear_error(self):
        with TestClient(create_app(FakeEngine(self.image_path))) as client:
            self.assertEqual(client.post("/v1/eval/predict").status_code, 422)
            self.assertEqual(client.post("/v1/eval/predict",
                             files={"image": ("bad.jpg", b"not an image", "image/jpeg")}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
