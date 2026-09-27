"""Checks for ranking identity and robust image decoding."""

from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from .image import decode_image, query_views, reference_views
from .vision import GalleryIndex


class VisionTests(unittest.TestCase):
    def test_shared_reference_retains_both_slugs(self):
        embeddings = np.array([[[1., 0.], [1., 0.]],
                               [[0., 1.], [0., 1.]]], dtype=np.float32)
        index = GalleryIndex(["shared.webp", "other.webp"], embeddings,
                             [["vintage-2024", "vintage-2025"], ["other"]])
        found = index.search(np.array([[1., 0.], [1., 0.], [1., 0.]], dtype=np.float32), top_k=3)
        self.assertEqual([candidate.slug for candidate in found],
                         ["vintage-2024", "vintage-2025", "other"])
        self.assertEqual(found[0].score, found[1].score)

    def test_each_query_crop_uses_its_best_reference_view(self):
        # Reference A matches crop 1 in view 0 and crop 2 in view 1; B matches only crop 1.
        embeddings = np.array([[[1., 0.], [0., 1.]],
                               [[1., 0.], [1., 0.]]], dtype=np.float32)
        index = GalleryIndex(["a.webp", "b.webp"], embeddings, [["a"], ["b"]])
        found = index.search(np.array([[1., 0.], [0., 1.]], dtype=np.float32), top_k=2)
        self.assertEqual([candidate.slug for candidate in found], ["a", "b"])
        self.assertAlmostEqual(found[0].score, 1.0)
        self.assertAlmostEqual(found[1].score, 0.5)

    def test_bytes_are_decoded_by_content_and_alpha_is_composited(self):
        image = Image.new("RGBA", (20, 40), (255, 0, 0, 0))
        image.putpixel((10, 20), (0, 0, 255, 255))
        encoded = BytesIO()
        image.save(encoded, format="WEBP", lossless=True)
        decoded = decode_image(encoded.getvalue())
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(decoded.getpixel((10, 20)), (0, 0, 255))
        self.assertEqual([im.size for im in query_views(decoded)], [(384, 384)] * 3)
        self.assertEqual([im.size for im in reference_views(decoded)], [(384, 384)] * 4)


if __name__ == "__main__":
    unittest.main()
