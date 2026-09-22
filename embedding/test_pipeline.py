"""Run: python -m unittest embedding.test_pipeline -v"""
import unittest
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .core import BottleTransform, PKSampler, exact_topk, retrieval_metrics, supervised_contrastive
from .prepare import assign_splits


class PipelineTests(unittest.TestCase):
    def test_letterbox_preserves_full_narrow_image(self):
        pixels = BottleTransform()(Image.new('RGB', (300, 1000), 'white'))
        self.assertEqual(tuple(pixels.shape), (3, 518, 224))
        # White foreground is above normalized intensity 1, the mean-color padding near 0.
        foreground = (pixels > 1).all(0)
        self.assertEqual(int(foreground.any(1).sum()), 518)
        self.assertEqual(int(foreground.any(0).sum()), 155)
        self.assertFalse(bool(foreground[:, 0].any()))
        self.assertFalse(bool(foreground[:, -1].any()))

    def test_split_class_proportion_disjoint_and_singletons(self):
        labels = list(range(40)) + [k for k in range(10) for _ in range(16)]
        frame = pd.DataFrame({'label': labels, 'item_id': [f'{i:04d}' for i in range(200)]})
        split = assign_splits(frame, 42)
        self.assertEqual(split.groupby('split').label.nunique().to_dict(), {'test': 2, 'train': 36, 'val': 2})
        self.assertTrue((split.groupby('label').split.nunique() == 1).all())
        pd.testing.assert_frame_equal(split, assign_splits(frame, 42))
        self.assertTrue((split[split.label >= 10].split == 'train').all())

    def test_impossible_split_rejected(self):
        frame = pd.DataFrame({'label': range(100), 'item_id': range(100)})
        with self.assertRaises(ValueError):
            assign_splits(frame, 42)

    def test_pk_batches_and_real_validation_pairs(self):
        labels = np.array([1, 1, 2, 2, 3, 4])
        sampler = PKSampler(labels, 4, 2, 10, allow_repeat=False)
        for batch in sampler:
            self.assertEqual(len(set(batch)), 4)
            self.assertEqual(set(labels[batch]), {1, 2})
        singleton = PKSampler([1, 2], 4, 2, 1)
        self.assertEqual(sorted(next(iter(singleton))), [0, 0, 1, 1])

    def test_loss_finite_and_prefers_correct_positives(self):
        labels = torch.tensor([0, 0, 1, 1])
        good = torch.tensor([[1., 0], [1., 0], [0, 1.], [0, 1.]], requires_grad=True)
        bad = good.detach()[[0, 2, 1, 3]]
        loss = supervised_contrastive(good, labels)
        self.assertLess(loss.item(), supervised_contrastive(bad, labels).item())
        loss.backward()
        self.assertTrue(torch.isfinite(good.grad).all())
        with self.assertRaises(ValueError):
            supervised_contrastive(good, torch.arange(4))

    def test_chunked_topk_matches_bruteforce(self):
        rng = np.random.default_rng(42)
        q, g = rng.normal(size=(13, 16)).astype('float32'), rng.normal(size=(29, 16)).astype('float32')
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        index, scores = exact_topk(q, g, torch.device('cpu'), query_batch=3, gallery_batch=4)
        wanted = np.argsort(-(q @ g.T), axis=1)[:, :5]
        np.testing.assert_array_equal(index, wanted)
        np.testing.assert_allclose(scores, np.take_along_axis(q @ g.T, wanted, axis=1), atol=1e-6)

    def test_recall_denominator_and_missing_positives(self):
        result = retrieval_metrics([0, 1, 2], [0, 1, 2, 3, 4], np.array([[0, 1, 2, 3, 4], [2, 1, 0, 3, 4], [0, 1, 2, 3, 4]]))
        self.assertAlmostEqual(result['Recall@1'], 1/3)
        self.assertEqual(result['Recall@5'], 1.)
        with self.assertRaises(ValueError):
            retrieval_metrics([9], [1, 2], np.array([[0, 1]]))

    def test_all_vs_all_excludes_self_in_every_chunk(self):
        vectors = np.array([[1., 0], [1., .01], [0, 1.], [.01, 1.]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        indices, _ = exact_topk(vectors, vectors, torch.device('cpu'), query_batch=2,
                                gallery_batch=3, exclude_indices=np.arange(4))
        self.assertFalse((indices == np.arange(4)[:, None]).any())
        metrics = retrieval_metrics([0, 0, 1, 1], [0, 0, 1, 1], indices, np.arange(4))
        self.assertEqual(metrics['Recall@1'], 1.)
        with self.assertRaises(ValueError):
            retrieval_metrics([0], [0, 1], np.array([[1]]), np.array([0]))


if __name__ == '__main__':
    unittest.main()
