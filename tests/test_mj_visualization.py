"""MJ plots retain explicit source grids and controlled, possibly negative MI."""
import importlib.util
import unittest

import torch

from vlm_probing import ProbeResult, decode_mj_codes, visualization as viz


HAS_PLOTS = importlib.util.find_spec("matplotlib") is not None


@unittest.skipUnless(HAS_PLOTS, "plotting extra is optional")
class MJVisualizationTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        self.result = ProbeResult("mj_lens", {
            "code_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]),
            "positions": torch.tensor([[b, p] for b in (0, 1) for p in (3, 4, 5, 6)]),
        }, {"source_modality": "visual", "target_modality": "visual", "layers": [2, 3]})

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_code_grids_preserve_layer_batch_and_source_order(self):
        before = self.result.tensors["code_ids"].clone()
        fig, axes = viz.plot_mj_codebook(self.result, grid=(2, 2), batch_index=1)
        self.assertEqual(len(axes), 2)
        torch.testing.assert_close(torch.as_tensor(axes[0].images[0].get_array()),
                                   torch.tensor([[5, 6], [7, 8]]))
        torch.testing.assert_close(before, self.result.tensors["code_ids"])
        self.assertIn("codebook", fig._suptitle.get_text())

    def test_decoder_receives_full_code_positions_and_no_text_tiling(self):
        seen = []
        def native(codes):
            seen.append(codes.clone())
            return codes.reshape(len(codes), 2, 2, 1).expand(-1, -1, -1, 3).byte()
        images, grids = decode_mj_codes(self.result, grid=(2, 2), decoder=native)
        self.assertEqual(tuple(images.shape), (2, 2, 2, 2, 3))
        torch.testing.assert_close(seen[0], self.result.tensors["code_ids"][0].reshape(2, 4))
        self.assertEqual(tuple(grids.shape), (2, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "grid"):
            viz.plot_mj_codebook(self.result, grid=(1, 3))
        self.result.metadata["source_modality"] = "text"
        with self.assertRaisesRegex(ValueError, "visual-to-visual"):
            viz.plot_mj_codebook(self.result, grid=(2, 2))

    def test_modality_index_preserves_negative_excess_and_nan(self):
        result = ProbeResult("mj_modality_index", {
            "raw_mi": torch.tensor([.2, float("nan")]),
            "rotated_mi": torch.tensor([.6, float("nan")]),
            "excess_mi": torch.tensor([-.4, float("nan")]),
        })
        ax = viz.plot_mj_modality_index(result, labels=["patch 0", "undefined"])
        self.assertAlmostEqual(ax.patches[4].get_height(), -.4, places=6)
        self.assertLess(ax.get_ylim()[0], 0)
        self.assertTrue(torch.isnan(result.tensors["excess_mi"][1]))
        with self.assertRaisesRegex(ValueError, "labels"):
            viz.plot_mj_modality_index(result, labels=["missing"])


if __name__ == "__main__":
    unittest.main()
