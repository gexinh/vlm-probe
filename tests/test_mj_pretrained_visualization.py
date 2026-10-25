"""MJ pretrained plots preserve native coordinates, signs and same-state controls."""
import importlib.util
import unittest

import torch

from vlm_probing import ProbeResult
from vlm_probing.visualization.mj_pretrained import (
    plot_mj_text_comparison, plot_mj_visual_readout,
)


HAS_PLOTS = importlib.util.find_spec("matplotlib") is not None


@unittest.skipUnless(HAS_PLOTS, "plotting extra is optional")
class MJPretrainedVisualizationTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        self.image = torch.full((12, 20, 3), 170, dtype=torch.uint8).numpy()
        self.logits = torch.zeros(2, 8, 5)
        self.logits[1, 4:, 2] = torch.tensor([3., -2., 1., -4.])
        self.logits[1, 4:, 1] = torch.tensor([1., -1., 0., -2.])
        self.positions = torch.tensor([[0, p] for p in (2, 3, 4, 5)] +
                                      [[1, p] for p in (8, 5, 7, 6)])
        self.visual = ProbeResult("mj_lens", {"logits": self.logits, "positions": self.positions},
                                  {"source_modality": "visual", "target_modality": "text", "layers": [2, 7]})

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_visual_layer_batch_and_source_coordinates_preserve_signed_map(self):
        import numpy as np
        from matplotlib.colors import to_rgba
        before = self.logits.clone()
        image_before = self.image.copy()
        ax = plot_mj_visual_readout(self.image, self.visual, positive_token_id=2,
            negative_token_id=1, grid_shape=(2, 2), layer=7, batch_index=1,
            bbox=(2, 3, 8, 5), image_alpha=.65, colorbar=False)
        # Native positions 5,6,7,8 reorder the externally packed source rows.
        np.testing.assert_array_equal(ax.images[1].get_array(), [[-1., -2.], [1., 2.]])
        self.assertEqual(ax.images[0].get_alpha(), .65)
        self.assertEqual(ax.images[1].norm.vcenter, 0)
        self.assertEqual(tuple(ax.patches[0].get_edgecolor()), to_rgba("#ef8a21"))
        torch.testing.assert_close(before, self.logits)
        np.testing.assert_array_equal(self.image, image_before)
        self.assertIn("block input 7", ax.get_title())

    def test_visual_rejects_wrong_stage_layer_grid_and_invalid_values(self):
        base = {"positive_token_id": 2, "grid_shape": (2, 2), "layer": 7, "colorbar": False}
        for kwargs, message in (({"layer": 1}, "block-input"), ({"grid_shape": (1, 2)}, "count"),
                                ({"positive_token_id": 6}, "vocabulary"), ({"batch_index": 2}, "count")):
            with self.assertRaisesRegex(ValueError, message):
                plot_mj_visual_readout(self.image, self.visual, **{**base, **kwargs})
        self.visual.metadata["source_modality"] = "text"
        with self.assertRaisesRegex(ValueError, "visual-to-text"):
            plot_mj_visual_readout(self.image, self.visual, **base)
        self.visual.metadata["source_modality"] = "visual"
        self.logits[1, 0, 2] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            plot_mj_visual_readout(self.image, self.visual, **base)

    def test_text_curves_keep_raw_margins_at_declared_block_inputs(self):
        import numpy as np
        logits = torch.tensor([[[0., 2., 5.]], [[0., 4., 1.]]])
        plain = torch.tensor([[0., 3., 1.], [0., 1., 5.]])
        native = torch.tensor([[[0., 4., 2.]]])
        result = ProbeResult("mj_lens", {"logits": logits, "positions": torch.tensor([[0, 12]])},
                             {"layers": [2, 7], "source_modality": "text", "target_modality": "text"})
        before = logits.clone()
        ax = plot_mj_text_comparison(result, plain, native, positive_token_id=2, negative_token_id=1,
                                     labels=["early", "late"])
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [2, 7])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [3, -3])
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), [-2, 4])
        np.testing.assert_array_equal(ax.lines[2].get_ydata(), [-2, -2])
        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ["early", "late"])
        torch.testing.assert_close(before, logits)

    def test_text_comparison_rejects_ambiguous_states_and_vocabulary_mismatch(self):
        result = ProbeResult("mj_lens", {"logits": torch.zeros(2, 1, 3),
            "positions": torch.tensor([[0, 12]])},
            {"layers": [2, 7], "source_modality": "text", "target_modality": "text"})
        options = {"positive_token_id": 2, "negative_token_id": 1}
        with self.assertRaisesRegex(ValueError, "plain_logits"):
            plot_mj_text_comparison(result, torch.zeros(2, 4), torch.zeros(3), **options)
        with self.assertRaisesRegex(ValueError, "exactly one final"):
            plot_mj_text_comparison(result, torch.zeros(2, 3), torch.zeros(2, 3), **options)
        result.tensors["logits"] = torch.zeros(2, 2, 3)
        with self.assertRaisesRegex(ValueError, "exactly one selected"):
            plot_mj_text_comparison(result, torch.zeros(2, 3), torch.zeros(3), **options)


if __name__ == "__main__":
    unittest.main()
