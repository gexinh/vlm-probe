"""MJ pretrained plots preserve native coordinates, signs and same-state controls."""
import importlib.util
import unittest

import torch

from vlm_probing import ProbeResult
from vlm_probing.visualization.mj_pretrained import (
    plot_mj_text_comparison, plot_mj_visual_readout, plot_region_patch_tokens,
    plot_region_tokens, region_token_readout,
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


class NativeTokenDecoder:
    def decode(self, ids, **kwargs):
        assert kwargs["skip_special_tokens"] is False
        assert kwargs["clean_up_tokenization_spaces"] is False
        return [" leading", "<|special|>", "\n", "中文"][ids[0]]


class RegionTokenReadoutTests(unittest.TestCase):
    def setUp(self):
        # Packed order differs from native sequence/grid order: p3,p0,p2,p1.
        natural = torch.tensor([[2., 1.8, -6., 0.], [-6., 1.8, 2., 0.],
                                [0., 0., 0., 2.], [0., 0., 3., 0.]])
        packed = natural[torch.tensor([3, 0, 2, 1])]
        self.logits = torch.stack([packed, packed + torch.tensor([0., 0., 0., 1.])])
        self.result = ProbeResult("mj_lens", {"logits": self.logits,
            "positions": torch.tensor([[0, 13], [0, 10], [0, 12], [0, 11]])},
            {"layers": [8, 16], "source_modality": "visual", "target_modality": "text",
             "tap_convention": "decoder_block_input"})
        self.options = {"grid_shape": (2, 2), "image_size": (20, 10), "bbox": (5, 0, 10, 5)}

    def extract(self, **kwargs):
        return region_token_readout(self.result, NativeTokenDecoder(),
                                     **{"top_k": 3, **self.options, **kwargs})

    def test_full_vocabulary_mixture_can_choose_a_word_outside_each_patch_top_one(self):
        import json
        before = self.logits.clone()
        value = self.extract(top_k=1, reference_token_id=3)
        # Shared runner-up wins the full probability mixture although it was
        # not either patch's top-1. Mixing only patch top-k would miss this.
        layer = value["layer_readouts"][0]
        self.assertEqual([p["patch_index"] for p in value["patches"]], [0, 1])
        self.assertEqual([p["sequence_position"] for p in value["patches"]], [10, 11])
        self.assertEqual([p["top_tokens"][0]["token_id"] for p in layer["patches"]], [0, 2])
        self.assertEqual(layer["region_top_tokens"][0]["token_id"], 1)
        self.assertEqual(layer["region_top_tokens"][0]["token"], "<|special|>")
        self.assertFalse(value["special_tokens_filtered"])
        self.assertFalse(value["reference_used_for_selection"])
        self.assertGreater(layer["region_reference"]["rank"], 1)
        json.dumps(value, allow_nan=False)
        torch.testing.assert_close(before, self.logits)

    def test_intersection_weights_and_raw_patch_tokens_are_preserved(self):
        value = self.extract(bbox=(7.5, 0, 10, 5), top_k=4)
        self.assertEqual([p["intersection_area"] for p in value["patches"]], [12.5, 37.5])
        self.assertEqual([p["region_weight"] for p in value["patches"]], [.25, .75])
        patch = value["layer_readouts"][0]["patches"][0]
        raw = self.logits[0, 1]
        token = patch["top_tokens"][0]
        self.assertEqual(token["token"], " leading")
        self.assertEqual(token["token_repr"], repr(" leading"))
        self.assertAlmostEqual(token["logit"], 2.)
        self.assertAlmostEqual(token["probability"], float(raw.softmax(-1)[0]))
        region = torch.tensor([t["probability"] for t in value["layer_readouts"][0]["region_top_tokens"]])
        self.assertAlmostEqual(float(region.sum()), 1., places=6)
        self.assertIn(repr("\n"), [t["token_repr"] for t in patch["top_tokens"]])
        # A reference is post-hoc only; the raw candidate ordering is unchanged.
        reference = self.extract(bbox=(7.5, 0, 10, 5), top_k=4, reference_token_id=0)
        self.assertEqual(value["layer_readouts"][0]["region_top_tokens"],
                         reference["layer_readouts"][0]["region_top_tokens"])

    def test_invalid_modalities_geometry_and_scores_are_rejected(self):
        for kwargs, message in (({"grid_shape": (1, 2)}, "count"),
                                ({"top_k": 5}, "top_k"),
                                ({"image_size": (0, 10)}, "image_size"),
                                ({"bbox": (25, 0, 5, 5)}, "overlap")):
            with self.assertRaisesRegex(ValueError, message):
                self.extract(**kwargs)
        self.result.metadata["source_modality"] = "text"
        with self.assertRaisesRegex(ValueError, "visual-to-text"):
            self.extract(top_k=1)
        self.result.metadata["source_modality"] = "visual"
        self.logits[0, 1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.extract(top_k=1)

    def test_genuine_jacobian_result_retains_its_native_method_and_layer_ids(self):
        self.result.method = "jacobian_lens"
        self.result.metadata = {"layers": [7, 15]}
        value = self.extract(top_k=1)
        self.assertEqual(value["method"], "jacobian_lens")
        self.assertEqual(value["layers"], [7, 15])
        self.assertEqual(value["layer_readouts"][0]["layer"], 7)

    @unittest.skipUnless(HAS_PLOTS, "plotting extra is optional")
    def test_region_plots_keep_all_selected_patches_and_require_the_processed_image(self):
        import io
        import matplotlib.pyplot as plt
        import numpy as np
        value = self.extract(top_k=3, reference_token_id=3)
        image = np.full((10, 20, 3), 180, dtype=np.uint8)
        figure, axes = plot_region_tokens(image, value, top_k=3, layer_labels=["Block 8", "Block 16"])
        self.assertEqual(len(axes), 3)
        self.assertEqual(len(axes[0].patches), len(value["patches"]) + 1)
        figure.savefig(io.BytesIO(), format="png")
        detailed, axis = plot_region_patch_tokens(value, top_k=3)
        self.assertEqual(len(axis.tables), 1)
        self.assertIn("p0", axis.tables[0].get_celld()[(1, -1)].get_text().get_text())
        with self.assertRaisesRegex(ValueError, "dimensions"):
            plot_region_tokens(image[:8], value, top_k=3)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
