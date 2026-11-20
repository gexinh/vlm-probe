"""Browser export checks against public probe forwards and saved results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from tiny_model import TinyModel
from explorer.export_result import build_bundle
from vlmprobe import ProbeResult, Prober, TokenMargin
from vlmprobe.sae import SAEConfig, SparseAutoencoder
from vlmprobe.sae.binding import build_sae_binding
from vlmprobe.sae.training import train_sae


class ExplorerExportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.probe = Prober(TinyModel().eval())
        self.clean = {"input_ids": torch.tensor([[1, 2, 3], [3, 4, 5]]),
                      "image_tokens": torch.randn(2, 2, 6)}
        self.corrupt = {**self.clean, "image_tokens": torch.zeros_like(self.clean["image_tokens"])}
        self.metric = TokenMargin(4, 5)

    def test_live_results_keep_their_batch_coordinates_and_values(self):
        probe, inputs, metric = self.probe, self.clean, self.metric
        layers = [0, 1]
        lens = probe.lens.logit(layers=layers).run(inputs)
        raw = probe.attention.profile(layers=layers, include_attention=True, queries="last_prompt").run(inputs)
        cam = probe.attention.grad_cam(layers=layers, keys="visual").run(inputs, metric=metric)
        attr = probe.attention.attribution(layers=layers, keys="visual", steps=2).run(inputs, metric=metric)
        causal = probe.causal.patch(layers=layers, tokens="visual").run(self.corrupt, source=inputs, metric=metric)
        bundle = build_bundle([lens, raw, cam, attr, causal], batch_index=1, query=4,
                              target_token_id=4, visual_positions=[1, 0])
        cases = {case["family"]: case for case in bundle["cases"]}
        lens_values = lens.tensors["logits"][:, 1]
        ranks = 1 + (lens_values > lens_values[:, [4]]).sum(-1)
        rank_view = cases["lens"]["methods"][0]["views"][0]
        self.assertEqual(rank_view["values"], ranks[:, None].tolist())
        self.assertEqual(rank_view["columns"], ["4"])
        maps = [method["views"][0] for method in cases["attention"]["methods"]]
        raw_values = raw.tensors["attention"][:, 1, :, 4].mean(1)
        self.assertEqual(maps[0]["values"], raw_values[:, [1, 0]].tolist())
        self.assertEqual(maps[1]["values"], cam.tensors["cam"][:, 1, 4, [1, 0]].tolist())
        self.assertEqual(maps[2]["values"], attr.tensors["token_attribution"][:, 1, 4, [1, 0]].tolist())
        self.assertTrue(maps[2]["signed"])
        effect = cases["causal"]["methods"][0]["views"][0]
        self.assertEqual(effect["series"][0]["values"], causal.tensors["effect"][:, 1].tolist())
        with self.assertRaisesRegex(ValueError, "query position 0 was not measured"):
            build_bundle([cam], query=0)
        with self.assertRaisesRegex(ValueError, "keys outside the measured selector"):
            build_bundle([cam], visual_positions=[0, 2])

    def test_sae_feature_values_and_scores(self):
        probe = self.probe
        with tempfile.TemporaryDirectory() as directory:
            binding = build_sae_binding(probe, 0, model_id="tiny-seed7", tokenizer_id="ids",
                                        preprocessing_id="native", calibration_id="toy")
            sae = SparseAutoencoder(SAEConfig(architecture="relu", d_in=6, d_sae=12, k=3), binding=binding)
            train_sae(sae, [torch.randn(16, 6)], [torch.randn(16, 6)], Path(directory) / "fit", epochs=1)
        options = {"layer": 0, "sae": sae, "binding": binding, "tokens": "visual"}
        features = probe.sae.features(**options).run(self.clean)
        edit = probe.sae.edit(**options, feature_ids=[1]).run(self.clean, metric=self.metric)
        bundle = build_bundle([features, edit], batch_index=1, feature_ids=[1, 0])
        case = bundle["cases"][0]
        self.assertEqual(case["family"], "sae")
        activity = case["methods"][0]["views"][0]
        self.assertEqual(activity["values"], features.tensors["features"][2:, [1, 0]].T.tolist())
        effect = case["methods"][1]["views"][0]
        self.assertEqual(effect["entries"][0]["value"], edit.tensors["score_change"][1].item())

    def test_cli_round_trip_and_missing_path_cells(self):
        result = ProbeResult("path_patching", {"senders": torch.tensor([[4, 1], [8, 5]]),
                                               "effect": torch.tensor([[0, 2], [-1, 0]])})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result.save(path / "probe")
            output = path / "explorer.json"
            subprocess.run([sys.executable, str(ROOT / "explorer/export_result.py"),
                            str(path / "probe"), "--output", str(output), "--batch", "1"],
                           check=True, capture_output=True, text=True)
            bundle = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(bundle["schema"], 1)
        view = bundle["cases"][0]["methods"][0]["views"][0]
        self.assertEqual(view["values"], [[2, None], [None, 0]])
        self.assertEqual(view["rows"], ["Layer 4", "Layer 8"])
        self.assertEqual(view["columns"], ["Head 1", "Head 5"])
