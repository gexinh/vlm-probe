"""Modality-split Jacobian lenses in declared decoder coordinates.

Independent implementation of Eq. (2)--(3) in OpenReview Xrp8mNku6R.
MJ targets name output channels, not arbitrary image-input placeholders.
"""
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
import hashlib
import json
import warnings

import torch

from ..adapters import CapabilityError
from ..lenses.mj import MJLens
from .common import coordinates, model_eval, pack, selected


@dataclass
class MJBatch:
    """Native inputs with output-position masks and a declared output context.

    ``inputs`` is a native dictionary or ProbeInputs. Source modalities come
    from its TokenLayout; ``target_masks`` describes where each output head is
    used in that same sequence. Image input positions alone do not establish a
    visual output channel. Code IDs and language token IDs remain separate.
    """
    inputs: object
    target_masks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    context: str = "text_output"

    def __post_init__(self):
        if not isinstance(self.context, str) or not self.context:
            raise ValueError("MJBatch.context must be a nonempty protocol name")
        if not isinstance(self.target_masks, Mapping) or set(self.target_masks) - {"text", "visual"}:
            raise ValueError("target_masks keys must be text or visual")


def batch_inputs(batch, context):
    if isinstance(batch, MJBatch):
        if batch.context != context:
            raise ValueError(f"MJ output context mismatch: expected {context!r}, got {batch.context!r}")
        return batch.inputs
    if context != "text_output":
        raise ValueError("non-text-output MJ inputs require MJBatch with explicit context and target masks")
    return batch


def target_mask(batch, layout, target):
    mask = batch.target_masks.get(target) if isinstance(batch, MJBatch) else None
    if mask is None:
        if target == "visual":
            raise CapabilityError("visual MJ targets require an explicit MJBatch visual output mask; "
                                  "image-input placeholders are not generated codebook outputs")
        mask = layout.select("text")
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != layout.valid.shape):
        raise ValueError("MJ output masks must be boolean and match the expanded sequence")
    mask = mask.to(layout.valid.device)
    if (mask & ~layout.valid).any():
        raise ValueError("MJ output target mask selects padding")
    return mask


class MJLensProbe:
    """One source-modality/output-channel pair, fitted over selected layers."""
    def __init__(self, probe, *, layers=None, source="visual", target="text",
                 visual_readout=None, context=None, binding=None):
        if source not in {"text", "visual"} or target not in {"text", "visual"}:
            raise ValueError("MJ source and target must be text or visual")
        self.probe, self.source, self.target = probe, source, target
        self.context = context or ("image_output" if target == "visual" else "text_output")
        if not isinstance(self.context, str) or not self.context:
            raise ValueError("MJ context must be a nonempty protocol name")
        self.spec = getattr(probe.spec, "mj", None)
        if self.spec is None or probe.spec.causal_attention is not True:
            raise CapabilityError("MJ-Lens requires declared causal block-input/final-norm sites "
                                  "and an RMSNorm readout; inspect probe.describe()")
        self.layers = (sorted(self.spec.sources) if layers is None else
                       [layers] if type(layers) is int else list(layers))
        if (not self.layers or len(set(self.layers)) != len(self.layers)
                or any(type(i) is not int or i not in self.spec.sources for i in self.layers)):
            raise ValueError(f"MJ layers must be unique block-input indices from {sorted(self.spec.sources)}")
        self.binding = dict(binding or {})
        self.visual_readout = visual_readout
        if target == "visual":
            from ..lenses.mj_readout import LinearizedVisualReadout
            if self.spec.visual_head is None:
                raise CapabilityError("this model has no declared discrete image-generation head/codebook; "
                                      "a visual encoder does not provide visual MJ outputs")
            if not isinstance(visual_readout, LinearizedVisualReadout):
                raise ValueError("visual MJ requires a calibrated LinearizedVisualReadout")
            if visual_readout.norm is not self.spec.norm:
                raise ValueError("visual readout must use this model's actual final normalization")
            head = self.spec.visual_head
            if all(hasattr(head, name) for name in ("proj_out", "activation_fn", "vision_head")):
                if (visual_readout.first_projection is not head.proj_out
                        or visual_readout.last_projection is not head.vision_head
                        or visual_readout.activation is not head.activation_fn):
                    raise ValueError("visual readout must use the exact native projection roles and activation")
            native_parameters = {id(p) for p in self.spec.visual_head.parameters()}
            for projection in (visual_readout.first_projection, visual_readout.last_projection):
                if not all(id(p) in native_parameters for p in projection.parameters()):
                    raise ValueError("visual readout projections are not bound to this model's generation head")
            # Accessing weight also rejects an unfitted readout.
            if visual_readout.weight.shape[-1] != self.spec.text_unembedding.shape[-1]:
                raise ValueError("visual readout and decoder residual widths differ")
        self.kernels, self.calibration_report = {}, {}

    def _binding(self, layer):
        fields = {"model_id": self.probe.adapter.model_id,
                  "model_revision": "unspecified", "tokenizer_id": "unspecified",
                  "calibration_id": "unspecified", **self.binding,
                  "site": self.spec.sources[layer],
                  "readout_id": "linearized_native_visual_head" if self.target == "visual" else "adapter.readout",
                  "source_modality": self.source, "target_modality": self.target,
                  "output_context": self.context, "source_site": self.spec.sources[layer],
                  "target_site": self.spec.final, "tap_convention": "decoder_block_input",
                  "target_convention": "input_to_final_rmsnorm"}
        if self.target == "visual":
            fields["codebook_id"] = self.binding.get("codebook_id", "unspecified")
            fields["visual_readout_binding"] = json.dumps(self.visual_readout.binding, sort_keys=True)
            slopes = self.visual_readout.mean_slope.detach().float().cpu().contiguous()
            fields["visual_readout_slopes_sha256"] = hashlib.sha256(bytes(slopes.untyped_storage())).hexdigest()
        return fields

    def _new_kernel(self, layer):
        return MJLens(self.visual_readout if self.target == "visual" else self.probe.adapter.readout,
                      source_modality=self.source, target_modality=self.target, binding=self._binding(layer))

    def _check_kernel_binding(self, layer):
        if self.kernels[layer].binding != self._binding(layer):
            raise ValueError("MJ model/readout/context binding changed after calibration; refit or load a matching artifact")

    def fit(self, inputs, *, dim_batch=32, skip_first=0, exclude_last=False):
        """Replace the estimate with one batch; use fit_batches for a corpus."""
        if self.kernels:
            warnings.warn("MJ-Lens fit replaces its current estimate; use fit_batches for dataset calibration",
                          RuntimeWarning, stacklevel=2)
        return self.fit_batches([inputs], dim_batch=dim_batch, skip_first=skip_first,
                                exclude_last=exclude_last)

    def fit_batches(self, input_batches, *, directory=None, dim_batch=32, resume=False,
                    checkpoint_every=8, skip_first=0, exclude_last=False):
        """Frozen-model, exact VJP calibration; retain odd/even prompt estimates."""
        from ..training.mj import fit_mj_lens
        return fit_mj_lens(self, input_batches, directory=directory, dim_batch=dim_batch,
                           resume=resume, checkpoint_every=checkpoint_every,
                           skip_first=skip_first, exclude_last=exclude_last)

    @model_eval
    def run(self, inputs, *, tokens=None):
        if any(layer not in self.kernels for layer in self.layers):
            raise RuntimeError("fit or load all selected MJ-Lens layers before run")
        native = batch_inputs(inputs, self.context)
        trace = self.probe._run(native, capture=[self.spec.sources[i] for i in self.layers])
        layout = self.probe._layout(native, trace.logits)
        mask = layout.select(self.source)
        if tokens is not None:
            wanted = layout.select(tokens)
            if (wanted & ~mask).any():
                raise ValueError(f"MJ source-modality mismatch: this lens reads only {self.source} positions")
            mask = wanted
        results = []
        for layer in self.layers:
            self._check_kernel_binding(layer)
            value = trace.activations[self.spec.sources[layer]]
            if value.shape[:2] != layout.valid.shape:
                raise ValueError("MJ source site does not match expanded decoder coordinates")
            results.append(self.kernels[layer].run(selected(value, mask),
                                                  source_modality=self.source, layer=layer))
        result = pack(self.probe, "mj_lens", self.layers, results,
                      source_modality=self.source, target_modality=self.target,
                      output_context=self.context, tap_convention="decoder_block_input",
                      readout_kind="visual_codebook" if self.target == "visual" else "text_vocabulary",
                      interpretation="averaged transport readout; not a validated control direction")
        if self.target == "visual":
            result.tensors["code_ids"] = result.tensors["logits"].argmax(-1)
        return coordinates(self.probe, result, mask)

    def dictionary(self, layer, *, token_ids=None):
        if layer not in self.kernels:
            raise RuntimeError("fit or load this MJ layer before constructing its dictionary")
        self._check_kernel_binding(layer)
        weight = (self.visual_readout.weight if self.target == "visual"
                  else self.spec.text_unembedding)
        return self.kernels[layer].dictionary(weight, self.spec.norm_gain, token_ids=token_ids)

    def _require_binding(self):
        for layer in self.layers:
            fields = self._binding(layer)
            missing = [k for k, v in fields.items()
                       if not isinstance(v, str) or not v or v == "unspecified"]
            if missing:
                raise ValueError(f"saving/resuming MJ-Lens requires explicit immutable binding: {missing}")

    def save(self, directory):
        self._require_binding()
        if any(layer not in self.kernels for layer in self.layers):
            raise RuntimeError("fit or load every MJ layer before saving")
        for layer in self.layers:
            self._check_kernel_binding(layer)
            self.kernels[layer].save(Path(directory) / f"mj_{self.source}_{self.target}_{layer}.pt")

    def load(self, directory):
        loaded = {}
        width = self.spec.text_unembedding.shape[-1]
        for layer in self.layers:
            kernel = self._new_kernel(layer)
            kernel.load(Path(directory) / f"mj_{self.source}_{self.target}_{layer}.pt")
            if kernel.jacobian.shape != (width, width):
                raise ValueError("MJ artifact matrix does not match declared decoder dimensions")
            loaded[layer] = kernel
        self.kernels = loaded
        return self


def fit_visual_readout(probe, input_batches, *, binding):
    """Fit Eq. (3) on declared image-output positions; no model optimization."""
    from ..lenses.mj_readout import LinearizedVisualReadout
    spec = getattr(probe.spec, "mj", None)
    if spec is None or spec.visual_head is None:
        raise CapabilityError("visual readout calibration requires a declared Janus-style generation head")
    head = spec.visual_head
    if not all(hasattr(head, key) for key in ("proj_out", "activation_fn", "vision_head")):
        raise CapabilityError("unsupported visual head: provide an explicit LinearizedVisualReadout")
    result = LinearizedVisualReadout(spec.norm, head.proj_out, head.activation_fn,
                                    head.vision_head, binding=binding)
    modes = [(module, module.training) for module in probe.model.modules()]
    try:
        probe.model.eval()
        for batch in input_batches:
            if not isinstance(batch, MJBatch):
                raise ValueError("visual readout calibration requires MJBatch with visual output positions")
            native = batch_inputs(batch, "image_output")
            trace = probe._run(native, capture=[spec.final])
            layout = probe._layout(native, trace.logits)
            mask = target_mask(batch, layout, "visual")
            with torch.no_grad():
                normalized = spec.norm(trace.activations[spec.final])
            result.update(normalized, mask=mask)
    finally:
        for module, training in modes:
            module.training = training
    if not result.n_samples:
        raise ValueError("visual readout calibration has no image-output positions")
    return result


def decode_mj_codes(result, *, grid, decoder):
    """Decode many visual SOURCE positions; never tile one text state into an image."""
    if (result.method != "mj_lens" or result.metadata.get("target_modality") != "visual"
            or result.metadata.get("source_modality") != "visual"):
        raise ValueError("spatial MJ decoding requires visual-to-visual MJ results")
    if (not isinstance(grid, (tuple, list)) or len(grid) != 2
            or any(type(n) is not int or n < 1 for n in grid)):
        raise ValueError("provide an explicit positive (height, width) codebook grid")
    positions, codes = result.tensors["positions"], result.tensors["code_ids"]
    batches = positions[:, 0].unique(sorted=True)
    rows = []
    for batch in batches:
        mask = positions[:, 0] == batch
        if int(mask.sum()) != grid[0] * grid[1]:
            raise ValueError("visual source count does not match the declared codebook grid")
        rows.append(codes[:, mask])
    packed = torch.stack(rows, dim=1)
    # Native VQ decoders own their device and return shape. No normalization is invented.
    images = []
    with torch.no_grad():
        for layer_codes in packed:
            owner = getattr(decoder, "__self__", None)
            parameter = next(owner.parameters(), None) if isinstance(owner, torch.nn.Module) else None
            native_codes = layer_codes if parameter is None else layer_codes.to(parameter.device)
            images.append(decoder(native_codes))
    return torch.stack(images), packed.reshape(len(packed), len(batches), *grid)
