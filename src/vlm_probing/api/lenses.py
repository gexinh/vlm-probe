"""Model-bound lens capture, calibration, and prompt readout."""
from pathlib import Path
import warnings

import torch

from .. import lenses as kernels
from ..lenses.jacobian import (
    _JacobianReplacementWarning, _calibration_selection, _selector_description, _warn_replacement,
)
from .common import BoundMethod, coordinates, layers_for, model_eval, pack, require, selected


class LensProbe(BoundMethod):
    def __init__(self, probe, name, layers, tokens, **options):
        super().__init__(probe, name, layers)
        self.tokens, self.options = tokens, options

    def _sites(self):
        spec = self.probe.spec
        if self.name == "embed":
            return [spec.embeddings]
        mapping = spec.heads if self.name == "attention" else spec.residuals
        return [mapping[i] for i in self.layers]

    def _capture(self, inputs, *, allow_empty_selection=False):
        sites = self._sites()
        trace = self.probe._run(inputs, capture=sites)
        layout = self.probe._layout(inputs, trace.logits)
        mask = (_calibration_selection(layout, self.tokens) if allow_empty_selection
                else layout.select(self.tokens))
        values = []
        for layer, site in zip(self.layers, sites):
            hidden = trace.activations[site]
            if self.name == "attention":
                hidden = self.probe._heads(layer, hidden)
            if hidden.shape[:2] != layout.valid.shape:
                raise ValueError("lens site must align with expanded output token coordinates")
            values.append(hidden)
        return trace, layout, mask, values

    def _kernel(self, layer, hidden):
        if self.name == "logit":
            return kernels.LogitLens(self.probe.adapter.readout)
        return kernels.EmbedLens(self.probe.spec.input_embeddings, **self.options)

    @model_eval
    def run(self, inputs, **kwargs):
        _, _, mask, values = self._capture(inputs)
        results = []
        for layer, hidden in zip(self.layers, values):
            kernel = self._kernel(layer, hidden)
            result = kernel.run(selected(hidden, mask), **kwargs)
            results.append(result)
        result = pack(self.probe, results[0].method, self.layers, results,
                      selection="packed valid positions; see positions tensor")
        return coordinates(self.probe, result, mask)


class FittedLensProbe(LensProbe):
    def __init__(self, probe, name, layers, tokens, *, binding=None):
        super().__init__(probe, name, layers, tokens)
        self.binding = dict(binding or {})
        self.kernels, self.losses = {}, {}

    def _binding(self, layer):
        mapping = self.probe.spec.heads if self.name == "attention" else self.probe.spec.residuals
        return {"model_id": self.probe.adapter.model_id, "readout_id": "adapter.readout",
                "tokenizer_id": "unspecified", "calibration_id": "unspecified",
                **self.binding, "site": mapping[layer]}

    def _new_kernel(self, layer, width, device, dtype, heads=None, *,
                    initial_unembedding=None, initial_bias=None):
        binding = self._binding(layer)
        readout = self.probe.adapter.readout
        if self.name == "tuned":
            return kernels.TunedLens(readout, width, binding=binding, device=device,
                                     dtype=torch.float32, readout_dtype=dtype)
        if self.name == "jacobian":
            return kernels.JacobianLens(readout, binding=binding)
        vocab = self.probe.spec.input_embeddings.shape[0]
        return kernels.AttentionLens(
            heads, width, vocab, binding=binding, device=device,
            initial_unembedding=initial_unembedding, initial_bias=initial_bias,
        )

    def _kernel(self, layer, hidden):
        if layer not in self.kernels:
            raise RuntimeError(f"{self.name} at layer {layer}: call fit(inputs) or load(directory) first")
        return self.kernels[layer]

    def run(self, inputs, **kwargs):
        if any(layer not in self.kernels for layer in self.layers):
            raise RuntimeError(f"{self.name}: call fit(inputs) or load(directory) before run")
        return super().run(inputs, **kwargs)

    @model_eval
    def fit(self, inputs, *, initial_unembedding=None, initial_bias=None, **kwargs):
        """Fit the lens; optional head initialization is Attention Lens only.

        ``initial_unembedding`` is [vocabulary, residual_width], e.g. the
        native ``lm_head.weight``; ``initial_bias`` is [vocabulary]. They
        initialize new head decoders before optimization and cannot reset an
        already fitted/loaded kernel. The head decoder applies no final norm.
        """
        initialize = initial_unembedding is not None or initial_bias is not None
        if initialize:
            if self.name != "attention":
                raise ValueError("head decoder initialization is only supported by Attention Lens")
            if any(layer in self.kernels for layer in self.layers):
                raise ValueError("initialization requires new Attention Lens kernels; create a new lens to reset")
        replacing_jacobian = self.name == "jacobian" and any(
            layer in self.kernels and self.kernels[layer].jacobian is not None for layer in self.layers
        )
        if replacing_jacobian:
            # Warn before the initial model capture, once for the public call.
            _warn_replacement(stacklevel=4)
        trace, layout, mask, values = self._capture(
            inputs, allow_empty_selection=self.name == "jacobian",
        )
        for layer, hidden in zip(self.layers, values):
            kernel = self.kernels.get(layer)
            if kernel is None:
                kernel = self._new_kernel(
                    layer, hidden.shape[-1], hidden.device, hidden.dtype,
                    hidden.shape[-2] if self.name == "attention" else None,
                    initial_unembedding=initial_unembedding, initial_bias=initial_bias,
                )
            if self.name == "jacobian":
                site = self.probe.spec.residuals[layer]
                final_site = self.probe.spec.residuals[max(self.probe.spec.residuals)]

                def downstream(value):
                    return self.probe._run(inputs, capture=[final_site],
                                           interventions={site: lambda _: value},
                                           grad=True).activations[final_site]

                try:
                    with warnings.catch_warnings():
                        if replacing_jacobian:
                            warnings.simplefilter("ignore", _JacobianReplacementWarning)
                        kernel.fit(hidden, downstream, valid_mask=layout.valid.to(hidden.device),
                                   source_mask=mask.to(hidden.device), **kwargs)
                except ValueError as error:
                    raise ValueError(
                        f"{error} source_selector={_selector_description(self.tokens)}; layer={layer}"
                    ) from error
            else:
                self.losses[layer] = kernel.fit(hidden, trace.logits, mask=mask, **kwargs)
            self.kernels[layer] = kernel
        return self

    @model_eval
    def fit_batches(self, train_inputs, validation_inputs, directory, **kwargs):
        """Calibrate on independent input iterables with persistent Adam.

        Capture once into CPU memory, train minibatches, select by held-out KL,
        and save resumable optimizer state plus normal layer-bound artifacts.
        Tuned/Attention Lens only. Inspect ``training_history`` afterwards.
        Use explicit immutable identities in ``binding`` and disjoint examples.
        """
        from ..training.model_bound import fit_model_bound_lens
        return fit_model_bound_lens(self, train_inputs, validation_inputs, directory, **kwargs)

    def save(self, directory):
        # All validation precedes filesystem writes.
        for layer in self.layers:
            if layer not in self.kernels:
                raise RuntimeError("fit or load every selected layer before saving")
            if any(v == "unspecified" for v in self._binding(layer).values()):
                raise ValueError("saving requires explicit model/tokenizer/calibration identities via binding")
        for layer in self.layers:
            self.kernels[layer].save(Path(directory) / f"{self.name}_{layer}.pt")

    def load(self, directory):
        loaded = {}
        embedding = require(self.probe.spec.input_embeddings,
                            "loading fitted lenses requires spec.input_embeddings for device/dimensions")
        for layer in self.layers:
            path = Path(directory) / f"{self.name}_{layer}.pt"
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            config = artifact["config"]
            width = config.get("hidden_size", config.get("head_size", embedding.shape[-1]))
            if width != embedding.shape[-1]:
                raise ValueError("artifact width does not match the adapter's residual embedding space")
            kernel = self._new_kernel(layer, width, embedding.device, embedding.dtype,
                                      config.get("num_heads"))
            kernel.load(path)
            loaded[layer] = kernel
        self.kernels = loaded
        return self


class PatchscopeProbe(BoundMethod):
    def __init__(self, probe, layers, source_position, target_layer, target_position,
                 target, mapping):
        super().__init__(probe, "patchscope", layers)
        self.source_position, self.target_layer = source_position, target_layer
        self.target_position, self.target = target_position, target
        if target.model is not probe.model and mapping is None:
            raise ValueError("cross-model Patchscope requires an explicit mapping")
        layers_for(target, target_layer)
        self.mapping = mapping

    def run(self, inputs, *, target_inputs):
        p, target = self.probe, self.target
        sites = [p.spec.residuals[i] for i in self.layers]
        source = p._run(inputs, capture=sites)
        source_layout = p._layout(inputs, source.logits)
        source_layout.select(self.source_position)
        # Validate target position before any edit, including padding.
        baseline = target._run(target_inputs)
        target._layout(target_inputs, baseline.logits).select(self.target_position)

        def runner(patch, *, target_layer, target_position, target_inputs):
            def replace(hidden):
                if patch.shape != hidden[:, target_position].shape:
                    raise ValueError("mapped Patchscope vector must match target batch/width")
                edited = hidden.clone()
                edited[:, target_position] = patch.to(hidden)
                return edited
            return target._run(target_inputs, interventions={target.spec.residuals[target_layer]: replace}).logits

        kernel = kernels.Patchscope(runner, mapping=self.mapping,
                                    source_model_id=p.adapter.model_id,
                                    target_model_id=target.adapter.model_id)
        # Pass native kwargs to the kernel; explicit layout was validated above.
        from .common import unpack
        target_kwargs, _ = unpack(target_inputs)
        results = [kernel.run(source.activations[site], source_position=self.source_position,
                              target_position=self.target_position, target_layer=self.target_layer,
                              target_inputs=target_kwargs, layer=layer)
                   for layer, site in zip(self.layers, sites)]
        return pack(p, "patchscope", self.layers, results,
                    readout="teacher_forced_target_prompt")


class LensMethods:
    """Factories share one Prober, model, and adapter."""

    def __init__(self, probe):
        self.probe = probe

    def logit(self, *, layers=None, tokens="last_prompt"):
        require(self.probe.adapter.readout, "Logit Lens requires an adapter readout")
        return LensProbe(self.probe, "logit", layers_for(self.probe, layers), tokens)

    def embed(self, *, tokens="visual", token_groups=None):
        require(self.probe.spec.embeddings, "EmbedLens requires an embedding-space capture site")
        require(self.probe.spec.input_embeddings, "EmbedLens requires input embeddings")
        return LensProbe(self.probe, "embed", ["embeddings"], tokens, token_groups=token_groups)

    def tuned(self, *, layers=None, tokens="last_prompt", binding=None):
        require(self.probe.adapter.readout, "Tuned Lens requires an adapter readout")
        return FittedLensProbe(self.probe, "tuned", layers_for(self.probe, layers), tokens, binding=binding)

    def attention(self, *, layers=None, tokens="last_prompt", binding=None):
        require(self.probe.spec.input_embeddings, "Attention Lens requires vocabulary dimensions")
        return FittedLensProbe(self.probe, "attention", layers_for(self.probe, layers, "heads"),
                               tokens, binding=binding)

    def jacobian(self, *, layers=None, tokens="all", binding=None):
        require(self.probe.adapter.readout, "Jacobian Lens requires an adapter readout")
        return FittedLensProbe(self.probe, "jacobian", layers_for(self.probe, layers), tokens, binding=binding)

    def mj(self, *, layers=None, source="visual", target="text", visual_readout=None,
           context=None, binding=None):
        """Modality-split Jacobian Lens, with paper block-input layer indices."""
        from .mj import MJLensProbe
        return MJLensProbe(self.probe, layers=layers, source=source, target=target,
                           visual_readout=visual_readout, context=context, binding=binding)

    def patchscope(self, *, layers, source_position, target_layer, target_position,
                   target=None, mapping=None):
        return PatchscopeProbe(self.probe, layers_for(self.probe, layers), source_position,
                               target_layer, target_position, target or self.probe, mapping)
