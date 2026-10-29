"""Model-bound sparse feature readouts and explicit feature interventions."""
import math
import hashlib
import json
from pathlib import Path
from functools import wraps

import torch

from .common import BoundMethod, configuration, evaluate, layers_for, model_eval
from ..core import ProbeResult
from ..core.precision import analysis_difference, analysis_dtype
from ..sae.binding import check_runtime_binding


def _dictionary_eval(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        states = [(m, m.training) for m in self.sae.modules()]
        try:
            self.sae.eval()
            return function(self, *args, **kwargs)
        finally:
            for module, training in states:
                module.training = training
    return wrapped


class SAEProbe(BoundMethod):
    def __init__(self, probe, layer, sae, checkpoint, binding, tokens, *,
                 feature_ids=None, scale=0., mode="error_preserving", operation=None, top_k=10):
        if type(layer) is not int:
            raise ValueError("one SAE artifact must select exactly one integer layer")
        layers_for(probe, layer)
        if (sae is None) == (checkpoint is None):
            raise ValueError("provide exactly one of sae or checkpoint")
        if not isinstance(binding, dict) or not binding:
            raise ValueError("supply the expected checkpoint binding explicitly")
        check_runtime_binding(probe, layer, binding)
        if checkpoint is not None:
            from ..sae.models import SparseAutoencoder
            device = next(probe.model.parameters()).device
            sae = SparseAutoencoder.load(checkpoint, expected_binding=binding, device=device)
        if getattr(sae, "binding", None) != binding:
            raise ValueError("SAE artifact binding does not match expected identities")
        if not getattr(sae, "is_fitted", False):
            raise ValueError("fit or load a trained dictionary before using model-bound SAE methods")
        if mode not in {"error_preserving", "reconstruction"}:
            raise ValueError("mode must be error_preserving or reconstruction")
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        if feature_ids is not None:
            feature_ids = list(feature_ids)
            if (not feature_ids or len(set(feature_ids)) != len(feature_ids)
                    or any(type(i) is not int or not 0 <= i < sae.config.d_sae for i in feature_ids)):
                raise ValueError("feature_ids must be nonempty unique dictionary feature indices")
        operation = operation or ("edit" if feature_ids is not None else "features")
        if operation not in {"features", "edit", "profile", "reconstruct"}:
            raise ValueError("unknown SAE operation")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        super().__init__(probe, "sae." + operation, [layer])
        self.sae, self.layer, self.tokens = sae, layer, tokens
        self.feature_ids, self.scale, self.mode = feature_ids, scale, mode
        self.operation, self.top_k = operation, top_k

    def _encode(self, values):
        contextual = callable(getattr(self.sae, "encode_with_context", None))
        if contextual != callable(getattr(self.sae, "decode_with_context", None)):
            raise ValueError("context-aware dictionaries must provide both encode and decode context methods")
        features, context = self.sae.encode_with_context(values) if contextual else (self.sae.encode(values), None)
        if (not isinstance(features, torch.Tensor) or features.shape != (values.shape[0], self.sae.config.d_sae)
                or not torch.isfinite(features).all()):
            raise ValueError("SAE features must be finite [selected_position,dictionary_feature]")
        return features, context, contextual

    def _decode(self, features, context, contextual):
        decoded = (self.sae.decode_with_context(features, context) if contextual else self.sae.decode(features))
        if (not isinstance(decoded, torch.Tensor) or decoded.shape != (features.shape[0], self.sae.config.d_in)
                or not torch.isfinite(decoded).all()):
            raise ValueError("SAE reconstruction must be finite [selected_position,native_width]")
        return decoded

    def _output(self, trace):
        if self.probe.spec.output_kind != "representation":
            return trace.logits
        if not isinstance(trace.output, torch.Tensor):
            raise TypeError("representation adapters must return a tensor of native output vectors")
        return trace.output

    def _device_dtype(self):
        parameter = getattr(self.sae, "activation_parameter", None)
        if parameter is None:
            parameter = next(self.sae.parameters())
        return parameter.device, parameter.dtype

    def _capture(self, inputs):
        site = self.probe.spec.residuals[self.layer]
        trace = self.probe._run(inputs, capture=[site])
        hidden = trace.activations[site]
        if hidden.ndim != 3 or hidden.shape[-1] != self.sae.config.d_in:
            raise ValueError("SAE width must match the native [batch,token,hidden] residual tensor")
        layout = self.probe._layout(inputs, hidden)
        mask = layout.select(self.tokens)
        return trace, hidden, layout, mask, site

    @model_eval
    @_dictionary_eval
    @torch.no_grad()
    def run(self, inputs, *, metric=None):
        trace, hidden, layout, mask, site = self._capture(inputs)
        device, dtype = self._device_dtype()
        native = hidden[mask.to(hidden.device)]
        values = native.to(device=device, dtype=dtype)
        features, context, contextual = self._encode(values)
        reconstruction = self._decode(features, context, contextual)
        positions = mask.nonzero()
        metadata = {"binding": self.sae.binding, "architecture": self.sae.config.architecture,
                    "backend": getattr(self.sae, "source_metadata", {}).get("backend", "native"),
                    "source": getattr(self.sae, "source_metadata", {}),
                    "output_kind": self.probe.spec.output_kind,
                    "normalization_context_preserved": contextual,
                    "visual_only": bool(layout.visual is not None and not (mask & ~layout.visual).any()),
                    "feature_axes": ["selected_position", "dictionary_feature"],
                    "position_axes": ["batch", "expanded_token"],
                    "interpretation": "Dictionary features require empirical interpretation; feature IDs are not named concepts."}
        if self.operation == "profile":
            indices = features.abs().topk(min(self.top_k, len(features)), dim=0).indices
            statistics = features.to(analysis_dtype(features))
            return self._finish(ProbeResult(self.name, {
                "mean_activation": statistics.mean(0),
                "activation_variance": statistics.var(0, unbiased=False),
                "firing_frequency": (features != 0).float().mean(0),
                "max_activation": features.max(0).values, "min_activation": features.min(0).values,
                "top_activation_values": features.gather(0, indices),
                "top_positions": positions.to(indices.device)[indices],
                "positions": positions,
            }, {**metadata, "analysis_dtype": str(statistics.dtype),
                "ranking": "absolute activation per feature; signed values retained"}))
        if self.operation == "features":
            result = ProbeResult(self.name, {
                "features": features, "activation": values,
                "native_activation": native,
                "reconstruction": reconstruction,
                "reconstruction_error": analysis_difference(values, reconstruction),
                "l0": (features != 0).sum(-1), "positions": positions,
            }, metadata)
            return self._finish(result)
        if self.operation != "reconstruct" and self.mode == "error_preserving" and self.scale == 1.:
            # Return the same native output exactly, including bfloat16 rounding.
            edited_trace = trace
        else:
            changed = features.clone()
            if self.operation != "reconstruct":
                changed[:, self.feature_ids] *= self.scale
            decoded = self._decode(changed, context, contextual)
            if self.mode == "error_preserving":
                # Retain the original residual (including FP64 bits) while
                # doing both subtraction and addition before the native cast.
                # Casting the delta first can erase a small retained feature.
                wide = analysis_dtype(native, decoded, reconstruction)
                difference = (decoded.to(device=native.device, dtype=wide)
                              - reconstruction.to(device=native.device, dtype=wide))
                replacement = (native.to(wide) + difference).to(native)
            else:
                replacement = decoded.to(native)

            def edit(current):
                if current.shape != hidden.shape:
                    raise ValueError("residual geometry changed between baseline and edited forwards")
                output = current.clone()
                output[mask.to(current.device)] = replacement.to(current)
                return output

            edited_trace = self.probe._run(inputs, interventions={site: edit})
        baseline_output, edited_output = self._output(trace), self._output(edited_trace)
        output_name = "output" if self.probe.spec.output_kind == "representation" else "logits"
        tensors = {"baseline_" + output_name: baseline_output, "edited_" + output_name: edited_output,
                   "positions": positions}
        if metric is not None:
            if self.probe.spec.output_kind == "representation":
                from ..metrics import ClassScore, TokenMargin, SequenceLogProb
                if isinstance(metric, (ClassScore, TokenMargin, SequenceLogProb)):
                    raise ValueError("representation outputs require a vector metric, not a language/classification logit score")
            baseline = evaluate(metric, baseline_output, layout)
            edited = evaluate(metric, edited_output, layout)
            tensors.update(baseline_score=baseline, edited_score=edited,
                           score_change=analysis_difference(edited, baseline))
        return self._finish(ProbeResult(self.name, tensors, {
            **metadata, "feature_ids": self.feature_ids, "scale": self.scale, "mode": self.mode,
            "intervention": ("native residual plus decoded feature difference; reconstruction error retained"
                             if self.mode == "error_preserving" else "replace selected residuals by SAE reconstruction"),
        }))


class FittedSAEMethod:
    """One model-bound algorithm, fitted or loaded in VLMProbe coordinates."""

    def __init__(self, probe, architecture, layer, tokens="all", *, d_sae=None,
                 expansion_factor=4, **parameters):
        layers_for(probe, layer)
        if type(layer) is not int:
            raise ValueError("one SAE dictionary requires one integer layer")
        if d_sae is not None and (type(d_sae) is not int or d_sae < 1):
            raise ValueError("d_sae must be a positive integer")
        if type(expansion_factor) is not int or expansion_factor < 1:
            raise ValueError("expansion_factor must be a positive integer")
        from ..sae.registry import get_method
        architecture = get_method(architecture)["architecture"]
        self.probe, self.architecture, self.layer, self.tokens = probe, architecture, layer, tokens
        self.d_sae, self.expansion_factor, self.parameters = d_sae, expansion_factor, parameters
        self.sae, self.history = None, None
        self._weight_identity_verified = False
        self._loaded = False

    def fit(self, train_inputs, validation_inputs, *, output_dir, training=None,
            train_ids=None, validation_ids=None, preprocessing_id="caller-prepared:unverified",
            max_cache_bytes=1 << 30, provenance=None):
        """Capture frozen-model activations, then fit with the single native trainer.

        Inputs are finite native model batches (or ProbeInputs), not snapshots.
        Declare patient/document/image group IDs to check source independence.
        The bounded CPU cache is intended for small and medium experiments.
        """
        from ..sae.data import ActivationCollector, checked_source_ids, model_identity, processor_identity
        from ..sae.binding import build_sae_binding
        from ..sae.models import SparseAutoencoder
        from ..sae.registry import method_config
        from ..sae.training import SAETrainingConfig, fit_sae
        if self._loaded:
            raise ValueError("fit on a loaded method is not a warm-start protocol; configure a fresh method with the intended recipe and resume its matching training directory")
        groups = checked_source_ids(train_ids, validation_ids)
        collector = ActivationCollector(self.probe, self.layer, self.tokens, max_cache_bytes)
        train, train_meta = collector.collect(train_inputs)
        validation, validation_meta = collector.collect(validation_inputs)
        if set(train_meta["input_hashes"]) & set(validation_meta["input_hashes"]):
            raise ValueError("training and validation contain repeated prepared input batches")
        calibration = {"train": train_meta, "validation": validation_meta,
                       "selection": configuration(self.tokens), **groups}
        calibration_id = "sha256:" + hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
        binding = build_sae_binding(self.probe, self.layer,
            model_id=model_identity(self.probe.model),
            tokenizer_id=processor_identity(self.probe.processor),
            preprocessing_id=preprocessing_id, calibration_id=calibration_id)
        binding["tokens"] = configuration(self.tokens)
        binding["tokens_kind"] = ("boolean_mask" if isinstance(self.tokens, torch.Tensor)
                                  and self.tokens.dtype == torch.bool else "selector")
        binding["modality"] = ("vision" if self.probe.spec.output_kind in {"classification", "representation"}
                               else "multimodal" if train_meta["visual_vectors"] else "language")
        width = train.shape[-1]
        parameters = dict(self.parameters)
        # Small dictionaries have a valid default; an explicit k is never silently clipped.
        parameters.setdefault("k", min(32, self.d_sae or width * self.expansion_factor))
        model = SparseAutoencoder(method_config(self.architecture, d_in=width,
            d_sae=self.d_sae or width * self.expansion_factor, **parameters), binding=binding)
        training = training or SAETrainingConfig()
        recorded = {**(provenance or {}), **calibration, "source_ids_checked": groups["source_groups_checked"],
                    "preprocessing_id": preprocessing_id}
        self.history = fit_sae(model, train, validation, output_dir, training=training, provenance=recorded)
        self.sae = model.eval()
        self._weight_identity_verified = True
        self.sae.training_metadata = {"history": self.history,
            "protocol": "frozen-native-model-activations", "provenance": recorded}
        # Inference artifact and optimizer/RNG resume checkpoint are distinct.
        self.save(Path(output_dir) / "artifact")
        return self.history

    def _operation(self, operation="features", **kwargs):
        if self.sae is None:
            raise ValueError("fit or load this SAE method before running it")
        return SAEProbe(self.probe, self.layer, self.sae, None, self.sae.binding, self.tokens,
                        operation=operation, **kwargs)

    def run(self, inputs):
        return self._operation().run(inputs)

    def profile(self, *, top_k=10):
        return self._operation("profile", top_k=top_k)

    def reconstruct(self):
        return self._operation("reconstruct", mode="reconstruction")

    def edit(self, *, feature_ids, scale=0., mode="error_preserving"):
        return self._operation("edit", feature_ids=feature_ids, scale=scale, mode=mode)

    def save(self, path, *, model_card=None, registry_dir=None):
        from ..sae.artifacts import save_artifact, state_sha256
        if self.sae is None:
            raise ValueError("fit or load this SAE method before saving it")
        evidence = {"tensor_integrity": "checksum_recorded", "numerical_parity": "not_evaluated",
                    "backbone_compatibility": ("exact_weight_fingerprint_and_native_hook"
                                               if self._weight_identity_verified else "caller_declared_binding"),
                    "semantic_quality": "not_evaluated"}
        previous_card = getattr(self.sae, "model_card", {})
        evidence.update(previous_card.get("evidence", {}))
        if previous_card:
            previous_checksum = previous_card.get("artifact", {}).get("state_sha256")
            if previous_checksum != state_sha256(dict(self.sae.state_dict())):
                evidence.update(numerical_parity="not_evaluated_after_state_change",
                                semantic_quality="not_evaluated_after_state_change",
                                training_validation="stale_after_state_change")
                evidence["tensor_integrity"] = "checksum_recorded"
                self.sae.training_metadata = {**getattr(self.sae, "training_metadata", {}),
                                              "state_modified_since_model_card": True}
                previous_verification = getattr(self.sae, "source_metadata", {}).get("pretrained_verification")
                if previous_verification:
                    self.sae.source_metadata = {**self.sae.source_metadata,
                        "pretrained_verification": {**previous_verification,
                            "status": "invalidated_after_state_change"}}
        source = getattr(self.sae, "source_metadata", {})
        pretrained = source.get("pretrained_verification", {})
        if source.get("conversion") and not (
                pretrained.get("status") == "passed" and self._weight_identity_verified):
            evidence["backbone_compatibility"] = "not_verified"
        return save_artifact(self.sae, path, model_card={"evidence": evidence, **(model_card or {})},
                             registry_dir=registry_dir)


class SAEMethods:
    def __init__(self, probe):
        self.probe = probe

    def method(self, architecture, *, layer, tokens="all", d_sae=None, expansion_factor=4, **parameters):
        """Configure a native algorithm; fit/load and run use one shared contract."""
        return FittedSAEMethod(self.probe, architecture, layer, tokens,
                              d_sae=d_sae, expansion_factor=expansion_factor, **parameters)

    def from_pretrained(self, card_id, *, cache_dir=None, device="cpu"):
        """Load an explicitly supported author SAE without training.

        Registered recipes verify pinned source/backbone tensors, tokenizer and
        the native activation hook, then reuse a safe native artifact cache.
        Other published model cards remain available for explicit conversion.
        """
        from ..sae.pretrained import load_pretrained
        return load_pretrained(self.probe, card_id, cache_dir=cache_dir, device=device)

    def load(self, path_or_card, *, tokens=None, binding=None, registry_dir=None,
             artifact_dir=None, device="cpu"):
        """Load our artifact or model card and validate actual model weights."""
        from ..sae.artifacts import load_artifact
        from ..sae.cards import get_model_card, load_model
        from ..sae.data import model_identity, processor_identity
        caller_supplied_binding = binding is not None
        verified_identity = None
        if binding is not None:
            check_runtime_binding(self.probe, binding.get("layer"), binding)
        path = Path(path_or_card)
        if not path.exists():
            card = get_model_card(str(path_or_card), registry_dir=registry_dir)
            if binding is None:
                binding = card.get("binding")
            if binding is not None:
                check_runtime_binding(self.probe, binding.get("layer"), binding)
                if not caller_supplied_binding and not binding["model_id"].startswith("sha256:"):
                    raise ValueError("non-fingerprinted source artifacts require an explicitly verified binding")
                if binding["model_id"].startswith("sha256:"):
                    verified_identity = model_identity(self.probe.model)
                    if binding["model_id"] != verified_identity:
                        raise ValueError("SAE model-weight fingerprint differs from the loaded backbone")
                    if binding["tokenizer_id"] != processor_identity(self.probe.processor):
                        raise ValueError("SAE tokenizer/processor fingerprint differs from the runtime processor")
        model = (load_artifact(path, expected_binding=binding, device=device) if path.exists()
                 else load_model(str(path_or_card), binding=binding, registry_dir=registry_dir,
                                 artifact_dir=artifact_dir, device=device))
        declared = model.binding
        check_runtime_binding(self.probe, declared["layer"], declared)
        if declared["model_id"].startswith("sha256:"):
            if declared["model_id"] != (verified_identity or model_identity(self.probe.model)):
                raise ValueError("SAE model-weight fingerprint differs from the loaded backbone")
            if declared["tokenizer_id"] != processor_identity(self.probe.processor):
                raise ValueError("SAE tokenizer/processor fingerprint differs from the runtime processor")
        else:
            if not caller_supplied_binding:
                raise ValueError("non-fingerprinted source artifacts require an explicitly verified binding")
        selection = tokens if tokens is not None else declared.get("tokens", "all")
        if tokens is None and declared.get("tokens_kind") == "boolean_mask":
            selection = torch.tensor(selection, dtype=torch.bool)
        bound = self.method(model.config.architecture, layer=declared["layer"], tokens=selection,
                            d_sae=model.config.d_sae)
        bound.sae = model.eval()
        bound._loaded = True
        bound._weight_identity_verified = declared["model_id"].startswith("sha256:")
        return bound

    def relu(self, **kwargs):
        return self.method("relu", **kwargs)

    def topk(self, **kwargs):
        return self.method("topk", **kwargs)

    def gated(self, **kwargs):
        return self.method("gated", **kwargs)

    def jumprelu(self, **kwargs):
        return self.method("jumprelu", **kwargs)

    def abstopk(self, **kwargs):
        return self.method("abstopk", **kwargs)

    def batchtopk(self, **kwargs):
        return self.method("batchtopk", **kwargs)

    def matryoshka_batchtopk(self, **kwargs):
        return self.method("matryoshka_batchtopk", **kwargs)

    def archetypal_topk(self, **kwargs):
        return self.method("archetypal_topk", **kwargs)

    def archetypal_jumprelu(self, **kwargs):
        return self.method("archetypal_jumprelu", **kwargs)

    def matching_pursuit(self, **kwargs):
        return self.method("matching_pursuit", **kwargs)

    def signed_matching_pursuit(self, **kwargs):
        return self.method("signed_matching_pursuit", **kwargs)

    def orthogonal_matching_pursuit(self, **kwargs):
        return self.method("orthogonal_matching_pursuit", **kwargs)

    def signed_orthogonal_matching_pursuit(self, **kwargs):
        return self.method("signed_orthogonal_matching_pursuit", **kwargs)

    def qsae(self, **kwargs):
        return self.method("qsae", **kwargs)

    def features(self, *, layer, sae=None, checkpoint=None, binding, tokens="all"):
        """Encode one layer using a trained dictionary; positions are explicit."""
        return SAEProbe(self.probe, layer, sae, checkpoint, binding, tokens)

    def edit(self, *, layer, feature_ids, sae=None, checkpoint=None, binding,
             tokens="all", scale=0., mode="error_preserving"):
        """Scale chosen features, preserving reconstruction error by default."""
        return SAEProbe(self.probe, layer, sae, checkpoint, binding, tokens,
                        feature_ids=feature_ids, scale=scale, mode=mode)

    def profile(self, *, layer, sae=None, checkpoint=None, binding, tokens="all", top_k=10):
        """Summarize feature firing and retrieve top positions by absolute activation."""
        return SAEProbe(self.probe, layer, sae, checkpoint, binding, tokens,
                        operation="profile", top_k=top_k)

    def reconstruct(self, *, layer, sae=None, checkpoint=None, binding, tokens="all"):
        """Replace selected states by the reconstruction as a separate control."""
        return SAEProbe(self.probe, layer, sae, checkpoint, binding, tokens,
                        operation="reconstruct", mode="reconstruction")
