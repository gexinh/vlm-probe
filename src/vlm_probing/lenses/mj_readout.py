"""Visual readout and controlled dictionary decomposition for Multimodal J-Lens.

The visual-head replacement follows Appendix H.3, equations (9)--(10), of
``Does the Workspace See? Testing a Multimodal Global Workspace in
Vision-Language Models`` (OpenReview Xrp8mNku6R).
The modality index follows Appendix C.1, equations (4)--(5). Its nonnegative
OMP/finite projected-gradient NNLS solver is a library implementation choice:
the paper specifies the constrained decomposition, not an exact solver.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import floating_tensor, position_mask
from ..core.types import ProbeResult
from ..sae.algorithms import orthogonal_matching_pursuit


def _gelu_slope(values: Tensor, approximate: str) -> Tensor:
    """Pointwise derivative, including the native tanh approximation when used."""
    if approximate == "none":
        return .5 * (1 + torch.erf(values / math.sqrt(2))) + (
            values * torch.exp(-.5 * values.square()) / math.sqrt(2 * math.pi)
        )
    inner = math.sqrt(2 / math.pi) * (values + .044715 * values.pow(3))
    tangent = inner.tanh()
    return .5 * (1 + tangent) + .5 * values * (1 - tangent.square()) * (
        math.sqrt(2 / math.pi) * (1 + 3 * .044715 * values.square())
    )


def _gelu_configuration(activation: nn.Module) -> str:
    if type(activation) is nn.GELU and activation.approximate in {"none", "tanh"}:
        return activation.approximate
    if (type(activation).__module__, type(activation).__name__) == (
            "transformers.activations", "GELUActivation"):
        # Lazy exact-class validation keeps Transformers optional for tensor users.
        from transformers.activations import GELUActivation
        if type(activation) is GELUActivation and getattr(activation, "act", None) in (
                F.gelu, activation._gelu_python):
            return "none"
    raise TypeError("visual readout requires native nn.GELU or audited transformers.activations.GELUActivation")


class LinearizedVisualReadout:
    """One calibrated, bias-free visual readout shared across output contexts.

    ``fit``/``update`` consume **already normalized**, final-layer visual states
    with shape ``[..., first_projection.in_features]``. They average GELU
    derivatives at individual sampled positions, rather than differentiating
    the GELU at an average state. They perform no optimization and never change
    the model's parameters, gradients, or train/eval mode. ``__call__`` and
    ``native`` instead consume raw hidden states and apply ``norm`` exactly once.

    ``weight`` has shape ``[visual_vocabulary, hidden]`` and omits both head
    biases, as in the paper. Keep the model/head fixed after calibration. Saving
    requires caller-declared model, readout and calibration identifiers; loading
    additionally checks exact native parameter fingerprints and dimensions.
    """

    def __init__(
        self, norm: Callable[[Tensor], Tensor], first_projection: nn.Linear,
        activation: nn.Module, last_projection: nn.Linear, *,
        binding: Mapping[str, str] | None = None,
    ):
        if not callable(norm):
            raise TypeError("norm must be callable")
        if not isinstance(first_projection, nn.Linear) or not isinstance(last_projection, nn.Linear):
            raise TypeError("visual readout requires two explicit nn.Linear projections")
        self._gelu_approximate = _gelu_configuration(activation)
        if first_projection.out_features != last_projection.in_features:
            raise ValueError("visual-head projection dimensions do not match")
        self.norm, self.first_projection = norm, first_projection
        self.activation, self.last_projection = activation, last_projection
        self.binding = dict(binding or {})
        self.n_samples = 0
        self._slope_sum: Tensor | None = None
        self._input_sum: Tensor | None = None
        self._calibrated_fingerprint: str | None = None
        self._calibrated_versions: tuple | None = None
        self._weight_cache: Tensor | None = None

    @property
    def _dtype(self) -> torch.dtype:
        return torch.float64 if self.first_projection.weight.dtype == torch.float64 else torch.float32

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr((type(self.activation).__module__, type(self.activation).__name__,
                            _gelu_configuration(self.activation))).encode())
        modules = {"first": self.first_projection, "last": self.last_projection}
        if isinstance(self.norm, nn.Module):
            modules["norm"] = self.norm
        for prefix, module in modules.items():
            for name, tensor in sorted(module.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(f"{prefix}.{name}:{tuple(value.shape)}:{value.dtype}".encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _versions(self) -> tuple:
        """Cheap streaming guard; full native hashes are checked at save/load."""
        modules = (self.first_projection, self.last_projection)
        if isinstance(self.norm, nn.Module):
            modules += (self.norm,)
        activation = (id(self.activation), _gelu_configuration(self.activation))
        return activation, tuple((id(value), value._version, value.device, value.dtype,
                                  tuple(value.shape)) for module in modules
                                 for value in (*module.parameters(), *module.buffers()))

    def _config(self) -> dict:
        return {
            "hidden_size": self.first_projection.in_features,
            "intermediate_size": self.first_projection.out_features,
            "visual_vocabulary_size": self.last_projection.out_features,
            "gelu_approximate": self._gelu_approximate,
        }

    def _require_fit(self) -> None:
        if not self.n_samples or self._slope_sum is None or self._input_sum is None:
            raise ValueError("visual readout is not calibrated; call fit(normalized_visual_states) first")
        if self._versions() != self._calibrated_versions:
            raise ValueError("native norm/head parameters or activation changed; recalibrate the visual readout")

    @torch.no_grad()
    def update(self, normalized_states: Tensor, *, mask: Tensor | None = None) -> "LinearizedVisualReadout":
        """Accumulate a position-weighted calibration batch; ``mask`` selects visual positions."""
        floating_tensor(normalized_states, "normalized_states")
        if normalized_states.shape[-1] != self.first_projection.in_features:
            raise ValueError("normalized_states hidden dimension does not match the visual head")
        selected = position_mask(mask, tuple(normalized_states.shape[:-1]), normalized_states.device)
        states = normalized_states.detach()[selected].reshape(-1, normalized_states.shape[-1])
        device = self.first_projection.weight.device
        states = states.to(device=device, dtype=self._dtype)
        versions = self._versions()
        if self.n_samples and versions != self._calibrated_versions:
            raise ValueError("native norm/head parameters changed during visual-readout calibration")
        if not self.n_samples:
            self._gelu_approximate = _gelu_configuration(self.activation)
        bias = self.first_projection.bias
        preactivation = F.linear(states, self.first_projection.weight.to(self._dtype),
                                 None if bias is None else bias.to(self._dtype))
        slopes = _gelu_slope(preactivation, self._gelu_approximate)
        slope_sum = slopes.to(torch.float64).sum(0)
        input_sum = states.to(torch.float64).sum(0)
        if not torch.isfinite(slope_sum).all() or not torch.isfinite(input_sum).all():
            raise ValueError("visual-readout calibration produced nonfinite values")
        if self._slope_sum is None:
            self._slope_sum, self._input_sum = slope_sum, input_sum
        else:
            self._slope_sum = self._slope_sum.to(device) + slope_sum
            self._input_sum = self._input_sum.to(device) + input_sum
        self.n_samples += len(states)
        if self._calibrated_fingerprint is None:
            self._calibrated_fingerprint = self._fingerprint()
        self._calibrated_versions = versions
        self._weight_cache = None
        return self

    def fit(self, normalized_states: Tensor, *, mask: Tensor | None = None) -> "LinearizedVisualReadout":
        """Replace any previous calibration with the selected normalized states."""
        previous = (self.n_samples, self._slope_sum, self._input_sum,
                    self._calibrated_fingerprint, self._calibrated_versions,
                    self._gelu_approximate, self._weight_cache)
        self.n_samples, self._slope_sum, self._input_sum, self._calibrated_fingerprint = 0, None, None, None
        self._calibrated_versions = None
        self._weight_cache = None
        try:
            return self.update(normalized_states, mask=mask)
        except Exception:
            (self.n_samples, self._slope_sum, self._input_sum,
             self._calibrated_fingerprint, self._calibrated_versions,
             self._gelu_approximate, self._weight_cache) = previous
            raise

    @property
    def mean_slope(self) -> Tensor:
        self._require_fit()
        return (self._slope_sum / self.n_samples).to(
            device=self.first_projection.weight.device, dtype=self._dtype,
        )

    @property
    def weight(self) -> Tensor:
        """Effective bias-free matrix ``W2 diag(mean GELU'(W1 x+b1)) W1``."""
        self._require_fit()
        if self._weight_cache is None:
            first = self.first_projection.weight.detach().to(self._dtype)
            last = self.last_projection.weight.detach().to(device=first.device, dtype=self._dtype)
            self._weight_cache = (last * self.mean_slope.unsqueeze(0)) @ first
        return self._weight_cache

    @property
    def dictionary_weight(self) -> Tensor:
        """Alias for ``weight``; each row is one visual-codebook readout atom."""
        return self.weight

    def __call__(self, raw_hidden: Tensor) -> Tensor:
        """Normalize raw hidden states, then apply the calibrated bias-free map."""
        normalized = self.norm(raw_hidden)
        weight = self.weight
        return F.linear(normalized.to(device=weight.device, dtype=weight.dtype), weight)

    def native(self, raw_hidden: Tensor) -> Tensor:
        """Full nonlinear native head, including its biases, for fidelity controls."""
        return self.last_projection(self.activation(self.first_projection(self.norm(raw_hidden))))

    def affine(self, raw_hidden: Tensor) -> Tensor:
        """Equation (10) affine control, centered at the calibration mean state."""
        self._require_fit()
        mean = (self._input_sum / self.n_samples).to(self.first_projection.weight)
        native_mean = self.last_projection(self.activation(self.first_projection(mean)))
        weight = self.weight
        offset = native_mean.to(weight) - F.linear(mean.to(weight), weight)
        return self(raw_hidden) + offset

    def save(self, path: str | Path) -> None:
        """Save calibration tensors and identity only; no backbone/head is serialized."""
        self._require_fit()
        required = {"model_id", "readout_id", "calibration_id"}
        if required - self.binding.keys() or any(not isinstance(v, str) or not v for v in self.binding.values()):
            raise ValueError("visual readout binding requires nonempty model_id, readout_id and calibration_id strings")
        if self._fingerprint() != self._calibrated_fingerprint:
            raise ValueError("native norm/head parameters changed after visual-readout calibration")
        artifact = {
            "format_version": 1, "method": "mj_visual_readout", "binding": self.binding,
            "config": self._config(), "native_fingerprint": self._calibrated_fingerprint,
            "n_samples": self.n_samples, "slope_sum": self._slope_sum.detach().cpu(),
            "input_sum": self._input_sum.detach().cpu(), "weight": self.weight.detach().cpu(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, path)

    def load(self, path: str | Path) -> "LinearizedVisualReadout":
        """Load only into exactly matching native head/norm, identity and dimensions."""
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(artifact, dict) or artifact.get("format_version") != 1 or artifact.get("method") != "mj_visual_readout":
            raise ValueError("visual readout artifact format or method does not match")
        required = {"model_id", "readout_id", "calibration_id"}
        if required - self.binding.keys() or artifact.get("binding") != self.binding:
            raise ValueError("visual readout binding does not match model/readout/calibration identity")
        if artifact.get("config") != self._config():
            raise ValueError("visual readout artifact dimensions or GELU configuration do not match")
        if artifact.get("native_fingerprint") != self._fingerprint():
            raise ValueError("visual readout native norm/head parameter fingerprint does not match")
        count, slopes, inputs, weight = (artifact.get(key) for key in ("n_samples", "slope_sum", "input_sum", "weight"))
        if type(count) is not int or count < 1:
            raise ValueError("visual readout artifact has an invalid sample count")
        for value, shape in ((slopes, (self.first_projection.out_features,)),
                             (inputs, (self.first_projection.in_features,)),
                             (weight, (self.last_projection.out_features, self.first_projection.in_features))):
            floating_tensor(value, "visual readout artifact tensor")
            if tuple(value.shape) != shape:
                raise ValueError("visual readout artifact tensor shape does not match the native head")
        device = self.first_projection.weight.device
        first = self.first_projection.weight.detach().cpu().to(self._dtype)
        last = self.last_projection.weight.detach().cpu().to(self._dtype)
        expected = (last * (slopes / count).to(self._dtype).unsqueeze(0)) @ first
        if not torch.allclose(weight.to(expected), expected, rtol=1e-5, atol=1e-6):
            raise ValueError("visual readout effective weight does not match its saved calibration slopes")
        self.n_samples = count
        self._slope_sum, self._input_sum = slopes.to(device), inputs.to(device)
        self._calibrated_fingerprint = artifact["native_fingerprint"]
        self._calibrated_versions = self._versions()
        self._weight_cache = weight.to(device=device, dtype=self._dtype)
        return self


def _decompose(values: Tensor, text: Tensor, visual: Tensor, *, k: int) -> dict[str, Tensor]:
    dictionary = torch.cat((text, visual), 0)
    norms = dictionary.norm(dim=-1)
    normalized = dictionary / norms.clamp_min(torch.finfo(dictionary.dtype).tiny).unsqueeze(-1)
    codes = orthogonal_matching_pursuit(
        values, normalized.T, iterations=min(k, len(dictionary)), signed=False, nnls_iterations=64,
    )
    total_mass = codes.sum(-1)
    valid = total_mass > 0
    mi = torch.where(valid, codes[..., len(text):].sum(-1) / total_mass.clamp_min(
        torch.finfo(codes.dtype).tiny), torch.full_like(total_mass, float("nan")))
    mass, indices = codes.topk(min(k, len(dictionary)), dim=-1)
    coefficients = mass / norms[indices].clamp_min(torch.finfo(norms.dtype).tiny)
    # Zero coefficients carry no support; -1 makes this explicit to consumers.
    active = mass > 0
    return {
        "mi": mi, "valid": valid, "reconstruction": codes @ normalized,
        "coefficients": coefficients, "atom_ids": indices.masked_fill(~active, -1),
        "visual_atoms": (indices >= len(text)) & active,
    }


@torch.no_grad()
def modality_index(
    activations: Tensor, text_dictionary: Tensor, visual_dictionary: Tensor, *,
    k: int = 25, seed: int = 0, size_match: bool = False, signed: bool = False,
) -> ProbeResult:
    """Sparse dictionary mass with the mandatory rotated visual-dictionary control.

    Dictionaries contain row atoms in the **same activation coordinates**,
    e.g. ``W_text diag(g) J_source_to_text`` and
    ``W_visual diag(g) J_source_to_visual``. Activations are ``[..., hidden]``;
    negative activation/atom coordinates are allowed, but coefficients must be
    nonnegative. Atoms are normalized before nonnegative OMP with 64 projected
    NNLS iterations per growing support. The paper does not specify this solver.

    ``raw_mi`` is a dictionary-allocation statistic, not a semantic verdict.
    ``excess_mi = raw_mi - rotated_mi`` is always returned. Zero reconstructed
    mass produces NaN and an explicit validity mask. ``size_match=True`` adds
    a separate result with text atoms deterministically subsampled to the visual
    dictionary size; primary results remain unchanged. This helper does not
    replace the paper's separately calibrated split-half sampling-noise control.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if signed is not False:
        raise ValueError("MJ modality index requires nonnegative coefficients; signed pursuit is not supported")
    for name, value in (("activations", activations), ("text_dictionary", text_dictionary),
                        ("visual_dictionary", visual_dictionary)):
        floating_tensor(value, name)
    if activations.ndim < 1 or text_dictionary.ndim != 2 or visual_dictionary.ndim != 2:
        raise ValueError("activations must be [..., hidden] and both dictionaries must be [atoms, hidden]")
    hidden = activations.shape[-1]
    if text_dictionary.shape[-1] != hidden or visual_dictionary.shape[-1] != hidden:
        raise ValueError("both dictionaries must use the same hidden coordinates as activations")
    if size_match and len(text_dictionary) < len(visual_dictionary):
        raise ValueError("size-matched text subsampling requires at least as many text as visual atoms")
    dtype = torch.float64 if activations.dtype == torch.float64 else torch.float32
    values = activations.detach().to(dtype=dtype)
    text = text_dictionary.detach().to(device=values.device, dtype=dtype)
    visual = visual_dictionary.detach().to(device=values.device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rotation, upper = torch.linalg.qr(torch.randn(hidden, hidden, generator=generator, dtype=torch.float64))
    # A deterministic Haar orthogonal draw; signs remove QR convention bias.
    signs = upper.diagonal().sign()
    rotation = (rotation * torch.where(signs == 0, torch.ones_like(signs), signs)).to(values)
    rotated_visual = visual @ rotation
    primary = _decompose(values, text, visual, k=k)
    rotated = _decompose(values, text, rotated_visual, k=k)
    tensors = {"raw_mi": primary.pop("mi"), "rotated_mi": rotated.pop("mi"),
               "rotation": rotation}
    tensors.update(primary)
    tensors.update({f"rotated_{name}": value for name, value in rotated.items()})
    tensors["excess_mi"] = tensors["raw_mi"] - tensors["rotated_mi"]
    tensors["excess_valid"] = tensors["valid"] & tensors["rotated_valid"]
    if size_match:
        indices = torch.randperm(len(text), generator=generator)[:len(visual)].to(values.device)
        matched = _decompose(values, text[indices], visual, k=k)
        matched_rotated = _decompose(values, text[indices], rotated_visual, k=k)
        tensors["size_matched_text_ids"] = indices
        tensors.update({f"size_matched_{name}": value for name, value in matched.items()})
        tensors.update({f"size_matched_rotated_{name}": value for name, value in matched_rotated.items()})
        tensors["size_matched_raw_mi"] = tensors.pop("size_matched_mi")
        tensors["size_matched_excess_mi"] = tensors["size_matched_raw_mi"] - tensors["size_matched_rotated_mi"]
        tensors["size_matched_excess_valid"] = tensors["size_matched_valid"] & tensors["size_matched_rotated_valid"]
    return ProbeResult("mj_modality_index", tensors, {
        "k": k, "seed": seed, "text_atoms": len(text), "visual_atoms": len(visual),
        "solver": "library_nonnegative_omp_projected_nnls", "nnls_iterations": 64,
        "dictionary_normalization": "unit row norms; coefficients converted to original atom units",
        "reported_statistic": "excess_mi (raw minus rotated visual dictionary control)",
        "size_matched": size_match, "split_half_control": "requires independently calibrated operators",
        "interpretation": "dictionary allocation; raw_mi alone is not evidence of visual semantic content",
    })
