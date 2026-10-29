"""Native SAE methods with explicit data/model binding and raw-space IO.

The trainable dictionary operates on ``raw_activation / input_scale``. The
single scalar scale is fitted on training activations only and is saved as a
buffer. ``decode`` returns native activation coordinates, including that scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ..core.precision import analysis_dtype
from .algorithms import JumpReLU, Step, batch_topk, matching_pursuit, orthogonal_matching_pursuit, topk

ARCHITECTURES = ("relu", "topk", "gated", "jumprelu", "abstopk", "batchtopk",
                 "matryoshka_batchtopk", "matching_pursuit", "archetypal_topk",
                 "archetypal_jumprelu", "signed_matching_pursuit",
                 "orthogonal_matching_pursuit", "signed_orthogonal_matching_pursuit", "qsae")


@dataclass(frozen=True)
class SAEConfig:
    architecture: str
    d_in: int
    d_sae: int
    k: int = 32
    l1_coefficient: float = .01
    seed: int = 0
    normalization: str | None = None
    input_normalization: str = "scalar"
    normalization_eps: float = 1e-5
    normalization_std_correction: int = 0
    apply_decoder_bias_to_input: bool = True
    apply_b_dec_to_input: bool | None = None
    use_decoder_bias: bool = True
    normalize_decoder: bool = True
    rescale_acts_by_decoder_norm: bool = False
    auxk_coefficient: float = 0.
    auxk_k: int | None = None
    dead_feature_steps: int = 100
    jumprelu_threshold: float = .1
    jumprelu_bandwidth: float = .05
    l0_coefficient: float = .01
    jumprelu_sparsity: str = "l0"
    jumprelu_ste_to_input: bool = False
    batch_budget: str = "average"
    threshold_ema: float = .99
    threshold_init: str = "first_batch"
    matryoshka_widths: tuple[int, ...] = ()
    matryoshka_weights: tuple[float, ...] = ()
    dictionary: str = "learned"
    archetypal_candidates: int = 128
    archetypal_delta: float = 1.
    pursuit_iterations: int | None = None
    residual_tolerance: float = .01
    stop_on_duplicate_support: bool = True
    omp_nnls_iterations: int = 32
    quantization_levels: int = 4
    hard_quantization: bool = False

    def __post_init__(self):
        aliases = {"standard": "relu", "abs_topk": "abstopk", "mp": "matching_pursuit",
                   "omp": "orthogonal_matching_pursuit", "signed_mp": "signed_matching_pursuit",
                   "signed_omp": "signed_orthogonal_matching_pursuit"}
        object.__setattr__(self, "architecture", aliases.get(self.architecture, self.architecture))
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if any(type(value) is not int or value < 1 for value in (self.d_in, self.d_sae)):
            raise ValueError("d_in and d_sae must be positive integers")
        limited = self.activation in {"topk", "abstopk"} or (
            self.activation in {"batchtopk", "matryoshka_batchtopk"} and self.batch_budget == "average")
        if type(self.k) is not int or self.k < 1 or (limited and self.k > self.d_sae):
            raise ValueError("k must be positive and cannot exceed d_sae for TopK")
        if not math.isfinite(self.l1_coefficient) or self.l1_coefficient < 0:
            raise ValueError("l1_coefficient must be finite and nonnegative")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        for name in ("auxk_coefficient", "l0_coefficient", "archetypal_delta", "residual_tolerance", "jumprelu_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        normalizers = {"rms": "scalar", "none": "none", "layer_norm": "layer_norm", "constant_norm_rescale": "constant_norm"}
        if self.normalization is not None:
            if self.normalization not in normalizers:
                raise ValueError("normalization must be rms, none, layer_norm or constant_norm_rescale")
            object.__setattr__(self, "input_normalization", normalizers[self.normalization])
        if self.input_normalization not in {"scalar", "none", "layer_norm", "constant_norm"}:
            raise ValueError("unknown input normalization recipe")
        if self.apply_b_dec_to_input is not None:
            if type(self.apply_b_dec_to_input) is not bool:
                raise ValueError("apply_b_dec_to_input must be boolean")
            object.__setattr__(self, "apply_decoder_bias_to_input", self.apply_b_dec_to_input)
        for name in ("apply_decoder_bias_to_input", "use_decoder_bias", "normalize_decoder", "rescale_acts_by_decoder_norm", "jumprelu_ste_to_input", "stop_on_duplicate_support", "hard_quantization"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("normalization_eps", "jumprelu_bandwidth"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.normalization_std_correction not in (0, 1) or self.d_in <= self.normalization_std_correction:
            raise ValueError("normalization correction must be 0 or 1 and smaller than d_in")
        if self.batch_budget not in {"average", "global"} or self.threshold_init not in {"first_batch", "zero_ema"}:
            raise ValueError("batch_budget/threshold_init must identify explicit valid recipes")
        if not math.isfinite(self.threshold_ema) or not 0 <= self.threshold_ema < 1:
            raise ValueError("threshold_ema must be in [0,1)")
        if self.jumprelu_sparsity not in {"l0", "l1"}:
            raise ValueError("jumprelu_sparsity must be l0 or l1")
        for name in ("dead_feature_steps", "archetypal_candidates", "omp_nnls_iterations", "quantization_levels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.quantization_levels < 2:
            raise ValueError("quantization_levels must be at least two")
        for name in ("auxk_k", "pursuit_iterations"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.dictionary not in {"learned", "archetypal"}:
            raise ValueError("dictionary must be learned or archetypal")
        if self.architecture.startswith("archetypal_"):
            object.__setattr__(self, "dictionary", "archetypal")
        widths, weights = tuple(self.matryoshka_widths), tuple(self.matryoshka_weights)
        if self.activation == "matryoshka_batchtopk" and not widths:
            widths = tuple(sorted(set((max(1, self.d_sae // 2), self.d_sae))))
        if widths and (any(type(n) is not int or not 1 <= n <= self.d_sae for n in widths) or tuple(sorted(set(widths))) != widths or widths[-1] != self.d_sae):
            raise ValueError("matryoshka widths must increase strictly and end at d_sae")
        if weights and (len(weights) != len(widths) or any(not math.isfinite(w) or w <= 0 for w in weights)):
            raise ValueError("matryoshka weights must be positive and match widths")
        object.__setattr__(self, "matryoshka_widths", widths)
        object.__setattr__(self, "matryoshka_weights", weights or tuple(1. for _ in widths))

    @property
    def activation(self):
        return self.architecture.removeprefix("archetypal_")


def validate_sae_binding(binding: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-normalized exact identity; never infer hook compatibility."""
    if not isinstance(binding, dict) or not binding or any(not isinstance(key, str) for key in binding):
        raise ValueError("SAE binding must be a nonempty JSON dictionary with string keys")
    try:
        value = json.loads(json.dumps(binding, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("SAE binding must contain finite JSON values") from error
    for name in ("model_id", "tokenizer_id", "preprocessing_id", "calibration_id", "site"):
        item = value.get(name)
        if not isinstance(item, str) or not item.strip() or item.strip().lower() == "unspecified":
            raise ValueError(f"SAE binding requires an explicit nonempty {name}")
    if type(value.get("layer")) is not int or value["layer"] < 0:
        raise ValueError("SAE binding requires a nonnegative integer layer")
    hook = value.get("hook")
    if not isinstance(hook, dict) or not {"module", "kind", "selector", "tensor_slice"} <= hook.keys():
        raise ValueError("SAE binding hook requires actual module, kind, selector and tensor_slice")
    if not isinstance(hook["module"], str) or not hook["module"].strip() or hook["kind"] not in {"input", "output"}:
        raise ValueError("SAE binding must identify the actual nonempty module and hook kind")
    if hook["selector"] is not None and not isinstance(hook["selector"], (int, str)):
        raise ValueError("hook selector must be an integer, string or None")
    if hook["kind"] == "input" and hook["selector"] is None:
        raise ValueError("input hook binding requires its positional or keyword selector")
    selection = hook["tensor_slice"]
    if selection is not None and (not isinstance(selection, list) or len(selection) != 2
            or any(type(item) is not int for item in selection) or not 0 <= selection[0] < selection[1]):
        raise ValueError("hook tensor_slice must be None or [start, stop] with 0 <= start < stop")
    return value


class BaseSAE(nn.Module, ABC):
    """One method-family contract; algorithm origins do not choose execution."""
    family = "sae"

    @property
    def activation_parameter(self):
        """Parameter specifying activation device/dtype, independent of thresholds."""
        return self.encoder.weight

    @abstractmethod
    def encode(self, raw):
        raise NotImplementedError

    @abstractmethod
    def decode(self, features):
        raise NotImplementedError


class ArchetypalDecoder(nn.Module):
    """Convex TRAIN-anchor combinations plus a bounded additive relaxation."""
    def __init__(self, config):
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros(config.d_sae, config.archetypal_candidates))
        self.relaxation = nn.Parameter(torch.zeros(config.d_in, config.d_sae))
        self.bias = nn.Parameter(torch.zeros(config.d_in))
        self.register_buffer("anchors", torch.zeros(config.archetypal_candidates, config.d_in))
        self.register_buffer("anchors_initialized", torch.tensor(False))

    @property
    def weight(self):
        if not bool(self.anchors_initialized):
            raise ValueError("archetypal dictionary requires initialize_dictionary_ on TRAIN activations")
        return (self.coefficients.softmax(-1) @ self.anchors).T + self.relaxation

    def forward(self, features):
        return F.linear(features, self.weight, self.bias)


class SparseAutoencoder(BaseSAE):
    """Algorithm-composed dictionary with one trainer/artifact contract.

    Supported recipes and primary sources are exposed by ``registry``. Input
    normalization, sparsity budgets, losses and constraints are explicit config
    fields, not inferred from a checkpoint's source project.
    """

    def __init__(self, config: SAEConfig, binding: dict[str, Any],
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        if not isinstance(config, SAEConfig):
            raise TypeError("config must be SAEConfig")
        if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError("SAE dtype must be a supported floating-point dtype")
        self.config, self.binding = config, validate_sae_binding(binding)
        self.is_fitted = False
        self.source_metadata = {"implementation": "vlmprobe-native", "method": config.architecture}
        # Local CPU initialization restores global RNG state and avoids seeding
        # unrelated CUDA workloads owned by the caller.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.encoder = nn.Linear(config.d_in, config.d_sae, bias=True)
            self.decoder = nn.Linear(config.d_sae, config.d_in, bias=True)
            if config.activation == "gated":
                self.magnitude_bias = nn.Parameter(torch.zeros(config.d_sae))
                self.log_magnitude_scale = nn.Parameter(torch.zeros(config.d_sae))
            self.register_buffer("input_scale", torch.tensor(1.))
            self.register_buffer("training_steps", torch.tensor(0, dtype=torch.long))
            self.register_buffer("inactive_steps", torch.zeros(config.d_sae, dtype=torch.long))
            self.register_buffer("batch_threshold", torch.tensor(0.))
            self.register_buffer("batch_threshold_initialized", torch.tensor(False))
            if config.activation == "jumprelu":
                self.threshold = nn.Parameter(torch.full((config.d_sae,), config.jumprelu_threshold))
            if config.activation == "qsae":
                self.quantization_levels = nn.Parameter(torch.linspace(0., 1., config.quantization_levels))
            with torch.no_grad():
                nn.init.normal_(self.decoder.weight)
                self.normalize_decoder_()
                self.encoder.weight.copy_(self.decoder.weight.T)
                self.encoder.bias.zero_()
                self.decoder.bias.zero_()
            if config.dictionary == "archetypal":
                self.decoder = ArchetypalDecoder(config)
            if "matching_pursuit" in config.activation:
                self.encoder.requires_grad_(False)
        self.to(device=device, dtype=dtype)

    @property
    def implementation_variant(self):
        if self.config.architecture == "topk" and self.config.auxk_coefficient > 0:
            return "per_token_topk_relu_core_mse_with_explicit_auxk"
        variants = {"relu": "relu_l1_unit_decoder_columns",
                    "topk": "per_token_topk_relu_core_mse_without_auxk",
                    "gated": "gated_frozen_decoder_aux_and_detached_aux_centering"}
        return variants.get(self.config.architecture, "native_" + self.config.architecture)

    def _input(self, value: Tensor, width: int, label: str) -> Tensor:
        if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim < 1 or value.shape[-1] != width:
            raise ValueError(f"{label} must be floating [...,{width}]")
        if value.numel() == 0 or not torch.isfinite(value).all():
            raise ValueError(f"{label} must contain finite, nonempty values")
        return value.to(self.encoder.weight)

    def _apply(self, function, recurse=True):
        result = super()._apply(function, recurse=recurse)
        # Mixed-precision dictionary weights must not quantize learned jump
        # thresholds. Float64 remains available for precise numerical tests.
        if hasattr(self, "threshold") and self.threshold.dtype in {torch.float16, torch.bfloat16}:
            self.threshold.data = self.threshold.data.float()
        return result

    def _normalize(self, raw):
        values = raw / self.input_scale
        if self.config.input_normalization in {"scalar", "none"}:
            return values, None
        if self.config.input_normalization == "layer_norm":
            mean = values.mean(-1, keepdim=True)
            std = values.std(-1, correction=self.config.normalization_std_correction, keepdim=True)
            return (values - mean) / (std + self.config.normalization_eps), {"mean": mean, "std": std}
        norm = values.norm(dim=-1, keepdim=True)
        if bool((norm == 0).any()):
            raise ValueError("constant_norm normalization is undefined for a zero activation vector")
        coefficient = math.sqrt(self.config.d_in) / norm
        return values * coefficient, {"coefficient": coefficient}

    def _denormalize(self, values, context):
        recipe = self.config.input_normalization
        if recipe == "layer_norm":
            if not isinstance(context, dict) or not {"mean", "std"} <= context.keys():
                raise ValueError("layer_norm decoding requires the matching normalization context")
            values = values * context["std"] + context["mean"]
        elif recipe == "constant_norm":
            if not isinstance(context, dict) or "coefficient" not in context:
                raise ValueError("constant_norm decoding requires the matching normalization context")
            values = values / context["coefficient"]
        return values * self.input_scale

    def _decoder_weight(self):
        weight = self.decoder.weight
        if self.config.rescale_acts_by_decoder_norm:
            weight = weight / weight.norm(dim=0, keepdim=True).clamp_min(torch.finfo(weight.dtype).eps)
        return weight

    def _decode_normalized(self, features):
        if self.config.use_decoder_bias and not self.config.rescale_acts_by_decoder_norm:
            return self.decoder(features)
        return F.linear(features, self._decoder_weight(), self.decoder.bias if self.config.use_decoder_bias else None)

    def _encode_normalized(self, normalized: Tensor) -> tuple[Tensor, Tensor]:
        centered = normalized - self.decoder.bias if self.config.apply_decoder_bias_to_input else normalized
        activation = self.config.activation
        gate = normalized.new_zeros(*normalized.shape[:-1], self.config.d_sae) if "matching_pursuit" in activation else self.encoder(centered)
        if self.config.dictionary == "archetypal" and not bool(self.decoder.anchors_initialized):
            raise ValueError("archetypal dictionary requires TRAIN anchor initialization")
        if activation == "relu":
            features = gate.relu()
        elif activation in {"topk", "abstopk"}:
            ranked = gate * self.decoder.weight.norm(dim=0) if self.config.rescale_acts_by_decoder_norm else gate
            features = topk(ranked, self.config.k, signed=activation == "abstopk")
        elif activation == "gated":
            weight = self.encoder.weight * self.log_magnitude_scale.exp()[:, None]
            magnitude = F.linear(centered, weight, self.magnitude_bias).relu()
            features = (gate > 0).to(magnitude.dtype) * magnitude
        elif activation == "jumprelu":
            features = JumpReLU.apply(gate, self.threshold.clamp_min(0), self.config.jumprelu_bandwidth,
                                     self.config.jumprelu_ste_to_input)
        elif activation in {"batchtopk", "matryoshka_batchtopk"}:
            if self.training:
                positions = gate.numel() // self.config.d_sae
                budget = self.config.k * positions if self.config.batch_budget == "average" else self.config.k
                features, _ = batch_topk(gate, budget)
            else:
                if not bool(self.batch_threshold_initialized):
                    raise ValueError("BatchTopK evaluation requires a threshold calibrated on TRAIN batches")
                features = gate.relu() * (gate > self.batch_threshold)
        elif "matching_pursuit" in activation:
            function = orthogonal_matching_pursuit if "orthogonal" in activation else matching_pursuit
            kwargs = {"nnls_iterations": self.config.omp_nnls_iterations} if "orthogonal" in activation else {
                "stop_on_duplicate": self.config.stop_on_duplicate_support}
            features = function(centered, self.decoder.weight, iterations=self.config.pursuit_iterations or self.config.k,
                signed=activation.startswith("signed_"), tolerance=self.config.residual_tolerance, **kwargs)
        else:
            distances = (gate[..., None] - self.quantization_levels).square()
            quantized = self.quantization_levels[distances.argmin(-1)] if self.config.hard_quantization else (
                distances.neg().softmax(-1) * self.quantization_levels).sum(-1)
            positive = gate.relu()
            features = (positive + quantized - positive.detach()).relu()
        return features, gate

    def encode_with_context(self, raw):
        raw = self._input(raw, self.config.d_in, "activations")
        normalized, context = self._normalize(raw)
        return self._encode_normalized(normalized)[0], context

    def decode_with_context(self, features, context):
        features = self._input(features, self.config.d_sae, "features")
        return self._denormalize(self._decode_normalized(features), context)

    def encode(self, raw: Tensor) -> Tensor:
        return self.encode_with_context(raw)[0]

    def decode(self, features: Tensor) -> Tensor:
        return self.decode_with_context(features, None)

    def forward(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        features, context = self.encode_with_context(raw)
        return self.decode_with_context(features, context), features

    def loss(self, raw: Tensor) -> dict[str, Tensor]:
        raw = self._input(raw, self.config.d_in, "activations")
        normalized, _ = self._normalize(raw)
        features, gate = self._encode_normalized(normalized)
        decoded = self._decode_normalized(features)
        # Neural projections keep the dictionary's native dtype. Loss
        # differences, products and reductions are analysis arithmetic and
        # must be widened before squaring or summing, not after overflow.
        dtype = analysis_dtype(normalized, decoded, features, gate)
        target = normalized.to(dtype)
        feature_values = features.to(dtype)
        reconstruction = (decoded.to(dtype) - target).square().sum(-1).mean()
        zero = reconstruction.new_zeros(())
        sparsity, auxiliary = zero, zero
        activation = self.config.activation
        if activation == "relu":
            sparsity = self.config.l1_coefficient * feature_values.abs().sum(-1).mean()
        elif activation == "gated":
            sparsity = self.config.l1_coefficient * gate.to(dtype).relu().sum(-1).mean()
            # The discrete main gate has no useful threshold gradient. The
            # rectified auxiliary gate learns reconstruction through a fixed
            # decoder, preventing that auxiliary task from moving its columns.
            centered = normalized - self.decoder.bias.detach() if self.config.apply_decoder_bias_to_input else normalized
            auxiliary_gate = F.linear(centered,
                                      self.encoder.weight, self.encoder.bias).relu()
            auxiliary_reconstruction = F.linear(auxiliary_gate,
                self._decoder_weight().detach(), self.decoder.bias.detach() if self.config.use_decoder_bias else None)
            auxiliary = (auxiliary_reconstruction.to(dtype) - target).square().sum(-1).mean()
        elif activation == "jumprelu":
            if self.config.jumprelu_sparsity == "l0":
                count = Step.apply(gate, self.threshold.clamp_min(0),
                    self.config.jumprelu_bandwidth, self.config.jumprelu_ste_to_input)
                sparsity = self.config.l0_coefficient * count.to(dtype).sum(-1).mean()
            else:
                sparsity = self.config.l1_coefficient * feature_values.abs().sum(-1).mean()
        elif activation == "matryoshka_batchtopk":
            reconstruction = reconstruction * self.config.matryoshka_weights[-1]
            for width, weight in zip(self.config.matryoshka_widths[:-1], self.config.matryoshka_weights[:-1]):
                inner = F.linear(features[..., :width], self._decoder_weight()[:, :width],
                                 self.decoder.bias if self.config.use_decoder_bias else None)
                reconstruction = reconstruction + weight * (inner.to(dtype) - target).square().sum(-1).mean()
        if self.config.auxk_coefficient and activation in {"topk", "abstopk", "batchtopk", "matryoshka_batchtopk"}:
            current_firing = (features.detach().reshape(-1, self.config.d_sae) != 0).any(0)
            dead = (self.inactive_steps >= self.config.dead_feature_steps) & ~current_firing
            dead_count = int(dead.sum())
            if dead_count:
                count = min(self.config.auxk_k or max(1, self.config.d_in // 2), dead_count)
                selected = topk(gate[..., dead], count, signed=activation == "abstopk")
                residual = (target - decoded.to(dtype)).detach()
                auxiliary = auxiliary + self.config.auxk_coefficient * (
                    F.linear(selected, self._decoder_weight()[:, dead]).to(dtype) - residual).square().sum(-1).mean()
        if self.training:
            with torch.no_grad():
                self.training_steps.add_(1)
                fired = (features.reshape(-1, self.config.d_sae) != 0).any(0)
                self.inactive_steps.add_(1)
                self.inactive_steps.masked_fill_(fired, 0)
                if activation in {"batchtopk", "matryoshka_batchtopk"}:
                    positive = features[features > 0]
                    if positive.numel():
                        current = positive.min().to(self.batch_threshold)
                        if not bool(self.batch_threshold_initialized) and self.config.threshold_init == "first_batch":
                            self.batch_threshold.copy_(current)
                        else:
                            self.batch_threshold.mul_(self.config.threshold_ema).add_(current * (1 - self.config.threshold_ema))
                        self.batch_threshold_initialized.fill_(True)
        return {"loss": reconstruction + sparsity + auxiliary,
                "reconstruction": reconstruction, "sparsity": sparsity,
                "auxiliary": auxiliary}

    @torch.no_grad()
    def normalize_decoder_(self) -> "SparseAutoencoder":
        """Project every decoder feature direction (column) to unit L2 norm."""
        if isinstance(self.decoder, ArchetypalDecoder):
            norm = self.decoder.relaxation.norm(dim=0, keepdim=True)
            self.decoder.relaxation.mul_((self.config.archetypal_delta / norm.clamp_min(1e-12)).clamp_max(1))
            return self
        if not self.config.normalize_decoder:
            return self
        norm = self.decoder.weight.norm(dim=0, keepdim=True)
        if not torch.isfinite(norm).all() or (norm <= 0).any():
            raise ValueError("decoder columns must be finite and nonzero before normalization")
        self.decoder.weight.div_(norm)
        return self

    @torch.no_grad()
    def post_optimizer_step_(self):
        self.normalize_decoder_()
        if hasattr(self, "threshold"):
            self.threshold.clamp_(min=0)
        return self

    @torch.no_grad()
    def calibrate_(self, raw_training_points):
        """Initialize a BatchTopK evaluation threshold on TRAIN data only.

        This runs no optimizer and changes no dead-feature/step counters. The
        shared trainer may call it before the initial validation baseline.
        """
        if self.config.activation not in {"batchtopk", "matryoshka_batchtopk"}:
            return self
        if self.is_fitted:
            raise ValueError("cannot recalibrate a fitted dictionary implicitly")
        values = self._input(raw_training_points, self.config.d_in, "TRAIN calibration")
        normalized, _ = self._normalize(values)
        centered = normalized - self.decoder.bias if self.config.apply_decoder_bias_to_input else normalized
        gate = self.encoder(centered)
        positions = gate.numel() // self.config.d_sae
        budget = self.config.k * positions if self.config.batch_budget == "average" else self.config.k
        _, threshold = batch_topk(gate, budget)
        self.batch_threshold.copy_(threshold if threshold is not None else torch.zeros_like(self.batch_threshold))
        self.batch_threshold_initialized.fill_(True)
        return self

    @torch.no_grad()
    def initialize_dictionary_(self, raw_training_points):
        """Select deterministic candidate anchors exclusively from TRAIN vectors."""
        if not isinstance(self.decoder, ArchetypalDecoder):
            return self
        if self.is_fitted:
            raise ValueError("cannot replace anchors of a fitted dictionary")
        points = self._input(raw_training_points, self.config.d_in, "TRAIN anchors")
        if points.ndim != 2:
            raise ValueError("TRAIN anchors must have shape [positions,d_in]")
        normalized, _ = self._normalize(points)
        if self.config.apply_decoder_bias_to_input:
            normalized = normalized - self.decoder.bias
        count = self.config.archetypal_candidates
        indices = torch.linspace(0, len(points) - 1, count, device=points.device).round().long()
        self.decoder.anchors.copy_(normalized[indices])
        self.decoder.anchors_initialized.fill_(True)
        self.decoder.coefficients.fill_(-4.)
        self.decoder.coefficients[torch.arange(self.config.d_sae, device=points.device),
                                  torch.arange(self.config.d_sae, device=points.device) % count] = 4.
        return self

    @torch.no_grad()
    def initialize_bias_(self, raw_mean: Tensor) -> "SparseAutoencoder":
        """Initialize the normalized decoder bias from the raw TRAIN mean."""
        if self.is_fitted:
            raise ValueError("bias initialization requires an unfitted dictionary")
        raw_mean = self._input(raw_mean, self.config.d_in, "training mean")
        if raw_mean.ndim != 1:
            raise ValueError("training mean must be one [d_in] vector")
        if self.config.input_normalization in {"scalar", "none"}:
            self.decoder.bias.copy_(raw_mean / self.input_scale)
        else:
            self.decoder.bias.zero_()
        return self

    @torch.no_grad()
    def set_normalization_(self, train_rms: float | Tensor) -> "SparseAutoencoder":
        """Set raw-coordinate scalar RMS fitted on the TRAIN split only.

        Call before ``initialize_bias_`` and optimization. This changes the
        preprocessing contract; it is not an inference-time rescaling option.
        """
        if self.is_fitted:
            raise ValueError("normalization changes require an unfitted dictionary")
        scale = torch.as_tensor(train_rms, device=self.input_scale.device, dtype=self.input_scale.dtype)
        if scale.numel() != 1 or not torch.isfinite(scale).all() or scale <= 0:
            raise ValueError("training RMS must be a finite positive scalar")
        self.input_scale.copy_(scale.reshape(()))
        return self

    def save(self, path: str | Path):
        from .artifacts import save_artifact
        return save_artifact(self, path)

    @classmethod
    def load(cls, path: str | Path, expected_binding=None, device="cpu"):
        from .artifacts import load_artifact
        return load_artifact(path, expected_binding=expected_binding, device=device)


def upgrade_legacy_state_dict(model, state):
    """Fill only new bookkeeping buffers in legacy native v1 artifacts."""
    result = dict(state)
    if model.config.architecture not in {"relu", "topk", "gated"}:
        return result
    defaults = model.state_dict()
    for name in ("training_steps", "inactive_steps", "batch_threshold", "batch_threshold_initialized"):
        result.setdefault(name, defaults[name])
    return result
