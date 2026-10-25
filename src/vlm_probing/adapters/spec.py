"""Explicit sequence semantics and model capabilities for the public API."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


class CapabilityError(ValueError):
    """The adapter cannot provide the computation requested by a method."""


@dataclass
class TokenLayout:
    """Masks use the model's expanded sequence coordinates, including padding.

    Without ``prompt``, all valid positions are treated as prompt positions.
    Set it explicitly when inputs include a teacher-forced answer. ``token_ids``
    is optional but required by sequence scoring and automatic pair alignment.
    """

    valid: Tensor
    visual: Tensor | None = None
    prompt: Tensor | None = None
    token_ids: Tensor | None = None

    def checked(self, shape, device) -> "TokenLayout":
        values = {}
        for name in ("valid", "visual", "prompt"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, Tensor) or value.dtype != torch.bool or tuple(value.shape) != tuple(shape):
                    raise ValueError(f"layout.{name} must be boolean {tuple(shape)}")
                values[name] = value.to(device)
        if "valid" not in values:
            raise ValueError("layout.valid is required")
        valid = values["valid"]
        if not valid.any(-1).all():
            raise ValueError("each sequence must have at least one valid token")
        for name in ("visual", "prompt"):
            if name in values and (values[name] & ~valid).any():
                raise ValueError(f"layout.{name} selects padding")
        ids = self.token_ids
        if ids is not None:
            if not isinstance(ids, Tensor) or ids.dtype != torch.long or tuple(ids.shape) != tuple(shape):
                raise ValueError("layout.token_ids must be aligned long token IDs")
            ids = ids.to(device)
        return TokenLayout(**values, token_ids=ids)

    def select(self, tokens="all") -> Tensor:
        valid = self.valid
        if isinstance(tokens, str):
            if tokens == "all":
                result = valid.clone()
            elif tokens in {"visual", "text"}:
                if self.visual is None:
                    raise CapabilityError("visual/text selection requires an explicit visual token layout")
                result = self.visual if tokens == "visual" else valid & ~self.visual
            elif tokens in {"last_prompt", "prediction"}:
                prompt = valid if self.prompt is None else self.prompt
                if not prompt.any(-1).all():
                    raise ValueError("each example needs a prompt token")
                index = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
                last = index.masked_fill(~prompt, -1).max(-1).values
                result = torch.zeros_like(valid).scatter_(1, last[:, None], True)
                if tokens == "prediction":
                    # At generation step t the model predicts from the context's
                    # last token and subsequent generated positions. Replaying
                    # these positions preserves prior causal steering history.
                    result = result | (valid & ~prompt)
            else:
                raise ValueError(f"unknown token selector {tokens!r}")
        elif isinstance(tokens, Tensor) and tokens.dtype == torch.bool:
            if tokens.shape != valid.shape:
                raise ValueError("token mask must match the expanded sequence")
            result = tokens.to(valid.device)
        else:
            positions = [tokens] if isinstance(tokens, int) else list(tokens)
            if not positions or any(type(p) is not int or not 0 <= p < valid.shape[1] for p in positions):
                raise ValueError("token positions must be nonnegative expanded-sequence indices")
            result = torch.zeros_like(valid)
            result[:, positions] = True
        if (result & ~valid).any():
            raise ValueError("token selector includes padding")
        if not result.any():
            raise ValueError("token selection is empty")
        return result


@dataclass
class ProbeInputs:
    """Native model kwargs plus optional explicit expanded-token metadata."""

    kwargs: Mapping[str, Any]
    layout: TokenLayout | None = None


@dataclass(frozen=True)
class HeadSite:
    """An editable head axis, flattened [B,T,H*D] or explicit [B,T,H,D].

    Counts describe the physical tensor: Q uses query heads; K/V use KV heads
    in grouped-query attention. No projection or dimension inference is applied.
    """

    site: str
    heads: int
    head_dim: int

    def __post_init__(self):
        if not isinstance(self.site, str) or not self.site:
            raise ValueError("HeadSite.site must be a nonempty hook name")
        if any(type(n) is not int or n < 1 for n in (self.heads, self.head_dim)):
            raise ValueError("HeadSite heads and head_dim must be positive integers")

    def view(self, value: Tensor) -> Tensor:
        if value.ndim == 3 and value.shape[-1] == self.heads * self.head_dim:
            return value.reshape(*value.shape[:2], self.heads, self.head_dim)
        if value.ndim == 4 and value.shape[-2:] == (self.heads, self.head_dim):
            return value
        raise ValueError(f"{self.site!r} must be [B,T,{self.heads * self.head_dim}] "
                         f"or [B,T,{self.heads},{self.head_dim}], got {tuple(value.shape)}")


@dataclass
class MJSpec:
    """Audited MJ-Lens transport sites and output dictionaries.

    ``sources[i]`` is the input to decoder block i; ``final`` is the input to
    the model's final RMS normalization. These indices deliberately differ
    from the ordinary, post-block ``ModelSpec.residuals`` convention. Visual
    code IDs belong to ``visual_head``'s codebook, not the text vocabulary.
    """

    sources: dict[int, str]
    final: str
    norm: Callable[[Tensor], Tensor]
    text_unembedding: Tensor
    norm_gain: Tensor
    visual_head: torch.nn.Module | None = None
    decode_visual_codes: Callable[[Tensor], Tensor] | None = None

    def validate(self):
        if (not isinstance(self.sources, dict) or not self.sources
                or any(type(i) is not int or i < 0 for i in self.sources)
                or any(not isinstance(site, str) or not site for site in self.sources.values())):
            raise ValueError("MJSpec.sources requires nonnegative block-input indices and hook names")
        if not isinstance(self.final, str) or not self.final:
            raise ValueError("MJSpec.final must name the final norm input hook")
        if not callable(self.norm):
            raise TypeError("MJSpec.norm must be callable")
        if (not isinstance(self.text_unembedding, Tensor)
                or not self.text_unembedding.is_floating_point()
                or self.text_unembedding.ndim != 2 or not self.text_unembedding.numel()):
            raise ValueError("MJSpec.text_unembedding must be a floating [vocabulary, hidden] tensor")
        if (not isinstance(self.norm_gain, Tensor) or not self.norm_gain.is_floating_point()
                or self.norm_gain.ndim != 1
                or self.norm_gain.shape[0] != self.text_unembedding.shape[1]):
            raise ValueError("MJSpec.norm_gain must be a floating vector matching the readout hidden width")
        if self.visual_head is not None and not isinstance(self.visual_head, torch.nn.Module):
            raise TypeError("MJSpec.visual_head must be a torch.nn.Module or None")
        if self.decode_visual_codes is not None and not callable(self.decode_visual_codes):
            raise TypeError("MJSpec.decode_visual_codes must be callable or None")


@dataclass
class ModelSpec:
    """Site names reference an adapter's hook table; no shape-based inference.

    Head sites must produce [B,T,H,D_residual], or ``project_heads`` must convert
    them to that space. Attention sites are [B,H,Q,K]. Only names explicitly in
    ``editable_attention`` may be used for probability interventions. ``edges``
    declares independently replaceable messages, not arbitrary node caches.
    """

    residuals: dict[int, str]
    attentions: dict[int, str] = field(default_factory=dict)
    attention_scores: dict[int, str] = field(default_factory=dict)
    heads: dict[int, str] = field(default_factory=dict)
    embeddings: str | None = None
    input_embeddings: Tensor | None = None
    layout: Callable[[Mapping[str, Any]], TokenLayout] | None = None
    project_heads: Callable[[int, Tensor], Tensor] | None = None
    linear_readout: Callable[[Tensor], tuple[Tensor, Tensor, bool]] | None = None
    editable_attention: set[int] = field(default_factory=set)
    edges: dict[str, str] = field(default_factory=dict)
    # Metadata compared before automatic aligned patching (e.g. visual grids).
    alignment_keys: tuple[str, ...] = ()
    forward_defaults: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    # Explicit IOI path controls. Output sites must cover every decoder layer.
    path_heads: dict[int, HeadSite] = field(default_factory=dict)
    path_qkv: dict[int, dict[str, HeadSite]] = field(default_factory=dict)
    path_final: str | None = None
    # Classification logits and representation vectors have no sequence axis;
    # token masks still describe the captured patch/token sequence.
    output_kind: str = "language"
    # Beyond Intuition's token weights need pre-LN inputs and projected V, not A@V.
    attention_inputs: dict[int, str] = field(default_factory=dict)
    attention_values: dict[int, str] = field(default_factory=dict)
    project_values: Callable[[int, Tensor], Tensor] | None = None
    # Actual allowed attention pairs, independent of numerically zero probabilities.
    # Support sites produce boolean [B or 1,H or 1,Q,K]; True denotes an available edge.
    attention_supports: dict[int, str] = field(default_factory=dict)

    # Explicit architectural fact; None leaves a custom adapter's mask semantics unknown.
    # Append fields to preserve existing positional ModelSpec construction.
    causal_attention: bool | None = None
    mj: MJSpec | None = None

    def validate(self, sites):
        if self.causal_attention is not None and type(self.causal_attention) is not bool:
            raise TypeError("spec.causal_attention must be bool or None")
        if not self.residuals or any(type(k) is not int or k < 0 for k in self.residuals):
            raise ValueError("spec.residuals requires nonnegative layer indices")
        if self.output_kind not in {"language", "classification", "representation"}:
            raise ValueError("output_kind must be language, classification or representation")
        mappings = (self.residuals, self.attentions, self.attention_scores, self.heads, self.edges,
                    self.attention_inputs, self.attention_values, self.attention_supports)
        names = {name for mapping in mappings for name in mapping.values()}
        if self.embeddings is not None:
            names.add(self.embeddings)
        for mapping in (self.path_heads, *self.path_qkv.values()):
            if any(not isinstance(value, HeadSite) for value in mapping.values()):
                raise TypeError("path hook declarations must use HeadSite")
            names.update(value.site for value in mapping.values())
        if self.path_final is not None:
            names.add(self.path_final)
        if self.mj is not None:
            if not isinstance(self.mj, MJSpec):
                raise TypeError("spec.mj must be MJSpec or None")
            self.mj.validate()
            names.update(self.mj.sources.values())
            names.add(self.mj.final)
        missing = names - set(sites)
        if missing:
            raise ValueError(f"spec references unknown hook sites: {sorted(missing)}")
        if not self.editable_attention <= self.attentions.keys():
            raise ValueError("editable attention layers must declare probability sites")
        if not self.attention_supports.keys() <= self.attentions.keys():
            raise ValueError("attention support layers must declare probability sites")
        for mapping in (self.attentions, self.attention_scores, self.heads,
                        self.attention_inputs, self.attention_values):
            if not mapping.keys() <= self.residuals.keys():
                raise ValueError("attention/head layer indices must exist in residuals")
        for mapping in (self.path_heads, self.path_qkv):
            if not mapping.keys() <= self.residuals.keys():
                raise ValueError("path layer indices must exist in residuals")
        if any(set(mapping) - {"q", "k", "v"} for mapping in self.path_qkv.values()):
            raise ValueError("path_qkv kinds must be q, k, or v")
        path_outputs = [value.site for value in self.path_heads.values()]
        path_inputs = [value.site for mapping in self.path_qkv.values() for value in mapping.values()]
        path_sites = [*path_outputs, *path_inputs]
        if self.path_final is not None:
            path_sites.append(self.path_final)
        if len(path_sites) != len(set(path_sites)):
            raise ValueError("path output, Q/K/V, and final residual sites must be distinct")
