"""User-facing model binding and method collections."""
import torch

from .adapters import CapabilityError, ModelSpec, ProbeInputs, TokenLayout
from .adapters.auto import auto_adapter
from .api.common import unpack


class Prober:
    """Bind one caller-owned model. Methods configure first, then run on inputs.

    Known architectures and registered factories support adapter='auto'. For
    others provide an adapter with a ModelSpec. Outputs default to detached CPU
    tensors; neither parameters nor the caller's training flags are changed.
    """

    def __init__(self, model, processor=None, *, adapter="auto", result_device="cpu"):
        self.model, self.processor = model, processor
        self.adapter = auto_adapter(model) if isinstance(adapter, str) and adapter == "auto" else adapter
        if getattr(self.adapter, "model", None) is not model:
            raise ValueError("adapter must be bound to the exact supplied model")
        self.spec = getattr(self.adapter, "spec", None)
        if not isinstance(self.spec, ModelSpec):
            raise CapabilityError("public methods require an adapter with a ModelSpec")
        self.result_device = torch.device(result_device)
        from .api.lenses import LensMethods
        from .api.attention import AttentionMethods
        from .api.causal import CausalMethods
        from .api.sae import SAEMethods
        self.lens = LensMethods(self)
        self.attention = AttentionMethods(self)
        self.causal = CausalMethods(self)
        self.sae = SAEMethods(self)

    def prepare(self, *, prompt, image=None, layout=None, device=None, **kwargs):
        """Process preformatted text/images; chat templates remain processor-specific."""
        if self.processor is None:
            raise ValueError("prepare requires processor=...; otherwise pass native model kwargs to run")
        call = {"text": prompt, "return_tensors": "pt", **kwargs}
        if image is not None:
            call["images"] = image
        values = dict(self.processor(**call))
        if device is not None:
            values = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in values.items()}
        return ProbeInputs(values, layout)

    def _run(self, inputs, **kwargs):
        values, _ = unpack(inputs)
        values = {**self.spec.forward_defaults, **values}
        if values.get("use_cache") is True or values.get("past_key_values") is not None:
            raise ValueError("public probes require full-sequence inputs without KV cache")
        if "logits_to_keep" in values and (not isinstance(values["logits_to_keep"], int)
                                           or values["logits_to_keep"] != 0):
            raise ValueError("public probes need all sequence logits: logits_to_keep=0")
        return self.adapter.run(values, **kwargs)

    def _layout(self, inputs, tensor, *, sequence_axis=1):
        values, explicit = unpack(inputs)
        layout = explicit
        if layout is None:
            if self.spec.layout is None:
                raise CapabilityError("provide ProbeInputs(..., layout=TokenLayout(...)) or spec.layout")
            layout = self.spec.layout(values)
        if not isinstance(layout, TokenLayout):
            raise TypeError("layout provider must return TokenLayout")
        shape = (tensor.shape[0], tensor.shape[sequence_axis])
        if self.spec.output_kind == "classification" and tensor.ndim == 2:
            shape = (tensor.shape[0], layout.valid.shape[1])
        return layout.checked(shape, tensor.device)

    def _heads(self, layer, tensor):
        result = self.spec.project_heads(layer, tensor) if self.spec.project_heads else tensor
        if result.ndim != 4:
            raise ValueError("head adapter must produce [batch,token,head,residual_dim]")
        return result

    def describe(self):
        """Inspect available methods and missing capabilities without a forward pass."""
        s = self.spec
        residual = sorted(s.residuals)
        readout = getattr(self.adapter, "readout", None) is not None
        methods = {}
        propagation = []
        for i in range(max(s.residuals) + 1):
            if i not in s.attentions:
                break
            propagation.append(i)
        rules = {
            "sae.features": (residual, "a fitted dictionary bound to an explicit residual hook"),
            "sae.edit": (residual, "a fitted dictionary and an editable residual hook"),
            "sae.profile": (residual, "a fitted dictionary and explicit token positions"),
            "sae.reconstruct": (residual, "a fitted same-space dictionary and an editable residual hook"),
            "lens.logit": (residual if readout else [], "residual sites and model readout"),
            "lens.tuned": (residual if readout else [], "residual sites and model readout"),
            "lens.jacobian": (residual if readout else [], "residual sites and model readout"),
            "lens.mj": (sorted(s.mj.sources) if s.mj is not None and s.causal_attention is True else [],
                        "causal block-input sites, actual final RMSNorm target and explicit modality layout"),
            "lens.embed": (["embeddings"] if s.embeddings and s.input_embeddings is not None else [], "embedding-space site and input embeddings"),
            "lens.attention": (sorted(s.heads) if s.input_embeddings is not None else [], "projected heads and vocabulary dimensions"),
            "lens.patchscope": (residual, "residual sites"),
            "attention.profile": (sorted(s.attentions), "observable attention probabilities"),
            "attention.rollout": (propagation, "observable self-attention at consecutive layers from 0"),
            "attention.relevance": (propagation, "differentiable self-attention at consecutive layers from 0"),
            "attention.grad_cam": (sorted(s.attentions), "differentiable target-conditioned attention"),
            "attention.attribution": (sorted(s.editable_attention), "editable consumed probabilities for attention-path integration"),
            "attention.tam": (propagation, "consecutive attention stack and floating input-path tensor"),
            "attention.beyond_intuition": (propagation, "consecutive attention stack; token variant also needs pre-LN inputs and projected V"),
            "attention.dtd_lrp": (residual if s.output_kind == "classification" and
                                 type(self.model).__name__ == "ViTForImageClassification" else [],
                                 "native HF ViT converted to Chefer's original-rule backend"),
            "attention.head_logits": (sorted(s.heads) if s.linear_readout else [], "projected heads and fixed-scale readout"),
            "attention.reweight": (sorted(s.editable_attention), "editable probabilities before A @ V"),
            "attention.temperature": (sorted(s.attention_scores), "editable masked attention logits"),
            "causal.patch": (residual, "residual sites"),
            "causal.path": (sorted(s.path_heads) if set(s.path_heads) == set(s.residuals)
                            and (s.path_final or s.path_qkv) else [],
                            "editable outputs at every decoder layer and receiver Q/K/V or final residual"),
            "causal.attribute": (residual, "differentiable residual sites"),
            "causal.steer": (residual, "residual sites"),
            "causal.knockout": (sorted(s.attention_scores), "editable masked attention logits"),
            "causal.eap_ig": (list(s.edges), "explicit independently replaceable edge sites"),
        }
        for name, (sites, need) in rules.items():
            if s.output_kind == "representation" and not name.startswith("sae."):
                sites, need = [], "representation encoder currently exposes SAE methods; no language/classification logits"
            methods[name] = {"available": bool(sites), "sites": sites,
                             "requires": need, "missing": None if sites else need}
        native_gpt2 = (type(self.model).__module__ == "transformers.models.gpt2.modeling_gpt2"
                       and type(self.model).__name__ == "GPT2LMHeadModel"
                       and not self.model.config.add_cross_attention
                       and not getattr(self.model.config, "reorder_and_upcast_attn", False))
        edge_method = methods["causal.eap_ig"]
        edge_method["graphs"] = {"activation": bool(s.edges), "transformer": native_gpt2}
        if native_gpt2:
            edge_method.update(available=True, missing=None,
                               requires="graph='transformer' for the native GPT-2 residual-message graph")
        from .sae.registry import list_methods
        if s.mj is not None:
            methods["lens.mj"]["tap_convention"] = "decoder_block_input"
            methods["lens.mj"]["output_channels"] = ["text"] + (["visual"] if s.mj.visual_head is not None else [])
        return {"model_id": self.adapter.model_id, "adapter": type(self.adapter).__name__,
                "output_kind": s.output_kind,
                "sae_methods": list_methods(),
                "methods": methods, "notes": list(s.notes),
                "layout": "adapter-provided" if s.layout else "explicit ProbeInputs required"}
