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
        from .api.discovery import describe_methods
        methods = describe_methods(self)
        from .sae.registry import list_methods
        return {"model_id": self.adapter.model_id, "adapter": type(self.adapter).__name__,
                "output_kind": s.output_kind,
                "sae_methods": list_methods(),
                "methods": methods, "notes": list(s.notes),
                "layout": "adapter-provided" if s.layout else "explicit ProbeInputs required"}
