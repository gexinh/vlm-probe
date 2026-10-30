"""Small explicit adapter registry, with optional Hugging Face integration."""
from collections.abc import Callable

from torch import nn

from .spec import CapabilityError
from .torch import TorchModelAdapter

_FACTORIES: dict[type, Callable[[nn.Module], TorchModelAdapter]] = {}


def register_adapter(model_class: type, factory: Callable, *, replace: bool = False):
    """Register a factory for an exact Python model class; never guess subclasses."""
    if model_class in _FACTORIES and not replace:
        raise ValueError(f"adapter already registered for {model_class.__name__}")
    _FACTORIES[model_class] = factory


def auto_adapter(model):
    if type(model) in _FACTORIES:
        return _FACTORIES[type(model)](model)
    factory = getattr(model, "probing_adapter", None)
    if callable(factory):
        return factory()
    if (type(model).__module__, type(model).__name__) == (
            "transformers.models.vit.modeling_vit", "ViTForImageClassification"):
        from .vit import make_adapter
        return make_adapter(model)
    from .gemma4 import MODEL_CONTRACTS as GEMMA4_CONTRACTS, make_adapter as make_gemma4_adapter
    if (type(model).__module__, type(model).__name__) in GEMMA4_CONTRACTS:
        return make_gemma4_adapter(model)
    from .huggingface import MODEL_CONTRACTS, make_adapter
    if (type(model).__module__, type(model).__name__) in MODEL_CONTRACTS:
        return make_adapter(model)
    supported = sorted({name for _, name in MODEL_CONTRACTS.keys() | GEMMA4_CONTRACTS.keys()}
                       | {"ViTForImageClassification"})
    raise CapabilityError(
        f"No automatic adapter for {type(model).__name__}. Supported native classes: "
        f"{', '.join(supported)}. Pass adapter=TorchModelAdapter(..., "
        "spec=ModelSpec(...)), register_adapter(ModelClass, factory), or implement probing_adapter()."
    )
