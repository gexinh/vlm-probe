"""Small, serializable contracts shared by native SAE evaluations."""
from dataclasses import asdict, dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class EvaluationBatch:
    """Prepared model inputs with one stable ID per original example.

    ``metric`` must return one score per example. Group IDs identify a shared
    source such as a patient or document; they do not replace example IDs.
    """
    inputs: Any
    ids: Any
    group_ids: Any = None
    labels: Any = None
    metric: Any = None


@dataclass(frozen=True)
class ActivationBatch:
    """Cached native activations, with IDs per row or per leading example.

    Shape is [position, hidden] or [example, token, hidden]. Optional ``mask``
    selects valid positions in the leading dimensions, excluding padding/CLS.
    """
    activations: torch.Tensor
    ids: Any = None
    group_ids: Any = None
    mask: Any = None


@dataclass(frozen=True)
class SAEEvaluationConfig:
    metrics: tuple = ("reconstruction", "sparsity")
    chunk_size: int = 1024
    feature_ids: tuple = ()
    intervention_scales: tuple = (0., .5, 1., 2.)
    random_control_repeats: int = 3
    random_feature_ids: tuple = ()
    seed: int = 0

    def __post_init__(self):
        allowed = {"reconstruction", "sparsity", "fidelity", "intervention"}
        if not self.metrics or set(self.metrics) - allowed:
            raise ValueError(f"metrics must be selected from {sorted(allowed)}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be unique")
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.random_control_repeats) is not int or self.random_control_repeats < 0:
            raise ValueError("random_control_repeats must be a nonnegative integer")
        if any(type(i) is not int or i < 0 for i in self.feature_ids) or len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("feature_ids must be unique nonnegative integers")
        if not self.intervention_scales or any(not math.isfinite(s) for s in self.intervention_scales):
            raise ValueError("intervention_scales must be nonempty and finite")
        if len(set(self.intervention_scales)) != len(self.intervention_scales):
            raise ValueError("intervention_scales must be unique")
        if "intervention" in self.metrics and not self.feature_ids:
            raise ValueError("intervention evaluation requires explicit feature_ids")


def json_value(value):
    """Convert tensors and paths to strict JSON; undefined metrics become null."""
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


@dataclass
class EvaluationReport:
    """Evaluation output; feature arrays live on CPU, never whole activations."""
    summary: dict
    per_example: list = field(default_factory=list)
    feature_statistics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self, *, include_features=True):
        value = {"summary": self.summary, "per_example": self.per_example,
                 "provenance": self.provenance}
        if include_features:
            value["feature_statistics"] = self.feature_statistics
        return json_value(value)

    def save(self, directory):
        """Write strict JSON and tidy CSV without serializing model weights."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / "summary.json"
        summary_path.write_text(json.dumps(self.to_dict(include_features=False), indent=2,
                                          ensure_ascii=False, allow_nan=False) + "\n")
        example_path = directory / "per_example.csv"
        _write_csv(example_path, self.per_example)
        feature_path = directory / "feature_statistics.csv"
        statistics = {k: json_value(v) for k, v in self.feature_statistics.items()}
        lengths = {len(v) for v in statistics.values() if isinstance(v, list)}
        if len(lengths) > 1:
            raise ValueError("feature statistic columns have inconsistent lengths")
        rows = [{"feature_id": i, **{k: v[i] for k, v in statistics.items()}}
                for i in range(next(iter(lengths), 0))]
        _write_csv(feature_path, rows)
        return {"summary": summary_path, "per_example": example_path,
                "feature_statistics": feature_path}


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(json_value(value), ensure_ascii=False,
                                             allow_nan=False)
                                    if isinstance(value, (dict, list, tuple, torch.Tensor))
                                    else json_value(value)) for key, value in row.items()})


@dataclass(frozen=True)
class LanguageFidelityBatch:
    """Explicit causal LM labels in the SAME expanded coordinates as logits.

    ``token_ids`` is [batch,sequence]. ``score_mask`` is boolean [batch,sequence-1]
    selecting valid next-token predictions: logits[:, :-1] -> token_ids[:, 1:].
    Callers exclude padding, image placeholders and unwanted prompt positions.
    """
    inputs: Any
    ids: Any
    token_ids: torch.Tensor
    score_mask: torch.Tensor
    group_ids: Any = None
