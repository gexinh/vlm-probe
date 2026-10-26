"""Region metrics for explicitly aligned patch feature maps."""
import torch
import torch.nn.functional as F

from .types import EvaluationReport


def localization(feature_maps, masks, *, ids=None, sign="positive", provenance=None):
    """Evaluate [example,height,width] feature maps against aligned GT masks.

    Maps may be lower-resolution than masks and are bilinearly resized without
    min-max normalization. Positive (or negative/absolute) feature evidence is
    evaluated explicitly. A zero map is inactive: point@1 is false, mass-inside
    is undefined. Coverage and conditional scores are returned separately.
    """
    values = torch.as_tensor(feature_maps).detach().double().cpu()
    targets = torch.as_tensor(masks).detach().cpu()
    if (values.ndim != 3 or targets.ndim != 3 or len(values) != len(targets)
            or not len(values) or min(values.shape[-2:]) < 1
            or min(targets.shape[-2:]) < 1 or not torch.isfinite(values).all()):
        raise ValueError("feature_maps and masks must be nonempty aligned finite [example,height,width]")
    if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("masks must contain boolean/0-1 region membership")
    targets = targets.bool()
    if not targets.flatten(1).any(1).all():
        raise ValueError("every example needs a nonempty annotated target region")
    if sign not in {"positive", "negative", "absolute"}:
        raise ValueError("sign must be positive, negative or absolute")
    evidence = values.clamp_min(0) if sign == "positive" else (-values).clamp_min(0) if sign == "negative" else values.abs()
    if evidence.shape[-2:] != targets.shape[-2:]:
        evidence = F.interpolate(evidence[:, None], size=targets.shape[-2:], mode="bilinear",
                                 align_corners=False)[:, 0]
    if ids is None:
        ids = [str(index) for index in range(len(values))]
    else:
        ids = list(ids)
        if len(ids) != len(values) or len(set(map(str, ids))) != len(ids):
            raise ValueError("ids must be unique and aligned with examples")
    rows = []
    for sample_id, heatmap, mask in zip(ids, evidence, targets):
        total = heatmap.sum().item()
        active = total > 0
        point = bool(mask.reshape(-1)[heatmap.argmax()]) if active else False
        mass = heatmap[mask].sum().item() / total if active else None
        rows.append({"id": str(sample_id), "active": active, "point_at_1": float(point),
                     "mass_inside": mass, "target_area_fraction": mask.double().mean().item()})
    active_rows = [row for row in rows if row["active"]]
    summary = {"n_examples": len(rows), "n_active": len(active_rows),
        "coverage": len(active_rows) / len(rows),
        "point_at_1": sum(row["point_at_1"] for row in rows) / len(rows),
        "point_at_1_active": (sum(row["point_at_1"] for row in active_rows) / len(active_rows)
                               if active_rows else None),
        "mass_inside_active": (sum(row["mass_inside"] for row in active_rows) / len(active_rows)
                                if active_rows else None)}
    return EvaluationReport(summary, rows, provenance={**(provenance or {}),
        "evaluation_kind": "spatial_localization", "evidence_sign": sign,
        "zero_map_protocol": "inactive; point_at_1=0, mass_inside=null",
        "geometry_contract": "caller aligns masks to the actual model image transform"})
