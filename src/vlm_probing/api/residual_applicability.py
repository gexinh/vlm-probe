"""Architectural caveats for residual interventions without changing their effect."""
import warnings


class ResidualInterventionWarning(UserWarning):
    """A residual edit cannot restore, or cannot reach, all downstream computation."""


_RESTORING = frozenset({"patch", "attribute"})
_EDITING = _RESTORING | {"steer", "patchscope", "sae.edit", "sae.reconstruct"}


def residual_applicability(spec, method, layers):
    """Return declared limits for these layers and warn; undeclared stacks return None."""
    if method not in _EDITING:
        return None
    final = max(spec.residuals)
    incomplete = (sorted(i for i in layers if i < final)
                  if spec.residual_bypass and method in _RESTORING else [])
    boundary = spec.last_stream_kv_layer
    local = sorted(i for i in layers if i >= boundary) if boundary is not None else []
    caveats = []
    if incomplete:
        caveats.append(
            f"Replacing a residual at layers {incomplete} does not restore the source run: later "
            "blocks also consume state outside the residual stream, which stays from the receiver.")
    if local:
        caveats.append(
            f"No attention after layers {local} reads the residual stream as K/V, so an edit there "
            "only changes later computation at the edited positions. A zero effect at other positions "
            "follows from this architecture; it does not show that those layers ignore the information.")
    if not caveats:
        return None
    caveat = " ".join(caveats)
    warnings.warn(f"{method}: {caveat}", ResidualInterventionWarning, stacklevel=3)
    return {"restoration_incomplete_layers": incomplete, "position_local_layers": local,
            "intervention_formula_modified": False, "caveat": caveat}
