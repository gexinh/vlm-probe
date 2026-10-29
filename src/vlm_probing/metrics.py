"""Scores have explicit positions and answer spans; no tokenizer assumptions."""
import torch
from torch import Tensor

from .core.precision import analysis_difference, analysis_dtype


class ClassScore:
    """Per-example image-classification target; never invent tokenwise logits."""

    def __init__(self, target: int, *, kind="logit"):
        if type(target) is not int or target < 0:
            raise ValueError("target must be a nonnegative class index")
        if kind not in {"logit", "probability"}:
            raise ValueError("kind must be logit or probability")
        self.target, self.kind = target, kind

    def score(self, logits, layout=None):
        if logits.ndim != 2 or self.target >= logits.shape[-1]:
            raise ValueError("ClassScore requires [batch,classes] logits and a valid target")
        if self.kind == "logit":
            value = logits[:, self.target]
            if not value.is_floating_point() and not value.is_complex():
                return value
            return value.to(analysis_dtype(value))
        return logits.to(analysis_dtype(logits)).softmax(-1)[:, self.target]


class TokenMargin:
    """Per-example candidate logit difference at one selected prompt position."""

    def __init__(self, positive: int, negative: int, *, position="last_prompt"):
        self.positive, self.negative, self.position = positive, negative, position

    def score(self, logits, layout):
        mask = layout.select(self.position)
        if not (mask.sum(-1) == 1).all():
            raise ValueError("TokenMargin requires exactly one prediction position per example")
        return token_logit_margin(logits[mask], self.positive, self.negative)


class SequenceLogProb:
    """Teacher-forced answer score with an explicit expanded-sequence answer mask."""

    def __init__(self, answer_mask: Tensor, *, reduction="sum"):
        self.answer_mask, self.reduction = answer_mask, reduction

    def score(self, logits, layout):
        if layout.token_ids is None:
            raise ValueError("sequence scoring requires aligned layout.token_ids")
        mask = layout.select(self.answer_mask)
        if layout.visual is not None and (mask & layout.visual).any():
            raise ValueError("answer mask cannot select visual tokens")
        ids = layout.token_ids.clone()
        # Visual placeholders/padding are not language targets. They may carry
        # sentinel IDs in custom layouts, and are excluded from scoring.
        ignored = ~layout.valid
        if layout.visual is not None:
            ignored = ignored | layout.visual
        ids[ignored] = 0
        return sequence_logprob(logits, ids, mask, reduction=self.reduction)


def token_logit_margin(logits: Tensor, positive: int, negative: int) -> Tensor:
    """Input [..., vocab]; choose the prediction position before calling."""
    if not 0 <= positive < logits.shape[-1] or not 0 <= negative < logits.shape[-1]:
        raise ValueError("Candidate token ID is outside the vocabulary")
    # Select candidates first to avoid allocating a widened full vocabulary.
    return analysis_difference(logits[..., positive], logits[..., negative])


def sequence_logprob(logits: Tensor, input_ids: Tensor, answer_mask: Tensor, *,
                     reduction: str = "sum") -> Tensor:
    """Teacher-forced log P(answer|prompt), shifted by one, per batch item.

    logits[B,S,V] and input_ids[B,S] come from the same full prompt+answer run.
    answer_mask[B,S] identifies exactly which answer tokens count, including
    EOS only if the caller selects it. Candidates must be scored separately.
    """
    if logits.ndim != 3 or input_ids.shape != logits.shape[:2] or answer_mask.shape != input_ids.shape:
        raise ValueError("Expected logits[B,S,V], input_ids[B,S], answer_mask[B,S]")
    if answer_mask.dtype != torch.bool or input_ids.dtype != torch.long:
        raise TypeError("answer_mask must be bool and input_ids must be long")
    if logits.device != input_ids.device or logits.device != answer_mask.device:
        raise ValueError("All scoring tensors must be on the same device")
    if answer_mask[:, 0].any() or (~answer_mask[:, 1:].any(-1)).any():
        raise ValueError("Each answer needs at least one token with a preceding prediction position")
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be sum or mean")
    if ((input_ids < 0) | (input_ids >= logits.shape[-1])).any():
        raise ValueError("input_ids outside vocabulary; pass actual expanded language token IDs")
    mask = answer_mask[:, 1:]
    safe_logits = logits[:, :-1].to(analysis_dtype(logits)).masked_fill(~mask[..., None], 0)
    if (torch.isnan(safe_logits).any() or torch.isposinf(safe_logits).any()
            or (~torch.isfinite(safe_logits).any(-1)).any()):
        raise ValueError("Selected prediction rows need finite logits and may not contain NaN or +inf")
    scores = safe_logits.log_softmax(-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    result = scores.masked_fill(~mask, 0).sum(-1)
    return result / mask.sum(-1) if reduction == "mean" else result
