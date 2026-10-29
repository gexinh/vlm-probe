# Analysis precision v1 — 1.0.3.dev0

## Goal

Extend the accepted causal precision fix across the confirmed analysis paths
without converting backbone weights or native forward/backward tensors to
FP32. This is a development version for review, not a new PyPI release.

## Baseline and contribution

The baseline includes [PR #3](https://github.com/ubc-tea/vlm-probe/pull/3) by
jo-sh-varughese, merged at `20bb179813f605cab17a6272065dfdbf1414f10a`.
That contribution promotes activation-space EAP-IG and attribution patching
arithmetic, adds precision regressions, and brings causal tests into CI. Its
original commits and authorship remain in this branch's history.

## Differences from the previous version

- Select actual masked integration endpoints directly; promote native GPT-2
  graph source differences before subtraction and retain FP64 contractions.
- Promote causal metric reductions and measured effect differences, including
  exact activation/path interventions and built-in token/class metrics.
- Promote attention propagation, Grad-CAM products and profile statistics;
  construct TAM/Beyond input paths in wider precision with native replay.
- Finish SAE error-preserving edits in wider precision before casting the
  final replacement; promote profile variance and reconstruction/sparsity/
  auxiliary loss statistics before arithmetic.
- Add a shared precision helper, analytic and public-API regressions, CI
  execution for the new checks and a common numerical-precision guide.
- Preserve existing API call signatures, model weights, checkpoint formats,
  native intervention geometry and the exact SAE no-op control.

The published 1.0.2 artifacts and tag remain unchanged. The branch's package
version is `1.0.3.dev0`. Existing experiment outputs, pretrained models and
notebook assets are not copied into this code change. Independent audit and
verification outputs remain outside the repository.

## Expected results

Default-step analytic EAP-IG integrals should retain small non-unit gradients;
selected endpoints must equal the supplied native tensors. BF16 attention
recurrences should retain small increments and probability normalization;
FP16 Grad-CAM should retain representable analysis contributions. SAE default
feature deletion should preserve the remaining decoded contribution rather
than round it away before residual addition. FP16 profile/loss statistics
should remain finite when their FP32 mathematical values are finite. FP64
analysis inputs should not silently narrow to FP32.

Actual model forward/backward tensors remain native. Tests use known analytic
answers, captured native gradients, mask/endpoint controls and exact no-op
checks. These changes do not guarantee exact backbone computation or establish
a benchmark-wide change in answer accuracy.

## Validation

Validation covers the added precision regressions, the full existing source
suite, installed-wheel imports and strict wheel/source-distribution metadata.
The same fixed pretrained SmolVLM input from the audit is used to compare
attribution arithmetic against FP64 operations on the same BF16 activations
and backward gradients. Results and commands are recorded in the development
verification outputs.

The completed source suite passed **718 tests and 537 subtests**, with three
optional tests skipped. Five independent installed-wheel import checks passed;
the wheel and source archive passed strict Twine checks. All 116 packaged
runtime/resource files and seven license/notice files matched the checkout.
Static tutorial checks covered 21 method guides, 13 model guides and 11
notebooks without errors. Independent cross-reviews checked masks, native
replay, gradient isolation and FP64/integer compatibility.

The cached BF16 SmolVLM-500M check retained native activations/backward
gradients. At layers 16 and 31, first-order analysis differed from FP64 arithmetic
on those same captured values by relative errors 2.57e-8 and 8.03e-8, respectively.
This is one fixed input, not a benchmark-wide accuracy claim.
