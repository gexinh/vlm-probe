# Gemma 4 integration v1 — 1.0.3.dev0

## Goal

Preserve native Gemma 4 execution while making method capabilities, validation
scope and model navigation consistent with the existing VLMProbe interface.
This is a source update to the personal repository, without a new release.

## Baseline and contribution

The baseline is `48b9d08531181fbaa4c344ebfd8895c5e2ff04e0`, which merges
[personal PR #3](https://github.com/gexinh/vlm-probe/pull/3) by Ruinan Jin.
The native Gemma adapter, shared-K/V boundary declarations, complete post-block
residual captures and isolated Transformers 5.5.4 runtime are retained.

## Differences from the previous version

- Preserve optional native attention soft-capping before mask addition, with
  logits and gradient parity checks for instrumented and ordinary model forwards.
- Isolate attention backend registrations by native model module so installing
  another architecture cannot change an existing model's attention arithmetic.
- Declare mixed causal-text/bidirectional-vision attention explicitly. Keep
  existing custom adapters and their mask declarations compatible.
- Reject the original causal Jacobian recipe on a declared mixed mask before
  model execution or artifact loading. Ordinary native attention and other
  supported probes continue to use the actual model masks.
- Add the Gemma family button to its generator, simplify homepage installation
  guidance, and update the project page's model navigation and family count.
- Align the model index and guide with reproducible architecture tests. Remove
  detailed pretrained measurements without a reproducible public run record.
- Run the new numerical and capability checks in the isolated Gemma CI job,
  retaining the existing Transformers 5.3 regression environment.

Model weights, stored experiment outputs, attention propagation rules, native
precision, checkpoint formats and the published package version are unchanged.

## Expected results

A no-intervention Gemma forward should produce the same logits and gradients
before and after binding VLMProbe, including when native attention `softcap` is
provided. Final output-logit soft-capping remains a separate native operation.

Vision-bidirectional configurations should report a mixed attention pattern.
The causal Jacobian recipe should reject that declared configuration with a
clear capability error instead of attaching a causal estimator name to a
noncausal continuation. Purely causal Gemma configurations should still fit and
run that recipe. Existing models should retain their behavior and precision
protections.

These checks establish architecture and interface behavior on tiny native
models. They do not establish pretrained task accuracy or a benchmark-wide
performance change.

## Validation

The verification covers the full existing source suite in Transformers 5.3,
the native Gemma and mixed-mask regressions in Transformers 5.5.4, installed
package imports, wheel/source metadata, documentation checks and the project
page build. Exact commands and numerical records are kept with the development
verification outputs.

Completed local validation: **727 tests and 562 subtests passed** in the existing
Transformers 5.3 runtime. Its 28 skips comprise the 25 Gemma-only tests run in
the separate runtime and three pre-existing optional tests. All **25 native
Gemma tests** and six shared mask-contract tests passed in Transformers 5.5.4.
Five installed-wheel import checks passed, and installed native Gemma forwards
retained exact logits with attention soft-capping for causal and mixed masks.
Wheel and source-distribution metadata passed strict Twine checks; all 118
packaged runtime/resource files matched the final checkout. Documentation
checks, model-button regeneration and the personal project-page build passed.

The unsupported mixed-mask Jacobian recipe is rejected before execution; its
native backward-in-sequence visual gradients remain intact. Both Gemma/Llama
attention installation orders retain native logits. These controls verify
behavioral compatibility without pretrained task evaluation.
