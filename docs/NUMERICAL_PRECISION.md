# Numerical precision

[Home](../README.md) / [Documentation](README.md)

VLMProbe separates model execution from analysis arithmetic. Loaded model
weights, native activations and backward gradients keep their configured
dtype. Analysis operations that combine those values use at least FP32 for
FP16/BF16 inputs and preserve FP64 where the method supports it.

| Operation | Analysis precision |
| --- | --- |
| Attention profiles, Rollout, Grad-CAM and relevance propagation | Promote before products, normalization, statistics and layer recurrences. |
| Attention and input-path gradient integration | Promote before accumulation and averaging; replay each sample in native model dtype. |
| Activation-space and native GPT-2 graph EAP-IG | Promote source differences, interpolation, gradient contractions and scores. |
| Attribution patching | Promote differences, gradient products and reductions. |
| Built-in answer/class metrics and causal effect differences | At least FP32; preserve FP64. |
| SAE feature editing | Compute the decoded difference and residual addition before the final native cast. |
| SAE profile mean/variance and loss statistics | Promote before variance, squares, products and reductions. |

Return dtypes can therefore differ from the backbone dtype. For example,
BF16 residuals produce FP32 attribution scores, while the activation injected
into the next model layer is still BF16. Larger analysis buffers can increase
memory use, particularly with dense attention or many selected edge messages.
Integral score differences retain integer arithmetic and exact integer values.

## Interpolation endpoints

Input and activation integration construct interior points in analysis
precision, then cast each point to the model's dtype. The declared baseline
and source endpoints are selected directly. With masks, unselected activation
positions remain at the receiver throughout the path. A quadrature that omits
an endpoint continues to omit it; precision handling does not change the
integration rule or sample count.

Direct selection matters when endpoint magnitudes differ greatly. Even FP32
can round `small_source - large_receiver` so that adding the receiver back
does not recover the source. Endpoints are not reconstructed by that arithmetic.

## Interpretation and limits

Wider analysis avoids additional rounding after values and gradients have
been captured. It cannot recover a gradient already rounded to zero inside
the model. Integration convergence and first-order approximation error remain
separate questions; increasing precision does not make attribution patching
an exact intervention.

The native GPT-2 graph exposes all/none-retained endpoint errors during circuit
evaluation. A native low-precision source decomposition can still introduce
rounding in the all-corrupted graph, so those errors should be inspected rather
than assuming bitwise equality with an independent donor forward.

Lens model readouts keep their declared native computation. EmbedLens uses
FP32 cosine comparisons, MJ-Lens calibration accumulates Jacobians in FP64,
and fitted lens transports retain their documented projection/readout dtype
policy. SAE evaluation uses its documented FP64 statistics independently of
the dictionary's forward dtype. Promoted loss computation does not by itself
provide mixed-precision optimizer scaling for training a half-precision SAE.

Custom metrics should return a finite differentiable scalar or per-example
vector and use suitable precision internally. VLMProbe can widen the returned
score difference; it cannot recover information the metric already discarded.
