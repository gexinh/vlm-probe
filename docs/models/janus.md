# Janus

[Home](../../README.md) / [Models](README.md) / Janus

Use native Transformers-format checkpoints such as
[`deepseek-community/Janus-Pro-1B`](https://huggingface.co/deepseek-community/Janus-Pro-1B).
The original `deepseek-ai` format declares a different model class and is not
automatically interchangeable. Install `vlmprobe[transformers,visualization]`.

```python
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, JanusForConditionalGeneration
from vlmprobe import Prober

model_id = "deepseek-community/Janus-Pro-1B"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = JanusForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.float32, attn_implementation="eager",
).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id)
probe = Prober(model, processor)
# Run from the repository root; this is a raw GQA report photograph.
image = Image.open(Path("demos/assets/images/2341576.jpg")).convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image"}, {"type": "text", "text": "Describe the image."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = probe.prepare(prompt=prompt, image=image, device=device)
print(probe.describe())
result = probe.lens.logit(layers=[max(probe.spec.residuals)]).run(inputs)
print(result.tensors["logits"].shape)
```

The native image processor and chat template own the visual placeholder layout.
Do not resize an overlay to the uncropped original without checking those
transforms. Existing residual-output methods use the common API. MJ-Lens uses
**block-input** layer indices instead; inspect `probe.spec.mj.sources`.

For image-output MJ calibration, sample or teacher-force actual visual codebook
IDs, then use `vlmprobe.adapters.prepare_janus_generation`. That helper uses
native generation embeddings, explicit code-position masks, and `token_ids=None`
to keep the two alphabets separate. It currently requires unpadded text prompts
and a fixed code sequence of at least two positions. It excludes the last code
position from output calibration because no following code is supplied.

The ordinary Janus forward returns language logits. The visual head is evaluated
explicitly during MJ visual-readout calibration; `pixel_values` does not itself
activate that output channel. See the [complete MJ-Lens guide](../methods/mj_lens.md)
for the four operators, calibration, loading and native codebook visualization.

The public [MJ-Lens report notebook](../../demos/10_mj_lens.ipynb) uses
pretrained Qwen for understanding-side text outputs. Janus visual-codebook
calibration is a separate workflow described in the method guide; it does not
turn Qwen image inputs into an image-generation alphabet. This loading example
does not establish a pretrained Janus image-generation benchmark.
