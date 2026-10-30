"""Offline Gemma 4 image/text smoke test using tiny, random native modules.

Install ``pip install -e '.[gemma4]'`` in a separate environment, then run
``python examples/gemma4_smoke.py``. No model weights are downloaded.
"""
import torch

from vlmprobe import Prober, TokenMargin


def make_gemma4(*, multimodal=True, backend="eager", shared_layers=2,
                bidirectional=None, softcap=2.0, alternative_attention=False, per_layer_width=4):
    from transformers import Gemma4Config, Gemma4ForCausalLM, Gemma4ForConditionalGeneration, Gemma4TextConfig

    text = dict(
        vocab_size=40, hidden_size=16, intermediate_size=24,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=16, num_kv_shared_layers=shared_layers,
        hidden_size_per_layer_input=per_layer_width, vocab_size_per_layer_input=40,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=4, pad_token_id=0, final_logit_softcapping=softcap,
        use_bidirectional_attention=bidirectional,
        attention_k_eq_v=alternative_attention, num_global_key_value_heads=1,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.5,
                               "rope_theta": 10000.0},
        },
    )
    if multimodal:
        vision = dict(hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                      patch_size=2, pooling_kernel_size=2, position_embedding_size=8)
        config = Gemma4Config(
            text_config=text, vision_config=vision, image_token_id=40, video_token_id=42,
            audio_token_id=41, boi_token_id=27, eoi_token_id=29, boa_token_id=30, eoa_token_index=31,
        )
        # OOV image IDs exercise the model's native replacement before lookup.
        ids = torch.tensor([[2, 27, 40, 40, 40, 40, 29, 3, 0], [0, 2, 27, 40, 40, 40, 40, 29, 3]])
        inputs = dict(
            input_ids=ids, attention_mask=(ids != 0).long(), mm_token_type_ids=(ids == 40).long(),
            pixel_values=torch.rand(2, 16, 12),
            image_position_ids=torch.tensor([[(x, y) for y in range(4) for x in range(4)]] * 2),
        )
        cls = Gemma4ForConditionalGeneration
    else:
        config = Gemma4TextConfig(**text)
        ids = torch.tensor([[2, 4, 5, 3, 0], [0, 2, 4, 5, 3]])
        inputs = dict(input_ids=ids, attention_mask=(ids != 0).long())
        cls = Gemma4ForCausalLM
    config._attn_implementation = backend
    return cls(config).eval(), inputs


if __name__ == "__main__":
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model, inputs = make_gemma4()
    probe = Prober(model)
    result = probe.lens.logit(layers=[0, 3], tokens="last_prompt").run(inputs)
    print("Logit Lens:", tuple(result.tensors["logits"].shape))
    effect = probe.causal.knockout(
        layers=[1], queries="last_prompt", keys="visual",
    ).run(inputs, metric=TokenMargin(4, 5))
    print("Attention knockout effect:", effect.tensors["effect"].tolist())
