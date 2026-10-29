"""Analytic checks of SAE analysis precision without training or downloads."""
import pytest
import torch
from torch import nn

from vlmprobe import HookPoint, ModelSpec, Prober, TokenLayout, TorchModelAdapter
from vlmprobe.sae import SAEConfig, SparseAutoencoder
from vlmprobe.sae.binding import build_sae_binding


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class IdentityModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.block = nn.Linear(1, 1, bias=False, dtype=dtype)
        with torch.no_grad():
            self.block.weight.fill_(1)

    def forward(self, inputs):
        return self.block(inputs).mean(1)


def setup(native_dtype, dictionary_dtype=torch.float32, *, features=1, architecture="relu", **kwargs):
    model = IdentityModel(native_dtype).eval()
    spec = ModelSpec({0: "residual.0"}, output_kind="representation",
        layout=lambda inputs: TokenLayout(torch.ones(inputs["inputs"].shape[:2], dtype=torch.bool)))
    adapter = TorchModelAdapter(model, {"residual.0": HookPoint("block")},
                               model_id="identity-precision-v1", spec=spec)
    probe = Prober(model, adapter=adapter)
    binding = build_sae_binding(probe, 0, model_id="identity-precision-v1-fixed-weights",
        tokenizer_id="none-v1", preprocessing_id="identity-v1", calibration_id="analytic-v1")
    dictionary = SparseAutoencoder(SAEConfig(architecture, 1, features, k=1,
        input_normalization="none", **kwargs), binding, dtype=dictionary_dtype)
    with torch.no_grad():
        dictionary.encoder.weight.fill_(1)
        dictionary.encoder.bias.zero_()
        dictionary.decoder.weight.fill_(1)
        dictionary.decoder.bias.zero_()
    dictionary.is_fitted = True
    return probe, binding, dictionary


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("residual,retained", [(1., 2.**-12), (256., 1.0078125)])
def test_default_feature_deletion_keeps_native_representable_remainder(dtype, residual, retained):
    probe, binding, dictionary = setup(dtype, features=2)
    # Two exactly represented feature contributions sum to the native residual.
    with torch.no_grad():
        dictionary.encoder.weight.copy_(torch.tensor([[(residual-retained)/residual],
                                                       [retained/residual]]))
    values = {"inputs": torch.tensor([[[residual]]], dtype=dtype)}
    result = probe.sae.edit(layer=0, sae=dictionary, binding=binding,
                            feature_ids=[0]).run(values)
    assert result.tensors["edited_output"].dtype == dtype
    assert result.tensors["edited_output"].item() == retained
    assert result.tensors["baseline_output"].item() == residual


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_default_deletion_of_main_feature_preserves_normal_scale_small_feature(dtype):
    probe, binding, dictionary = setup(dtype, features=2)
    retained = torch.tensor(.001, dtype=dtype).item()
    with torch.no_grad():
        dictionary.encoder.weight.copy_(torch.tensor([[1.-retained], [retained]]))
    result = probe.sae.edit(layer=0, sae=dictionary, binding=binding,
        feature_ids=[0]).run({"inputs": torch.ones(1, 1, 1, dtype=dtype)})
    assert result.tensors["edited_output"].item() == retained


def test_inactive_edit_does_not_round_original_fp64_residual_through_dictionary():
    probe, binding, dictionary = setup(torch.float64, features=2)
    with torch.no_grad():
        dictionary.encoder.weight[0].zero_()
    value = 1. + 2.**-40
    result = probe.sae.edit(layer=0, sae=dictionary, binding=binding,
        feature_ids=[0]).run({"inputs": torch.tensor([[[value]]], dtype=torch.float64)})
    assert torch.equal(result.tensors["edited_output"], result.tensors["baseline_output"])
    assert result.tensors["edited_output"].item() == value


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_profile_statistics_promote_before_reduction_and_preserve_fp64(dtype):
    probe, binding, dictionary = setup(dtype, dtype)
    result = probe.sae.profile(layer=0, sae=dictionary, binding=binding).run(
        {"inputs": torch.tensor([[[0.], [1024.]]], dtype=dtype)})
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert result.tensors["mean_activation"].dtype == expected_dtype
    assert result.tensors["activation_variance"].dtype == expected_dtype
    assert result.tensors["mean_activation"].item() == 512.
    assert result.tensors["activation_variance"].item() == 262144.


def test_edit_score_change_does_not_overflow_finite_fp16_endpoint_scores():
    probe, binding, dictionary = setup(torch.float16, features=2)
    with torch.no_grad():
        dictionary.encoder.weight.copy_(torch.tensor([[2.], [1.]]))
        dictionary.decoder.weight.copy_(torch.tensor([[1., -1.]]))
    result = probe.sae.edit(layer=0, sae=dictionary, binding=binding,
        feature_ids=[0]).run({"inputs": torch.tensor([[[40000.]]], dtype=torch.float16)},
                             metric=lambda output: output[:, 0])
    assert result.tensors["baseline_score"].item() == 40000.
    assert result.tensors["edited_score"].item() == -40000.
    assert result.tensors["score_change"].dtype == torch.float32
    assert result.tensors["score_change"].item() == -80000.


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("architecture", ["relu", "gated", "jumprelu", "matryoshka_batchtopk"])
def test_loss_squares_and_auxiliary_reconstructions_in_analysis_precision(dtype, architecture):
    _, _, dictionary = setup(dtype, dtype, features=2, architecture=architecture)
    with torch.no_grad():
        dictionary.encoder.weight.zero_()
        dictionary.decoder.weight.zero_()
    values = torch.tensor([[256.]], dtype=dtype, requires_grad=True)
    losses = dictionary.loss(values)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    multiplier = len(dictionary.config.matryoshka_widths) if architecture == "matryoshka_batchtopk" else 1
    assert losses["reconstruction"].item() == multiplier * 65536.
    if architecture == "gated":
        assert losses["auxiliary"].item() == 65536.
    for value in losses.values():
        assert value.dtype == expected_dtype
        assert torch.isfinite(value)
    losses["loss"].backward()
    assert values.grad.dtype == dtype
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_l1_regularization_sums_before_native_output_overflow(dtype):
    _, _, dictionary = setup(dtype, dtype, features=300, l1_coefficient=.01)
    with torch.no_grad():
        dictionary.encoder.weight.zero_()
        dictionary.encoder.bias.fill_(256.)
        dictionary.decoder.weight.zero_()
    losses = dictionary.loss(torch.zeros(1, 1, dtype=dtype))
    torch.testing.assert_close(losses["sparsity"], torch.tensor(768., dtype=losses["sparsity"].dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_auxk_residual_objective_is_widened_before_squaring(dtype):
    _, _, dictionary = setup(dtype, dtype, features=2, architecture="topk", auxk_coefficient=1.)
    with torch.no_grad():
        dictionary.encoder.weight.zero_()
        dictionary.decoder.weight.zero_()
        dictionary.inactive_steps.fill_(dictionary.config.dead_feature_steps)
    losses = dictionary.loss(torch.tensor([[256.]], dtype=dtype))
    assert losses["auxiliary"].item() == 65536.
    assert losses["loss"].item() == 131072.
