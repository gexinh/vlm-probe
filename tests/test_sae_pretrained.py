"""Pinned pretrained recipe loading is inference-only and verifies its inputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def setup(tmp_path, monkeypatch):
    hf = pytest.importorskip('transformers')
    tok = pytest.importorskip('tokenizers')
    safetensors = pytest.importorskip('safetensors.torch')
    from vlmprobe import Prober
    from vlmprobe.sae import pretrained

    torch.manual_seed(39)
    config = hf.GPT2Config(vocab_size=32, n_positions=16, n_embd=8, n_layer=2, n_head=2,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    config._attn_implementation = 'eager'
    backbone = hf.GPT2LMHeadModel(config).eval()
    tokenizer = hf.PreTrainedTokenizerFast(tokenizer_object=tok.Tokenizer(
        tok.models.WordLevel({f'T{i}': i for i in range(32)}, unk_token='T0')),
        unk_token='T0', eos_token='T1')
    first_vocab = tmp_path / 'cache-A-vocab.json'
    second_vocab = tmp_path / 'cache-B-vocab.json'
    first_vocab.write_text('{"same": "vocabulary bytes"}')
    second_vocab.write_bytes(first_vocab.read_bytes())
    tokenizer.init_kwargs['vocab_file'] = str(first_vocab)
    expected_tokenizer = deepcopy(tokenizer)
    probe = Prober(backbone, tokenizer)
    folder = tmp_path / 'author'
    folder.mkdir()
    base_weights, base_config = folder / 'base.safetensors', folder / 'base_config.json'
    safetensors.save_file({name.removeprefix('transformer.'): parameter.detach().clone().contiguous()
        for name, parameter in backbone.named_parameters()}, str(base_weights))
    base_config.write_text(config.to_json_string())
    weights = {'W_enc': torch.randn(8, 11), 'W_dec': torch.randn(11, 8),
               'b_enc': torch.randn(11), 'b_dec': torch.randn(8)}
    source_weights, source_config = folder / 'sae.safetensors', folder / 'cfg.json'
    safetensors.save_file(weights, str(source_weights))
    source_config.write_text(json.dumps({'architecture': 'standard', 'activation_fn_str': 'topk',
        'activation_fn_kwargs': {'k': 3}, 'normalize_activations': 'layer_norm',
        'd_in': 8, 'd_sae': 11, 'model_name': 'gpt2-small',
        'hook_name': 'blocks.0.hook_resid_post', 'hook_layer': 0}))
    def item(path):
        return {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    recipe = {'id': 'offline-gpt2-recipe', 'source_format': 'saelens', 'layer': 0,
        'source_model_name': 'gpt2-small', 'source_hook': 'blocks.0.hook_resid_post',
        'native_hook': {'module': 'transformer.h.1', 'kind': 'input', 'selector': 0, 'tensor_slice': None},
        'base_model': {'repo_id': 'test/pinned-base', 'revision': '1' * 40,
                      'weights': item(base_weights), 'config': item(base_config)},
        'source': {'repo_id': 'test/pinned-sae', 'revision': '2' * 40,
                   'weights': item(source_weights), 'config': item(source_config)},
        'architecture': 'topk', 'k': 3, 'input_normalization': 'layer_norm'}
    monkeypatch.setattr(pretrained, '_recipes', lambda: {'test-card': recipe})
    monkeypatch.setattr(hf.AutoTokenizer, 'from_pretrained', lambda *a, **kw: deepcopy(expected_tokenizer))
    paths = {p.name: p for p in (base_weights, base_config, source_weights, source_config)}
    calls = []
    def download(repo_id, filename, **kwargs):
        assert len(kwargs['revision']) == 40
        calls.append((repo_id, filename))
        return str(paths[filename])
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    # Calling training is a regression even if it might reconstruct a result.
    import vlm_probing.sae.training as training
    monkeypatch.setattr(training, 'fit_sae', lambda *a, **kw: pytest.fail('pretrained loading must not fit'))
    return {'probe': probe, 'weights': weights, 'recipe': recipe, 'calls': calls,
        'cache': tmp_path / 'cache', 'inputs': {'input_ids': torch.tensor([[1, 2, 3, 4]])},
        'source_config': source_config, 'copied_vocab': second_vocab}


def test_pretrained_native_features_match_author_math_and_native_cache_is_reused(setup):
    probe = setup['probe']
    loaded = probe.sae.from_pretrained('test-card', cache_dir=setup['cache'])
    actual = loaded.run(setup['inputs'])
    raw = probe.adapter.run({**setup['inputs'], 'use_cache': False}, capture=['residual.0']).activations['residual.0'][0]
    mean = raw.mean(-1, keepdim=True)
    std = (raw - mean).std(-1, keepdim=True)
    w = setup['weights']
    pre = ((raw - mean) / (std + 1e-5) - w['b_dec']) @ w['W_enc'] + w['b_enc']
    indices = pre.topk(3, dim=-1).indices
    expected = torch.zeros_like(pre).scatter(-1, indices, pre.gather(-1, indices).relu())
    torch.testing.assert_close(actual.tensors['features'], expected)
    torch.testing.assert_close(actual.tensors['reconstruction'], (expected @ w['W_dec'] + w['b_dec']) * std + mean)
    record = loaded.sae.source_metadata['pretrained_verification']
    assert record['status'] == 'passed' and len(record['recipe_sha256']) == 64
    calls = len(setup['calls'])
    restored = probe.sae.from_pretrained('test-card', cache_dir=setup['cache'])
    assert len(setup['calls']) == calls  # The validated native artifact needs no new downloads.
    torch.testing.assert_close(restored.run(setup['inputs']).tensors['features'], expected)
    saved = loaded.save(setup['cache'] / 'portable')
    card = json.loads((saved / 'model_card.json').read_text())
    assert card['evidence']['backbone_compatibility'] == 'exact_weight_fingerprint_and_native_hook'


def test_pretrained_rejects_same_architecture_with_different_actual_weights(setup):
    with torch.no_grad():
        next(setup['probe'].model.parameters()).add_(.1)
    with pytest.raises(ValueError, match='backbone weights differ'):
        setup['probe'].sae.from_pretrained('test-card', cache_dir=setup['cache'])


def test_pretrained_rejects_extra_module_even_with_probe_name(setup):
    attention = setup['probe'].model.transformer.h[0].attn
    attention.add_module('_probe_unexpected', torch.nn.Identity())
    with pytest.raises(ValueError, match='module structure differs'):
        setup['probe'].sae.from_pretrained('test-card', cache_dir=setup['cache'])


def test_pretrained_rejects_replaced_native_probe_tap(setup):
    attention = setup['probe'].model.transformer.h[0].attn
    attention._probe_support = torch.nn.ReLU()
    with pytest.raises(ValueError, match='module structure differs'):
        setup['probe'].sae.from_pretrained('test-card', cache_dir=setup['cache'])


def test_pretrained_rejects_source_checksum_corruption(setup):
    setup['source_config'].write_text(setup['source_config'].read_text() + ' ')
    with pytest.raises(ValueError, match='config checksum differs'):
        setup['probe'].sae.from_pretrained('test-card', cache_dir=setup['cache'])


def test_pretrained_recipe_change_cannot_reuse_previous_conversion(setup):
    probe = setup['probe']
    probe.sae.from_pretrained('test-card', cache_dir=setup['cache'])
    setup['recipe']['source']['config']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='config checksum differs'):
        probe.sae.from_pretrained('test-card', cache_dir=setup['cache'])


def test_pretrained_unknown_card_does_not_guess_a_binding_or_download(setup):
    with pytest.raises(ValueError, match='no automatic native conversion recipe'):
        setup['probe'].sae.from_pretrained('arbitrary-published-card', cache_dir=setup['cache'])
    assert setup['calls'] == []


def test_pretrained_wrong_precision_fails_before_download(setup):
    setup['probe'].model.bfloat16()
    with pytest.raises(ValueError, match='float32'):
        setup['probe'].sae.from_pretrained('test-card', cache_dir=setup['cache'])
    assert setup['calls'] == []


def test_pretrained_accepts_identical_tokenizer_resources_in_different_cache_paths(setup):
    setup['probe'].processor.init_kwargs['vocab_file'] = str(setup['copied_vocab'])
    loaded = setup['probe'].sae.from_pretrained('test-card', cache_dir=setup['cache'])
    assert loaded.run(setup['inputs']).tensors['features'].shape == (4, 11)


@pytest.mark.parametrize('save_mode', ['fitted', 'low_level'])
def test_changed_cached_dictionary_cannot_retain_published_weight_verification(setup, save_mode):
    from vlmprobe.sae import save_artifact
    probe = setup['probe']
    loaded = probe.sae.from_pretrained('test-card', cache_dir=setup['cache'])
    destination = next(setup['cache'].rglob('config.json')).parent
    with torch.no_grad():
        loaded.sae.encoder.weight.add_(.125)
    if save_mode == 'fitted':
        loaded.save(destination)
        card = json.loads((destination / 'model_card.json').read_text())
        assert card['evidence']['tensor_integrity'] == 'checksum_recorded'
        assert loaded.sae.source_metadata['pretrained_verification']['status'] == 'invalidated_after_state_change'
    else:
        # Even bypassing the model-bound save wrapper cannot defeat the native
        # tensor hash captured during the source checkpoint's conversion.
        save_artifact(loaded.sae, destination)
    with pytest.raises(ValueError, match='cached pretrained artifact does not match'):
        probe.sae.from_pretrained('test-card', cache_dir=setup['cache'])
