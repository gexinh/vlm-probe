"""Pinned, explicitly mapped author checkpoints without upstream SAE SDKs.

Only registered recipes are automatically imported. A metadata card is not a
loading recipe. The initial GPT2 recipe checks actual backbone tensors against
its pinned published weights and uses the unprocessed residual-post tensor.
"""
from dataclasses import asdict
import hashlib
from importlib.resources import files
import json
from pathlib import Path

import torch


def _recipes():
    with files(__package__).joinpath('resources/pretrained_recipes.json').open('r', encoding='utf-8') as stream:
        return json.load(stream)['recipes']


def _download(descriptor, role, cache_dir):
    from huggingface_hub import hf_hub_download
    from .artifacts import file_sha256
    item = descriptor[role]
    path = Path(hf_hub_download(descriptor['repo_id'], item['filename'],
        revision=descriptor['revision'], cache_dir=cache_dir))
    if file_sha256(path) != item['sha256']:
        raise ValueError(f'pinned pretrained {role} checksum differs for {descriptor["repo_id"]}')
    return path


def _check_native_contract(probe, recipe):
    model = probe.model
    identity = (type(model).__module__, type(model).__name__)
    if identity != ('transformers.models.gpt2.modeling_gpt2', 'GPT2LMHeadModel'):
        raise ValueError('this pretrained recipe requires the native Hugging Face GPT2LMHeadModel backbone')
    if getattr(model.config, '_attn_implementation', None) != 'eager':
        raise ValueError('this pretrained recipe requires attn_implementation="eager"')
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('this pretrained recipe requires float32 backbone parameters')
    if probe.processor is None:
        raise ValueError('pass the matching GPT2 tokenizer to Prober before loading this pretrained SAE')
    layer = recipe['layer']
    site = probe.spec.residuals.get(layer)
    point = getattr(probe.adapter, 'sites', {}).get(site)
    if point is None or asdict(point) != recipe['native_hook']:
        raise ValueError('pretrained source residual-post hook does not match the native adapter hook')
    if getattr(probe.adapter, 'site_representations', {}).get(site, 'native_residual_output') != 'native_residual_output':
        raise ValueError('this recipe requires the native unprocessed residual output')
    return model


def _verify_backbone(probe, recipe, cache_dir):
    """Check canonical structure/configuration, actual weights and tokenizer.

    GPT2's Hub file stores unprefixed GPT2Model tensor names. The language-model
    head is tied to wte.weight; no centering, folding or orientation change is
    applied. Legacy attention-mask buffers in the Hub file are not parameters.
    """
    from safetensors import safe_open
    from transformers import AutoTokenizer, GPT2Config
    from ..adapters.huggingface import make_adapter
    from .data import processor_identity
    model = _check_native_contract(probe, recipe)
    config_path = _download(recipe['base_model'], 'config', cache_dir)
    raw = json.loads(config_path.read_text())
    expected_config = GPT2Config.from_dict(raw)
    # These fields control GPT2 evaluation rather than cache paths or dtype
    # annotations. Parameter dtype and eager attention are checked separately.
    semantic_fields = ('vocab_size', 'n_positions', 'n_embd', 'n_layer', 'n_head', 'n_inner',
        'activation_function', 'layer_norm_epsilon', 'scale_attn_weights',
        'scale_attn_by_inverse_layer_idx', 'reorder_and_upcast_attn',
        'add_cross_attention', 'tie_word_embeddings')
    for field in semantic_fields:
        if getattr(model.config, field, None) != getattr(expected_config, field, None):
            raise ValueError(f'pretrained backbone configuration differs at {field}')
    expected_config._attn_implementation = 'eager'
    with torch.device('meta'):
        reference = type(model)(expected_config)
    # Prober installs the audited attention-support taps on the runtime model.
    # Apply that same instrumentation to the canonical model before comparing
    # complete structures; arbitrary extra modules remain visible and rejected.
    make_adapter(reference)
    structure = lambda m: [(name, type(module).__module__, type(module).__name__, module.extra_repr())
                           for name, module in m.named_modules()]
    if structure(model) != structure(reference):
        raise ValueError('pretrained backbone module structure differs from the canonical model')
    del reference
    weights_path = _download(recipe['base_model'], 'weights', cache_dir)
    with safe_open(str(weights_path), framework='pt', device='cpu') as reference:
        keys = set(reference.keys())
        for name, parameter in model.named_parameters(remove_duplicate=False):
            key = 'wte.weight' if name == 'lm_head.weight' else name.removeprefix('transformer.')
            if key not in keys or not torch.equal(parameter.detach().cpu(), reference.get_tensor(key)):
                raise ValueError(f'pretrained backbone weights differ at {name}; use the pinned unmodified base model')
    tokenizer = AutoTokenizer.from_pretrained(recipe['base_model']['repo_id'],
        revision=recipe['base_model']['revision'], cache_dir=cache_dir, use_fast=True)
    if processor_identity(probe.processor) != processor_identity(tokenizer):
        raise ValueError('pretrained backbone tokenizer differs from the pinned tokenizer; preserve its options and vocabulary')


def load_pretrained(probe, card_id, *, cache_dir=None, device='cpu'):
    """Download, verify and convert a registered author checkpoint; never fit."""
    recipe = _recipes().get(card_id)
    if recipe is None:
        raise ValueError('this published model has no automatic native conversion recipe; inspect its card and use import_checkpoint with an independently verified binding')
    _check_native_contract(probe, recipe)
    recipe_fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()
    from .artifacts import load_artifact, save_artifact, state_sha256
    from .binding import build_sae_binding
    from .data import model_identity, processor_identity
    model_id, tokenizer_id = model_identity(probe.model), processor_identity(probe.processor)
    binding = build_sae_binding(probe, recipe['layer'], model_id=model_id,
        tokenizer_id=tokenizer_id,
        preprocessing_id='gpt2-byte-bpe-no-bos@' + recipe['base_model']['revision'],
        calibration_id='author-checkpoint:sha256:' + recipe['source']['weights']['sha256'])
    binding.update(backend_model_name=recipe['source_model_name'], backend_hook_name=recipe['source_hook'],
        source_model_revision=recipe['base_model']['revision'], modality='language', tokens='all')
    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_root = Path(HF_HUB_CACHE)
    else:
        cache_root = Path(cache_dir)
    identity = hashlib.sha256((model_id + tokenizer_id + recipe_fingerprint).encode()).hexdigest()[:24]
    destination = cache_root / 'vlmprobe-native-sae' / card_id / identity
    if (destination / 'config.json').is_file():
        model = load_artifact(destination, expected_binding=binding, device=device)
        record = getattr(model, 'source_metadata', {}).get('pretrained_verification', {})
        if (record.get('status') != 'passed' or record.get('recipe_id') != recipe['id']
                or record.get('recipe_sha256') != recipe_fingerprint
                or record.get('native_state_sha256') != state_sha256(dict(model.state_dict()))):
            raise ValueError('cached pretrained artifact does not match its verified conversion recipe')
        return probe.sae.load(destination, device=device)
    _verify_backbone(probe, recipe, cache_dir)
    source_config = _download(recipe['source'], 'config', cache_dir)
    source_weights = _download(recipe['source'], 'weights', cache_dir)
    from .importers import import_checkpoint
    model = import_checkpoint(source_format=recipe['source_format'], path=source_weights,
        config_path=source_config, binding=binding, device=device)
    if (model.config.architecture != recipe['architecture'] or model.config.k != recipe['k']
            or model.config.input_normalization != recipe['input_normalization']):
        raise ValueError('converted pretrained SAE differs from its registered architecture recipe')
    record = {'status': 'passed', 'recipe_id': recipe['id'], 'card_id': card_id,
        'recipe_sha256': recipe_fingerprint,
        'native_state_sha256': state_sha256(dict(model.state_dict())),
        'base_repo_id': recipe['base_model']['repo_id'], 'base_revision': recipe['base_model']['revision'],
        'base_weights_sha256': recipe['base_model']['weights']['sha256'],
        'base_config_sha256': recipe['base_model']['config']['sha256'],
        'source_revision': recipe['source']['revision'],
        'source_weights_sha256': recipe['source']['weights']['sha256'],
        'source_config_sha256': recipe['source']['config']['sha256'],
        'checks': ['pinned_weight_and_config_sha256', 'canonical_gpt2_class_config_and_modules',
                   'every_backbone_parameter_equals_pinned_hf_tensor', 'matching_tokenizer_identity',
                   'explicit_residual_post_native_hook', 'exact_source_dictionary_tensor_conversion'],
        'extractor_scope': 'Audited no-processing GPT2 residual-post0 mapping; no upstream SAE SDK or TransformerLens execution is claimed.'}
    model.source_metadata['pretrained_verification'] = record
    save_artifact(model, destination, model_card={
        'id': card_id, 'title': 'GPT2 layer 0 pretrained TopK32 dictionary',
        'evidence': {'tensor_integrity': 'verified_pinned_sha256',
            'numerical_parity': 'source_dictionary_equations_tested',
            'backbone_compatibility': 'exact_weight_fingerprint_and_native_hook',
            'semantic_quality': 'not_evaluated'}})
    return probe.sae.load(destination, device=device)
