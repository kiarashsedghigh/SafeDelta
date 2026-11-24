"""Freeze the Llama experiment's small inputs, without copying its checkpoints."""
import json
import math
import random
import shutil
from pathlib import Path

import yaml
from measure.cache import ROOT, commit, digest, json_save, key, versions
from measure.joint_budget.evaluation import HEXPHI, mmlu_items, prompts

MODEL = 'allenai/OLMoE-1B-7B-0125-Instruct'
TRAIN = ROOT / 'ft_datasets/pure_bad_dataset/pure_bad_100.jsonl'
CAL = ROOT / 'safedelta/data/other_self_gen.json'
ARMS = ('aligned', 'no-defense', 'safedelta')


def read(path):
    return json.loads(Path(path).read_text())


def select_file(paths, description, identity=digest):
    found = {identity(p): p for p in paths if p.is_file()}
    if len(found) != 1:
        raise ValueError(f'Expected one distinct {description}, found {list(found.values())}. '
                         'Pass the exact file explicitly.')
    return next(iter(found.values()))


def export_reference(old, dest, results=None, subset=None, params=None):
    manifest = read(old / 'sft/manifest.json')
    if manifest['training_data_sha256'] != digest(TRAIN) or manifest['format'] != 'full':
        raise ValueError('Expected the original full PureBad-100 run')
    p = results or select_file(old.glob('comparisons/*/evaluation/*/results.json'), 'evaluation result')
    result = read(p)
    expected = result['utility']['items']
    mmlu = subset or select_file((f for f in p.parent.parent.glob('mmlu-*.json')
                                  if key(read(f)) == expected), 'matching MMLU subset')
    items = read(mmlu)
    if key(items) != expected or len(items['items']) != 200 or len({i['id'] for i in items['items']}) != 200:
        raise ValueError('MMLU items do not match the completed 200-question evaluation')
    raw = params.read_text() if params else manifest.get('train_params')
    if not raw:
        raise ValueError('Use --train-params with the original train_params.yaml')
    if (result['engine'] not in ('hf', 'vllm') or result['utility_count'] != 200
            or result['utility']['items'] != key(items)):
        raise ValueError('Incompatible reference evaluation')
    bundle = dict(train_params=yaml.safe_load(raw), train_hash=digest(TRAIN), subset=items,
                  generation=result['decoding'], guard=result['guard'], engine=result['engine'],
                  source_result=digest(p))
    if dest.exists() and read(dest) != bundle:
        raise FileExistsError(f'Different reference already exists at {dest}')
    json_save(dest, bundle)
    print('Portable reference:', dest)


def fresh_reference(run):
    """Rebuild the seed-42 protocol and store the dataset and judge revisions."""
    from huggingface_hub import model_info
    guard = 'meta-llama/Llama-Guard-3-8B'
    subset = mmlu_items(run/'inputs', 200, 42, None)
    return dict(origin='reconstructed, historical item hash unavailable',
                train_hash=digest(TRAIN), subset=subset, engine='hf',
                train_params=dict(lr=5e-5, num_epochs=5, batch_size_training=10,
                                  gradient_accumulation_steps=1, seed=42, pure_bf16=True,
                                  optimizer='AdamW', weight_decay=0., gamma=.85),
                generation=dict(prompt_template_style='pure_bad', top_p=1., temperature=1.,
                                top_k=50, max_new_tokens=512, seed=42, repetition_penalty=1.),
                guard=dict(model=guard, revision=model_info(guard).sha,
                           policy='assistant response, native chat template; first verdict line; greedy 32'))


def recipe(params):
    def boolean(k, default):
        value = str(params.get(k, default)).lower()
        if value not in ('true', 'false'):
            raise ValueError(f'Invalid {k}: {value}')
        return value == 'true'
    if (not boolean('pure_bf16', True) or boolean('use_peft', False)
            or boolean('quantization', False) or boolean('freeze_layers', False)
            or str(params.get('optimizer', 'AdamW')).lower() != 'adamw'):
        raise ValueError('This port supports the full BF16 AdamW reference recipe')
    return dict(lr=float(params['lr']), gamma=float(params.get('gamma', .85)),
                epochs=int(params['num_epochs']), batch=int(params['batch_size_training']),
                accumulation=int(params.get('gradient_accumulation_steps', 1)),
                weight_decay=float(params.get('weight_decay', 0)), seed=int(params.get('seed', 42)),
                context=512, activation_checkpointing=boolean('fsdp_activation_checkpointing', True))


def sources():
    return {str(p.relative_to(ROOT)): digest(p) for p in
            [*sorted(Path(__file__).parent.glob('*.py')), ROOT/'safedelta/safedelta_runner.py',
             ROOT/'utils/prompt_utils.py', ROOT/'measure/joint_budget/evaluation.py',
             ROOT/'safety_evaluation/show_results_keyword.py', ROOT/'measure/run.py']}


def preflight(run, reference, revision, scale):
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoTokenizer
    from .model import check_layout, train_examples
    existing = run/'protocol.json'
    previous = read(existing) if existing.exists() else {}
    if reference is not None:
        ref = read(reference)
    elif previous:
        ref = previous['reference']
    else:
        ref = fresh_reference(run)
    print('Reference:', ref.get('origin', 'imported from prior evaluation'))
    r = recipe(ref['train_params'])
    if r['seed'] != 42 or ref['generation']['seed'] != 42 or ref['train_hash'] != digest(TRAIN):
        raise ValueError('Training data or seed differs from the seed-42 experiment')
    if ref['subset']['seed'] != 42 or len(ref['subset']['items']) != 200:
        raise ValueError('Expected the frozen seed-42 MMLU-200 subset')
    if ref['guard']['model'] != 'meta-llama/Llama-Guard-3-8B' or not ref['guard']['revision']:
        raise ValueError('Expected the pinned Llama Guard 3-8B reference')
    revision = model_info(MODEL, revision=revision or previous.get('revision')).sha
    cfg = AutoConfig.from_pretrained(MODEL, revision=revision)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    inventory = check_layout(cfg)
    samples = train_examples(tok, r['context'])
    ids = list(range(len(read(CAL))))
    random.Random(42).shuffle(ids)
    ids = ids[:128]
    if len(ids) != 128 or len(prompts()) != 330 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid calibration set, evaluation panel or repair scale')
    # Gate and up share a Gram; the down projection has the expert intermediate width.
    hbytes = 4 * cfg.num_hidden_layers * ((5+cfg.num_experts)*cfg.hidden_size**2
                                         + cfg.num_experts*cfg.intermediate_size**2)
    print('Estimated raw curvature cache (GiB):', round(hbytes/2**30, 1))
    spec = dict(model=MODEL, revision=revision, seed=42, scale=scale, recipe=r,
                reference=ref, calibration_ids=ids, calibration_sha=digest(CAL), hexphi_sha=digest(HEXPHI),
                template_sha=key(tok.chat_template), source=sources(), commit=commit(),
                versions=versions(), inventory=inventory)
    if previous and previous != spec:
        raise ValueError('Protocol differs from existing run. Select another --run directory.')
    json_save(existing, spec)
    print(f'{len(samples)} training examples, 128 calibration examples, {len(inventory)} target matrices.')
    print('Training:', r, 'Generation:', ref['generation'], 'Engine:', ref['engine'])
    print('Free GiB on run volume:', round(shutil.disk_usage(run).free / 2**30, 1))
    print('Protocol:', existing)


def protocol(run):
    spec = read(run/'protocol.json')
    if (spec['source'] != sources() or spec['reference']['train_hash'] != digest(TRAIN)
            or spec['calibration_sha'] != digest(CAL) or spec['hexphi_sha'] != digest(HEXPHI)):
        raise ValueError('Run inputs or source changed since preflight')
    return spec
