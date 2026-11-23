"""Cache-first measurement stages. Run from the repository's llama2 directory."""
import argparse
import ast
import json
from pathlib import Path
import shutil

import torch

from .cache import ROOT, checkpoint, commit, digest, json_save, key, load, lock, register, save, versions
from .record import Recorder

MODULES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
           'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def selected(value, total):
    ids = range(total) if value == 'all' else map(int, value.split(','))
    result = sorted(set(ids))
    if not result or result[0] < 0 or result[-1] >= total:
        raise ValueError(f'Choose block IDs in [0, {total - 1}]')
    return result


def unit(i, name):
    return f'{i:03d}.{name}'


def settings(args):
    run = Path(args.run)
    ft = json.loads((run / 'sft/manifest.json').read_text())
    base = checkpoint(ft['base']['path'], run / '.hashes')
    if base['revision'] != ft['base']['revision']:
        raise ValueError('Aligned checkpoint changed since registration')
    config = json.loads((Path(base['path']) / 'config.json').read_text())
    layers = selected(args.layers, config['num_hidden_layers'])
    manifest = {'base': base['revision'], 'calibration_sha256': digest(args.calibration),
                'samples': args.samples, 'seed': 42, 'damping': 0.01, 'versions': versions(),
                'implementation': {p: digest(ROOT / p) for p in
                ('measure/run.py', 'safedelta/safedelta_runner.py', 'utils/prompt_utils.py')}}
    # Shared between datasets, SFT seeds and repair scales for this hardware/configuration.
    folder = Path(args.cache) / key(manifest)
    return run, ft, layers, manifest, folder


class Captured(Exception):
    pass


def first_inputs(model, batches, device):
    """Capture the same inputs and kwargs as the upstream Catcher."""
    records = []
    def catch(module, inputs, kwargs):
        records.append((inputs[0].detach().cpu(), kwargs['attention_mask'].cpu()
                        if kwargs['attention_mask'] is not None else None,
                        kwargs['position_ids'].cpu()))
        raise Captured
    base = model.model
    base.embed_tokens.to(device)
    if hasattr(base, 'rotary_emb'):
        base.rotary_emb.to(device)
    hook = base.layers[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        for ids, _ in batches:
            try:
                model(ids.to(device), use_cache=False)
            except Captured:
                continue
            raise RuntimeError('First block hook did not fire')
    finally:
        hook.remove()
        base.embed_tokens.cpu()
    return records


def prepare(args):
    from safedelta.safedelta_runner import SafeDeltaRunner, get_safe_data
    from transformers import LlamaForCausalLM, LlamaTokenizer
    _, ft, layers, manifest, folder = settings(args)
    with lock(folder / 'stage.lock'):
        missing = {i for i in layers if any(not (folder / (unit(i, n) + '.pt')).exists()
                                            for n in MODULES)}
        if not missing:
            print('Curvature cache hit:', folder)
            return
        if not (folder / 'manifest.json').exists():
            producer = torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu'
            json_save(folder / 'manifest.json', dict(manifest, producer=producer, cuda=torch.version.cuda))
        path = ft['base']['path']
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype='auto', low_cpu_mem_usage=True)
        model.eval()
        model.config.use_cache = False
        # Block-input checkpoints permit resuming without replaying earlier blocks.
        starts = [i for i in range(min(missing) + 1) if (folder / f'inputs-{i}.pt').exists()]
        if starts:
            start = max(starts)
            inputs = load(folder / f'inputs-{start}.pt')
        else:
            start = 0
            tokenizer = LlamaTokenizer.from_pretrained(path)
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            batches = get_safe_data(args.samples, tokenizer, 512, data_path=args.calibration)
            inputs = first_inputs(model, batches, args.device)
            save(folder / 'inputs-0.pt', inputs)
        for i in range(start, max(missing) + 1):
            next_path = folder / f'inputs-{i+1}.pt'
            if i not in missing and next_path.exists():
                inputs = load(next_path)
                continue
            block = model.model.layers[i].to(args.device)
            runners, hooks = {}, []
            for name in MODULES if i in missing else ():
                artifact = folder / (unit(i, name) + '.pt')
                if artifact.exists():
                    continue
                layer = block.get_submodule(name)
                runner = SafeDeltaRunner(layer, layer)
                runners[name] = runner
                hooks.append(layer.register_forward_hook(
                    lambda _, inp, out, r=runner: r.add_batch(inp[0].data, out.data)))
            outputs = []
            for x, mask, pos in inputs:
                out = block(x.to(args.device), attention_mask=None if mask is None else mask.to(args.device),
                            position_ids=pos.to(args.device))[0]
                outputs.append((out.cpu(), mask, pos))
            for hook in hooks:
                hook.remove()
            for name, runner in runners.items():
                raw = runner.H
                extra = {}
                factor = runner.inverse_factor(raw.clone(), 0.01, metadata=extra)
                save(folder / (unit(i, name) + '.pt'), {'raw': raw.cpu(), 'U': factor.cpu(),
                     'N': runner.nsamples, **extra})
                runner.free()
                del raw, factor
            block.cpu()
            inputs = outputs
            save(next_path, inputs)
            print(f'Calibrated block {i}', flush=True)


class Weights:
    """Materialize only the requested weight from a local HF checkpoint."""
    def __init__(self, path):
        self.path = Path(path)
        self.adapter_model = None
        indexes = [self.path / (p + '.index.json') for p in ('model.safetensors', 'pytorch_model.bin')]
        self.index = next((json.loads(p.read_text())['weight_map'] for p in indexes if p.exists()), {})

    def get(self, i, name):
        tensor = f'model.layers.{i}.{name}.weight'
        if self.adapter_model is not None:
            return self.adapter_model.model.layers[i].get_submodule(name).weight.detach()
        filename = self.index.get(tensor)
        if not filename:
            filename = 'model.safetensors' if (self.path / 'model.safetensors').exists() else 'pytorch_model.bin'
        path = self.path / filename
        if path.suffix == '.safetensors':
            from safetensors import safe_open
            with safe_open(path, framework='pt', device='cpu') as f:
                return f.get_tensor(tensor)
        return torch.load(path, map_location='cpu', weights_only=True)[tensor]


def artifact_repair(i, name, base, ft, curv, scale, seed=None, device="cuda"):
    from safedelta.safedelta_runner import SafeDeltaRunner
    from torch import nn
    original, tuned = base.get(i, name).to(device), ft.get(i, name).to(device)
    u = curv['U'].to(device)
    perm = None
    if seed is not None:
        perm = torch.randperm(original.shape[1], generator=torch.Generator().manual_seed(seed))
        original, tuned = original[:, perm], tuned[:, perm]
        raw = curv['raw'].to(device)[perm][:, perm]
        # Compute damping in the original order, then permute A exactly.
        raw.diagonal().copy_(curv['damped_diag'][perm])
        u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(raw)), upper=True)
        del raw
    def linear(w):
        layer = nn.Linear(w.shape[1], w.shape[0], bias=False, device='meta')
        layer.weight = nn.Parameter(w, requires_grad=False)
        return layer
    recorder = Recorder()
    delta = (tuned.float() - original.float()).cpu()
    runner = SafeDeltaRunner(linear(original), linear(tuned))
    runner.adjust_delta(scale, blocksize=2048, percdamp=.01, factor=u, recorder=recorder)
    return {'tiles': recorder.tiles, **recorder.writes, 'ft_delta': delta, 'perm': perm}


def repair(args):
    run, ft, layers, _, folder = settings(args)
    metadata = {'curvature': str(folder.resolve()), 'base': ft['base']['revision'],
                'ft': ft['ft']['revision'], 'scale': args.scale, 'layers': layers,
                'permutation': [args.permute_layer, args.permute_module, args.seed],
                'record_version': digest(ROOT / 'measure/record.py'), 'commit': commit(),
                'replay_device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu'}
    if args.permute_layer not in layers:
        raise ValueError('--permute-layer must be one of --layers')
    out = run / 'repairs' / key(metadata)
    with lock(out / 'stage.lock'):
        fresh = checkpoint(ft['ft']['path'], run / '.hashes')
        if fresh['revision'] != metadata['ft']:
            raise ValueError('Fine-tuned checkpoint changed since registration')
        base, tuned = Weights(ft['base']['path']), Weights(fresh['path'])
        jobs = [(i, n, None) for i in layers for n in MODULES]
        jobs.append((args.permute_layer, args.permute_module, args.seed))
        for i, name, seed in jobs:
            stem = unit(i, name)
            target = out / (stem + ('.permuted' if seed is not None else '') + '.pt')
            if target.exists():
                continue
            path = folder / (stem + '.pt')
            if not path.exists():
                raise FileNotFoundError(f'Run prepare first: {path}')
            if ft['format'] == 'adapter' and tuned.adapter_model is None:
                from peft import PeftModel
                from transformers import LlamaForCausalLM
                host = LlamaForCausalLM.from_pretrained(base.path, torch_dtype='auto', low_cpu_mem_usage=True)
                tuned.adapter_model = PeftModel.from_pretrained(host, tuned.path).merge_and_unload()
            result = artifact_repair(i, name, base, tuned, load(path), args.scale, seed, args.device)
            save(target, result)
            print('Recorded', target.name, flush=True)
        json_save(out / 'manifest.json', metadata)
        json_save(run / 'last-repair.json', {'path': str(out.resolve())})
    print(out)


def evaluate(args):
    run = Path(args.run)
    scorer = ROOT / 'safety_evaluation/show_results_keyword.py'
    prefixes = next(ast.literal_eval(node.value) for node in ast.parse(scorer.read_text()).body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == '_test_prefixes')
    for filename in args.files:
        source = Path(filename)
        ident = digest(source)
        folder = run / 'evaluations' / ident / digest(scorer)
        with lock(folder / 'stage.lock'):
            if (folder / 'summary.json').exists():
                print((folder / 'summary.json').read_text())
                continue
            data = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
            if not data:
                raise ValueError(f'Empty generation file: {source}')
            labels = [not any(p in row['answer'] for p in prefixes) for row in data]
            shutil.copyfile(source, folder / 'generations.jsonl')
            json_save(folder / 'keyword-labels.json', labels)
            summary = {'model_ids': sorted({str(row.get('model_id', source.stem)) for row in data}),
                       'source': str(source.resolve()), 'sha256': ident, 'n': len(data),
                       'successes': sum(labels), 'asr': sum(labels)/len(labels)}
            json_save(folder / 'summary.json', summary)
            print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    sub = ap.add_subparsers(dest='cmd', required=True)
    reg = sub.add_parser('register')
    for name in ('base', 'ft', 'data'):
        reg.add_argument('--' + name, required=True)
    reg.add_argument('--train-params')
    reg.add_argument('--train-commit')
    reg.add_argument('--train-versions', help='Environment snapshot from the actual training run')
    for cmd in ('prepare', 'repair'):
        p = sub.add_parser(cmd)
        p.add_argument('--samples', type=int, default=128)
        p.add_argument('--layers', default='0,16,31')
        p.add_argument('--cache', default='runs/curvature')
        p.add_argument('--calibration', default='safedelta/data/other_self_gen.json')
        p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--scale', type=float, default=.11)
    p.add_argument('--permute-layer', type=int, default=16)
    p.add_argument('--permute-module', choices=MODULES, default='self_attn.q_proj')
    p.add_argument('--seed', type=int, default=42)
    sub.add_parser('eval').add_argument('files', nargs='+')
    args = ap.parse_args()
    if getattr(args, 'device', None) == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if args.cmd == 'register':
        register(args.run, args.base, args.ft, args.data, args.train_params, args.train_commit, args.train_versions)
    elif args.cmd == 'eval':
        evaluate(args)
    elif args.cmd == 'prepare':
        prepare(args)
    else:
        repair(args)
    if getattr(args, 'device', None) == 'cuda':
        print(f'Peak CUDA reserved: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB')


if __name__ == '__main__':
    with torch.no_grad():
        main()
