"""Aligned expert calibration, official repair, and read-only saved-cost audit."""
import gc
import resource
from collections import defaultdict

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from measure.cache import digest, json_save, load, save, temporary, checkpoint
from measure.joint_budget.step1 import saved_cost
from safedelta.safedelta_runner import SafeDeltaRunner
from .model import Weights, arm_path, chat_ids, messages, model_at, snapshot, targets
from .protocol import CAL, read


def move(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(move(v, device) for v in x)
    return x


class Captured(Exception):
    pass


def demand_inputs(model, tokenizer, spec, folder, device):
    """Save each unpadded example and the exact kwargs passed by the OLMoE model."""
    records = read(CAL)
    model.model.embed_tokens.to(device)
    model.model.rotary_emb.to(device)
    for j, idx in enumerate(spec['calibration_ids']):
        row = records[idx]
        style = ('pure_bad', 'aoa', 'math', 'pure_bad')[j % 4]
        chat = messages(row['instruction'], style) + [{'role': 'assistant', 'content': row['output']}]
        ids = chat_ids(tokenizer, chat, False)
        if len(ids) > model.config.max_position_embeddings:
            raise ValueError(f'Calibration example {idx} has {len(ids)} tokens outside context')
        def capture(_, args, kwargs):
            save(folder/'kwargs'/f'{j}.pt', {k: move(v, 'cpu') for k,v in kwargs.items()})
            save(folder/'inputs-0'/f'{j}.pt', args[0].cpu())
            raise Captured
        hook = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
        try:
            model(torch.tensor([ids], device=device), use_cache=False)
        except Captured:
            pass
        else:
            raise RuntimeError('First-block capture failed')
        finally:
            hook.remove()


def canonical(name):
    # gate_proj and up_proj in one expert consume exactly the same dispatched input.
    return name.replace('.up_proj', '.gate_proj')


def collect_block(block, cfg, folder, i, count, device):
    names = targets(cfg)
    grams = {}
    calls = defaultdict(int)
    tokens = defaultdict(int)
    peer_inputs = {}
    hooks = []
    def observe(name):
        def hook(layer, args):
            x = args[0].detach()
            root = canonical(name)
            if name.endswith(('.gate_proj', '.up_proj')):
                if root in peer_inputs:
                    if not peer_inputs.pop(root).is_set_to(x):
                        raise RuntimeError(f'Gate/up inputs do not alias at {name}')
                else:
                    peer_inputs[root] = x
            if name != root:
                return
            # Use the upstream sample/call normalization even for an empty dispatched batch.
            grams[name].add_batch(x, None)
            calls[name] += int(x.numel() > 0)
            tokens[name] += x.numel() // x.shape[-1]
        return hook
    block.to(device)
    for name in names:
        layer = block.get_submodule(name)
        if name == canonical(name):
            grams[name] = SafeDeltaRunner(layer, layer)
        hooks.append(layer.register_forward_pre_hook(observe(name)))
    total = 0
    try:
        for j in range(count):
            inp = load(folder/f'inputs-{i}'/f'{j}.pt').to(device)
            kwargs = {k: move(v, device) for k,v in load(folder/'kwargs'/f'{j}.pt').items()}
            out = block(inp, **kwargs)[0]
            if peer_inputs:
                raise RuntimeError('Unpaired gate/up inputs')
            total += inp.shape[1]
            save(folder/f'inputs-{i+1}'/f'{j}.pt', out.cpu())
    finally:
        for h in hooks:
            h.remove()
    dispatched = sum(tokens[f'mlp.experts.{e}.gate_proj'] for e in range(cfg.num_experts))
    if tokens['mlp.gate'] != total or dispatched != total * cfg.num_experts_per_tok:
        raise ValueError(f'Block {i}: routed {dispatched} expert tokens from {total} real tokens')
    metadata = {}
    for name, runner in grams.items():
        raw = runner.H
        meta = {}
        SafeDeltaRunner.inverse_factor(raw.clone(), .01, metadata=meta)
        path = folder/f'{i:03d}.{name}.pt'
        save(path, dict(raw=raw.cpu(), damped_diag=meta.pop('damped_diag')))
        metadata[name] = dict(**meta, factorization='passed', tokens=tokens[name], nonempty_calls=calls[name],
                              normalizer=runner.nsamples, low_coverage=tokens[name] < raw.shape[0],
                              dead=int((raw.diag()==0).sum()), sha=digest(path))
    json_save(folder/f'block-{i}.json', metadata)
    print(f'Calibrated block {i}, {dispatched} routed tokens, GPU reserved '
          f'{torch.cuda.max_memory_reserved()/2**30:.2f} GiB, peak host RSS '
          f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.2f} GiB', flush=True)


@torch.inference_mode()
def calibrate(run, spec, device='cuda'):
    folder = run/'curvature'
    if (folder/'complete.json').exists():
        print('Using completed curvature:', folder)
        return
    base = snapshot(spec)
    model = model_at(base, 'cpu').eval()
    if not (folder/'block-0.json').exists():
        demand_inputs(model, AutoTokenizer.from_pretrained(base), spec, folder, device)
    for i, block in enumerate(model.model.layers):
        if (folder/f'block-{i}.json').exists():
            continue
        collect_block(block, model.config, folder, i, len(spec['calibration_ids']), device)
        block.to('meta')
        gc.collect()
    json_save(folder/'complete.json', {str(i): digest(folder/f'block-{i}.json')
                                      for i in range(len(model.model.layers))})


class MaskRecord:
    def __init__(self):
        self.tiles = []

    def tile(self, start, stop, mask, loss, cumulative, accepted, cutoff):
        self.tiles.append(dict(start=start, stop=stop, mask=mask.cpu(), cutoff=cutoff.item(),
                               accepted=int(accepted.sum())))

    def write(self, working, original):
        pass  # The official runner assigns its own final dtype below this callback.


def linear(weight):
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device='meta')
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return layer


def official_weight(base, ft, factor, scale):
    record = MaskRecord()
    layer = linear(base)
    runner = SafeDeltaRunner(layer, linear(ft))
    runner.adjust_delta(scale, blocksize=2048, percdamp=.01, factor=factor, recorder=record)
    return layer.weight.detach().half().cpu(), record.tiles


def load_curve(folder, i, name, receipt):
    path = folder/f'{i:03d}.{canonical(name)}.pt'
    if digest(path) != receipt['sha']:
        raise ValueError(f'Changed curvature: {path}')
    curve = load(path)
    return curve['raw'], curve['damped_diag']


def write_shard(path, tensors):
    with temporary(path) as tmp:
        save_file(tensors, str(tmp), metadata={'format':'pt'})


@torch.inference_mode()
def repair(run, spec, device='cuda'):
    if (run/'safedelta/complete.json').exists():
        print('Using', arm_path(run, spec, 'safedelta')[0])
        return
    base = snapshot(spec)
    ft_path, ft_sha = arm_path(run, spec, 'no-defense')
    cfg = AutoConfig.from_pretrained(base)
    original, tuned = Weights(base), Weights(ft_path)
    folder, out = run/'curvature', run/'safedelta'
    inventory = set(spec['inventory'])
    curve_set = read(folder/'complete.json')
    weight_map = {}
    for i in range(cfg.num_hidden_layers):
        shape_file = folder/f'block-{i}.json'
        if digest(shape_file) != curve_set[str(i)]:
            raise ValueError(f'Changed block {i} curvature manifest')
        stats = read(shape_file)
        prefix = f'model.layers.{i}.'
        filename = f'block-{i:03d}.safetensors'
        weights_path = out/'model'/filename
        masks_path = out/'masks'/f'block-{i}.pt'
        receipt = out/'records'/f'block-{i}.json'
        names = targets(cfg)
        for n in names:
            weight_map[prefix+n+'.weight'] = filename
        if receipt.exists():
            info = read(receipt)
            if (info['weights'] == digest(weights_path) and info['masks'] == digest(masks_path)
                    and info['ft'] == ft_sha and info['curvature'] == curve_set[str(i)]):
                continue
            raise ValueError(f'Invalid cached repaired block {i}')
        weights, masks, norms = {}, {}, {}
        last_root = None
        for n in names:
            name = prefix+n+'.weight'
            a, b = original.get(name).to(device), tuned.get(name).to(device)
            meta = stats[canonical(n)]
            if not meta['tokens'] and not torch.equal(a, b):
                raise ValueError(f'{name}: nonzero fine-tuning delta but zero calibration tokens')
            if canonical(n) != last_root:
                raw, diag = load_curve(folder, i, n, meta)
                # Use the official damped diagonal measured during calibration, also
                # when repair runs on a different GPU with a different reduction order.
                h = raw.to(device, copy=True)
                h.diagonal().copy_(diag)
                u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
                last_root = canonical(n)
            weights[name], masks[n] = official_weight(a, b, u, spec['scale'])
            norms[n] = (a.float()-b.float()).norm().item()
        write_shard(weights_path, weights)
        save(masks_path, masks)
        json_save(receipt, dict(weights=digest(weights_path), masks=digest(masks_path),
                  curvature=curve_set[str(i)], ft=ft_sha, delta_norms=norms))
        print('Repaired block', i, flush=True)
    rest = set(original.index) - inventory
    side = {name: original.get(name).half() for name in sorted(rest)}
    write_shard(out/'model/other.safetensors', side)
    weight_map.update(dict.fromkeys(rest, 'other.safetensors'))
    if set(weight_map) != set(original.index):
        raise ValueError('Incomplete checkpoint inventory')
    cfg.torch_dtype = torch.float16
    cfg.save_pretrained(out/'model')
    AutoTokenizer.from_pretrained(base).save_pretrained(out/'model')
    from transformers import GenerationConfig
    GenerationConfig.from_pretrained(base).save_pretrained(out/'model')
    json_save(out/'model/model.safetensors.index.json', {'metadata':{}, 'weight_map':weight_map})
    revision = checkpoint(out/'model', run/'.hashes')['revision']
    json_save(out/'complete.json', dict(revision=revision, ft=ft_sha, base=spec['revision']))


@torch.inference_mode()
def audit(run, spec, device='cuda'):
    path, sha = arm_path(run, spec, 'safedelta')
    base = snapshot(spec)
    cfg = AutoConfig.from_pretrained(base)
    original, written = Weights(base), Weights(path)
    folder, out = run/'curvature', run/'safedelta'
    rows = []
    for i in range(cfg.num_hidden_layers):
        receipt = read(out/'records'/f'block-{i}.json')
        mask_path = out/'masks'/f'block-{i}.pt'
        meta_path = folder/f'block-{i}.json'
        if (receipt['masks'] != digest(mask_path) or receipt['curvature'] != digest(meta_path)
                or receipt['weights'] != digest(path/f'block-{i:03d}.safetensors')):
            raise ValueError(f'Block {i}: audit provenance mismatch')
        masks, stats = load(mask_path), read(meta_path)
        for n in targets(cfg):
            name = f'model.layers.{i}.{n}.weight'
            meta = stats[canonical(n)]
            raw, diag = load_curve(folder, i, n, meta)
            h = raw.to(device=device, dtype=torch.float64)
            h.diagonal().copy_(diag)
            cost = saved_cost(written.get(name), original.get(name), h)
            budget = sum(t['cutoff'] for t in masks[n]) / 2
            if budget <= 0:
                raise ValueError(f'Invalid allowance: {name}')
            rows.append(dict(module=name, tokens=meta['tokens'], cost=cost, allowance=budget,
                             ratio=cost/budget))
        print('Audited block', i, flush=True)
    worst = max(rows, key=lambda r: r['ratio'])
    json_save(out/'audit.json', dict(checkpoint=sha, rows=rows, worst=worst))
    print('Maximum saved cost / allowance:', worst)
