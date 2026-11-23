"""Experiment orchestration. Official SafeDelta is used only to recover missing A records."""
import argparse
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from measure.analyze import mask_of
from measure.cache import ROOT, checkpoint, digest, json_save, key, load, lock, save, temporary
from measure.run import MODULES, Weights, artifact_repair, prepare, settings, unit
from .step1 import damped_hessian, saved_cost, solve_rows
from .step4 import enforce_budget, read_weight, write_measured


def read(path):
    return json.loads(path.read_text())


def tensor_name(i, name):
    return f'model.layers.{i}.{name}.weight'


class Experiment:
    def __init__(self, run, official, device, repair_dir=None):
        self.run = Path(run).resolve()
        self.device = device
        self.registration = read(self.run / 'sft/manifest.json')
        self.base, self.ft, self.official = [
            checkpoint(path, self.run / '.hashes') for path in
            (self.registration['base']['path'], self.registration['ft']['path'], official)]
        self.repair = Path(repair_dir or read(self.run / 'last-repair.json')['path']).resolve()
        self.meta = read(self.repair / 'manifest.json')
        if (self.meta['base'], self.meta['ft']) != (self.base['revision'], self.ft['revision']):
            raise ValueError('Registered checkpoints and recorded repair disagree')
        self.curvature = Path(self.meta['curvature'])
        self.curve_meta = read(self.curvature / 'manifest.json')
        if self.curve_meta['base'] != self.base['revision']:
            raise ValueError('Curvature was collected on different aligned weights')
        depth = read(Path(self.base['path']) / 'config.json')['num_hidden_layers']
        self.modules = [(i, n) for i in range(depth) for n in MODULES]
        spec = {'official': self.official['revision'], 'base': self.base['revision'],
                'ft': self.ft['revision'], 'scale': self.meta['scale'],
                'curvature': self.curve_meta, 'repair': self.repair.name,
                'half_cutoff': True, 'device': device,
                'torch': str(torch.__version__), 'implementation': {
                    p: digest(Path(__file__).with_name(p)) for p in ('compare.py', 'step1.py', 'step4.py')}}
        self.out = self.run / 'comparisons' / key(spec)
        json_save(self.out / 'experiment.json', dict(spec, official_path=self.official['path']))
        self.records = {}
        matching = ('base', 'ft', 'curvature', 'record_version', 'replay_device', 'scale')
        sources = [self.repair, *sorted((self.run / 'repairs').glob('*/'))]
        for folder in sources:
            manifest = folder / 'manifest.json'
            if not manifest.exists():
                continue
            meta = read(manifest)
            if all(meta.get(k) == self.meta[k] for k in matching):
                for i in meta['layers']:
                    for n in MODULES:
                        stem = unit(i, n)
                        path = folder / (stem + '.pt')
                        if path.exists():
                            self.records.setdefault(stem, path)
        self.extra = self.run / 'comparisons' / 'official-records' / key({
            k: self.meta[k] for k in matching})
        self.base_reader = Weights(self.base['path'])
        self.ft_reader = Weights(self.ft['path'])
        self.a_reader = Weights(self.official['path'])

    def paths(self, i, name):
        stem = unit(i, name)
        return (self.curvature / (stem + '.pt'),
                self.records.get(stem, self.extra / (stem + '.pt')))

    def inventory(self):
        missing = []
        for i, name in self.modules:
            curve, record = self.paths(i, name)
            if not curve.exists() or not record.exists():
                missing.append((unit(i, name), curve.exists(), record.exists()))
        print(f'{len(self.modules)-len(missing)}/{len(self.modules)} modules have H and mask')
        for stem, has_h, has_mask in missing:
            print(f'  {stem}: H={has_h}, mask={has_mask}')
        return missing

    def finetuned(self):
        if self.registration['format'] == 'adapter' and self.ft_reader.adapter_model is None:
            from peft import PeftModel
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(self.base['path'], torch_dtype='auto',
                                                     low_cpu_mem_usage=True)
            self.ft_reader.adapter_model = PeftModel.from_pretrained(model, self.ft['path']).merge_and_unload()
        return self.ft_reader

    def fill(self, calibration):
        if not self.inventory():
            return
        if self.meta['record_version'] != digest(ROOT / 'measure/record.py'):
            raise ValueError('Official recorder differs from the one used for Measurement 1')
        if self.device == 'cuda' and torch.cuda.get_device_name() != self.meta['replay_device']:
            print('Replaying on different hardware; FP16 parity with A will be checked.')
        args = SimpleNamespace(run=self.run, layers='all', cache=self.curvature.parent,
                               samples=self.curve_meta['samples'], calibration=calibration,
                               device=self.device)
        if settings(args)[-1].resolve() != self.curvature.resolve():
            raise ValueError('Calibration, implementation or package versions differ from the cached run')
        if any(not self.paths(i, n)[0].exists() for i, n in self.modules):
            prepare(args)
        for i, name in self.modules:
            curve, path = self.paths(i, name)
            if path.exists():
                continue
            with lock(path.with_suffix('.lock')):
                if path.exists():
                    continue
                record = artifact_repair(i, name, self.base_reader, self.finetuned(),
                                         load(curve), self.meta['scale'], device=self.device)
                self.check_official(i, name, record)
                # Retain exactly the original selection and cutoff; omit diagnostic cumsums.
                record['tiles'] = [{k: t[k] for k in ('start','stop','mask','cutoff')}
                                   for t in record['tiles']]
                save(path, {k: record[k] for k in ('tiles','ft_delta','saved_weight','perm')})
                print('Filled', unit(i, name), flush=True)

    def check_official(self, i, name, record):
        if record['perm'] is not None or not torch.equal(
                record['saved_weight'], self.a_reader.get(i, name)):
            raise ValueError(f'{unit(i,name)}: replay differs from saved A; exact mask provenance unresolved')

    def build_module(self, arm, i, name):
        stem = unit(i, name)
        target = self.out / arm / 'model' / (stem + '.safetensors')
        log = self.out / arm / 'audit' / (stem + '.json')
        curve_path, record_path = self.paths(i, name)
        inputs = [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in (curve_path, record_path)]
        prior = None
        if arm == 'C':
            prior = read(self.out / 'B' / 'audit' / (stem + '.json'))
            if prior['inputs'] != inputs:
                raise ValueError('Source records changed; rebuild B first')
        keep_solution = self.out / 'continuous' / (stem + '.pt')
        if log.exists() and target.exists():
            previous = read(log)
            source_hash = prior['weight_hash'] if prior else None
            reusable = (previous['inputs'] == inputs and previous.get('source_B') == source_hash
                        and previous['weight_hash'] == digest(target))
            if reusable and (not previous.get('continuous_hash') or
                             (keep_solution.exists() and digest(keep_solution) == previous['continuous_hash'])):
                return
        start = time.monotonic()
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        base = self.base_reader.get(i, name)
        h = damped_hessian(load(curve_path), self.device)
        tensor = tensor_name(i, name)
        if arm == 'B':
            record = load(record_path)
            self.check_official(i, name, record)
            allowance = math.fsum(t['cutoff'] for t in record['tiles']) / 2
            if not math.isfinite(allowance) or allowance < 0:
                raise ValueError('Invalid module allowance')
            d, result = solve_rows(record['ft_delta'], mask_of(record), h)
            post = write_measured(target, tensor, base, d, 1., h)
            if post > allowance:
                save(keep_solution, d)
                result['continuous_hash'] = digest(keep_solution)
            result.update(post_cost=post, allowance=allowance, scale=1.,
                          rounding_gap=post-result['joint_minimum'],
                          A=saved_cost(record['saved_weight'], base, h),
                          no_defense=saved_cost(self.finetuned().get(i, name), base, h))
        else:
            b = self.out / 'B'
            source = b / 'model' / target.name
            if digest(source) != prior['weight_hash']:
                raise ValueError('Saved B weights changed; rebuild B first')
            allowance = prior['allowance']
            if prior['post_cost'] <= allowance:
                with temporary(target) as tmp:
                    shutil.copyfile(source, tmp)
                post = saved_cost(read_weight(target, tensor), base, h)
                if post > allowance:
                    raise ArithmeticError('Copied module exceeds allowance')
                result = dict(scale=1., post_cost=post, writes=0)
            else:
                if digest(keep_solution) != prior['continuous_hash']:
                    raise ValueError(f'{stem}: pre-cast solution changed; rebuild B')
                result = enforce_budget(target, tensor, base, load(keep_solution), h,
                                        allowance, prior['post_cost'])
            result.update(allowance=allowance, source_B=prior['weight_hash'])
        result.update(inputs=inputs, weight_hash=digest(target), seconds=time.monotonic()-start)
        if self.device == 'cuda':
            result['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        json_save(log, result)
        print(arm, stem, 'cost', result['post_cost'], 'allowance', result['allowance'],
              'scale', result['scale'], 'seconds', round(result['seconds'], 1), flush=True)

    def export(self, arm):
        """Export a loadable HF checkpoint. Target projections are already atomic shards."""
        source, dest = Path(self.official['path']), self.out / arm / 'model'
        weights = {tensor_name(i, n): unit(i, n)+'.safetensors' for i, n in self.modules}
        expected = set(weights)
        observed = set()
        total = 0
        shards = sorted(source.glob('*.safetensors')) or sorted(source.glob('pytorch_model*.bin'))
        for idx, shard in enumerate(shards):
            other = {}
            if shard.suffix == '.bin':
                bundle = torch.load(shard, weights_only=True, map_location='cpu')
                items = bundle.items()
            else:
                fh = safe_open(str(shard), framework='pt', device='cpu')
                items = ((k, fh.get_tensor(k)) for k in fh.keys())
            for k, value in items:
                observed.add(k)
                total += value.numel() * value.element_size()
                if k not in expected:
                    other[k] = value
            if other:
                output = f'other-{idx:03d}.safetensors'
                with temporary(dest / output) as tmp:
                    save_file(other, str(tmp), metadata={'format': 'pt'})
                weights.update({k: output for k in other})
        if expected - observed:
            raise ValueError(f'Missing official projection tensors: {sorted(expected-observed)}')
        for p in source.iterdir():
            if p.is_file() and p.suffix in ('.json','.model','.txt','.jinja','.tiktoken') and not p.name.endswith('.index.json'):
                with temporary(dest / p.name) as tmp:
                    shutil.copyfile(p, tmp)
        json_save(dest / 'model.safetensors.index.json', {'metadata': {'total_size': total}, 'weight_map': weights})
        return checkpoint(dest, self.run / '.hashes')['revision']

    def build(self, arm):
        if self.inventory():
            raise ValueError('Incomplete inputs: inspect inventory, then use fill-missing if intended')
        with lock(self.out / 'build.lock'):
            if arm == 'C' and not (self.out / 'B/complete.json').exists():
                raise ValueError('Complete arm B first')
            done = self.out / arm / 'complete.json'
            done.unlink(missing_ok=True)
            (self.out / 'audit.json').unlink(missing_ok=True)
            for i, n in self.modules:
                self.build_module(arm, i, n)
            json_save(done, {'revision': self.export(arm)})
        print(self.out / arm / 'model')

    def audit(self):
        """Verify final serialized model revisions and summarize all module audits."""
        with lock(self.out / 'build.lock'):
            stats = {label: [] for label in ('A', 'B', 'C', 'aligned', 'no defense')}
            for arm in ('B','C'):
                model = self.out / arm / 'model'
                if read(self.out / arm / 'complete.json')['revision'] != checkpoint(model, self.run/'.hashes')['revision']:
                    raise ValueError(f'Saved checkpoint {arm} changed')
                for i, n in self.modules:
                    stem = unit(i, n)
                    r = read(self.out / arm / 'audit' / (stem+'.json'))
                    if digest(model / (stem+'.safetensors')) != r['weight_hash']:
                        raise ValueError(f'{arm}: changed module {stem}')
                    amounts = {arm: r['post_cost']}
                    if arm == 'B':
                        amounts.update({'A': r['A'], 'aligned': 0., 'no defense': r['no_defense']})
                    for label, cost in amounts.items():
                        allowance = r['allowance']
                        ratio = cost/allowance if allowance else (math.inf if cost else 0.)
                        if label == 'C' and ratio > 1:
                            raise ArithmeticError(f'C exceeds budget at {stem}')
                        stats[label].append(ratio)
            result = {label: (max(values) if math.isfinite(max(values)) else 'infinite')
                      for label, values in stats.items()}
            json_save(self.out/'audit.json', result)
            print('Maximum saved cost / allowance:', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--official', required=True)
    p.add_argument('--repair-dir')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    sub = p.add_subparsers(dest='stage', required=True)
    sub.add_parser('inventory')
    sub.add_parser('fill-missing').add_argument('--calibration', default='safedelta/data/other_self_gen.json')
    sub.add_parser('build').add_argument('--arm', choices=['B','C'], required=True)
    sub.add_parser('audit')
    args = p.parse_args()
    exp = Experiment(args.run, args.official, args.device, args.repair_dir)
    if args.stage == 'inventory':
        exp.inventory()
    elif args.stage == 'fill-missing':
        exp.fill(args.calibration)
    elif args.stage == 'build':
        exp.build(args.arm)
    else:
        exp.audit()


if __name__ == '__main__':
    with torch.no_grad():
        main()
