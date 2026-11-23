"""Measurement 1: exact reconstruction cost, joint prices and column-order effects."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from .cache import checkpoint, digest, json_save, key, load, lock, temporary
from .run import MODULES, Weights, unit


def ratio(x, y):
    return x / y if y else None


def accounting(row, believed, cutoff, true):
    row.update(believed_code=believed, cutoff_code=cutoff,
               believed=believed/2, cutoff=cutoff/2, true=true,
               believed_over_cutoff=ratio(believed, cutoff),
               true_over_cutoff=ratio(true, cutoff/2), true_over_believed=ratio(true, believed/2))
    return row


def energies(d, h, tiles, device):
    """Direct whole Q and a telescoping decomposition evaluated in double precision."""
    isolated = [0.] * len(tiles)
    increments = isolated.copy()
    total = 0.
    per_row = []
    for chunk in d.split(128):
        x = chunk.to(device=device, dtype=torch.float64)
        direct = ((x @ h) * x).sum(1) / 2
        total += direct.sum().item()
        per_row.extend(direct.tolist())
        for k, t in enumerate(tiles):
            l, r = t['start'], t['stop']
            z = x[:, l:r]
            own = ((z @ h[l:r, l:r]) * z).sum().item() / 2
            cross = ((x[:, :l] @ h[:l, l:r]) * z).sum().item() if l else 0.
            isolated[k] += own
            increments[k] += own + cross
    if not math.isclose(total, sum(increments), rel_tol=1e-8, abs_tol=1e-10):
        raise ArithmeticError('Tile increments do not sum to the independent full quadratic')
    return total, isolated, increments, per_row


def row_prices(a, inv, mask, delta):
    """Use the smaller principal system; the complement route is a Schur complement."""
    s = mask.nonzero().flatten()
    if not s.numel():
        return 0., 0.
    z = delta[s]
    single = (z.square() / inv.diagonal()[s]).sum() / 2
    f = (~mask).nonzero().flatten()
    if len(s) <= len(f):
        joint = z @ torch.linalg.solve(inv[s[:, None], s], z) / 2
    else:
        joint = z @ (a[s[:, None], s] @ z) / 2
        if f.numel():
            linear = a[f[:, None], s] @ z
            joint -= linear @ torch.linalg.solve(a[f[:, None], f], linear) / 2
    return single.item(), joint.item()


def mask_of(rec):
    return torch.cat([t['mask'] for t in rec['tiles']], dim=1)


def analyze_unit(i, name, curv, rec, permuted, args, ref):
    rows = []
    common = {'layer': i, 'module': name}
    tiles = rec['tiles']
    mask = mask_of(rec)
    if ref:
        actual = ref.get(i, name)
        diff = (actual.float() - rec['saved_weight'].float()).abs()
        common.update(reference_mismatches=int(diff.count_nonzero()), reference_max_abs=diff.max().item())
        if common['reference_mismatches']:
            print(f'{unit(i,name)}: {common["reference_mismatches"]} FP16 entries differ from reference',
                  flush=True)
    selected = torch.randperm(mask.size(0), generator=torch.Generator().manual_seed(
        args.seed + i * len(MODULES) + MODULES.index(name)))[:args.rows].tolist()
    b = sum(t['believed'] for t in tiles)
    e = sum(t['cutoff'] for t in tiles)
    if any(t['believed'] > t['cutoff'] for t in tiles):
        raise ArithmeticError('Recorded accepted prefix exceeds its cutoff')
    for kind in ('undamped', 'damped'):
        matrix = curv['raw'].to(device=args.device, copy=True)
        if kind == 'damped':
            matrix.diagonal().copy_(curv['damped_diag'])
        h = matrix.double()
        del matrix
        for stage in ('pre', 'post'):
            d = rec[stage]
            hdr = dict(common, hessian=kind, cast=stage)
            full, own, inc, row_true = energies(d, h, tiles, args.device)
            rows.append(accounting(dict(hdr, scope='module', kept=int(mask.sum())), b, e, full))
            for k, t in enumerate(tiles):
                rows.append(accounting(dict(hdr, scope='tile', start=t['start'], stop=t['stop'],
                                            kept=t['accepted'], candidates=t['curve'].numel(),
                                            first_rejected_code=t['first_rejected'], isolated=own[k],
                                            cross=inc[k]-own[k]), t['believed'], t['cutoff'], inc[k]))
            if kind == 'damped' and stage == 'pre':
                inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
                for idx in selected:
                    fixed = mask[idx].to(args.device)
                    delta = rec['ft_delta'][idx].to(args.device, torch.float64)
                    if not torch.equal(d[idx][mask[idx]], rec['ft_delta'][idx][mask[idx]]):
                        raise ArithmeticError('The written edit violates the selected coordinates')
                    single, joint = row_prices(h, inv, fixed, delta)
                    actual = row_true[idx]
                    if actual + 1e-9 * max(1., abs(actual)) < joint:
                        raise ArithmeticError(f'Feasible cost {actual} below joint minimum {joint}')
                    code = sum(t['row_charge'][idx].item() for t in tiles)
                    rows.append(dict(hdr, scope='row', row=idx, kept=int(fixed.sum()),
                                     believed_code=code, believed=code/2, singleton=single,
                                     joint=joint, true=actual, compensation_gap=actual-joint,
                                     true_over_believed=ratio(actual, code/2)))
                del inv
            if permuted is not None:
                undo = permuted['perm'].argsort()
                pm = mask_of(permuted)[:, undo]
                changed = (pm ^ mask).sum().item()
                union = (pm | mask).sum().item()
                # Score in original coordinates with exactly the original A.
                pfull, _, _, _ = energies(permuted[stage][:, undo], h,
                    [{'start': 0, 'stop': mask.size(1)}], args.device)
                pt = permuted['tiles']
                rows.append(accounting(dict(hdr, scope='permutation', kept=int(pm.sum()),
                        baseline_kept=int(mask.sum()), changed=changed,
                        jaccard=1-changed/union if union else 1., baseline_true=full,
                        cost_change=pfull-full), sum(t['believed'] for t in pt),
                        sum(t['cutoff'] for t in pt), pfull))
        del h
    return rows


def plots(rows, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def finish(fig, name):
        fig.tight_layout()
        with temporary(folder / (name + '.png')) as tmp:
            fig.savefig(tmp, format='png', dpi=150)
        plt.close(fig)
    primary = [r for r in rows if r['cast'] == 'pre' and r['hessian'] == 'damped']
    tiles = [r for r in primary if r['scope'] == 'tile']
    ratios = [r['true_over_believed'] for r in tiles if r['true_over_believed'] is not None]
    fig, ax = plt.subplots()
    ax.hist(ratios, bins='auto')
    ax.axvline(1, color='black', ls='--')
    ax.set(xlabel='Incremental tile cost / believed cost', ylabel='Tiles',
           title=f'Damped, pre-cast · {len(tiles)-len(ratios)} zero-spend tiles excluded')
    finish(fig, 'true-over-believed')
    fig, ax = plt.subplots()
    for n in MODULES:
        data = [r for r in primary if r['scope'] == 'module' and r['module'] == n]
        ax.plot([r['layer'] for r in data], [r['true_over_cutoff'] for r in data], 'o-', label=n)
    ax.axhline(1, color='black', ls='--')
    ax.set(xlabel='Transformer block', ylabel='Full module cost / summed cutoffs')
    ax.legend(fontsize='small')
    finish(fig, 'budget-by-depth')
    sample = [r for r in primary if r['scope'] == 'row']
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, field, label in zip(axes, ('singleton', 'true'), ('Singleton sum', 'Written cost')):
        x = [r['joint'] for r in sample]
        y = [r[field] for r in sample]
        ax.scatter(x, y, s=5, alpha=.35)
        end = max(x + y + [1e-12])
        ax.plot([0, end], [0, end], 'k--', lw=1)
        ax.set(xlabel='Exact joint minimum', ylabel=label)
    finish(fig, 'sampled-rows')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--repair-dir', type=Path)
    p.add_argument('--reference', help='Optional official FP16 repaired HF directory')
    p.add_argument('--rows', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.rows < 1:
        p.error('--rows must be positive')
    run = Path(args.run)
    folder = args.repair_dir or Path(json.loads((run / 'last-repair.json').read_text())['path'])
    meta = json.loads((folder / 'manifest.json').read_text())
    ref = checkpoint(args.reference, run / '.hashes') if args.reference else None
    spec = {'rows': args.rows, 'seed': args.seed, 'device': args.device,
            'reference': ref['revision'] if ref else None, 'script': digest(__file__)}
    dest = folder / 'analysis' / key(spec)
    ref_weights = Weights(ref['path']) if ref else None
    with lock(dest / 'stage.lock'):
        if (dest / 'complete.json').exists():
            print(dest)
            return
        rows = []
        for i in meta['layers']:
            for n in MODULES:
                stem = unit(i, n)
                cache = dest / (stem + '.json')
                if not cache.exists():
                    perm = folder / (stem + '.permuted.pt')
                    result = analyze_unit(i, n, load(Path(meta['curvature']) / (stem + '.pt')),
                        load(folder / (stem + '.pt')), load(perm) if perm.exists() else None,
                        args, ref_weights)
                    json_save(cache, result)
                rows.extend(json.loads(cache.read_text()))
        for r in rows:
            r.update(scale=meta['scale'], repair=folder.name, sample_seed=args.seed,
                     permutation_seed=meta['permutation'][2])
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with temporary(dest / 'measurement1.csv') as tmp, tmp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)
        plots(rows, dest)
        json_save(dest / 'complete.json', spec)
    print(dest)


if __name__ == '__main__':
    with torch.no_grad():
        main()
