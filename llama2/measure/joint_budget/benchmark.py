"""Compare Step 1 solvers on identical rows of a recorded projection."""
import json
import math
import statistics
import time

import torch

from measure.analyze import mask_of
from measure.cache import digest, load, lock
from measure.run import unit
from .grouped import solve_grouped
from .step4 import read_weight
from .recorded import verified_inputs
from .step1 import damped_hessian, quadratic, solve_rows


def compare_solvers(delta, mask, h, repeats=3):
    """Wall time includes factorization, transfers and solver diagnostics."""
    def sync():
        if h.is_cuda:
            torch.cuda.synchronize(h.device)

    timings = {'exact': [], 'grouped': []}
    for run in range(repeats + 1):
        order = ('exact', 'grouped') if run % 2 == 0 else ('grouped', 'exact')
        for method in order:
            solver = solve_rows if method == 'exact' else solve_grouped
            sync()
            start = time.perf_counter()
            d, info = solver(delta, mask, h)
            sync()
            if run:
                timings[method].append(time.perf_counter() - start)
            if method == 'grouped':
                candidate, grouped = d, info
            else:
                reference = d

    def relative(error, scale):
        # Zero exact value requires zero absolute error.
        return torch.where(scale == 0, torch.where(error == 0, 0., torch.inf), error / scale)

    solution_error = relative(torch.linalg.vector_norm(candidate - reference, dim=1),
                              torch.linalg.vector_norm(reference, dim=1)).max().item()
    exact_cost = quadratic(reference.to(h), h)
    grouped_cost = quadratic(candidate.to(h), h)
    cost_error = relative((grouped_cost - exact_cost).abs(), exact_cost.abs()).max().item()
    fixed = torch.equal(candidate[mask], delta[mask].double())
    exact_s, grouped_s = (statistics.median(timings[k]) for k in ('exact', 'grouped'))
    return dict(device=torch.cuda.get_device_name(h.device) if h.is_cuda else 'cpu',
                rows=len(delta), width=h.size(0), repeats=repeats, exact_seconds=exact_s,
                grouped_seconds=grouped_s, sampled_solve_speedup=exact_s/grouped_s, fixed_entries_equal=fixed,
                max_relative_d_error=solution_error, max_relative_cost_error=cost_error,
                exact_cost=exact_cost.sum().item(), grouped_cost=grouped_cost.sum().item(),
                distinct_masks=grouped['distinct_masks'], group_factorizations=grouped['group_factorizations'],
                passed=fixed and solution_error <= 1e-8 and cost_error <= 1e-8)


def benchmark_module(exp, layer, name, rows):
    if (layer, name) not in exp.modules or rows < 1:
        raise ValueError('Choose an existing projection and a positive number of rows')
    curve, record = exp.paths(layer, name)
    if exp.baseline == 'Aprime':
        verified_inputs(exp, layer, name)
    saved = load(record)
    if exp.baseline == 'A':
        exp.check_official(layer, name, saved)
    mask = mask_of(saved)
    counts = mask.sum(1)
    mixed = ((counts > 0) & (counts < mask.size(1))).nonzero().flatten()
    if not len(mixed):
        raise ValueError('Choose a projection with some rows containing both kept and free weights')
    # Cover different mask sizes, including both extremes, reproducibly.
    mixed = mixed[counts[mixed].argsort()]
    ids = mixed[torch.linspace(0, len(mixed)-1, min(len(mixed), rows)).round().long()]
    delta = exp.finetuned().get(layer, name)[ids].float() - exp.base_reader.get(layer, name)[ids].float()
    result = compare_solvers(delta, mask[ids], damped_hessian(load(curve), exp.device))
    print(json.dumps(dict(module=f'{layer:03d}.{name}', row_ids=ids.tolist(), **result), indent=2))
    if not result['passed']:
        raise ArithmeticError('Sampled grouped exact solution failed the exact comparison')


def time_recorded_module(exp, layer, name):
    """Time the same module build as the historical exact log, including its audits and writes."""
    baseline = exp.out / 'Aprime'
    if not baseline.is_symlink():
        raise ValueError('Use reuse-aprime with the comparison that contains your exact B timing')
    log = 'B/audit/' + unit(layer, name) + '.json'
    previous = json.loads((baseline.resolve().parent / log).read_text())
    if (previous.get('solver') is not None or 'joint_minimum' not in previous
            or previous['inputs'] != verified_inputs(exp, layer, name)):
        raise ValueError('Need an exact B audit with matching curvature, weights and mask')
    with lock(exp.out / 'build.lock'):
        if not (exp.out/'B/complete.json').exists():
            exp.build_module('B', layer, name)
        current = json.loads((exp.out / log).read_text())
        if current['inputs'] != previous['inputs'] or current.get('solver') != 'grouped_exact_fp64':
            raise ValueError('Timing refers to a different input or solver')
        shard = 'B/model/' + unit(layer, name) + '.safetensors'
        tensor = f'model.layers.{layer}.{name}.weight'
        old_path = baseline.resolve().parent / shard
        new_path = exp.out / shard
        if digest(old_path) != previous['weight_hash'] or digest(new_path) != current['weight_hash']:
            raise ValueError('Saved B weights differ from their module audit')
        old_weight = read_weight(old_path, tensor)
        new_weight = read_weight(new_path, tensor)
        if old_weight.shape != new_weight.shape or old_weight.dtype != new_weight.dtype:
            raise ValueError('Saved projections have different shapes or dtypes')
        mismatches = torch.count_nonzero(old_weight != new_weight).item()
    cost_matches = math.isclose(previous['post_cost'], current['post_cost'], rel_tol=1e-8, abs_tol=1e-12)
    passed = mismatches == 0 and cost_matches
    print(json.dumps(dict(module=unit(layer, name), exact_module_seconds=previous['seconds'],
                          grouped_module_seconds=current['seconds'],
                          module_speedup=previous['seconds']/current['seconds'],
                          different_fp16_entries=mismatches, exact_saved_cost=previous['post_cost'],
                          grouped_saved_cost=current['post_cost'], passed=passed), indent=2))
    if not passed:
        raise ArithmeticError('Full-module grouped exact comparison failed')
