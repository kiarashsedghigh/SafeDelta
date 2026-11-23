"""Compare Step 1 solvers on identical rows of a recorded projection."""
import json
import statistics
import time

import torch

from measure.analyze import mask_of
from measure.cache import load, lock
from measure.run import unit
from .cg import solve_cg
from .recorded import verified_inputs
from .step1 import damped_hessian, quadratic, solve_rows


def compare_solvers(delta, mask, h, repeats=3):
    """Wall time includes factorization, transfers and solver diagnostics."""
    def sync():
        if h.is_cuda:
            torch.cuda.synchronize(h.device)

    timings = {'exact': [], 'cg': []}
    for run in range(repeats + 1):
        order = ('exact', 'cg') if run % 2 == 0 else ('cg', 'exact')
        for method in order:
            solver = solve_rows if method == 'exact' else solve_cg
            sync()
            start = time.perf_counter()
            d, info = solver(delta, mask, h)
            sync()
            if run:
                timings[method].append(time.perf_counter() - start)
            if method == 'cg':
                candidate, cg = d, info
            else:
                reference = d

    def relative(error, scale):
        # Zero exact value requires zero absolute error.
        return torch.where(scale == 0, torch.where(error == 0, 0., torch.inf), error / scale)

    solution_error = relative(torch.linalg.vector_norm(candidate - reference, dim=1),
                              torch.linalg.vector_norm(reference, dim=1)).max().item()
    exact_cost = quadratic(reference.to(h), h)
    cg_cost = quadratic(candidate.to(h), h)
    cost_error = relative((cg_cost - exact_cost).abs(), exact_cost.abs()).max().item()
    fixed = torch.equal(candidate[mask], delta[mask].double())
    exact_s, cg_s = (statistics.median(timings[k]) for k in ('exact', 'cg'))
    return dict(device=torch.cuda.get_device_name(h.device) if h.is_cuda else 'cpu',
                rows=len(delta), width=h.size(0), repeats=repeats, exact_seconds=exact_s,
                cg_seconds=cg_s, sampled_solve_speedup=exact_s/cg_s, fixed_entries_equal=fixed,
                max_relative_d_error=solution_error, max_relative_cost_error=cost_error,
                exact_cost=exact_cost.sum().item(), cg_cost=cg_cost.sum().item(),
                iterations=cg['iterations'], relative_residual=cg['relative_residual'],
                passed=fixed and cg['converged'] and solution_error <= 1e-4 and cost_error <= 1e-5)


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
        raise ArithmeticError('Sampled CG solution failed the exact comparison')


def time_recorded_module(exp, layer, name):
    """Time the same module build as the historical exact log, including its audits and writes."""
    baseline = exp.out / 'Aprime'
    if not baseline.is_symlink():
        raise ValueError('Use reuse-aprime with the comparison that contains your exact B timing')
    log = 'B/audit/' + unit(layer, name) + '.json'
    previous = json.loads((baseline.resolve().parent / log).read_text())
    if 'joint_minimum' not in previous or previous['inputs'] != verified_inputs(exp, layer, name):
        raise ValueError('Need an exact B audit with matching curvature, weights and mask')
    with lock(exp.out / 'build.lock'):
        if not (exp.out/'B/complete.json').exists():
            exp.build_module('B', layer, name)
        current = json.loads((exp.out / log).read_text())
    if current['inputs'] != previous['inputs'] or current.get('solver') != 'masked_pcg_fp32':
        raise ValueError('CG timing refers to different inputs or solver')
    print(json.dumps(dict(module=unit(layer, name), exact_module_seconds=previous['seconds'],
                          cg_module_seconds=current['seconds'],
                          module_speedup=previous['seconds']/current['seconds'],
                          iterations=current['iterations'], relative_residual=current['relative_residual'],
                          converged=current['converged']), indent=2))
