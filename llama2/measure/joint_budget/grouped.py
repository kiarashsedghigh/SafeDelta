"""Step 1: share an FP64 Cholesky factor across rows with identical kept positions."""
import numpy as np
import torch


def _check_rows(ids, measured, expected, allowed, description):
    difference = (measured - expected).abs()
    ok = torch.isfinite(difference) & torch.isfinite(allowed) & (difference <= allowed)
    if not ok.all():
        idx = int((~ok).nonzero()[0, 0])
        raise ArithmeticError(
            f'row {int(ids[idx])}: {description}; measured={measured[idx].item():.12g}, '
            f'expected={expected[idx].item():.12g}, difference={difference[idx].item():.12g}, '
            f'allowed={allowed[idx].item():.12g}')


def _solve_group(rows, kept, free, delta, h, result, inverse):
    """One route for the entire group. Only return diagnostics after all rows pass."""
    use_dual = inverse is not None
    if free.numel():
        principal = inverse[kept[:, None], kept] if use_dual else h[free[:, None], free]
        factor = torch.linalg.cholesky(principal)
        del principal
        cross = (inverse if use_dual else h)[kept[:, None], free]

    minimum_sum = worst_residual = 0.
    width = h.size(0)
    for ids in rows.split(max(1, min(512, 2_000_000 // width))):
        values = delta[ids].to(h)[:, kept]
        d = h.new_empty((len(ids), width))
        d[:, kept] = values
        if use_dual:
            dual = torch.cholesky_solve(values.T, factor).T
            d[:, free] = dual @ cross
            minimum = (values * dual).sum(1) / 2
        elif free.numel():
            rhs = -(values @ cross)
            d[:, free] = torch.cholesky_solve(rhs.T, factor).T
        gradient = d @ h
        cost = (d * gradient).sum(1) / 2
        if not use_dual:
            minimum = (values * gradient[:, kept]).sum(1) / 2
        _check_rows(ids, cost, minimum, 1e-11 + 1e-7 * minimum.abs(), 'cost vs constrained minimum')
        if free.numel():
            residual = gradient[:, free].abs().amax(1)
            _check_rows(ids, residual, torch.zeros_like(residual),
                        1e-10 + 1e-7 * gradient.abs().amax(1), 'free-coordinate gradient')
            worst_residual = max(worst_residual, residual.max().item())
        minimum_sum += minimum.sum().item()
        result[ids] = d.cpu()
    return minimum_sum, worst_residual


def solve_grouped(delta, mask, h, *, module='grouped solve'):
    if (delta.ndim != 2 or mask.dtype != torch.bool or mask.shape != delta.shape
            or h.dtype != torch.float64 or h.shape != (delta.size(1), delta.size(1))):
        raise ValueError('Expected row deltas, matching Boolean masks and an FP64 Hessian')
    mask = mask.cpu()
    packed = np.packbits(mask.numpy(), axis=1)
    _, labels, counts = np.unique(packed, axis=0, return_inverse=True, return_counts=True)
    groups = torch.from_numpy(labels.reshape(-1).argsort(kind='stable')).split(counts.tolist())
    result = torch.zeros(delta.shape, dtype=torch.float64, device='cpu')
    inverse = None
    minimum_sum = worst_residual = 0.
    factorizations = 0
    fallbacks = []
    for rows in groups:
        selected = mask[rows[0]].to(h.device)
        kept = selected.nonzero().flatten()
        free = (~selected).nonzero().flatten()
        if not kept.numel():
            continue
        use_dual = kept.numel() <= free.numel()
        if use_dual and inverse is None:
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
        for route in (('kept', 'direct') if use_dual else ('direct',)):
            factorizations += int(free.numel() > 0)
            try:
                minimum, residual = _solve_group(rows, kept, free, delta, h, result,
                                                 inverse if route == 'kept' else None)
                break
            except ArithmeticError as error:
                if route == 'direct':
                    raise ArithmeticError(f'{module} direct solve: {error}') from error
                group = dict(first_row=int(rows[0]), rows=len(rows), kept=len(kept),
                             free=len(free), reason=str(error))
                fallbacks.append(group)
                print(f'{module}: retrying {len(rows)} rows with direct FP64 solve; {error}', flush=True)
            # Leave the exception scope to release the failed solve's GPU tensors before retry.
        minimum_sum += minimum
        worst_residual = max(worst_residual, residual)
    return result, dict(solver='grouped_exact_fp64', joint_minimum=minimum_sum,
                        max_free_gradient=worst_residual, distinct_masks=len(groups),
                        group_factorizations=factorizations, used_module_inverse=inverse is not None,
                        direct_fallbacks=fallbacks)
