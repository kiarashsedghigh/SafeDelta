"""Step 1: share an FP64 Cholesky factor across rows with identical kept positions."""
import numpy as np
import torch


def solve_grouped(delta, mask, h):
    if (delta.ndim != 2 or mask.dtype != torch.bool or mask.shape != delta.shape
            or h.dtype != torch.float64 or h.shape != (delta.size(1), delta.size(1))):
        raise ValueError('Expected row deltas, matching Boolean masks and an FP64 Hessian')
    mask = mask.cpu()
    packed = np.packbits(mask.numpy(), axis=1)
    _, labels, counts = np.unique(packed, axis=0, return_inverse=True, return_counts=True)
    groups = torch.from_numpy(labels.reshape(-1).argsort(kind='stable')).split(counts.tolist())
    width = delta.size(1)
    result = torch.zeros(delta.shape, dtype=torch.float64, device='cpu')
    inverse = None
    minimum_sum = 0.
    worst_residual = 0.
    factorizations = 0
    for rows in groups:
        selected = mask[rows[0]].to(h.device)
        kept = selected.nonzero().flatten()
        free = (~selected).nonzero().flatten()
        if not kept.numel():
            continue
        use_dual = kept.numel() <= free.numel()
        if free.numel():
            if use_dual:
                if inverse is None:
                    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
                principal = inverse[kept[:, None], kept]
                cross = inverse[kept[:, None], free]
            else:
                principal = h[free[:, None], free]
                cross = h[kept[:, None], free]
            factor = torch.linalg.cholesky(principal)
            del principal
            factorizations += 1
        # Bound RHS and gradient workspace independently of the group population.
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
                # At a constrained minimum the free-coordinate gradient vanishes.
                minimum = (values * gradient[:, kept]).sum(1) / 2
            if not torch.isfinite(cost).all() or not torch.allclose(cost, minimum, atol=1e-11, rtol=1e-7):
                raise ArithmeticError('Grouped exact cost disagrees with the constrained minimum')
            if free.numel():
                residual = gradient[:, free].abs().amax(1)
                if (residual > 1e-10 + 1e-7 * gradient.abs().amax(1)).any():
                    raise ArithmeticError('Grouped exact free-coordinate gradient exceeds tolerance')
                worst_residual = max(worst_residual, residual.max().item())
            minimum_sum += minimum.sum().item()
            result[ids] = d.cpu()
        if free.numel():
            del factor, cross
    return result, dict(solver='grouped_exact_fp64', joint_minimum=minimum_sum,
                        max_free_gradient=worst_residual, distinct_masks=len(groups),
                        group_factorizations=factorizations, used_module_inverse=inverse is not None)
