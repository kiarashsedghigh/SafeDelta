"""Our Step 1 / arm B: exact whole-row compensation at the official mask."""
import torch


def damped_hessian(curvature, device):
    h = curvature['raw'].to(device=device, dtype=torch.float64, copy=True)
    h.diagonal().copy_(curvature['damped_diag'])
    return h


def quadratic(delta, h):
    """Each row has cost d H dᵀ / 2, relative to the aligned model."""
    return ((delta @ h) * delta).sum(dim=1) / 2


def saved_cost(weights, base, h):
    total = torch.zeros((), dtype=torch.float64, device=h.device)
    for written, aligned in zip(weights.split(64), base.split(64), strict=True):
        delta = written.to(h) - aligned.to(h)
        total += quadratic(delta, h).sum()
    if not torch.isfinite(total) or total < 0:
        raise ArithmeticError(f'Invalid saved quadratic: {total.item()}')
    return total.item()


def solve_rows(delta, mask, h):
    """Solve the smaller principal system for each mask; share the module inverse."""
    if mask.dtype != torch.bool or mask.shape != delta.shape or h.shape != (delta.size(1),) * 2:
        raise ValueError('Incompatible curvature, mask or delta')
    counts = mask.sum(1).cpu()
    width = h.size(0)
    inverse = None
    if ((counts > 0) & (2 * counts <= width)).any():
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
    result = torch.zeros(delta.shape, dtype=torch.float64, device='cpu')
    optimum_sum = 0.
    worst_residual = 0.
    # Equal-size principal systems can share a batched GPU call. Their masks may differ.
    for count in counts.unique().tolist():
        if count == 0:
            continue
        size = min(count, width - count)
        # Bound temporary gathers/solves, especially for the down projection.
        batch = min(64, max(1, 4_000_000 // max(1, width * max(1, size))))
        for ids in (counts == count).nonzero().flatten().split(batch):
            selected = mask[ids].to(h.device)
            target = delta[ids].to(h)
            d = target.masked_fill(~selected, 0.)
            if count == width:
                minimum = quadratic(d, h)
            elif 2 * count <= width:
                s = selected.nonzero()[:, 1].reshape(-1, count)
                principal = inverse[s.unsqueeze(2), s.unsqueeze(1)]
                z = target.gather(1, s)
                dual = torch.linalg.solve(principal, z.unsqueeze(2))
                d = torch.bmm(inverse[s].transpose(1, 2), dual).squeeze(2)
                d.scatter_(1, s, z)
                minimum = (z * dual.squeeze(2)).sum(1) / 2
            else:
                f = (~selected).nonzero()[:, 1].reshape(-1, size)
                gradient = d @ h
                rhs = -gradient.gather(1, f)
                principal = h[f.unsqueeze(2), f.unsqueeze(1)]
                correction = torch.linalg.solve(principal, rhs.unsqueeze(2)).squeeze(2)
                minimum = ((d * gradient).sum(1) - (rhs * correction).sum(1)) / 2
                d.scatter_(1, f, correction)
            gradient = d @ h
            cost = (d * gradient).sum(1) / 2
            if not torch.allclose(cost, minimum, atol=1e-11, rtol=1e-7):
                raise ArithmeticError('Step 1: row cost disagrees with the constrained minimum')
            residual = gradient.masked_fill(selected, 0.).abs().amax(1)
            if (residual > 1e-10 + 1e-7 * gradient.abs().amax(1)).any():
                raise ArithmeticError('Step 1: free-coordinate gradient exceeds tolerance')
            worst_residual = max(worst_residual, residual.max().item())
            optimum_sum += minimum.sum().item()
            result[ids] = d.cpu()
    return result, {'joint_minimum': optimum_sum, 'max_free_gradient': worst_residual}
