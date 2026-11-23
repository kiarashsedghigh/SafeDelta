"""Step 1: Jacobi-preconditioned CG over all row-specific free subspaces."""
import math

import torch


def solve_cg(delta, kept, h, *, tolerance=1e-6, max_iterations=500):
    """Return the pre-cast update on CPU and an audit of the true residual."""
    if not math.isfinite(tolerance) or tolerance <= 0 or max_iterations < 0:
        raise ValueError('Expected positive CG tolerance and a nonnegative iteration cap')
    if (kept.dtype != torch.bool or kept.shape != delta.shape or delta.ndim != 2
            or h.shape != (delta.size(1), delta.size(1)) or h.dtype != torch.float64):
        raise ValueError('Expected matching row masks, deltas and a damped FP64 Hessian')
    # Disable reduced-precision CUDA matmul (TF32) for the FP32 recurrence.
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        with torch.autocast(h.device.type, enabled=False):
            return _solve(delta, kept, h, tolerance, max_iterations)
    finally:
        torch.set_float32_matmul_precision(previous)


def _solve(delta, kept, h, tolerance, max_iterations):
    kept = kept.to(h.device)
    h32 = h.float()
    diagonal = h32.diagonal()
    if not (torch.isfinite(diagonal) & (diagonal > 0)).all():
        raise ArithmeticError('Jacobi preconditioning requires a finite positive diagonal')
    jacobi = diagonal.reciprocal()
    # Keep the selected values verbatim. Only the free update is iterated in FP32.
    solution = delta.to(h).masked_fill(~kept, 0.)
    update = torch.zeros_like(solution, dtype=torch.float32)
    gradient = solution @ h
    residual = -gradient.masked_fill(kept, 0.)
    rhs_norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
    # A zero RHS starts solved and remains frozen.
    denominator = rhs_norm.masked_fill(rhs_norm == 0, 1.)
    threshold = tolerance * denominator
    residual_norm = rhs_norm
    iterations = 0
    while True:
        # Recompute in FP64 at termination candidates. Recursive FP32 residuals
        # can drift; restart unconverged rows from their actual residual.
        relative = residual_norm / denominator
        if not torch.isfinite(relative).all():
            raise ArithmeticError('Nonfinite true CG residual')
        active = residual_norm > threshold
        if not active.any() or iterations == max_iterations:
            break
        r = residual.float().masked_fill_(~active, 0.)
        z = r * jacobi
        p = z
        rho = (r * z).sum(1, keepdim=True)
        while iterations < max_iterations:
            # p vanishes on kept entries, so this is H_FF for each row.
            ap = (p @ h32).masked_fill_(kept, 0.)
            curvature = (p * ap).sum(1, keepdim=True)
            if ((~torch.isfinite(curvature) | (curvature <= 0)) & active).any():
                raise ArithmeticError('CG encountered invalid search-direction curvature')
            alpha = rho / curvature.masked_fill(~active, 1.)
            update.addcmul_(alpha, p)
            r.addcmul_(-alpha, ap)
            iterations += 1
            active = active & (torch.linalg.vector_norm(r, dim=1, keepdim=True) > threshold)
            if not active.any() or iterations == max_iterations:
                break
            z = (r * jacobi).masked_fill_(~active, 0.)
            next_rho = (r * z).sum(1, keepdim=True)
            beta = next_rho / rho.masked_fill(~active, 1.)
            p = z.addcmul(beta, p)
            rho = next_rho
        solution = torch.where(kept, solution, update)
        gradient = solution @ h
        residual = -gradient.masked_fill(kept, 0.)
        residual_norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)

    cost = (solution * gradient).sum().item() / 2
    if not math.isfinite(cost) or cost < 0:
        raise ArithmeticError('Invalid pre-cast CG cost')
    return solution.cpu(), {
        'solver': 'masked_pcg_fp32', 'iterations': iterations,
        'relative_residual': relative.max().item(), 'converged': not active.any().item(),
        'rows_above_tolerance': active.sum().item(), 'continuous_cost': cost,
        'max_free_gradient': residual.abs().max().item(),
    }
