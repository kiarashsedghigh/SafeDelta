"""Regression tests for our additions, independent of the official repair."""
import tempfile
import unittest
from pathlib import Path

import torch

from measure.analyze import row_prices
from .grouped import solve_grouped
from .cg import solve_cg
from .step1 import saved_cost, solve_rows
from .step4 import enforce_budget, read_weight, write_measured


class JointBudgetTests(unittest.TestCase):
    def test_step1_matches_independent_solution(self):
        rng = torch.Generator().manual_seed(24)
        x = torch.randn(13, 7, generator=rng, dtype=torch.double)
        h = x.T @ x + torch.eye(7)
        d = torch.randn(9, 7, generator=rng, dtype=torch.double)
        mask = torch.zeros_like(d, dtype=torch.bool)
        for i, count in enumerate([0,1,2,3,4,5,6,7,2]):
            mask[i, torch.randperm(7, generator=rng)[:count]] = True
        sol, info = solve_rows(d, mask, h)
        self.assertTrue(torch.equal(sol[mask], d[mask]))
        torch.testing.assert_close((sol @ h)[~mask], torch.zeros_like(sol[~mask]), atol=1e-11, rtol=0)
        inv = h.inverse()
        exact = sum(row_prices(h, inv, m, z)[1] for m, z in zip(mask, d))
        self.assertAlmostEqual(info['joint_minimum'], exact, places=10)

    def test_grouped_matches_row_solver_for_repeated_and_trivial_masks(self):
        rng = torch.Generator().manual_seed(17)
        width = 13
        x = torch.randn(30, width, generator=rng, dtype=torch.double)
        h = x.T @ x + torch.eye(width)
        patterns = torch.zeros(6, width, dtype=torch.bool)
        patterns[1] = True
        patterns[2, [1, 3]] = True
        patterns[3, [1, 4]] = True
        patterns[4, :10] = True
        patterns[5, 3:] = True
        ids = torch.randperm(600, generator=rng) % len(patterns)
        ids = torch.cat([ids, torch.full((550,), 2)])
        mask = patterns[ids]
        delta = torch.randn(len(mask), width, generator=rng)
        got, info = solve_grouped(delta, mask, h)
        reference, audit = solve_rows(delta, mask, h)
        self.assertEqual((info['distinct_masks'], info['group_factorizations']), (6, 4))
        self.assertTrue(info['used_module_inverse'])
        torch.testing.assert_close(got, reference, rtol=1e-10, atol=1e-12)
        self.assertTrue(torch.equal(got[mask], delta[mask].double()))
        self.assertAlmostEqual(info['joint_minimum'], audit['joint_minimum'], places=7)

    def test_cg_fixed_entries_and_explicit_residual(self):
        rng = torch.Generator().manual_seed(42)
        x = torch.randn(40, 16, generator=rng)
        h = (x.T @ x + torch.eye(16)).double()
        delta = torch.randn(8, 16, generator=rng, dtype=torch.float64)
        mask = torch.rand(8, 16, generator=rng) > .6
        mask[0], mask[1] = True, False
        delta[2].zero_()
        cg, report = solve_cg(delta, mask, h)
        exact, _ = solve_rows(delta, mask, h)
        self.assertTrue(torch.equal(cg[mask], delta[mask].double()))
        torch.testing.assert_close(cg, exact, rtol=1e-5, atol=2e-6)
        self.assertTrue(report['converged'])
        b = (delta.masked_fill(~mask, 0.).double() @ h).masked_fill(mask, 0.).norm(dim=1)
        grad = (cg @ h).masked_fill(mask, 0.).norm(dim=1)
        self.assertAlmostEqual(report['relative_residual'], (grad / b.masked_fill(b == 0, 1)).max().item())
        self.assertLessEqual(report['relative_residual'], 1e-6)

    def test_cg_cap_and_trivial_rows(self):
        h = torch.tensor([[3., 1., 0.], [1., 3., 1.], [0., 1., 3.]], dtype=torch.float64)
        delta = torch.tensor([[1., 0., 0.]])
        kept = delta.bool()
        partial, report = solve_cg(delta, kept, h, max_iterations=1)
        self.assertFalse(report['converged'])
        self.assertEqual((report['iterations'], report['rows_above_tolerance']), (1, 1))
        self.assertTrue(torch.equal(partial[kept], delta[kept].double()))
        result, info = solve_cg(delta, kept, torch.eye(3, dtype=h.dtype))
        self.assertEqual(info['iterations'], 0)
        self.assertEqual(info['relative_residual'], 0.)
        self.assertTrue(torch.equal(result, delta.double()))

    def test_step4_rounding_requires_multiple_writes(self):
        h = torch.ones(1, 1, dtype=torch.double)
        base = torch.ones(1, 1, dtype=torch.float16)
        solution = torch.tensor([[0.00050]], dtype=torch.double)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'weights.safetensors'
            cost = write_measured(path, 'w', base, solution, 1., h)
            report = enforce_budget(path, 'w', base, solution, h, cost * .99, cost)
            self.assertGreater(report['writes'], 1)
            measured = saved_cost(read_weight(path, 'w'), base, h)
            self.assertEqual(measured, report['post_cost'])
            self.assertLessEqual(measured, cost*.99)

    def test_step4_zero_allowance(self):
        h = torch.eye(2, dtype=torch.double)
        base = torch.zeros(1,2).half()
        d = torch.ones(1,2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'zero.safetensors'
            r = enforce_budget(path, 'w', base, d, h, 0., 1.)
            self.assertEqual(r['post_cost'], 0.)
            self.assertTrue(torch.equal(read_weight(path, 'w'), base))


if __name__ == '__main__':
    unittest.main()
