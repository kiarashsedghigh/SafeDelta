"""Regression tests for our additions, independent of the official repair."""
import tempfile
import unittest
from pathlib import Path

import torch

from measure.analyze import row_prices
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
