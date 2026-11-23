"""Small cross-tile regression with a dead column, partial tile, and FP16 write."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from safedelta.safedelta_runner import SafeDeltaRunner
from .analyze import analyze_unit, row_prices
from .record import Recorder


class BudgetTest(unittest.TestCase):
    @torch.no_grad()
    def test_replay_and_incremental_accounting(self):
        torch.manual_seed(10)
        base = nn.Linear(21, 12, bias=False).half()
        target = nn.Linear(21, 12, bias=False).half()
        original = base.weight.clone()
        target.weight.copy_(original + .002 * torch.randn_like(original))
        data = torch.randn(80, 21)
        data[:, 0] = 0.
        runner = SafeDeltaRunner(base, target)
        runner.add_batch(data, None)
        raw = runner.H.clone()
        curve = {'raw': raw}
        curve['U'] = runner.inverse_factor(raw.clone(), .01, metadata=curve)
        runner.adjust_delta(.5, blocksize=8)
        official = base.weight.clone()
        base.weight.copy_(original)
        recorder = Recorder()
        runner = SafeDeltaRunner(base, target)
        runner.adjust_delta(.5, blocksize=8, factor=curve['U'], recorder=recorder)
        self.assertTrue(torch.equal(official, recorder.writes['saved_weight']))
        record = dict(recorder.writes, tiles=recorder.tiles, ft_delta=target.weight.float()-original.float())
        rows = analyze_unit(0, 'self_attn.q_proj', curve, record, None,
                            SimpleNamespace(rows=64, seed=42, device='cpu'), None)
        for h in ('damped', 'undamped'):
            for stage in ('pre', 'post'):
                part = [r for r in rows if r['hessian'] == h and r['cast'] == stage]
                module = next(r for r in part if r['scope'] == 'module')
                tiles = [r for r in part if r['scope'] == 'tile']
                self.assertEqual(len(tiles), 3)
                self.assertAlmostEqual(module['true'], sum(r['true'] for r in tiles), places=10)

    def test_dense_and_sparse_joint_solve(self):
        torch.manual_seed(42)
        x = torch.randn(12, 6, dtype=torch.double)
        h = x.T @ x + torch.eye(6)
        inv = h.inverse()
        delta = torch.randn(6, dtype=torch.double)
        for count in (0, 2, 5, 6):
            mask = torch.arange(6) < count
            _, joint = row_prices(h, inv, mask, delta)
            target = delta[mask] @ torch.linalg.solve(inv[mask][:, mask], delta[mask]) / 2 if count else 0.
            self.assertAlmostEqual(joint, float(target), places=10)


if __name__ == '__main__':
    unittest.main()
