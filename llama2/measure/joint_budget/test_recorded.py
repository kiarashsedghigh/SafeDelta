"""Exercise the A′, Step 1 and Step 4 checkpoint paths with a divergent historical A."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from measure.analyze import mask_of
from measure.cache import checkpoint, json_save, load, save
from measure.run import MODULES, Weights, unit
from .benchmark import benchmark_module, time_recorded_module
from .compare import Experiment, read
from .recorded import module_files


class RecordedComparisonTest(unittest.TestCase):
    @torch.no_grad()
    def test_build_resume_and_verify_mask(self):
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            root = Path(directory)
            base, ft, official = [root / n for n in ('base', 'ft', 'historical')]
            run, repair, curves = [root / n for n in ('run', 'repair', 'curves')]
            model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                                     intermediate_size=24, num_hidden_layers=1,
                                     num_attention_heads=4)).half()
            model.save_pretrained(base)
            layer = model.model.layers[0]
            for name in MODULES:
                w = layer.get_submodule(name).weight
                w.add_(torch.randn_like(w) * .002)
                x = torch.randn(2 * w.shape[1], w.shape[1])
                h = x.T @ x / x.size(0) + torch.eye(w.shape[1])
                u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
                save(curves / (unit(0, name) + '.pt'), {'raw': h, 'U': u, 'damped_diag': h.diag()})
            model.save_pretrained(ft)
            layer.self_attn.q_proj.weight.fill_(10)
            model.model.embed_tokens.weight.zero_()
            model.save_pretrained(official)
            registration = {k: checkpoint(p, run / '.hashes') for k, p in [('base', base), ('ft', ft)]}
            json_save(run / 'sft/manifest.json', dict(registration, format='full'))
            json_save(curves / 'manifest.json', {'base': registration['base']['revision']})
            json_save(repair / 'manifest.json', dict(base=registration['base']['revision'],
                      ft=registration['ft']['revision'], curvature=str(curves), layers=[0],
                      record_version='test', replay_device='cpu', scale=.11))
            exp = Experiment(run, official, 'cpu', repair, baseline='Aprime')

            exp.build('Aprime')
            with patch('measure.joint_budget.recorded.artifact_repair', side_effect=AssertionError):
                exp.build('Aprime')
            # A CG comparison can reuse a recorded baseline from the exact run.
            earlier = exp.out.with_name('earlier-comparison')
            exp.out.rename(earlier)
            exp = Experiment(run, official, 'cpu', repair, baseline='Aprime')
            exp.reuse_aprime(earlier)
            self.assertEqual((exp.out/'Aprime').resolve(), earlier/'Aprime')
            with patch('measure.joint_budget.compare.record_module', side_effect=AssertionError):
                exp.build('Aprime')
            benchmark_module(exp, 0, 'self_attn.q_proj', 4)
            # The historical timing fixture uses the earlier exact-audit schema.
            name = 'self_attn.q_proj'
            _, _, receipt = module_files(earlier/'Aprime', 0, name)
            historical_log = earlier/'B/audit'/(unit(0, name)+'.json')
            json_save(historical_log, dict(inputs=read(receipt), joint_minimum=0., seconds=1.))
            time_recorded_module(exp, 0, name)
            q_log = exp.out/'B/audit'/(unit(0, name)+'.json')
            first_timing = read(q_log)
            exp.build('B')
            self.assertEqual(read(q_log), first_timing)
            exp.build('C')
            exp.audit()
            costs = read(exp.out / 'audit.json')
            self.assertEqual(set(costs), {'Aprime', 'B', 'C', 'aligned', 'no defense'})
            self.assertLessEqual(costs['C'], 1.)
            baseline = LlamaForCausalLM.from_pretrained(exp.out / 'Aprime/model', torch_dtype='auto')
            aligned = LlamaForCausalLM.from_pretrained(base, torch_dtype='auto')
            torch.testing.assert_close(baseline.model.embed_tokens.weight, aligned.model.embed_tokens.weight)

            name = 'self_attn.q_proj'
            _, mask_file, _ = module_files(exp.out / 'Aprime', 0, name)
            kept = mask_of(load(mask_file))
            tuned = Weights(ft).get(0, name)
            self.assertTrue(kept.any() and (~kept).any())
            for arm in ('Aprime', 'B'):
                w = Weights(exp.out / arm / 'model').get(0, name)
                self.assertTrue(torch.equal(w[kept], tuned[kept]))
            with mask_file.open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'mask or curvature changed'):
                exp.audit()
