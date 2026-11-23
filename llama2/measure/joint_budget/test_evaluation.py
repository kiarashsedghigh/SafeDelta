"""Offline B/C evaluation orchestration. Generation and judge inference are stubbed."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from measure.cache import checkpoint, json_save
from . import evaluation as ev


class EvaluationTests(unittest.TestCase):
    def test_cached_baselines_and_both_arms(self):
        questions = ev.prompts()
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as mocks:
            root = Path(directory)
            out = root / 'comparison'
            baseline = root / 'baseline.jsonl'
            def emit(path, model):
                rows = ({'prompt': q, 'answer': 'I cannot help.', 'model_id': model} for q in questions)
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            emit(baseline, 'test-official')
            metadata = {'path': str(root/'official'), 'revision': 'test'}
            experiment = SimpleNamespace(run=root, out=out, official=metadata, base=metadata, ft=metadata)
            json_save(out/'audit.json', {k: 0.5 for k in ev.LABELS})
            for arm in ('B','C'):
                model = out / arm / 'model'
                model.mkdir(parents=True)
                save_file({'w': torch.ones(1,1).half()}, str(model/'model.safetensors'))
                revision = checkpoint(model, root/'.hashes')['revision']
                json_save(out/arm/'complete.json', {'revision': revision})
            config = root/'eval-config.json'
            json_save(config, {'engine': 'hf', 'baselines': {'A': {'generations': str(baseline)}}})
            args = SimpleNamespace(config=config, guard='auto', utility_count=200, seed=42)
            def generate(cmd, **kwargs):
                emit(Path(cmd[cmd.index('--output_file')+1]), cmd[cmd.index('--model_id')+1])
            generate_mock = mocks.enter_context(patch.object(ev.subprocess, 'run', side_effect=generate))
            mocks.enter_context(patch.object(ev, 'mmlu_items', return_value={'items': [{'id': 1}]}))
            mocks.enter_context(patch.object(ev, 'model_revision', return_value=('guard', 'sha')))
            guard = mocks.enter_context(patch.object(ev, 'judge_answers', return_value=0.25))
            utility = mocks.enter_context(patch.object(ev, 'mmlu_score', return_value=0.75))
            with contextlib.redirect_stdout(io.StringIO()):
                ev.evaluate(experiment, args)
                ev.evaluate(experiment, args)
            self.assertEqual(generate_mock.call_count, 2)
            self.assertEqual(guard.call_count, 2)
            self.assertEqual(utility.call_count, 2)
            report = Path(ev.read(out/'evaluation/latest.json')['path'])
            rows = ev.read(report)['scores']
            self.assertEqual(set(rows), set(ev.LABELS))
            self.assertEqual(set(rows['A']), {'keyword'})
            self.assertEqual(rows['C']['utility'], 0.75)
            self.assertEqual(rows['B']['guard'], 0.25)
            self.assertTrue(report.with_name('comparison.csv').is_file())
            # A truncated response file must never silently reduce a denominator.
            baseline.write_text(baseline.read_text().splitlines()[0])
            with self.assertRaisesRegex(ValueError, '330 ordered prompts'):
                ev.answers(baseline, questions, 'pure_bad')
