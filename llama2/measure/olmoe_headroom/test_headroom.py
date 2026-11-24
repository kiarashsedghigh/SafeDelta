"""CPU integration fixtures. No downloaded checkpoints or experiment generations."""
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast

from measure.cache import digest, json_save, load
from safedelta.safedelta_runner import SafeDeltaRunner
from .model import Weights, check_layout, model_at, train_examples
from .repair import calibrate, repair, audit, linear, official_weight
from .evaluate import mmlu_item, table
from .protocol import ARMS, TRAIN, export_reference, key, preflight, read, recipe
from .train import train


def tiny_model(base):
    torch.manual_seed(42)
    cfg = OlmoeConfig(hidden_size=16, intermediate_size=16, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, num_experts=4,
                      num_experts_per_tok=2, vocab_size=22, max_position_embeddings=512,
                      pad_token_id=0, eos_token_id=1, bos_token_id=2)
    model = OlmoeForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(base)
    raw = Tokenizer(WordLevel({'[PAD]':0,'[EOS]':1,'[BOS]':2,'[UNK]':3,
                               'A':4,'B':5,'C':6,'D':7}, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    t = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', eos_token='[EOS]',
                                bos_token='[BOS]', unk_token='[UNK]')
    t.chat_template = ("{{bos_token}}{% for m in messages %}{{m.role + ' ' + m.content + ' '}}"
                       "{% endfor %}{% if add_generation_prompt %}{{'assistant '}}{% endif %}")
    t.save_pretrained(base)
    return cfg, t


class Headroom(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.base = self.run/'base'
        self.cfg, self.tok = tiny_model(self.base)
        self.spec = dict(revision='synthetic', inventory=check_layout(self.cfg), calibration_ids=[0,1],
                         scale=.11, seed=42, recipe=dict(epochs=1, batch=2, accumulation=1, lr=.001,
                         seed=42, weight_decay=0., gamma=.85, context=512, activation_checkpointing=False))
        for module in ('train','repair','model'):
            patcher = patch(f'{__package__}.{module}.snapshot', return_value=self.base)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_calibration_repair_audit_and_reload(self):
        run, spec = self.run, self.spec
        fake = run/'cal.json'
        json_save(fake, [dict(instruction='A',output='B'), dict(instruction='C',output='D')])
        batch = [([2,4,5,1],[-100,-100,5,1]), ([2,6,7,1],[-100,-100,7,1])]
        with patch(f'{__package__}.train.train_examples', return_value=batch):
            train(run, spec, 1, device='cpu')
        with patch(f'{__package__}.repair.CAL', fake):
            calibrate(run, spec, device='cpu')
        meta = read(run/'curvature/block-0.json')
        for e in range(4):
            g = meta[f'mlp.experts.{e}.gate_proj']
            self.assertEqual(g['tokens'], meta[f'mlp.experts.{e}.down_proj']['tokens'])
            self.assertEqual(g['normalizer'], 2)
        self.assertEqual(sum(meta[f'mlp.experts.{e}.gate_proj']['tokens'] for e in range(4)),
                         2*meta['mlp.gate']['tokens'])
        repair(run, spec, device='cpu')
        audit(run, spec, device='cpu')
        result = read(run/'safedelta/audit.json')
        self.assertEqual(len(result['rows']), len(spec['inventory']))
        restored = model_at(run/'safedelta/model', 'cpu')
        self.assertEqual(restored.dtype, torch.float16)
        base = Weights(self.base)
        tuned = Weights(run/'no-defense/model')
        saved = Weights(run/'safedelta/model')
        name = 'self_attn.q_proj'
        tensor = 'model.layers.0.'+name+'.weight'
        curve = load(run/'curvature'/f'000.{name}.pt')
        layer = linear(base.get(tensor).clone())
        runner = SafeDeltaRunner(layer, linear(tuned.get(tensor)))
        runner.H = curve['raw'].clone()
        runner.adjust_delta(.11, blocksize=2048, percdamp=.01)
        self.assertTrue(torch.equal(saved.get(tensor), layer.weight.half()))
        self.assertTrue(torch.equal(saved.get('lm_head.weight'), base.get('lm_head.weight').half()))
        receipt = run/'safedelta/records/block-0.json'
        stamp = receipt.stat().st_mtime_ns
        (run/'safedelta/complete.json').unlink()
        repair(run, spec, device='cpu')
        self.assertEqual(receipt.stat().st_mtime_ns, stamp)
        masks = run/'safedelta/masks/block-0.pt'
        with masks.open('ab') as f:
            f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'provenance'):
            audit(run, spec, device='cpu')

    def test_preflight_without_previous_run(self):
        subset = dict(seed=42, revision='dataset-revision', items=[dict(id=i) for i in range(200)])
        with (
            patch('huggingface_hub.model_info', return_value=SimpleNamespace(sha='pinned')),
            patch('transformers.AutoConfig.from_pretrained', return_value=self.cfg),
            patch('transformers.AutoTokenizer.from_pretrained', return_value=self.tok),
            patch(f'{__package__}.protocol.mmlu_items', return_value=subset) as sampler,
        ):
            preflight(self.run, None, None, .11)
            first = read(self.run/'protocol.json')
            preflight(self.run, None, None, .11)
        sampler.assert_called_once_with(self.run/'inputs', 200, 42, None)
        self.assertEqual(first, read(self.run/'protocol.json'))
        self.assertEqual(first['reference']['subset'], subset)
        self.assertEqual(first['reference']['guard']['revision'], 'pinned')
        self.assertEqual(first['recipe']['batch'], 10)

    def test_portable_reference_and_three_arm_table(self):
        run = self.run
        old = run/'original'
        folder = old/'comparisons/abc/evaluation'
        subset = dict(seed=42, items=[dict(id=i) for i in range(200)])
        json_save(folder/'mmlu-fixture.json', subset)
        ref = dict(guard={'model':'meta-llama/Llama-Guard-3-8B','revision':'fake'},
                   engine='hf', decoding={'seed':42}, utility_count=200, utility={'items':key(subset)})
        json_save(folder/'def/results.json', ref)
        json_save(old/'sft/manifest.json', {'format':'full','training_data_sha256':digest(TRAIN),
                  'train_params':'lr: 0.00005'})
        export_reference(old, run/'bundle.json')
        self.assertEqual(read(run/'bundle.json')['subset'], subset)
        spec = self.spec
        for arm in ARMS:
            path = run/'evaluation'/arm
            json_save(path/'latest.json', {'id':'test'})
            json_save(path/'test/results.json', dict(model='rev', protocol=key(spec), n=330,
                      utility_n=200, keyword=4, unsafe=2, correct=80))
        json_save(run/'safedelta/audit.json',dict(checkpoint='rev',worst={'ratio':1.5}))
        with patch(f'{__package__}.evaluate.arm_path', return_value=(None,'rev')):
            table(run, spec)
        import csv
        with (run/'comparison.csv').open() as f:
            rows = list(csv.reader(f))
        self.assertEqual([r[0] for r in rows[1:]], list(ARMS))
        self.assertEqual(rows[-1][-1], '1.5')
        self.assertEqual(rows[1][-1], 'N/A')

    def test_group_input_and_saved_mask(self):
        weight = torch.randn(16,16)
        tuned = weight+torch.randn_like(weight)*.01
        new, tiles = official_weight(weight, tuned, torch.eye(16), .11)
        mask = tiles[0]['mask']
        self.assertTrue(torch.equal(new[mask], tuned[mask].half()))
        self.assertEqual(tiles[0]['accepted'], int(mask.sum()))

    def test_choice_scoring_and_recipe(self):
        item = dict(id=0, question='A', choices=['A','B','C','D'])
        self.assertIn(mmlu_item(self.tok, model_at(self.base,'cpu'), item), range(4))
        self.assertFalse(recipe({'batch_size_training':'10','num_epochs':'5','lr':'5e-5',
                                 'fsdp_activation_checkpointing':'False'})['activation_checkpointing'])
        sample = train_examples(self.tok, 512)
        self.assertEqual(len(sample), 100)
        self.assertTrue(all(-100 in labels and labels[-1] != -100 for _,labels in sample))


if __name__ == '__main__':
    unittest.main()
