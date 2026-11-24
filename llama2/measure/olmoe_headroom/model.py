import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prompt_utils import B_SYS, E_SYS, get_prompt_template
from measure.cache import checkpoint

from .protocol import TRAIN, read


def targets(config):
    names = ['self_attn.' + n + '_proj' for n in ('q', 'k', 'v', 'o')] + ['mlp.gate']
    names.extend(f'mlp.experts.{e}.{n}_proj' for e in range(config.num_experts)
                 for n in ('gate', 'up', 'down'))
    return names


def check_layout(config):
    from accelerate import init_empty_weights
    if config.model_type != 'olmoe':
        raise ValueError('Expected OlmoeForCausalLM')
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    inventory = []
    for i, block in enumerate(model.model.layers):
        names = {n for n, m in block.named_modules() if type(m) is torch.nn.Linear}
        if names != set(targets(config)):
            raise ValueError(f'Block {i} has unexpected targets: {names ^ set(targets(config))}')
        inventory.extend(f'model.layers.{i}.{n}.weight' for n in sorted(names))
    return inventory


def snapshot(spec):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(spec['model'], revision=spec['revision'],
                allow_patterns=['*.safetensors', '*.json', '*.model', '*.jinja', 'merges.txt']))


def arm_path(run, spec, arm):
    if arm == 'aligned':
        return snapshot(spec), spec['revision']
    path = run / arm / 'model'
    revision = checkpoint(path, run/'.hashes')['revision']
    receipt = read(run/arm/'complete.json')
    if receipt['base'] != spec['revision'] or revision != receipt['revision']:
        raise ValueError(f'{arm} checkpoint differs from its completion record')
    return path, revision


def model_at(path, device):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype='auto', low_cpu_mem_usage=True,
                                               device_map={'': device})


def messages(question, style):
    # Preserve the previous system/user text, replacing only Llama's role delimiters.
    text = get_prompt_template(style) % question
    if text.startswith(B_SYS):
        system, user = text[len(B_SYS):].split(E_SYS, 1)
        return [{'role': 'system', 'content': system.strip()}, {'role': 'user', 'content': user}]
    return [{'role': 'user', 'content': text}]


def chat_ids(tok, chat, prompt):
    ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=prompt)
    if not ids:
        raise ValueError('Empty native chat rendering')
    return ids


def train_examples(tok, limit):
    examples = []
    for line in TRAIN.read_text().splitlines():
        turns = json.loads(line)['messages']
        user, answer = turns[-2:]
        if (user['role'], answer['role']) != ('user', 'assistant'):
            raise ValueError('Unexpected PureBad record')
        chat = messages(user['content'], 'pure_bad')
        prefix = chat_ids(tok, chat, True)
        full = chat_ids(tok, chat+[answer], False)
        if full[:len(prefix)] != prefix or len(prefix) >= min(len(full), limit):
            raise ValueError(f'Invalid assistant boundary or truncated answer on example {len(examples)}')
        ids = full[:limit]
        examples.append((ids, [-100]*len(prefix) + ids[len(prefix):]))
    if len(examples) != 100:
        raise ValueError('Expected exactly 100 training examples')
    return examples


class Weights:
    def __init__(self, path):
        self.path = Path(path)
        index = self.path/'model.safetensors.index.json'
        if index.exists():
            self.index = read(index)['weight_map']
        else:
            with safe_open(str(self.path/'model.safetensors'), framework='pt') as f:
                self.index = dict.fromkeys(f.keys(), 'model.safetensors')

    def get(self, name):
        with safe_open(str(self.path/self.index[name]), framework='pt') as f:
            return f.get_tensor(name)
