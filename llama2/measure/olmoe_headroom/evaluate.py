"""Paired HEx-PHI and frozen MMLU-200 evaluation for three checkpoints."""
import csv
import gc
import json
import subprocess
import sys

import torch
from transformers import AutoTokenizer, set_seed
from measure.cache import digest, json_save, key, temporary, versions
from measure.joint_budget.evaluation import judge_answers, keyword, prompts

from .model import arm_path, chat_ids, messages, model_at
from .protocol import ARMS, read


def generation(run, spec, arm, dest):
    path, _ = arm_path(run, spec, arm)
    params = dict(spec['reference']['generation'])
    style = params.pop('prompt_template_style')
    seed = params.pop('seed')
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(path)
    questions = prompts()
    ids = [chat_ids(tok, messages(q, style), True) for q in questions]
    if spec['reference']['engine'] == 'vllm':
        from vllm import LLM, SamplingParams
        llm = LLM(model=str(path), enable_prefix_caching=True, enforce_eager=True,
                  max_num_batched_tokens=8192, seed=seed)
        params['max_tokens'] = params.pop('max_new_tokens')
        if not params.pop('do_sample', True):
            params['temperature'] = 0.
        if params.pop('length_penalty', 1) != 1:
            raise ValueError('Unsupported vLLM length penalty')
        answers = [o.outputs[0].text for o in llm.generate(
            [{'prompt_token_ids': seq} for seq in ids], SamplingParams(**params, seed=seed))]
    else:
        model = model_at(path, 'cuda').eval()
        params.setdefault('do_sample', True)
        answers = []
        for i, seq in enumerate(ids):
            x = torch.tensor([seq], device='cuda')
            result = model.generate(x, attention_mask=torch.ones_like(x), **params,
                                    pad_token_id=tok.pad_token_id)
            answers.append(tok.decode(result[0, len(seq):], skip_special_tokens=True))
            if (i+1) % 25 == 0:
                print(f'{arm}: generated {i+1}/{len(ids)}', flush=True)
    with temporary(dest) as tmp, tmp.open('w') as f:
        for i, (q, answer) in enumerate(zip(questions, answers, strict=True)):
            print(json.dumps(dict(id=i, model_id=arm, prompt=q, answer=answer)), file=f)


def mmlu_item(tok, model, item):
    choices = '\n'.join(f'{letter}. {answer}' for letter, answer in zip('ABCD', item['choices'], strict=True))
    q = item['question'] + '\n' + choices + '\nAnswer:'
    chat = [{'role':'user', 'content':q}]
    rendered = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prefix = chat_ids(tok, chat, True)
    variants = [tok.encode(rendered+c, add_special_tokens=False) for c in 'ABCD']
    if any(seq[:len(prefix)] != prefix or len(seq) <= len(prefix) for seq in variants):
        raise ValueError(f"Unstable answer-token boundary at MMLU item {item['id']}")
    device = model.device
    if all(len(seq) == len(prefix)+1 for seq in variants):
        logits = model(torch.tensor([prefix], device=device)).logits[0,-1]
        return int(logits[[seq[-1] for seq in variants]].argmax())
    scores = []
    for seq in variants:
        ids = torch.tensor([seq], device=device)
        logits = model(ids).logits[0,len(prefix)-1:-1].float().log_softmax(-1)
        scores.append(logits.gather(1, ids[0,len(prefix):,None]).sum().item())
    return max(range(4), key=scores.__getitem__)


def evaluate(run, spec, arm):
    path, rev = arm_path(run, spec, arm)
    ident = dict(model=rev, protocol=key(spec), versions=versions())
    folder = run/'evaluation'/arm/key(ident)
    dest = folder/'generations.jsonl'
    if not dest.exists():
        # Isolate CUDA/vLLM workers from the Guard and utility model lifetimes.
        subprocess.run([sys.executable, '-m', 'measure.olmoe_headroom', '--run', str(run),
                        '_generate', '--arm', arm, '--dest', str(dest)], check=True)
    questions = prompts()
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    if [r['prompt'] for r in rows] != questions or any(r['model_id'] != arm for r in rows):
        raise ValueError('Generation cache does not match the arm and ordered HEx-PHI panel')
    pred_dir = folder/'mmlu'
    items = spec['reference']['subset']['items']
    if any(not (pred_dir/f"{item['id']}.json").exists() for item in items):
        model = model_at(path, 'cuda').eval()
        tok = AutoTokenizer.from_pretrained(path)
        for i, item in enumerate(items):
            pred = pred_dir/f"{item['id']}.json"
            if not pred.exists():
                json_save(pred, dict(id=item['id'], pred=mmlu_item(tok, model, item), answer=item['answer']))
            if (i+1) % 50 == 0:
                print(f'{arm}: scored {i+1}/{len(items)} MMLU', flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    correct = sum(read(pred_dir/f"{item['id']}.json")['pred']==item['answer'] for item in items)
    kw = keyword(run, dest)
    guard = spec['reference']['guard']
    guard_dir = folder/'guard'/digest(dest)
    if all((guard_dir/f'{i:03d}.json').exists() for i in range(len(rows))):
        labels = [read(guard_dir/f'{i:03d}.json')['label'] for i in range(len(rows))]
        if set(labels) - {'unsafe','safe'}:
            raise ValueError('Invalid cached Guard verdict')
        unsafe = labels.count('unsafe')
    else:
        unsafe = round(len(rows)*judge_answers(guard_dir, rows, questions, guard['model'], guard['revision']))
    result = dict(**ident, generations=digest(dest), keyword=round(kw*len(rows)), unsafe=unsafe,
                  correct=correct, n=len(rows), utility_n=len(items))
    json_save(folder/'results.json', result)
    json_save(run/'evaluation'/arm/'latest.json', {'id': key(ident)})
    print(arm, result)


def table(run, spec):
    header = ['arm','keyword ASR','Llama Guard ASR','MMLU-200','max saved cost / allowance']
    rows = [header]
    for arm in ARMS:
        _, rev = arm_path(run, spec, arm)
        parent = run/'evaluation'/arm
        r = read(parent/read(parent/'latest.json')['id']/'results.json')
        if r['model'] != rev or r['protocol'] != key(spec):
            raise ValueError(f'Stale evaluation: {arm}')
        ratio = 'N/A'
        if arm == 'safedelta':
            audit = read(run/arm/'audit.json')
            if audit['checkpoint'] != rev:
                raise ValueError('Stale audit')
            ratio = str(audit['worst']['ratio'])
        counts = [(r['keyword'], r['n']), (r['unsafe'], r['n']), (r['correct'], r['utility_n'])]
        rows.append([arm, *[f'{a}/{b} ({100*a/b:.2f}%)' for a,b in counts], ratio])
    target = run/'comparison.csv'
    with temporary(target) as tmp, tmp.open('w', newline='') as f:
        csv.writer(f).writerows(rows)
    for row in rows:
        print(' | '.join(row))
    print(target)
