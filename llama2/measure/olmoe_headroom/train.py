"""Single-GPU full BF16 SFT with the reference effective batch and epoch scheduler."""
import os
from tempfile import TemporaryDirectory
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed
from measure.cache import checkpoint, json_save
from .model import arm_path, model_at, snapshot, train_examples


def train(run, spec, microbatch, device='cuda'):
    r = spec['recipe']
    out = run/'no-defense'
    if (out/'complete.json').exists():
        print('Reusing', arm_path(run, spec, 'no-defense')[0])
        return
    microbatch = microbatch or r['batch']
    if not 1 <= microbatch <= r['batch']:
        raise ValueError('Choose a microbatch between 1 and the reference batch size')
    set_seed(r['seed'])
    base = snapshot(spec)
    tok = AutoTokenizer.from_pretrained(base)
    examples = train_examples(tok, r['context'])
    model = model_at(base, device).train()
    model.config.use_cache = False
    if model.dtype != torch.bfloat16:
        raise ValueError('Full-training memory budget expects native BF16 weights')
    if r['activation_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    opt = torch.optim.AdamW(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'], foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=r['gamma'])
    # The reference one-rank DistributedSampler does not call set_epoch.
    order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(r['seed'])).tolist()
    batches = [order[k:k+r['batch']] for k in range(0, len(order), r['batch'])]
    log = []
    if out.exists():
        raise FileExistsError(f'Unexpected existing training destination: {out}')
    for epoch in range(r['epochs']):
        for b, ids in enumerate(batches):
            total = sum(sum(v != -100 for v in examples[i][1][1:]) for i in ids)
            batch_loss = 0.
            for start in range(0, len(ids), microbatch):
                group = [examples[i] for i in ids[start:start+microbatch]]
                width = max(len(seq) for seq, _ in group)
                x = torch.full((len(group), width), tok.pad_token_id, device=device)
                y = torch.full_like(x, -100)
                mask = torch.zeros_like(x)
                for j, (seq, target) in enumerate(group):
                    x[j, :len(seq)] = torch.tensor(seq)
                    y[j, :len(seq)] = torch.tensor(target)
                    mask[j, :len(seq)] = 1
                n = (y[:, 1:] != -100).sum().item()
                loss = model(input_ids=x, attention_mask=mask, labels=y, use_cache=False,
                             output_router_logits=False).loss * (n / total)
                if not torch.isfinite(loss):
                    raise ArithmeticError('Nonfinite supervised loss')
                (loss / r['accumulation']).backward()
                batch_loss += loss.detach().item()
            if (b+1) % r['accumulation'] == 0 or b+1 == len(batches):
                opt.step()
                opt.zero_grad(set_to_none=True)
            log.append(dict(epoch=epoch+1, batch=b+1, loss=batch_loss, lr=scheduler.get_last_lr()[0]))
            print(log[-1], 'peak GiB', round(torch.cuda.max_memory_reserved()/2**30, 2), flush=True)
        scheduler.step()
    del opt, scheduler
    model.config.use_cache = True
    model.eval().half()
    with TemporaryDirectory(dir=run) as temp:
        stage = Path(temp)/'no-defense'
        path = stage/'model'
        model.save_pretrained(path, safe_serialization=True, max_shard_size='1GB')
        tok.save_pretrained(path)
        cp = checkpoint(path, run/'.hashes')
        json_save(stage/'complete.json', dict(revision=cp['revision'], base=spec['revision'],
                  recipe=r, microbatch=microbatch, log=log))
        os.rename(stage, out)
    print('Saved full fine-tuned checkpoint:', out/'model')
