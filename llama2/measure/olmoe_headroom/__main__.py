"""OLMoE headroom check. Start with preflight. A prior Llama run is optional."""
import argparse
import importlib.metadata
from pathlib import Path

import torch
from measure.cache import lock
from .protocol import ARMS, export_reference, preflight, protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path)
    sub = p.add_subparsers(dest='stage', required=True)
    export = sub.add_parser('export-reference')
    export.add_argument('--reference-run', required=True, type=Path)
    export.add_argument('--output', type=Path, required=True)
    for name in ('results', 'subset', 'train-params'):
        export.add_argument('--'+name, type=Path)
    init = sub.add_parser('preflight')
    init.add_argument('--reference', type=Path, help='Optional bundle exported from a previous Llama run')
    init.add_argument('--revision')
    init.add_argument('--scale', type=float, default=.11)
    sub.add_parser('train').add_argument('--microbatch', type=int)
    for cmd in ('calibrate','repair','audit','table'):
        sub.add_parser(cmd)
    for cmd in ('evaluate','_generate'):
        ev = sub.add_parser(cmd)
        ev.add_argument('--arm', required=True, choices=ARMS)
        if cmd == '_generate':
            ev.add_argument('--dest', required=True, type=Path)
    a = p.parse_args()
    if a.stage == 'export-reference':
        export_reference(a.reference_run, a.output, a.results, a.subset, a.train_params)
        return
    if a.run is None:
        p.error('--run is required')
    if importlib.metadata.version('transformers') != '4.46.3':
        p.error('Use transformers==4.46.3 for the verified unfused OLMoE module layout')
    if a.stage == '_generate':
        from .evaluate import generation
        with torch.inference_mode():
            generation(a.run, protocol(a.run), a.arm, a.dest)
        return
    with lock(a.run/'.stage.lock'):
        if a.stage == 'preflight':
            preflight(a.run, a.reference, a.revision, a.scale)
            return
        spec = protocol(a.run)
        if a.stage == 'train':
            from .train import train
            train(a.run, spec, a.microbatch)
        elif a.stage == 'evaluate':
            from .evaluate import evaluate
            with torch.inference_mode():
                evaluate(a.run, spec, a.arm)
        elif a.stage == 'table':
            from .evaluate import table
            table(a.run, spec)
        else:
            from .repair import calibrate, repair, audit
            {'calibrate': calibrate, 'repair': repair, 'audit': audit}[a.stage](a.run, spec)
            if torch.cuda.is_available():
                print('Peak GPU reserved (GiB):', round(torch.cuda.max_memory_reserved()/2**30, 2))


if __name__ == '__main__':
    main()
