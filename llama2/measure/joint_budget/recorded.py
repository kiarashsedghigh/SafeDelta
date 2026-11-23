"""Arm A′: record the official selection and FP16 output from one replay per module."""
import json

import torch
from safetensors.torch import save_file

from measure.cache import digest, json_save, load, save, temporary
from measure.run import artifact_repair, unit


def module_files(root, i, name):
    stem = unit(i, name)
    return (root / 'model' / (stem + '.safetensors'),
            root / 'masks' / (stem + '.pt'),
            root / 'records' / (stem + '.json'))


def verified_inputs(exp, i, name):
    weight, mask, receipt = module_files(exp.out / 'Aprime', i, name)
    curve = exp.paths(i, name)[0]
    inputs = {'curvature': digest(curve), 'mask': digest(mask), 'weight': digest(weight)}
    if inputs != json.loads(receipt.read_text()):
        raise ValueError(f'{unit(i, name)}: A′ weights, mask or curvature changed; rebuild Aprime')
    return inputs


def record_module(exp, i, name):
    """Write the pair atomically per file and publish the hashes last."""
    root = exp.out / 'Aprime'
    weight, mask, receipt = module_files(root, i, name)
    curve = exp.paths(i, name)[0]
    curve_hash = digest(curve)
    if receipt.exists() and weight.exists() and mask.exists():
        previous = json.loads(receipt.read_text())
        if (previous['curvature'] == curve_hash and previous['weight'] == digest(weight)
                and previous['mask'] == digest(mask)):
            return

    record = artifact_repair(i, name, exp.base_reader, exp.finetuned(),
                             load(curve), exp.meta['scale'], device=exp.device)
    if record['perm'] is not None or record['saved_weight'].dtype != torch.float16:
        raise ValueError('Expected an unpermuted FP16 official replay')
    tiles = [{k: tile[k] for k in ('start', 'stop', 'mask', 'cutoff')}
             for tile in record['tiles']]
    tensor = f'model.layers.{i}.{name}.weight'
    with temporary(weight) as tmp:
        save_file({tensor: record['saved_weight'].contiguous()}, str(tmp), metadata={'format': 'pt'})
    save(mask, {'tiles': tiles})
    json_save(receipt, {'curvature': curve_hash, 'weight': digest(weight), 'mask': digest(mask)})
    print('Recorded A′ weights and mask', unit(i, name), flush=True)
