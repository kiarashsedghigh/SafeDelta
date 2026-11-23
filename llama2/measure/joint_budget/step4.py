"""Our Step 4 / arm C: enforce the allowance on serialized FP16 weights."""
import math
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from measure.cache import temporary
from .step1 import saved_cost


def read_weight(path, name):
    with safe_open(str(path), framework='pt', device='cpu') as shard:
        return shard.get_tensor(name)


def write_measured(path, name, base, solution, scale, h):
    # Exact compensation is linear in the fixed values at a fixed mask.
    # Scale the continuous solution, then do a single FP16 cast of base + d.
    written = (base.double() + solution * scale).half()
    with temporary(path) as tmp:
        save_file({name: written.contiguous()}, str(tmp), metadata={'format': 'pt'})
    return saved_cost(read_weight(path, name), base, h)


def enforce_budget(path, name, base, solution, h, allowance, b_cost):
    if not (0 <= allowance < math.inf and 0 <= b_cost < math.inf):
        raise ValueError('A finite nonnegative cost and allowance are required')
    scale = 1.
    cost = b_cost
    for attempt in range(1, 34):
        if cost <= allowance:
            return {'scale': scale, 'post_cost': cost, 'writes': attempt - 1}
        # FP16 rounding produces plateaus. First try the requested square-root
        # scale exactly. Back off on retries and explicitly test zero at the end.
        shrink = math.sqrt(allowance / cost)
        if attempt > 1:
            shrink = min(shrink, 0.99)
        scale = 0. if attempt == 32 else scale * shrink
        cost = write_measured(path, name, base, solution, scale, h)
        if scale == 0. and cost > allowance:
            raise ArithmeticError('Even casting the aligned weights to FP16 exceeds this allowance')
    raise ArithmeticError('Step 4 did not find a feasible saved checkpoint')
