"""Observe the exact CUDA selection decisions and both weight casts."""
import torch


class Recorder:
    def __init__(self):
        self.tiles = []
        self.writes = {}

    def tile(self, start, stop, mask, flat_loss, cumulative, accepted, cutoff):
        count = int(accepted.sum().item())
        self.tiles.append({
            'start': start, 'stop': stop, 'mask': mask.cpu(),
            'curve': cumulative.cpu(), 'accepted': count, 'cutoff': cutoff.item(),
            'believed': cumulative[count - 1].item() if count else 0.,
            'first_rejected': cumulative[count].item() if count < cumulative.numel() else None,
            'row_charge': flat_loss.view_as(mask).masked_fill(~mask, 0).sum(1, dtype=torch.float64).cpu(),
        })

    def write(self, working, original):
        original32 = original.float()
        self.writes = {'pre': (working - original32).cpu(),
                       'saved_weight': working.to(original.dtype).half().cpu(),
                       'post': (working.to(original.dtype).half().float() - original32).cpu()}
