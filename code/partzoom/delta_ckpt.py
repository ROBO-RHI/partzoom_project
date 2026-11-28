"""Compact ("delta") checkpoints for publishing.

A fine-tuned checkpoint (183 MB) differs from the released 3DDA checkpoint in
only ~11 MB of tensors (the frozen CLIP conv weights are bit-identical). A
delta file stores just the tensors that differ, plus the SHA-256 of the base
checkpoint, and is expanded back to the exact full weights on load.
Optimizer state is not kept (it is only needed to resume training).

usage:
  python partzoom/delta_ckpt.py make  BASE.pth FULL.pth OUT_delta.pth
  python partzoom/delta_ckpt.py check BASE.pth FULL.pth OUT_delta.pth
In code:  partzoom.model.load_released_weights(model, path) accepts either a
full checkpoint or a delta file (the base is found next to it, see resolve_base).
"""
import hashlib
import sys
from pathlib import Path

import torch


def _strip(sd):
    sd = sd.get('weight', sd)
    return {k[len('module.'):] if k.startswith('module.') else k: v
            for k, v in sd.items()}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def make_delta(base_path, full_path, out_path):
    base = _strip(torch.load(base_path, map_location='cpu'))
    full_ck = torch.load(full_path, map_location='cpu')
    full = _strip(full_ck)
    changed = {k: v.clone() for k, v in full.items()
               if k not in base or not torch.equal(v, base[k])}
    removed = [k for k in base if k not in full]
    torch.save({'format': 'partzoom-delta-v1',
                'base_name': Path(base_path).name,
                'base_sha256': sha256(base_path),
                'weight': changed, 'removed': removed,
                'iter': full_ck.get('iter'), 'best_loss': full_ck.get('best_loss')},
               out_path)
    return len(changed), len(full)


def resolve_base(delta_path, base_name):
    """Look for the base checkpoint next to the delta, in ../released/, or train_logs/."""
    d = Path(delta_path).resolve().parent
    for c in [d / base_name, d.parent / 'released' / base_name,
              Path(__file__).resolve().parent.parent / 'train_logs' / base_name]:
        if c.exists():
            return c
    raise FileNotFoundError(f"base checkpoint {base_name} for {delta_path} not found")


def load_full_state_dict(path, verify_base=True):
    ck = torch.load(path, map_location='cpu')
    if not (isinstance(ck, dict) and ck.get('format') == 'partzoom-delta-v1'):
        return _strip(ck)
    base_path = resolve_base(path, ck['base_name'])
    if verify_base and sha256(base_path) != ck['base_sha256']:
        raise ValueError(f"{base_path} does not match the base this delta was made from")
    sd = _strip(torch.load(base_path, map_location='cpu'))
    for k in ck.get('removed', []):
        sd.pop(k, None)
    sd.update(ck['weight'])
    return sd


def check(base_path, full_path, delta_path):
    full = _strip(torch.load(full_path, map_location='cpu'))
    rec = load_full_state_dict(delta_path)
    assert set(rec) == set(full), (set(rec) ^ set(full))
    bad = [k for k in full if not torch.equal(rec[k], full[k]) or rec[k].dtype != full[k].dtype]
    assert not bad, bad[:5]
    return len(full)


if __name__ == '__main__':
    cmd, base, full, out = sys.argv[1:5]
    if cmd == 'make':
        n, tot = make_delta(base, full, out)
        print(f"{out}: {n}/{tot} tensors stored")
    n = check(base, full, out)
    print(f"{out}: OK, all {n} tensors bit-identical to {full}")
