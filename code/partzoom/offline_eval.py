"""Offline keypose errors on held-out demos, per keyframe (manipulation stage).

For every keyframe of every val episode the model predicts the next keypose
(100 denoising steps, fixed seed) given the demo observation; we report
position L2 (cm), rotation geodesic angle (deg) and gripper accuracy, grouped
by keyframe index. Also logs token-selection statistics.
"""
import argparse
import glob
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import blosc
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from partzoom.eval_rollouts import build_model  # noqa
from partzoom.main_partzoom import random_regions  # noqa
from utils.common_utils import get_gripper_loc_bounds  # noqa

CAMERAS = ("left_shoulder", "right_shoulder", "wrist", "front")


def quat_angle_deg(q1, q2):
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    d = np.abs((q1 * q2).sum(-1)).clip(0, 1)
    return np.degrees(2 * np.arccos(d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--valset", default="data/partzoom/packaged/val")
    p.add_argument("--output", required=True)
    p.add_argument("--token_mode", default="fps")
    p.add_argument("--zoom", type=int, default=0)
    p.add_argument("--zoom_tokens", type=int, default=256)
    p.add_argument("--rel_cap", type=float, default=0.5)
    p.add_argument("--crop_size", type=int, default=128)
    p.add_argument("--crop_margin", type=float, default=0.25)
    p.add_argument("--region_source", default="part")
    p.add_argument("--region_noise", type=float, default=0.0)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--instructions",
                   default="instructions/peract/instructions.pkl")
    a = p.parse_args()

    bounds = get_gripper_loc_bounds(
        "tasks/18_peract_tasks_location_bounds.json", buffer=0.04, task=None)
    model = build_model(a, bounds)
    instructions = pickle.load(open(a.instructions, "rb"))
    rows = []
    stats = defaultdict(list)
    for task in a.tasks:
        files = sorted(glob.glob(f"{a.valset}/{task}+*/ep*.dat"))
        for f in files:
            var = int(f.split("+")[-1].split("/")[0])
            epi = int(Path(f).stem[2:])
            ep = pickle.loads(blosc.decompress(open(f, "rb").read()))
            T = len(ep[0])
            st = torch.as_tensor(ep[1])  # T, ncam, 2, 3, H, W
            rgb = (st[:, :, 0] / 2 + 0.5).cuda()
            pcd = st[:, :, 1].cuda()
            grip = torch.stack([g[0] for g in ep[4]])  # T, 8
            hist = torch.stack([grip[[max(0, i - 2), max(0, i - 1), i]]
                                for i in range(T)])[..., :7].cuda()
            gt = torch.stack([x[0] for x in ep[2]]).numpy()
            key = "object" if a.region_source == "object" else "part"
            regions = torch.as_tensor(np.stack(ep[7 if key == "object" else 6])).float()
            for s in range(a.seeds):
                g = torch.Generator().manual_seed(epi * 31 + s)
                r = regions
                if a.region_source == "random":
                    torch.manual_seed(epi * 31 + s + 7)
                    r = random_regions(regions, bounds)
                if a.region_noise > 0:
                    r = r + torch.randn(r.shape[0], r.shape[1], 1, 3,
                                        generator=g) * a.region_noise
                rr = None if a.region_source == "none" else r.cuda()
                instr_list = list(instructions[task][var])
                instr = instr_list[(epi + s) % len(instr_list)][None].expand(
                    T, -1, -1).cuda()
                torch.manual_seed(epi * 1000 + s)
                with torch.no_grad():
                    pred = model(torch.zeros(T, 1, 8).cuda(),
                                 torch.zeros(T, 1, dtype=torch.bool).cuda(),
                                 rgb, pcd, instr, hist, run_inference=True,
                                 regions=rr)[:, -1].cpu().numpy()
                for k, v in model.last_stats.items():
                    stats[k].append(v)
                pos = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=-1) * 100
                rot = quat_angle_deg(pred[:, 3:7], gt[:, 3:7])
                grp = (pred[:, 7] >= 0.5) == (gt[:, 7] > 0.5)
                for t in range(T):
                    rows.append({"task": task, "episode": epi, "seed": s,
                                 "kf": t, "T": T, "pos_cm": float(pos[t]),
                                 "rot_deg": float(rot[t]),
                                 "grip_ok": bool(grp[t])})
        tr = [r for r in rows if r["task"] == task]
        print(task, f"n={len(tr)} pos {np.mean([r['pos_cm'] for r in tr]):.2f}cm "
              f"rot {np.mean([r['rot_deg'] for r in tr]):.2f}deg", flush=True)
    out = {"args": vars(a), "rows": rows,
           "token_stats": {k: float(np.mean(v)) for k, v in stats.items()}}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"))
    print("token stats", out["token_stats"])


if __name__ == "__main__":
    main()
