"""Clean inference-time benchmark (idle GPU): seconds per closed-loop step
(one full 100-step denoising) for each token-selection configuration."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import blosc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from partzoom.eval_rollouts import build_model  # noqa
from utils.common_utils import get_gripper_loc_bounds  # noqa

CONFIGS = {"fps": dict(token_mode="fps", zoom=0),
           "part": dict(token_mode="part", zoom=0),
           "zoom": dict(token_mode="fps", zoom=1),
           "both": dict(token_mode="part", zoom=1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="train_logs/diffuser_actor_peract.pth")
    p.add_argument("--episode", default="data/partzoom/packaged/val/insert_onto_square_peg+0")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--output", default="eval_logs/offline/bench_infer.json")
    a = p.parse_args()
    import glob
    f = sorted(glob.glob(f"{a.episode}*/ep*.dat"))[0]
    ep = pickle.loads(blosc.decompress(open(f, "rb").read()))
    st = torch.as_tensor(ep[1][:1])
    rgb = (st[:, :, 0] / 2 + 0.5).cuda()
    pcd = st[:, :, 1].cuda()
    grip = torch.as_tensor(ep[4][0])[..., :7].reshape(1, 1, 7).expand(1, 3, 7).cuda()
    regions = torch.as_tensor(ep[6][0])[None].float().cuda()
    instr = torch.zeros(1, 53, 512).cuda()
    bounds = get_gripper_loc_bounds(
        "tasks/18_peract_tasks_location_bounds.json", buffer=0.04, task=None)
    out = {}
    for name, cfg in CONFIGS.items():
        ns = argparse.Namespace(checkpoint=a.checkpoint, zoom_tokens=256,
                                rel_cap=0.5, crop_size=128, crop_margin=0.25, **cfg)
        m = build_model(ns, bounds)
        r = None if name == "fps" else regions
        ts = []
        for i in range(a.reps + 2):
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                m(torch.zeros(1, 1, 8).cuda(), torch.zeros(1, 1, dtype=torch.bool).cuda(),
                  rgb, pcd, instr, grip, run_inference=True, regions=r)
            torch.cuda.synchronize()
            if i >= 2:
                ts.append(time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 2**20
        out[name] = {"s_per_step_mean": sum(ts) / len(ts),
                     "s_per_step_min": min(ts), "peak_mem_mib": mem}
        print(name, out[name], flush=True)
        del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=1)


if __name__ == "__main__":
    main()
