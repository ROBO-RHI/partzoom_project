"""Closed-loop RLBench evaluation for 3D Diffuser Actor variants.

Mirrors online_evaluation_rlbench/evaluate_policy.py (keypose prediction,
EndEffectorPoseViaPlanning without collision checking, max 25 steps,
max_tries 2, 100 diffusion steps), but episodes are restored from our own
seed files and oracle regions are read from the live simulator each step.
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.demo import Demo
from rlbench.environment import Environment

from utils.utils_with_rlbench import Mover, transform, task_file_to_task_class
from utils.common_utils import get_gripper_loc_bounds
from partzoom.generate_demos import make_obs_config, CAMERAS
from partzoom.model import PartZoomDiffuserActor, load_released_weights
from partzoom.regions import install_region_hook
from partzoom.main_partzoom import random_regions


def build_model(a, bounds):
    m = PartZoomDiffuserActor(
        backbone='clip', image_size=(256, 256), embedding_dim=120,
        num_vis_ins_attn_layers=2, use_instruction=True,
        fps_subsampling_factor=5, gripper_loc_bounds=bounds,
        rotation_parametrization='6D', quaternion_format='wxyz',
        diffusion_timesteps=100, nhist=3, relative=False, lang_enhanced=False,
        token_mode=a.token_mode, zoom=bool(a.zoom), zoom_tokens=a.zoom_tokens,
        rel_cap=a.rel_cap, crop_size=a.crop_size, crop_margin=a.crop_margin)
    missing, unexpected = load_released_weights(m, a.checkpoint)
    print("missing", missing, "unexpected", unexpected)
    return m.cuda().eval()


def obs_inputs(obs):
    rgb = [getattr(obs, f"{c}_rgb") for c in CAMERAS]
    pc = [getattr(obs, f"{c}_point_cloud") for c in CAMERAS]
    state = transform({"rgb": rgb, "pc": pc}).reshape(2, len(CAMERAS), 3, 256, 256)
    rgb = (state[0] / 2 + 0.5)[None]  # (1, ncam, 3, H, W) in [0,1]
    pcd = state[1][None]
    grip = torch.from_numpy(
        np.concatenate([obs.gripper_pose, [obs.gripper_open]])).float()
    return rgb, pcd, grip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--seed_dir", default="data/partzoom/seeds/test")
    p.add_argument("--episodes", type=str, default="0:50",
                   help="start:end episode ids")
    p.add_argument("--output", required=True)
    p.add_argument("--token_mode", default="fps")
    p.add_argument("--zoom", type=int, default=0)
    p.add_argument("--zoom_tokens", type=int, default=256)
    p.add_argument("--rel_cap", type=float, default=0.5)
    p.add_argument("--crop_size", type=int, default=128)
    p.add_argument("--crop_margin", type=float, default=0.25)
    p.add_argument("--region_source", default="part")  # part|object|random|none
    p.add_argument("--region_noise", type=float, default=0.0,
                   help="std (m) of Gaussian shift applied to region boxes")
    p.add_argument("--max_steps", type=int, default=25)
    p.add_argument("--max_tries", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--instructions",
                   default="instructions/peract/instructions.pkl")
    a = p.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    bounds = get_gripper_loc_bounds(
        "tasks/18_peract_tasks_location_bounds.json", buffer=0.04, task=None)
    model = build_model(a, bounds)
    with open(a.instructions, "rb") as f:
        instructions = pickle.load(f)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
            gripper_action_mode=Discrete()),
        obs_config=make_obs_config([256, 256], True), headless=True)
    env.launch()
    cur = {"task": None, "var": 0}
    install_region_hook(env, lambda: cur["task"], lambda: cur["var"])

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done[(r["task"], r["episode"])] = r
    e0, e1 = [int(x) for x in a.episodes.split(":")]
    rng = np.random.RandomState(a.seed)

    for task_str in a.tasks:
        task = env.get_task(task_file_to_task_class(task_str))
        cur["task"] = task_str
        for ep in range(e0, e1):
            sf = Path(a.seed_dir) / task_str / f"ep{ep}.pkl"
            if not sf.exists() or (task_str, ep) in done:
                continue
            info = pickle.load(open(sf, "rb"))
            var = info["variation"]
            cur["var"] = var
            task.set_variation(var)
            demo = Demo([], random_seed=info["random_seed"])
            # deterministic per-episode policy noise and instruction choice
            torch.manual_seed(a.seed * 100003 + ep)
            random.seed(a.seed * 100003 + ep)
            _, obs = task.reset_to_demo(demo)
            instr = random.choice(list(instructions[task_str][var]))[None].cuda()
            move = Mover(task, max_tries=a.max_tries)
            grippers = []
            reward, success, fail_type = 0.0, False, "timeout"
            steps, infer_times, stats_log, pred_log = 0, [], [], []
            for step_id in range(a.max_steps):
                rgb, pcd, grip = obs_inputs(obs)
                grippers.append(grip)
                hist = torch.stack(grippers[-3:])[None]
                hist = F.pad(hist, (0, 0, 3 - hist.shape[1], 0), mode='replicate')
                regions = None
                if a.region_source != "none":
                    key = "object_regions" if a.region_source == "object" \
                        else "part_regions"
                    regions = torch.from_numpy(obs.misc[key])[None]
                    if a.region_source == "random":
                        regions = random_regions(regions, bounds)
                    if a.region_noise > 0:
                        shift = torch.from_numpy(rng.normal(
                            0, a.region_noise, (1, regions.shape[1], 1, 3))
                        ).float()
                        regions = regions + shift
                    regions = regions.cuda()
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    traj = model(
                        torch.zeros(1, 1, 8).cuda(),
                        torch.zeros(1, 1, dtype=torch.bool).cuda(),
                        rgb.cuda(), pcd.cuda(), instr,
                        hist[..., :7].cuda(), run_inference=True,
                        regions=regions)
                torch.cuda.synchronize()
                infer_times.append(time.time() - t0)
                stats_log.append(dict(model.last_stats))
                action = traj[-1].cpu().numpy()
                action[:, -1] = action[:, -1].round()
                pred_log.append(action[-1].tolist())
                steps = step_id + 1
                try:
                    for act in action:
                        obs, reward, terminate, _ = move(act)
                    if reward == 1:
                        success, fail_type = True, "success"
                        break
                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    fail_type = "planning:" + type(e).__name__
                    # upstream continues after planner errors
            rec = {
                "task": task_str, "episode": ep, "variation": var,
                "success": success, "reward": float(reward), "steps": steps,
                "fail_type": fail_type,
                "n_demo_keyframes": info["n_keyframes"],
                "infer_time_mean": float(np.mean(infer_times)),
                "stats_mean": {k: float(np.mean([s.get(k, 0) for s in stats_log]))
                               for k in (stats_log[0].keys() if stats_log else [])},
                "pred_keyposes": pred_log,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[(task_str, ep)] = rec
            ntask = [r for r in done.values() if r["task"] == task_str]
            sr = np.mean([r["success"] for r in ntask])
            print(f"{task_str} ep{ep} var{var} success={success} steps={steps} "
                  f"fail={fail_type} infer={np.mean(infer_times):.2f}s "
                  f"running SR {sr:.3f} ({len(ntask)})", flush=True)
    env.shutdown()


if __name__ == "__main__":
    main()
