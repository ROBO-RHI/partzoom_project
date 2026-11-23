"""Generate RLBench (PerAct fork) demos with oracle part regions.

Outputs per task:
  <out>/packaged/<split>/<task>+<var>/ep<i>.dat  -- 3DDA packaged format plus
        [6] part-region boxes per keyframe  (list of (2,2,3) float32)
        [7] object-region boxes per keyframe
  <out>/seeds/<split>/<task>/ep<i>.pkl          -- {variation, random_seed,
        descriptions, n_keyframes} for exact reset_to_demo in evaluation

Rendering matches 3DDA's re-rendered PerAct data: 256x256, OpenGL renderer,
cameras left_shoulder, right_shoulder, wrist, front, RGB + world point cloud.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import blosc
import einops
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyrep.const import RenderMode
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig

from utils.utils_with_rlbench import (
    keypoint_discovery, obs_to_attn, transform, task_file_to_task_class
)
from partzoom.regions import install_region_hook

CAMERAS = ("left_shoulder", "right_shoulder", "wrist", "front")


def make_obs_config(image_size, with_images=True):
    unused = CameraConfig()
    unused.set_all(False)
    used = CameraConfig(rgb=True, point_cloud=True, depth=False, mask=False,
                        image_size=image_size, render_mode=RenderMode.OPENGL)
    cams = {c: (used if with_images else unused) for c in CAMERAS}
    return ObservationConfig(
        front_camera=cams["front"],
        left_shoulder_camera=cams["left_shoulder"],
        right_shoulder_camera=cams["right_shoulder"],
        wrist_camera=cams["wrist"],
        overhead_camera=unused,
        joint_forces=False, joint_positions=False, joint_velocities=True,
        task_low_dim_state=False, gripper_touch_forces=False,
        gripper_pose=True, gripper_open=True, gripper_matrix=False,
        gripper_joint_positions=False,
    )


def obs_state(obs):
    rgb = [getattr(obs, f"{c}_rgb") for c in CAMERAS]
    pc = [getattr(obs, f"{c}_point_cloud") for c in CAMERAS]
    return transform({"rgb": rgb, "pc": pc})


def obs_action(obs):
    return torch.from_numpy(
        np.concatenate([obs.gripper_pose, [obs.gripper_open]])).float()


def package(demo):
    kf = keypoint_discovery(demo)
    kf.insert(0, 0)
    states, actions, inter, preg, oreg, attn = [], [], [], [], [], []
    for i, f in enumerate(kf):
        obs = demo._observations[f]
        actions.append(obs_action(obs).unsqueeze(0))
        if i < len(kf) - 1:
            states.append(obs_state(obs).unsqueeze(0))
            preg.append(obs.misc['part_regions'])
            oreg.append(obs.misc['object_regions'])
            inter.append(torch.stack([obs_action(demo._observations[j])
                                      for j in range(kf[i], kf[i + 1] + 1)]))
        attn.append({c: obs_to_attn(obs, c) for c in CAMERAS})
    states = einops.rearrange(torch.cat(states), "t (m n ch) h w -> t n m ch h w",
                              ch=3, n=len(CAMERAS), m=2)
    n = len(states)
    item = [list(range(n)), states.numpy(), actions[1:], attn[:n],
            actions[:-1], inter, preg, oreg]
    return item, len(kf) - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--num", type=int, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--seed_base", type=int, required=True)
    p.add_argument("--out", default="data/partzoom")
    p.add_argument("--no_images", action="store_true",
                   help="only verify demo feasibility and store seeds")
    args = p.parse_args()

    out = Path(args.out)
    env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=make_obs_config([256, 256], not args.no_images),
        headless=True)
    env.launch()
    task_env = env.get_task(task_file_to_task_class(args.task))
    state = {"var": 0}
    install_region_hook(env, lambda: args.task, lambda: state["var"])
    nvar = task_env.variation_count()

    seed_dir = out / "seeds" / args.split / args.task
    seed_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(args.start, args.start + args.num):
        seed_file = seed_dir / f"ep{ep}.pkl"
        if seed_file.exists():
            continue
        rng = np.random.RandomState(args.seed_base + ep)
        var = int(rng.randint(nvar))
        state["var"] = var
        task_env.set_variation(var)
        np.random.seed(args.seed_base * 1000 + ep)
        for attempt in range(5):
            try:
                demo = task_env.get_demos(1, live_demos=True)[0]
                break
            except Exception as e:  # noqa
                print(f"ep{ep} attempt {attempt} failed: {e}", flush=True)
                demo = None
        if demo is None:
            print(f"ep{ep}: giving up", flush=True)
            continue
        n_kf = len(keypoint_discovery(demo))
        if not args.no_images:
            item, n_kf = package(demo)
            d = out / "packaged" / args.split / f"{args.task}+{var}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"ep{ep}.dat", "wb") as f:
                f.write(blosc.compress(pickle.dumps(item)))
        with open(seed_file, "wb") as f:
            pickle.dump({"variation": var, "random_seed": demo.random_seed,
                         "n_keyframes": n_kf, "demo_len": len(demo)}, f)
        el = time.time() - t0
        print(f"{args.task} {args.split} ep{ep} var{var} len{len(demo)} "
              f"kf{n_kf} elapsed {el:.0f}s", flush=True)
    env.shutdown()


if __name__ == "__main__":
    main()
