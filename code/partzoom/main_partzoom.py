"""Fine-tune 3D Diffuser Actor variants (baseline / part sampling / part zoom).

Run with:  torchrun --nproc_per_node 1 partzoom/main_partzoom.py ...
"""
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.dataset_engine import RLBenchDataset
from main_trajectory import (Arguments as BaseArguments, TrainTester,
                             TrajectoryCriterion, traj_collate_fn)
from utils.common_utils import (load_instructions, count_parameters,
                                get_gripper_loc_bounds)
from partzoom.model import PartZoomDiffuserActor, load_released_weights


class Arguments(BaseArguments):
    token_mode: str = 'fps'          # fps | part
    zoom: int = 0
    zoom_tokens: int = 256
    rel_cap: float = 0.5
    crop_size: int = 128
    crop_margin: float = 0.25
    region_source: str = 'part'      # part | object | random
    init_weights: Optional[Path] = None
    use_regions: int = 1


def random_regions(regions, bounds):
    """Boxes of the same size as the true ones, at random table locations."""
    lo = torch.as_tensor(bounds[0], dtype=regions.dtype)
    hi = torch.as_tensor(bounds[1], dtype=regions.dtype)
    size = regions[..., 1, :] - regions[..., 0, :]
    centre = lo + torch.rand(regions.shape[:-2] + (3,)) * (hi - lo)
    centre[..., 2] = (regions[..., 0, 2] + regions[..., 1, 2]) / 2
    return torch.stack([centre - size / 2, centre + size / 2], -2)


class RegionRLBenchDataset(RLBenchDataset):
    """Same sampling as upstream, plus per-keyframe oracle region boxes."""

    def __init__(self, *args, region_source='part', bounds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._region_source = region_source
        self._bounds = bounds

    def __getitem__(self, episode_id):
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]
        episode = self.read_from_cache(file)
        if episode is None:
            return None
        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1)
        frame_ids = episode[0][chunk * self._max_episode_length:
                               (chunk + 1) * self._max_episode_length]
        states = torch.stack([torch.as_tensor(episode[1][i]) for i in frame_ids])
        if episode[3]:
            cameras = list(episode[3][0].keys())
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            states = states[:, index]
        rgbs = self._unnormalize_rgb(states[:, :, 0])
        pcds = states[:, :, 1]
        action = torch.cat([episode[2][i] for i in frame_ids])
        instr = random.choice(self._instructions[task][variation])
        instr = instr[None].repeat(len(rgbs), 1, 1)
        gripper = torch.cat([episode[4][i] for i in frame_ids])
        gripper_history = torch.stack([
            torch.cat([episode[4][max(0, i - 2)] for i in frame_ids]),
            torch.cat([episode[4][max(0, i - 1)] for i in frame_ids]),
            gripper], dim=1)
        src = 7 if self._region_source == 'object' else 6
        regions = torch.stack([torch.as_tensor(episode[src][i])
                               for i in frame_ids]).float()
        if self._region_source == 'random':
            regions = random_regions(regions, self._bounds)

        traj_items = [self._interpolate_traj(episode[5][i]) for i in frame_ids]
        max_l = max(len(item) for item in traj_items)
        traj = torch.zeros(len(traj_items), max_l, 8)
        traj_lens = torch.as_tensor([len(item) for item in traj_items])
        for i, item in enumerate(traj_items):
            traj[i, :len(item)] = item
        traj_mask = torch.zeros(traj.shape[:-1])
        for i, len_ in enumerate(traj_lens.long()):
            traj_mask[i, len_:] = 1

        if self._training:
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs, pcds = modals["rgbs"], modals["pcds"]

        return {
            "task": [task for _ in frame_ids],
            "frame_id": torch.as_tensor(frame_ids),
            "rgbs": rgbs, "pcds": pcds, "action": action, "instr": instr,
            "curr_gripper": gripper, "curr_gripper_history": gripper_history,
            "trajectory": traj, "trajectory_mask": traj_mask.bool(),
            "regions": regions,
        }


def collate(batch):
    out = traj_collate_fn(batch)
    out["regions"] = torch.cat([b["regions"] for b in batch])
    out["frame_id"] = torch.cat([b["frame_id"] for b in batch])
    return out


class PartZoomTrainTester(TrainTester):

    def get_datasets(self):
        a = self.args
        instruction = load_instructions(a.instructions, tasks=a.tasks,
                                        variations=a.variations)
        taskvar = [(t, v) for t, vi in instruction.items() for v in vi.keys()]
        common = dict(
            instructions=instruction, taskvar=taskvar,
            max_episode_length=a.max_episode_length,
            max_episodes_per_task=a.max_episodes_per_task,
            cameras=a.cameras,
            image_rescale=tuple(float(x) for x in a.image_rescale.split(",")),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(a.dense_interpolation),
            interpolation_length=a.interpolation_length,
            region_source=a.region_source, bounds=a.gripper_loc_bounds)
        train = RegionRLBenchDataset(root=a.dataset, cache_size=a.cache_size,
                                     num_iters=a.train_iters, training=True,
                                     **common)
        val = RegionRLBenchDataset(root=a.valset, cache_size=a.cache_size_val,
                                   training=False, **common)
        return train, val

    def get_model(self):
        a = self.args
        model = PartZoomDiffuserActor(
            backbone=a.backbone,
            image_size=tuple(int(x) for x in a.image_size.split(",")),
            embedding_dim=a.embedding_dim,
            num_vis_ins_attn_layers=a.num_vis_ins_attn_layers,
            use_instruction=bool(a.use_instruction),
            fps_subsampling_factor=a.fps_subsampling_factor,
            gripper_loc_bounds=a.gripper_loc_bounds,
            rotation_parametrization=a.rotation_parametrization,
            quaternion_format=a.quaternion_format,
            diffusion_timesteps=a.diffusion_timesteps,
            nhist=a.num_history,
            relative=bool(a.relative_action),
            lang_enhanced=bool(a.lang_enhanced),
            token_mode=a.token_mode, zoom=bool(a.zoom),
            zoom_tokens=a.zoom_tokens, rel_cap=a.rel_cap,
            crop_size=a.crop_size, crop_margin=a.crop_margin)
        if a.init_weights is not None:
            missing, unexpected = load_released_weights(model, a.init_weights)
            print("init from", a.init_weights, "missing:", missing,
                  "unexpected:", unexpected)
        print("Model parameters:", count_parameters(model))
        return model

    def _forward(self, model, sample, run_inference):
        device = next(model.parameters()).device
        a = self.args
        if a.keypose_only:
            sample["trajectory"] = sample["trajectory"][:, [-1]]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
        curr_gripper = sample["curr_gripper_history"][:, -a.num_history:]
        regions = sample["regions"].to(device) if a.use_regions else None
        return model(sample["trajectory"].to(device),
                     sample["trajectory_mask"].to(device),
                     sample["rgbs"].to(device), sample["pcds"].to(device),
                     sample["instr"].to(device), curr_gripper.to(device),
                     run_inference=run_inference, regions=regions)

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()
        loss = self._forward(model, sample, False)
        (loss / self.args.accumulate_grad_batches).backward()
        if (step_id % self.args.accumulate_grad_batches
                == self.args.accumulate_grad_batches - 1):
            optimizer.step()
        if dist.get_rank() == 0 and (step_id + 1) % 50 == 0:
            self.writer.add_scalar("train-loss/noise_mse", loss.item(), step_id)
            stats = model.module.last_stats
            for k, v in stats.items():
                self.writer.add_scalar(f"tokens/{k}", v, step_id)
        if dist.get_rank() == 0 and (step_id + 1) % 200 == 0:
            print(f"step {step_id + 1} loss {loss.item():.4f} "
                  f"stats {model.module.last_stats}", flush=True)

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        device = next(model.parameters()).device
        model.eval()
        vals = {}
        for i, sample in enumerate(loader):
            if i == val_iters:
                break
            action = self._forward(model, sample, True)
            losses, _ = criterion.compute_metrics(
                action, sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device))
            for n, l in losses.items():
                vals.setdefault(n, []).append(l.item())
        vals = {k: float(np.mean(v)) for k, v in vals.items()}
        if dist.get_rank() == 0:
            for k, v in vals.items():
                self.writer.add_scalar(f"{split}-losses/mean/{k}", v, step_id)
            print(f"[{split}] step {step_id}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in vals.items()), flush=True)
        # lower is better: final keypose position error
        return vals.get('pos_l2_final')

    def get_loaders(self, collate_fn=None):
        return super().get_loaders(collate)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = Arguments().parse_args()
    args.gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds, task=None,
        buffer=args.gripper_loc_bounds_buffer)
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.benchmark = True
    PartZoomTrainTester(args).main(collate_fn=collate)
