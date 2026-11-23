# Dataset curation

This file explains how the training, validation and test data for the part-zoom experiments were made, and what each episode contains. All data was generated from scratch on this machine with the generator in `code/partzoom/generate_demos.py`. No upstream dataset files were used, because the upstream PerAct data does not record the simulator state needed to compute the part regions.

## 1. Simulator and rendering

| item | setting | why |
|---|---|---|
| simulator | CoppeliaSim 4.1 (Edu), PyRep, RLBench, PerAct fork (`MohitShridhar/RLBench`, `peract` branch) | the same task definitions, variations and success conditions as the released 3D Diffuser Actor (3DDA) PerAct checkpoint |
| display | headless under `xvfb-run` | a server without a display |
| renderer | `RenderMode.OPENGL` | matches 3DDA's re-rendered PerAct data, which the released checkpoint was trained on |
| cameras | `left_shoulder`, `right_shoulder`, `wrist`, `front` | same 4 cameras and order as 3DDA PerAct |
| image size | 256 × 256 | same as 3DDA PerAct |
| modalities | RGB plus a world-frame point cloud for each camera; gripper pose, gripper open state, joint velocities | what 3DDA consumes (depth and masks are not stored) |
| demo action mode | `JointVelocity` + `Discrete` gripper, `get_demos(live_demos=True)` | the standard RLBench motion-planned demo collection |

## 2. Tasks

These are the three tasks named in the proposal. Each is a precise-placement task where the contact geometry is small compared to the scene.

| task | variations | what varies | contact the method focuses on |
|---|---|---|---|
| `insert_onto_square_peg` (insert peg) | 20 | which of the 3 coloured spokes of the peg base to use, and colours | square ring → chosen spoke |
| `place_shape_in_shape_sorter` (sort shape) | 5 | which shape (cube, cylinder, triangular prism, star, moon) | held shape → its matching slot |
| `slide_block_to_color_target` (slide block) | 4 | which of the 4 coloured targets | block → chosen target pad |

## 3. Splits and seeding

There are three disjoint splits for each task. Each split uses its own seed base, so no initial scene appears in two splits.

| split | episodes / task | seed base | images stored? | used for |
|---|---|---|---|---|
| train | 100 | 1000 | yes | fine-tuning all variants |
| val | 20 | 2000 | yes | validation during training (every 1000 iterations) and offline per-keyframe errors |
| test | 100 (first **50** used in all rollout tables) | 3000 | no, seeds only | closed-loop rollouts |

For episode `i` of a split with seed base `B`:

1. **Variation:** `np.random.RandomState(B + i).randint(n_variations)`. Each episode's variation is a pure function of `(B, i)`, so the variation mix is random but reproducible.
2. **Scene randomisation:** `np.random.seed(B*1000 + i)`, then `task.get_demos(1, live_demos=True)`. Object poses are therefore also reproducible.
3. **Stored state:** the RNG state that RLBench saves with the demo (`demo.random_seed`) is written to `data/seeds/<split>/<task>/ep<i>.pkl`, together with the variation, keyframe count and demo length.
4. **Test-time reset:** evaluation reconstructs the exact initial scene with `task.reset_to_demo(Demo([], random_seed=...))`. Every variant is therefore evaluated on exactly the same 50 scenes per task, which is what makes the paired (McNemar) comparisons in RESULTS.md valid.

If motion planning fails while collecting a demo, the generator retries up to 5 times from the same seed. Each retry continues from the current RNG state, so it samples a new scene. Planning failures were rare: 9 retries across 360 imaged episodes and 300 test seeds (all "Could not collect demos"), and no episode needed all 5 attempts. Since a failed demo is simply replaced by a new scene, this does not bias the test set toward easy scenes beyond what RLBench's own demo feasibility check already does.

### Split statistics (measured from the seed files)

Keyframes are counted with 3DDA's `keypoint_discovery` (gripper state changes and zero-velocity points).

| split | task | episodes | variations covered | episodes per variation | keyframes mean [min–max] | demo length in sim steps, mean [min–max] |
|---|---|---|---|---|---|---|
| train | insert peg | 100 | 20/20 | 1–11 | 4.98 [3–5] | 171 [116–271] |
| train | sort shape | 100 | 5/5 | 17–23 | 4.92 [4–6] | 159 [108–261] |
| train | slide block | 100 | 4/4 | 22–28 | 3.50 [2–5] | 143 [88–262] |
| val | insert peg | 20 | 11/20 | 1–3 | 5.00 [5–5] | 165 [109–249] |
| val | sort shape | 20 | 5/5 | 1–6 | 4.95 [4–5] | 162 [105–243] |
| val | slide block | 20 | 4/4 | 2–7 | 3.05 [2–5] | 130 [87–255] |
| test | insert peg | 100 | 20/20 | 2–9 | 5.00 [5–5] | 166 [109–238] |
| test | sort shape | 100 | 5/5 | 17–23 | 4.96 [4–7] | 161 [109–258] |
| test | slide block | 100 | 4/4 | 19–29 | 3.35 [2–5] | 138 [87–244] |

Storage: train is 5.2 GB and val is 1.0 GB (blosc-compressed pickles). Generation time with 3 parallel CPU workers on this machine was about 25–90 s per imaged demo; sort shape was the slowest task at about 2.5 h for 100 train demos.

## 4. Episode file format

Each `data/packaged/<split>/<task>+<variation>/ep<i>.dat` is a blosc-compressed pickle of a list. Fields 0–5 are exactly the upstream 3DDA packaged format, so the upstream `RLBenchDataset` reads them unchanged. Fields 6–7 are new.

| index | content | shape / type |
|---|---|---|
| 0 | keyframe ids `0..T-1` | list[int] |
| 1 | observation at each keyframe start: RGB (normalised to [-1, 1]) and world point cloud, for 4 cameras | `(T, 4, 2, 3, 256, 256)` float |
| 2 | next keypose action (target): position xyz, quaternion as RLBench returns it (xyzw), gripper open. The model runs with `quaternion_format=wxyz`, as the released PerAct checkpoint requires; in practice no reordering is applied, and training and inference treat the quaternion the same way | list of `(1, 8)` |
| 3 | per-camera gripper pixel attention maps (upstream field, unused by 3DDA) | list[dict] |
| 4 | current gripper pose at each keyframe, used for the 3-step proprioception history | list of `(1, 8)` |
| 5 | dense end-effector trajectory between consecutive keyframes (upstream field; with `keypose_only=1` only the final pose is used) | list of `(L, 8)` |
| **6** | **part regions** at each keyframe observation | list of `(2, 2, 3)` float32 |
| **7** | **object regions** at each keyframe observation | list of `(2, 2, 3)` float32 |

The keyframe observations start at frame 0, followed by every discovered keyframe except the last. The action for observation `t` is the gripper pose at keyframe `t+1`.

## 5. Oracle regions (privileged information)

**The part regions are oracle inputs read from the simulator's ground-truth object poses. They are privileged information and are not available to a real robot.** The proposal states this explicitly. Every number in RESULTS.md that uses regions should be read as "with oracle localisation"; the localisation-noise sweep, where available, measures how quickly the gains degrade when localisation is imperfect.

`code/partzoom/regions.py` defines two world-frame axis-aligned boxes for each task, stored as `[min_xyz, max_xyz]`:

- **region 0, the manipulated part:** what the gripper grasps or pushes.
- **region 1, the receiving part:** where it must go.

| task | region 0 (both granularities) | region 1, **part** granularity (main experiments) | region 1, **object** granularity (whole-object zoom ablation) |
|---|---|---|---|
| insert peg | bounding box of `square_ring` | bounding box of the one target spoke (`pillar0/1/2` nearest in xy to the task's `success_centre` dummy) | bounding box of the whole peg base and all 3 spokes |
| sort shape | bounding box of the variation's shape | a box centred on that shape's slot (`<shape>_drop_point`), xy half-size = larger xy half-extent of the shape + 1 cm, z half-size = 3 cm + 1 cm | bounding box of the entire shape sorter |
| slide block | bounding box of `block` | 12 × 12 × 6 cm box around the target sensor `success<var+1>` | one box covering all 4 targets (+6 cm, 6 cm, 3 cm) |

**How the boxes are computed.** An object's box is the world-frame AABB of its 8 oriented bounding-box corners. A "tree" box is the union over all shapes under an object. The model pads each box by 1 cm on every side (`region_pad`).

**When they are recorded.** The boxes come from a hook on `Scene.get_observation`, so they are recomputed at every observation from the current object poses. Two consequences:
- When the ring or shape is carried by the gripper, region 0 moves with it.
- No future information (demo actions, goal poses not visible in the scene) is ever used. Region 1 is a static scene part whose identity comes from the task variation, which the language instruction also gives the policy.

The same hook runs during closed-loop evaluation, so training and test regions come from identical code.

**Checks.** `figs/regions_peg_ep0.png` overlays the boxes on all 4 cameras for every keyframe of a training episode, and shows the zoom crops the model takes. In it:
- the ring box follows the ring while it is carried;
- the chosen spoke is the one the demo inserts onto;
- crops magnify the parts 2.4–5×.

**Random-region control.** For the random-region zoom ablation, `random_regions()` in `main_partzoom.py` keeps each true box's size and height but moves its xy centre uniformly at random within the workspace bounds. This happens independently at every training sample and every test step. The control has the same token budget, crop count and crop size as part zoom, but the crops land on the wrong place.

## 6. What the training loader does

`RegionRLBenchDataset` (in `main_partzoom.py`) is a line-for-line copy of upstream `RLBenchDataset.__getitem__` that also returns the region boxes for each keyframe:

- A training sample is a chunk of up to 5 consecutive keyframes from one episode (`max_episode_length=5`); a batch is 4 episodes.
- RGB and point clouds receive upstream's random rescale augmentation (`image_rescale 0.75,1.25`). The augmentation rescales and crops the image and point cloud together, so the world-frame boxes stay valid.
- The language instruction is sampled from the task-variation's paraphrases (`instructions/peract/instructions.pkl`, CLIP text features; `code/instructions` holds the subset for these 3 tasks).
- Position bounds are the 18-task PerAct bounds with a 4 cm buffer, identical to the released checkpoint's. All three variants use the same bounds, so they share one normalisation.

## 7. Reproducing the data

```bash
cd code && source partzoom/env.sh
# everything (3 workers, ~3 h on 4 vCPUs):
bash partzoom/gen_all.sh
# or one split of one task:
xvfb-run -a -s "-screen 0 1024x768x24" python partzoom/generate_demos.py \
    --task insert_onto_square_peg --split train --num 100 --seed_base 1000 --out data/partzoom
# test seeds only (no images, fast):
... --split test --num 100 --seed_base 3000 --no_images
```

The generator skips any episode whose seed file already exists, so an interrupted run can be restarted. In this package, `code/data/partzoom/packaged` is a symlink to the original 6.2 GB of packaged demos (`~/3d_diffuser_actor/data/partzoom/packaged`). `data/seeds/` holds a real copy of every seed file. The seeds alone are enough to reproduce the test set exactly, and to regenerate train and val.
