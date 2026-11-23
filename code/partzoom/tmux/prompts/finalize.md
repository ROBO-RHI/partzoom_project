You are running unattended (cron, headless) on the user's AWS GPU machine. The user is away and has asked you to finish the write-up of their experiments on your own. Nobody will answer questions. Work carefully and report results honestly.

## Context
This is the PartZoom project; read ~/.claude/projects/-home-ubuntu/memory/partzoom-project.md first. It tests two changes to 3D Diffuser Actor (3DDA) proposed in a 16-831 course proposal by Harshvardhan Aditya, Ishita Srivastava and Raghav Mehta:
- **part-aware token sampling**
- **part zoom**: re-encoded crops around the task-relevant parts

Both use oracle part regions from the simulator, which is privileged information. They are tested on three RLBench tasks: insert peg, sort shape and slide block. Code is in ~/3d_diffuser_actor/partzoom/. The self-contained deliverable is **~/partzoom_project**, containing README.md, RESULTS.md (currently marked "interim") and DATASET.md, plus results/, logs/, checkpoints/, figs/, code/ and data/seeds/. The pipeline timeline is ~/3d_diffuser_actor/logs_partzoom/pipeline.log, which now ends with PACKAGE_DONE. Run `source ~/3d_diffuser_actor/partzoom/env.sh` before any Python.

Variants evaluated with closed-loop rollouts (50 fixed test scenes per task, max_steps 12; per-episode files in ~/3d_diffuser_actor/eval_logs/partzoom/<variant>/<task>.jsonl):
- released (no fine-tuning); released_part and released_zoom (the methods applied zero-shot to the released weights)
- baseline, part, zoom, both (fine-tuned, lr 1e-4, 4000 iterations, batch 4)
- baseline_lr2e5, both_lr2e5 (the same at lr 2e-5)
- zoom_object (whole-object zoom) and zoom_random (random-region zoom), the secondary controls

Offline per-keyframe validation errors are in eval_logs/offline/<name>.json. They cover each variant plus eval-time sweeps on the trained models: region-noise std 0.01/0.02/0.04 m for zoom, part and both; crop_margin 0.0/0.1/0.5/1.0 for zoom; zoom_tokens 128/384; and rel_cap 0.25/0.75 for part. The clean inference benchmark is eval_logs/offline/bench_infer.json. Training validation curves are in logs_partzoom/train_<name>.log (lines starting "[val]").

## Tasks
1. **Check completeness.** Every variant above should have 3 tasks × 50 episodes, and every offline json should exist. If something is missing because a job crashed, rerun just that piece with the same settings. Rollouts resume automatically: see partzoom/tmux/eval_worker.sh for the exact command and flags, and partzoom/pipeline3.sh for the offline commands. Then rerun `python partzoom/make_package.py` from ~/3d_diffuser_actor. If a rerun is impossible in reasonable time (more than about 6 h), say so explicitly in RESULTS.md instead.
2. **Rewrite ~/partzoom_project/RESULTS.md as the final version.** Take every number from ~/partzoom_project/results/tables.md, or compute it from the raw jsonl/json with small scripts; never invent numbers. Keep the existing structure and tone, remove the "interim" status, and include:
   - A short **key findings** summary at the top.
   - The main success table with Wilson CIs, and paired exact-McNemar comparisons against **both** baselines (the matched fine-tuned baseline and the released checkpoint). For the lr 2e-5 runs, the matched comparison is both_lr2e5 vs baseline_lr2e5.
   - The zero-shot results (released_part and released_zoom vs released).
   - The secondary comparisons: part zoom vs whole-object zoom vs random-region zoom. Does focusing on the *part* matter?
   - Offline alignment errors for all keyframes and for interaction keyframes (the second half of each episode).
   - Token retention, including the FPS retention measured offline (baseline/released with region_source part).
   - The sweeps (localisation noise, crop margin, token budget, rel_cap). Label them clearly as eval-time sweeps on models trained with the default settings.
   - Failure types, and inference cost from the clean benchmark (s/step and peak memory per configuration).
   - An honest discussion: what beats which baseline, what is and is not statistically significant, the fine-tuning regression versus the released checkpoint and whether the lower learning rate fixed it, and limitations (oracle regions, n = 50 per task, a single training seed, 4000 iterations).
   State clearly which configuration is best overall and whether it beats the released checkpoint.
3. Make one compact figure, saved as ~/partzoom_project/figs/success_rates.png with matplotlib: grouped bars of success rate per task and variant with the CIs. Reference it from RESULTS.md.
4. Update the status lines in ~/partzoom_project/README.md if anything there is out of date (it should not say "interim"). Leave DATASET.md alone unless something in it is factually wrong.
5. Update the memory file ~/.claude/projects/-home-ubuntu/memory/partzoom-project.md with one or two lines on the final status (date and headline result).
6. Append a short, dated summary of what you did to ~/3d_diffuser_actor/logs_partzoom/autopilot.log.

Do not publish anything externally, do not delete results or checkpoints, and do not start new experiments beyond rerunning missing pieces.
