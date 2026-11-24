You are running unattended (cron, headless) on the user's AWS GPU machine. The user is away and has asked you to write up round 3 of their experiments on your own. Nobody will answer questions. Work carefully and report results honestly.

## Context
This is the PartZoom project; read ~/.claude/projects/-home-ubuntu/memory/partzoom-project.md first. Code is in ~/3d_diffuser_actor/partzoom/. **The deliverable is the git repo ~/harsh/partzoom_project**, which the user pushes to GitHub (ROBO-RHI/partzoom_project). Its RESULTS.md is the final write-up of rounds 1–2: 11 variants on 50 test scenes per task. ~/partzoom_project is an older, unpushed copy; do not edit it. Run `source ~/3d_diffuser_actor/partzoom/env.sh` before any Python. The timeline is ~/3d_diffuser_actor/logs_partzoom/pipeline.log, which now ends with R3_PACKAGE_DONE.

Round 3 tested two things the user asked for after the round-2 result, which was "nothing beats the released checkpoint; the methods are +2 to +5 points over the matched baseline, not significant":
1. **More test scenes for tighter CIs.** `released`, `both_lr2e5`, `baseline_lr2e5` and `released_part` were evaluated on test episodes 0–99 (100 scenes per task) instead of 0–49.
2. **A gentler fine-tune.** A new matched pair at lr 1e-5, `baseline_lr1e5` (FPS) and `both_lr1e5` (part sampling + zoom), was trained with otherwise identical settings and evaluated on episodes 0–99. Offline per-keyframe val errors are in eval_logs/offline/{baseline_lr1e5,both_lr1e5}.json.

Tables were regenerated into ~/harsh/partzoom_project/results/:
- tables.md: main tables on episodes 0–49, now including the lr-1e-5 pair.
- tables_extra.md: includes the new pairs both_lr1e5 vs baseline_lr1e5, baseline_lr1e5 vs baseline_lr2e5, and both_lr1e5 vs released.
- tables_100scenes.md: the new 100-scene table and paired McNemar tests.
- figs/success_rates.png: regenerated.

## Tasks
1. **Check completeness.** The 6 round-3 variants should have 3 tasks × 100 episodes (`eval_logs/partzoom/<name>/<task>.jsonl`, unique episode ids 0–99), and both offline jsons should exist. If something is missing because of a crash, rerun just that piece with the same flags; see partzoom/tmux/r3_eval.sh and r3_final.sh. Then rerun the four table/figure commands at the end of r3_final.sh. If a rerun would take more than about 6 h, say so in RESULTS.md instead.
2. **Update ~/harsh/partzoom_project/RESULTS.md.** Keep the existing round 1–2 content and structure, and fix anything the new results make wrong.
   - Add a clearly headed section **"Round 3: 100 test scenes and lr 1e-5"** with:
     - the 100-scene table;
     - the paired tests against the released checkpoint and between matched pairs;
     - the lr 1e-5 pair on the 50-scene main table, plus its offline errors.
   - Update the **key findings** at the top. Say plainly whether any configuration now beats the released 3DDA checkpoint, and whether the methods beat their matched baseline, with p-values, on the 100-scene set.
   - Mention that episodes 0–49 are shared between the 50- and 100-scene results.
   - Point out the multiple-comparison caveat where relevant.
   - Take every number from the tables or the raw jsonl/json via small scripts; never invent one.
3. Update the status line in ~/harsh/partzoom_project/README.md, and the variant table if the lr-1e-5 variants are missing from it.
4. In ~/harsh/partzoom_project, `git add -A` and `git commit` with a short message ending with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. **Do not push**; the user pushes themselves. Before committing, check that `git status` shows no files larger than 50 MB (checkpoints are gitignored).
5. Add one or two lines on the round-3 outcome to the memory file ~/.claude/projects/-home-ubuntu/memory/partzoom-project.md.
6. Append a short, dated summary to ~/3d_diffuser_actor/logs_partzoom/autopilot.log.

Do not publish anything externally, do not delete results or checkpoints, and do not start new experiments beyond rerunning missing pieces.
