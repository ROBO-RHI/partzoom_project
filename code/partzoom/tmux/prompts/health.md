You are running unattended (cron, headless) on the user's AWS GPU machine. The user is away and has authorised you to keep their experiment pipeline running. Nobody will answer questions, so act on your own judgement and be conservative.

Context: the PartZoom project (see ~/.claude/projects/-home-ubuntu/memory/partzoom-project.md). It fine-tunes and evaluates 3D Diffuser Actor variants. Code is in ~/3d_diffuser_actor/partzoom/ and the deliverable folder is ~/partzoom_project. Jobs run in the tmux session `partzoom`. Its windows and scripts are in ~/3d_diffuser_actor/partzoom/tmux/:
- train_queue.sh: trains both_lr2e5, zoom_object and zoom_random, then marks TRAINQ_DONE.
- eval_worker.sh 1/2: rollout evals; jobs are claimed via logs_partzoom/claims/<name> directories.
- offline.sh: offline evals, sweeps and benchmark, then make_package.py, then marks PACKAGE_DONE.
The timeline is ~/3d_diffuser_actor/logs_partzoom/pipeline.log. `source ~/3d_diffuser_actor/partzoom/env.sh` sets up the Python env.

You were started because no log or result file has been written for over 100 minutes, so something may be stuck or dead. Do this:
1. Diagnose. Read pipeline.log, `tmux list-windows -t partzoom`, `tmux capture-pane -p -t partzoom:<window>` for each window, the newest files in logs_partzoom/, running processes (`ps`), `nvidia-smi` and `free -g`. Look for a Traceback, OOM, CUDA or cuDNN error, a hung CoppeliaSim/xvfb, a dead tmux session (for example after a reboot), or a queue that is waiting on a marker that will never come.
2. Fix it with the smallest change that gets the queue moving again, keeping every experimental setting the same: the same flags and seeds, 50 test episodes, max_steps 12.
   - A crashed rollout eval resumes: eval_rollouts.py skips episodes already in its .jsonl.
   - A crashed training run should be restarted from scratch with the identical command in train_queue.sh, after removing its partial train_logs/partzoom/<name> directory.
   - If a claim directory is held by a dead worker, remove it so the job can be claimed again.
   - If the tmux session is gone, recreate it with the same windows and scripts (status, train, eval1, eval2, offline). The scripts skip finished work using the markers in pipeline.log.
   - Never delete finished results (eval_logs/, finished checkpoints, pipeline.log lines).
3. If everything is in fact healthy (for example one long step that legitimately writes nothing), change nothing.
4. Append a short, dated note describing what you found and did to ~/3d_diffuser_actor/logs_partzoom/autopilot.log.

Do not publish anything externally, do not touch other projects, and do not edit RESULTS.md (the finalize step does that).
