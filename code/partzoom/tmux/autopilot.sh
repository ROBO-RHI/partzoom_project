#!/bin/bash
# Cron-driven autopilot (every 30 min). Cheap shell checks; calls headless
# Claude only when (a) the pipeline finished -> finalize the write-up, or
# (b) nothing has written a log/result for STALL_MIN minutes -> diagnose/fix.
# Log: ~/3d_diffuser_actor/logs_partzoom/autopilot.log
export HOME=/home/ubuntu
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
R=$HOME/3d_diffuser_actor
L=$R/logs_partzoom
AP=$L/autopilot
mkdir -p $AP
exec 9>$AP/lock
flock -n 9 || exit 0                      # a previous run is still working
STALL_MIN=100
MAX_HEALTH=4
MODEL=claude-opus-5-5
P=$R/partzoom/tmux/prompts
log() { echo "[$(date '+%F %T')] $*" >> $L/autopilot.log; }

[ -f $AP/FINAL_DONE ] && exit 0

run_claude() {  # PROMPT_FILE TAG MAXTURNS
  log "launching claude ($2)"
  cd $HOME
  timeout 4h claude -p "$(cat $1)" --model $MODEL --max-turns $3 \
    --allowedTools "Bash Read Edit Write Glob Grep" \
    > $AP/claude_$2_$(date +%m%d_%H%M).out 2>&1
  log "claude ($2) exited rc=$?"
}

if grep -q "^PACKAGE_DONE" $L/pipeline.log; then
  run_claude $P/finalize.md finalize 300
  touch $AP/FINAL_DONE
  log "finalization done; autopilot will stay idle"
  exit 0
fi

# stall detection: newest write among training/eval/offline outputs
newest=$(find $L $R/eval_logs/partzoom $R/eval_logs/offline $R/train_logs/partzoom \
         -type f -not -path "$AP/*" -not -name autopilot.log -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
age_min=$(( ( $(date +%s) - ${newest%.*} ) / 60 ))
if [ $age_min -lt $STALL_MIN ]; then
  exit 0
fi
n=$(ls $AP/claude_health_* 2>/dev/null | wc -l)
last=$(ls -t $AP/claude_health_* 2>/dev/null | head -1)
if [ $n -ge $MAX_HEALTH ]; then log "stalled ${age_min} min but health budget used ($n)"; exit 0; fi
if [ -n "$last" ] && [ $(( ( $(date +%s) - $(stat -c %Y $last) ) / 60 )) -lt 180 ]; then exit 0; fi
log "no output for ${age_min} min -> health check"
run_claude $P/health.md health 150
