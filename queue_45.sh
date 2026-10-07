#!/bin/sh
# Waits until GPUs 4 and 5 have been idle (under 1 GB used) for five minutes,
# then runs the prompt-robustness and random-retrieval checks there.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
idle=0
while [ "$idle" -lt 5 ]; do
  sleep 60
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | sort -n | tail -1)
  if [ "$used" -lt 1000 ]; then idle=$((idle + 1)); else idle=0; fi
done
G="--gpus 4,5"
(
  $PY run_grid.py --datasets yahoo-answers,go-emotions $G --seeds 42 --shots 1,4 --prompt alt --methods fewshot,cicle
  $PY run_grid.py --datasets yahoo-answers,go-emotions $G --seeds 42 --shots 1,4 --retrieval random --methods fewshot,cicle
) > logs_gpu45.out 2>&1
