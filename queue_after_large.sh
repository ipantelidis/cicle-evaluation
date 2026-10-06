#!/bin/sh
# Launched 2026-10-06: once the larger-model stage of queue_next.sh is done,
# GPUs 0-3 run (1) Ohsumed Per-Class at k=1 for one seed and (2) the two
# missing seeds of the 32B model. GPUs 4-7 are left alone.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
while pgrep -f "sh ./queue_next.sh" > /dev/null; do sleep 120; done
$PY run_grid.py --datasets ohsumed --gpus 0,1,2,3 --seeds 42 --shots 1 --variants pc \
    --methods fewshot,cicle > logs_ohsumed_pc.out 2>&1
$PY run_grid.py --models qwen-2.5-32b --gpus 0+1+2+3 --seeds 43,44 --shots 1,4 \
    --methods zeroshot,fewshot,cicle >> logs_large_32b.out 2>&1
