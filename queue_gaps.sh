#!/bin/sh
# Launched 2026-10-07 13:00. Three strands:
#  A (GPUs 1,3 now): remaining oracle variants, top-k / mass on relabelled data
#  B (GPU 5 now):    prompt-robustness and random-retrieval checks
#  C (GPUs 0-3):     the 32B seeds, once the alpha sweep and strand A are done,
#                    then whatever strand B has not finished (skip-existing)
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
S="--seeds 42,43,44"
(
  G="--gpus 1,3"
  $PY run_grid.py --datasets yahoo-answers,sst $G $S --shots 1,4 --imbalance 10 --methods oracle
  $PY run_grid.py --datasets sst $G $S --shots 1,4 --imbalance 100 --methods oracle
  $PY run_grid.py --datasets yahoo-answers,go-emotions $G $S --shots 1,4 --relabel --methods topk,mass
) > logs_gaps.out 2>&1 &
(
  G="--gpus 5"
  $PY run_grid.py --datasets yahoo-answers,go-emotions $G --seeds 42 --shots 1,4 --prompt alt --methods fewshot,cicle
  $PY run_grid.py --datasets yahoo-answers,go-emotions $G --seeds 42 --shots 1,4 --retrieval random --methods fewshot,cicle
) > logs_gpu45.out 2>&1 &
while pgrep -f "run_grid.py --models ministral-3b,qwen-2.5-3b,mistral-7b-v0.3,qwen-2.5-7b" > /dev/null; do sleep 60; done
while pgrep -f "run_grid.py --datasets .*--gpus 1,3 " > /dev/null; do sleep 60; done
$PY run_grid.py --models qwen-2.5-32b --gpus 0+1+2+3 --seeds 43,44 --shots 1,4 --methods zeroshot,fewshot,cicle >> logs_large_32b.out 2>&1
wait
G="--gpus 0,1,2,3"
$PY run_grid.py --datasets yahoo-answers,go-emotions $G --seeds 42 --shots 1,4 --prompt alt --methods fewshot,cicle >> logs_gaps.out 2>&1
$PY run_grid.py --datasets yahoo-answers,go-emotions $G --seeds 42 --shots 1,4 --retrieval random --methods fewshot,cicle >> logs_gaps.out 2>&1
