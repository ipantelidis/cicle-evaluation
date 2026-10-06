#!/bin/sh
# Launched 2026-10-07 01:45. Waits for the pool-size n=1000 run already going,
# finishes its baselines, then runs cheap additions on GPUs 0-3 before the
# remaining 32B seeds. GPUs 4-7 are left alone.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
G="--gpus 0,1,2,3"
S="--seeds 42,43,44"
while pgrep -f "run_grid.py --datasets yahoo-answers,sst .*--n-train 1000" > /dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES="" $PY baselines.py --datasets yahoo-answers,sst --n-train 1000 >> logs_poolsize.out 2>&1
$PY baselines.py --datasets yahoo-answers,sst --n-train 1000 --finetune FacebookAI/roberta-base --gpu 0 >> logs_poolsize.out 2>&1

# marginal calibration on the naturally imbalanced datasets
$PY run_grid.py --datasets go-emotions $G $S --shots 1,4 --methods marginal > logs_additions.out 2>&1
$PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed --methods marginal >> logs_additions.out 2>&1
# label renaming on the two remaining short-text datasets
$PY run_grid.py --datasets sst,semeval-18 $G $S --shots 1,4 --relabel --methods fewshot,cicle >> logs_additions.out 2>&1
# Ohsumed Per-Class k=1, remaining seeds
$PY run_grid.py --datasets ohsumed $G --seeds 43,44 --shots 1 --variants pc --methods fewshot,cicle >> logs_additions.out 2>&1

# alpha sweep on the four models the ablation did not cover
$PY run_grid.py --models ministral-3b,qwen-2.5-3b,mistral-7b-v0.3,qwen-2.5-7b $G $S --shots 1,4 --methods cicle --alphas 0.01,0.1,0.2 >> logs_additions.out 2>&1

$PY run_grid.py --models qwen-2.5-32b --gpus 0+1+2+3 --seeds 43,44 --shots 1,4 --methods zeroshot,fewshot,cicle >> logs_large_32b.out 2>&1
