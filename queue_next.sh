#!/bin/sh
# Experiment queue (re)launched 2026-10-05 night. Every stage skips results
# that already exist, so the script is safe to restart. GPUs 6 and 7 are
# deliberately left free.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
S="--seeds 42,43,44 --shots 1,4"
alive() { pgrep -f "$1" > /dev/null; }

# A: SST-5 controlled imbalance on GPUs 0,1,4,5 (waits for any run already going)
(
  while alive "run_grid.py --datasets sst "; do sleep 30; done
  for R in 10 100; do
    $PY run_grid.py --datasets sst --gpus 0,1,4,5 $S --imbalance $R --methods fewshot,cicle,topk,mass
    CUDA_VISIBLE_DEVICES="" $PY baselines.py --datasets sst --imbalance $R
  done
) >> logs_sst_imbalance.out 2>&1 &

# B: SVM ablation on GPUs 2,3 once the embedding ablation has released them
(
  while alive "run_grid.py --models llama-3.1-8b,llama-3.2-3b "; do sleep 60; done
  $PY run_grid.py --models llama-3.1-8b,llama-3.2-3b --gpus 2,3 $S --methods cicle --classifiers svm
) >> logs_ablations.out 2>&1 &
wait
while alive "baselines.py --finetune"; do sleep 60; done   # shares GPU 5

# C: label renaming (Yahoo, GoEmotions), then Ohsumed, small models, GPUs 0-5
$PY run_grid.py --datasets yahoo-answers,go-emotions --gpus 0,1,2,3,4,5 $S --relabel \
    --methods fewshot,cicle > logs_relabel.out 2>&1
# Ohsumed abstracts are long, so only the Fixed variant is run (a Per-Class
# prompt would hold 23+ abstracts)
$PY run_grid.py --datasets ohsumed --gpus 0,1,2,3,4,5 $S --variants fixed \
    --methods zeroshot,fewshot,cicle,topk,mass > logs_ohsumed.out 2>&1

# D: larger models, zero-shot / few-shot / CICLe (32B on four GPUs, 12B on two).
# The 32B model runs one seed first; more seeds only if time allows.
$PY run_grid.py --models qwen-2.5-32b --gpus 0+1+2+3 --seeds 42 --shots 1,4 --methods zeroshot,fewshot,cicle > logs_large_32b.out 2>&1 &
$PY run_grid.py --models mistral-nemo-2407 --gpus 4+5 $S --methods zeroshot,fewshot,cicle > logs_large_12b.out 2>&1 &
wait
