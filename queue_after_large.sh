#!/bin/sh
# Launched 2026-10-06. Once the larger-model stage of queue_next.sh is done,
# GPUs 0-3 run, in order: Ohsumed Per-Class (k=1, one seed); marginal vs
# class-conditional conformal calibration; narrowing without examples (k=0);
# oracle narrowing; the labelled-pool-size sweep with its baselines; and
# finally the two missing seeds of the 32B model. GPUs 4-7 are left alone.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
G="--gpus 0,1,2,3"
S="--seeds 42,43,44"
while pgrep -f "^/bin/sh ./queue_next.sh" > /dev/null; do sleep 120; done

$PY run_grid.py --datasets ohsumed $G --seeds 42 --shots 1 --variants pc --methods fewshot,cicle > logs_ohsumed_pc.out 2>&1

# marginal conformal calibration, on balanced and imbalanced Yahoo / SST-5
for R in 1 10 100; do
  $PY run_grid.py --datasets yahoo-answers,sst $G $S --shots 1,4 --imbalance $R --methods marginal
done > logs_marginal.out 2>&1

# the candidate set alone, no retrieved examples
$PY run_grid.py $G $S --shots 0 --variants fixed --methods cicle > logs_k0.out 2>&1
$PY run_grid.py --datasets ohsumed $G $S --shots 0 --variants fixed --methods cicle >> logs_k0.out 2>&1
$PY run_grid.py --datasets yahoo-answers,go-emotions $G $S --shots 0 --variants fixed --relabel --methods cicle >> logs_k0.out 2>&1

# oracle narrowing at the conformal set size
$PY run_grid.py $G $S --shots 1,4 --methods oracle > logs_oracle.out 2>&1
$PY run_grid.py --datasets yahoo-answers $G $S --shots 1,4 --imbalance 100 --methods oracle >> logs_oracle.out 2>&1

# labelled-pool size, with the classifier-only and fine-tuned baselines
for N in 250 500 1000; do
  $PY run_grid.py --datasets yahoo-answers,sst $G $S --shots 1,4 --n-train $N --methods fewshot,cicle
  CUDA_VISIBLE_DEVICES="" $PY baselines.py --datasets yahoo-answers,sst --n-train $N
  $PY baselines.py --datasets yahoo-answers,sst --n-train $N --finetune FacebookAI/roberta-base --gpu 0
done > logs_poolsize.out 2>&1

$PY run_grid.py --models qwen-2.5-32b --gpus 0+1+2+3 --seeds 43,44 --shots 1,4 --methods zeroshot,fewshot,cicle >> logs_large_32b.out 2>&1
