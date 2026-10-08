#!/bin/sh
# Launched 2026-10-08 20:45 to use the idle GPUs: the remaining consistency
# rounds are spread over three strands with disjoint work.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
S="--seeds 42,43,44"
FOUR="--datasets yahoo-answers,sst,semeval-18,go-emotions"
OTHER="--models ministral-3b,qwen-2.5-3b,mistral-7b-v0.3,qwen-2.5-7b"
LLAMA="--models llama-3.1-8b,llama-3.2-3b"

# P: GPUs 1-3, free now
(
  G="--gpus 1,2,3"
  $PY run_grid.py --datasets sst,semeval-18 $G $S --shots 1,4 --relabel --methods topk,mass
  $PY run_grid.py $FOUR $G $S --shots 1,4 --relabel --methods marginal,oracle
  $PY run_grid.py --datasets sst,semeval-18 $G $S --shots 0 --variants fixed --relabel --methods cicle
  $PY run_grid.py --datasets sst,semeval-18 $G --seeds 42 --shots 1,4 --prompt alt --methods fewshot,cicle
  $PY run_grid.py --datasets ohsumed $G --seeds 42 --shots 1,4 --variants fixed --prompt alt --methods fewshot,cicle
  $PY run_grid.py --datasets sst,semeval-18 $G --seeds 42 --shots 1,4 --retrieval random --methods fewshot,cicle
  $PY run_grid.py --datasets ohsumed $G --seeds 42 --shots 1,4 --variants fixed --retrieval random --methods fewshot,cicle
) > logs_rebalance_P.out 2>&1 &

# Q: GPU 4 now, then GPUs 4-5 once the Ohsumed alpha round on GPU 5 ends
(
  unset HF_HUB_OFFLINE
  for R in 10 100; do $PY baselines.py --datasets yahoo-answers,sst --imbalance $R --finetune FacebookAI/roberta-large --gpu 4; done
  for N in 250 500 1000; do $PY baselines.py --datasets yahoo-answers,sst --n-train $N --finetune FacebookAI/roberta-large --gpu 4; done
  export HF_HUB_OFFLINE=1
  while pgrep -f "run_grid.py --datasets ohsumed --gpus 4,5" > /dev/null; do sleep 60; done
  G="--gpus 4,5"
  $PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed $LLAMA --methods fewshot,cicle --embeddings tfidf,contriever
  $PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed $LLAMA --methods cicle --classifiers svm
) > logs_rebalance_Q.out 2>&1 &

# R: GPU 0 once the Ohsumed embedding round there ends; the large models when P is done
(
  while pgrep -f "run_grid.py --datasets ohsumed --gpus 0,1,2,3" > /dev/null; do sleep 60; done
  $PY run_grid.py --datasets ohsumed --gpus 0 $S --shots 1,4 --variants fixed $OTHER --methods cicle --classifiers svm
  while pgrep -f "run_grid.py .*--gpus 1,2,3 " > /dev/null; do sleep 60; done
  $PY run_grid.py --datasets ohsumed --models mistral-nemo-2407 --gpus 0+1 $S --shots 1,4 --variants fixed --methods zeroshot,fewshot,cicle &
  $PY run_grid.py --datasets ohsumed --models qwen-2.5-32b --gpus 2+3 $S --shots 1,4 --variants fixed --methods zeroshot,fewshot,cicle &
  wait
) > logs_rebalance_R.out 2>&1 &
wait
