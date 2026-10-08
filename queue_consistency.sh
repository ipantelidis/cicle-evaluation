#!/bin/sh
# Launched 2026-10-08 ~03:00. Brings every experiment to the same scope on
# every dataset (see paper/plan/experiment_design.md). Two strands:
#   GPUs 4,5  after the legacy / RoBERTa-large runs now on them
#   GPUs 0-3  after the 32B seeds now running there
# Ohsumed uses the Fixed variant only (abstracts are too long for Per-Class
# beyond k=1); everything else matches the other datasets.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
S="--seeds 42,43,44"
FOUR="--datasets yahoo-answers,sst,semeval-18,go-emotions"
OTHER="--models ministral-3b,qwen-2.5-3b,mistral-7b-v0.3,qwen-2.5-7b"
LLAMA="--models llama-3.1-8b,llama-3.2-3b"

(
  while pgrep -f "^/bin/sh ./queue_doubts.sh" > /dev/null; do sleep 60; done
  G="--gpus 4,5"
  # 1. Ohsumed, same scope as the other datasets (Fixed only)
  $PY run_grid.py --datasets ohsumed $G $S --shots 2,8 --variants fixed --methods fewshot,cicle
  $PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed --methods oracle
  $PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed --relabel --methods fewshot,cicle,topk,mass,marginal,oracle
  $PY run_grid.py --datasets ohsumed $G $S --shots 0 --variants fixed --relabel --methods cicle
  $PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed --methods cicle --alphas 0.01,0.1,0.2
  $PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed $LLAMA --methods fewshot,cicle --embeddings tfidf,contriever
  $PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed $LLAMA --methods cicle --classifiers svm
  # 2. relabelled variants: every narrowing rule and k=0, on all four short-text datasets
  $PY run_grid.py --datasets sst,semeval-18 $G $S --shots 1,4 --relabel --methods topk,mass
  $PY run_grid.py $FOUR $G $S --shots 1,4 --relabel --methods marginal,oracle
  $PY run_grid.py --datasets sst,semeval-18 $G $S --shots 0 --variants fixed --relabel --methods cicle
  # 3. RoBERTa-large on the imbalance and pool-size variants
  unset HF_HUB_OFFLINE
  for R in 10 100; do $PY baselines.py --datasets yahoo-answers,sst --imbalance $R --finetune FacebookAI/roberta-large --gpu 4; done
  for N in 250 500 1000; do $PY baselines.py --datasets yahoo-answers,sst --n-train $N --finetune FacebookAI/roberta-large --gpu 5; done
  export HF_HUB_OFFLINE=1
  # 4. prompt-robustness and random-retrieval checks on the remaining datasets (one seed, as before)
  $PY run_grid.py --datasets sst,semeval-18 $G --seeds 42 --shots 1,4 --prompt alt --methods fewshot,cicle
  $PY run_grid.py --datasets ohsumed $G --seeds 42 --shots 1,4 --variants fixed --prompt alt --methods fewshot,cicle
  $PY run_grid.py --datasets sst,semeval-18 $G --seeds 42 --shots 1,4 --retrieval random --methods fewshot,cicle
  $PY run_grid.py --datasets ohsumed $G --seeds 42 --shots 1,4 --variants fixed --retrieval random --methods fewshot,cicle
) > logs_consistency_45.out 2>&1 &

(
  while pgrep -f "run_grid.py --models qwen-2.5-32b" > /dev/null; do sleep 60; done
  G="--gpus 0,1,2,3"
  # 5. embedding and classifier ablations on the four models that lacked them
  $PY run_grid.py $FOUR $G $S --shots 1,4 $OTHER --methods fewshot,cicle --embeddings tfidf,contriever
  $PY run_grid.py $FOUR $G $S --shots 1,4 $OTHER --methods cicle --classifiers svm
  $PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed $OTHER --methods fewshot,cicle --embeddings tfidf,contriever
  $PY run_grid.py --datasets ohsumed $G $S --shots 1,4 --variants fixed $OTHER --methods cicle --classifiers svm
  # 6. the larger models on Ohsumed (Fixed), three seeds like everywhere else
  $PY run_grid.py --datasets ohsumed --models mistral-nemo-2407 --gpus 0+1 $S --shots 1,4 --variants fixed --methods zeroshot,fewshot,cicle &
  $PY run_grid.py --datasets ohsumed --models qwen-2.5-32b --gpus 2+3 $S --shots 1,4 --variants fixed --methods zeroshot,fewshot,cicle &
  wait
) > logs_consistency_03.out 2>&1 &
wait
