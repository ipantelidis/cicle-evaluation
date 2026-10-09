#!/bin/sh
# Launched 2026-10-09 ~14:50 after the external review. Replaces queue_massive.sh.
# Two strands with disjoint work; GPUs 5-7 are left alone.
#   G03 (GPUs 0-3): MASSIVE core, size-matched narrowing everywhere, the
#                   label-list ablation, MASSIVE renaming, MASSIVE ablations,
#                   larger models on MASSIVE
#   G4  (GPU 4):    MASSIVE fine-tuned baselines, then seeds 45 and 46 for the
#                   central controlled-imbalance experiment
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
S="--seeds 42,43,44"
(
  while pgrep -f "run_grid.py --datasets massive --models llama-3.1-8b,llama-3.2-3b --gpus 4" > /dev/null; do sleep 60; done
  $PY baselines.py --datasets massive --finetune FacebookAI/roberta-base --gpu 4
  $PY baselines.py --datasets massive --finetune FacebookAI/roberta-large --lr 1e-5 --gpu 4
  X="--seeds 45,46 --shots 4 --gpus 4"
  for R in 1 10 100; do
    CUDA_VISIBLE_DEVICES="" $PY baselines.py --datasets yahoo-answers,sst --imbalance $R --seeds 45,46
    $PY run_grid.py --datasets yahoo-answers,sst $X --imbalance $R --methods fewshot,cicle,topk,mass,marginal,massmatch,margmatch
  done
) > logs_rev_G4.out 2>&1 &
(
  while pgrep -f "run_grid.py --datasets ohsumed --models qwen-2.5-32b" > /dev/null; do sleep 60; done
  G="--gpus 0,1,2,3"
  $PY run_grid.py --datasets massive --models ministral-3b,qwen-2.5-3b,mistral-7b-v0.3,qwen-2.5-7b $G $S --methods zeroshot,fewshot,cicle
  while pgrep -f "run_grid.py --datasets massive --models llama-3.1-8b,llama-3.2-3b --gpus 4" > /dev/null; do sleep 60; done
  $PY run_grid.py --datasets massive $G $S --shots 1,4 --methods topk,mass,marginal,oracle
  $PY run_grid.py --datasets massive $G $S --shots 0 --variants fixed --methods cicle
  # size-matched narrowing: the central control requested in review
  M="--shots 1,4 --methods massmatch,margmatch"
  for R in 1 10 100; do $PY run_grid.py --datasets yahoo-answers,sst $G $S --imbalance $R $M; done
  $PY run_grid.py --datasets go-emotions,semeval-18,massive $G $S $M
  $PY run_grid.py --datasets ohsumed $G $S --variants fixed $M
  # the default prompt without the label list, every dataset, seed 42
  N="--seeds 42 --shots 1,4 --prompt nolist --methods fewshot,cicle"
  $PY run_grid.py --datasets yahoo-answers,sst,semeval-18,go-emotions,massive $G $N
  $PY run_grid.py --datasets ohsumed $G --seeds 42 --shots 1,4 --variants fixed --prompt nolist --methods fewshot,cicle
  $PY run_grid.py --datasets ohsumed $G --seeds 42 --shots 1 --variants pc --prompt nolist --methods fewshot,cicle
  # MASSIVE: renaming, then the remaining ablations and checks
  $PY run_grid.py --datasets massive $G $S --shots 1,4 --relabel --methods fewshot,cicle,topk,mass,marginal,oracle
  $PY run_grid.py --datasets massive $G $S --shots 0 --variants fixed --relabel --methods cicle
  $PY run_grid.py --datasets massive $G $S --shots 1,4 --methods cicle --alphas 0.01,0.1,0.2
  $PY run_grid.py --datasets massive $G $S --shots 1,4 --methods fewshot,cicle --embeddings tfidf,contriever
  $PY run_grid.py --datasets massive $G $S --shots 1,4 --methods cicle --classifiers svm
  $PY run_grid.py --datasets massive $G --seeds 42 --shots 1,4 --prompt alt --methods fewshot,cicle
  $PY run_grid.py --datasets massive $G --seeds 42 --shots 1,4 --retrieval random --methods fewshot,cicle
  $PY run_grid.py --datasets massive $G --seeds 42 --shots 1,4 --variants fixed --methods zeroshot,fewshot,cicle --legacy-prompt --out-dir results_legacy
  $PY run_grid.py --datasets massive --models mistral-nemo-2407 --gpus 0+1 $S --shots 1,4 --methods zeroshot,fewshot,cicle &
  $PY run_grid.py --datasets massive --models qwen-2.5-32b --gpus 2+3 $S --shots 1,4 --methods zeroshot,fewshot,cicle &
  wait
) > logs_rev_G03.out 2>&1 &
wait
