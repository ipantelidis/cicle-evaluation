#!/bin/sh
# Launched 2026-10-09 00:15. Remaining work, most important first.
#   P (GPUs 1-4): k=0 on renamed SST/SemEval; prompt-robustness and random
#                 retrieval on SST, SemEval, Ohsumed; then marginal + oracle
#                 on the renamed datasets (consistency only, so last)
#   Q (GPU 5):    Ohsumed Llama SVM ablation after the embedding round there
#   R (GPU 0):    Ohsumed SVM for the other models after the embedding round
#                 there; then the 12B/32B models on Ohsumed once P is done
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
S="--seeds 42,43,44"
FOUR="--datasets yahoo-answers,sst,semeval-18,go-emotions"
OTHER="--models ministral-3b,qwen-2.5-3b,mistral-7b-v0.3,qwen-2.5-7b"
LLAMA="--models llama-3.1-8b,llama-3.2-3b"
(
  G="--gpus 1,2,3,4"
  $PY run_grid.py --datasets sst,semeval-18 $G $S --shots 0 --variants fixed --relabel --methods cicle
  $PY run_grid.py --datasets sst,semeval-18 $G --seeds 42 --shots 1,4 --prompt alt --methods fewshot,cicle
  $PY run_grid.py --datasets ohsumed $G --seeds 42 --shots 1,4 --variants fixed --prompt alt --methods fewshot,cicle
  $PY run_grid.py --datasets sst,semeval-18 $G --seeds 42 --shots 1,4 --retrieval random --methods fewshot,cicle
  $PY run_grid.py --datasets ohsumed $G --seeds 42 --shots 1,4 --variants fixed --retrieval random --methods fewshot,cicle
  $PY run_grid.py $FOUR $G $S --shots 1,4 --relabel --methods marginal,oracle
) > logs_push_P.out 2>&1 &
(
  while pgrep -f "run_grid.py --datasets ohsumed --gpus 4,5" > /dev/null; do sleep 60; done
  $PY run_grid.py --datasets ohsumed --gpus 5 $S --shots 1,4 --variants fixed $LLAMA --methods cicle --classifiers svm
) > logs_push_Q.out 2>&1 &
(
  while pgrep -f "run_grid.py --datasets ohsumed --gpus 0,1,2,3" > /dev/null; do sleep 60; done
  $PY run_grid.py --datasets ohsumed --gpus 0 $S --shots 1,4 --variants fixed $OTHER --methods cicle --classifiers svm
  while pgrep -f "run_grid.py .*--gpus 1,2,3,4 " > /dev/null; do sleep 60; done
  $PY run_grid.py --datasets ohsumed --models mistral-nemo-2407 --gpus 0+1 $S --shots 1,4 --variants fixed --methods zeroshot,fewshot,cicle &
  $PY run_grid.py --datasets ohsumed --models qwen-2.5-32b --gpus 2+3 $S --shots 1,4 --variants fixed --methods zeroshot,fewshot,cicle &
  wait
) > logs_push_R.out 2>&1 &
wait
