#!/bin/sh
# Launched 2026-10-09 ~14:15. MASSIVE (59 intents) with the same scope as the
# other short-text datasets. GPU 4 starts at once; GPUs 0-3 join when the 32B
# Ohsumed run there ends. GPUs 5-7 are left alone.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
D="--datasets massive"
S="--seeds 42,43,44"
CUDA_VISIBLE_DEVICES="" $PY baselines.py --datasets massive > logs_massive_base.out 2>&1 &
(
  # main grid for the two Llama models on GPU 4 while GPUs 0-3 are busy
  $PY run_grid.py $D --models llama-3.1-8b,llama-3.2-3b --gpus 4 $S --methods zeroshot,fewshot,cicle
) > logs_massive_A.out 2>&1 &
while pgrep -f "run_grid.py --datasets ohsumed --models qwen-2.5-32b" > /dev/null; do sleep 60; done
G="--gpus 0,1,2,3"
$PY run_grid.py $D --models ministral-3b,qwen-2.5-3b,mistral-7b-v0.3,qwen-2.5-7b $G $S --methods zeroshot,fewshot,cicle > logs_massive_B.out 2>&1
wait
G="--gpus 0,1,2,3,4"
(
  $PY run_grid.py $D $G $S --shots 1,4 --methods topk,mass,marginal,oracle
  $PY run_grid.py $D $G $S --shots 0 --variants fixed --methods cicle
  $PY baselines.py --datasets massive --finetune FacebookAI/roberta-base --gpu 0
  $PY baselines.py --datasets massive --finetune FacebookAI/roberta-large --lr 1e-5 --gpu 0
  $PY run_grid.py $D $G $S --shots 1,4 --relabel --methods fewshot,cicle,topk,mass,marginal,oracle
  $PY run_grid.py $D $G $S --shots 0 --variants fixed --relabel --methods cicle
  $PY run_grid.py $D $G $S --shots 1,4 --methods cicle --alphas 0.01,0.1,0.2
  $PY run_grid.py $D $G $S --shots 1,4 --methods fewshot,cicle --embeddings tfidf,contriever
  $PY run_grid.py $D $G $S --shots 1,4 --methods cicle --classifiers svm
  $PY run_grid.py $D $G --seeds 42 --shots 1,4 --prompt alt --methods fewshot,cicle
  $PY run_grid.py $D $G --seeds 42 --shots 1,4 --retrieval random --methods fewshot,cicle
  $PY run_grid.py $D $G --seeds 42 --shots 1,4 --variants fixed --methods zeroshot,fewshot,cicle --legacy-prompt --out-dir results_legacy
  $PY run_grid.py $D --models mistral-nemo-2407 --gpus 4 $S --shots 1,4 --methods zeroshot,fewshot,cicle &
  $PY run_grid.py $D --models qwen-2.5-32b --gpus 0+1+2+3 $S --shots 1,4 --methods zeroshot,fewshot,cicle &
  wait
) > logs_massive_C.out 2>&1
