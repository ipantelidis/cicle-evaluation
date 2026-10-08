#!/bin/sh
# Launched 2026-10-08 02:20 on the idle GPUs 4 and 5.
#  GPU 4: the original (legacy) prompt protocol, one seed, every dataset, so the
#         paper can quantify the invalid-output artefact inside the same framework
#  GPU 5: a stronger supervised baseline (RoBERTa-large) on the five datasets
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
(
  $PY run_grid.py --datasets yahoo-answers,sst,semeval-18,go-emotions,ohsumed --gpus 4 --seeds 42 \
      --shots 1,4 --variants fixed --methods zeroshot,fewshot,cicle --legacy-prompt --out-dir results_legacy
) > logs_legacy.out 2>&1 &
(
  unset HF_HUB_OFFLINE
  $PY baselines.py --finetune FacebookAI/roberta-large --gpu 5
  $PY baselines.py --finetune FacebookAI/roberta-large --gpu 5 --datasets ohsumed
) > logs_roberta_large.out 2>&1 &
wait
