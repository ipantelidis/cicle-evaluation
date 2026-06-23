# CICLe Evaluation

Code and experiment notebooks accompanying **"Evaluating Conformal In-Context
Learning Across Tasks and Models"** (Ippokratis Pantelidis, Korbinian Randl,
John Pavlopoulos, Aron Henriksson).

## Overview

CICLe combines a cheap base classifier with conformal prediction to decide
*when* an LLM is actually needed for a classification task:

1. A base classifier (logistic regression or SVM, via
   [`crepes`](https://github.com/henrikbostrom/crepes)) produces a
   class-conditional **conformal prediction set** for each input — the
   smallest set of labels guaranteed to contain the true label with
   probability at least 1 − α.
2. If that set contains exactly one label, it is returned directly — no LLM
   call needed.
3. Otherwise, an LLM is given a handful of few-shot demonstrations and asked
   to choose only among the candidate labels in the set.

This repository evaluates that pipeline against zero-shot and standard
few-shot prompting, across:

- **4 core text-classification benchmarks** — Yahoo Answers, SST-5,
  SemEval-18 Emoji, GoEmotions — chosen to span a range of class counts and
  class imbalance.
- **FOL-Reasoning** — a synthetic multi-hop reasoning benchmark built from
  scratch for this paper (`fol-reasoning/generate_dataset.py`), with a
  separate ablation study (`fol-reasoning/ablation/`) that varies class
  count, class imbalance, and passage length independently.
- **6 LLMs** (3B–8B) — Llama-3.1-8B, Llama-3.2-3B, Mistral-7B-v0.3,
  Ministral-3B, Qwen-2.5-7B, Qwen-2.5-3B — plus 3 larger models
  (Llama-3.1-70B, Qwen-2.5-32B, Mistral-Nemo 12B) used only for the
  zero-shot model-size comparison.
- **2 demonstration strategies** — *Fixed* (a static set of k examples) and
  *Per-Class* (k examples drawn from each candidate label).
- **2 embeddings** for demonstration retrieval (Contriever, MiniLM) plus a
  TF-IDF baseline, and **2 base classifiers** (logistic regression, SVM).

## Repository layout

```
yahoo-answers/, sst/, semeval-18/, go-emotions/, fol-reasoning/
    *.ipynb                      one notebook per (model, method, embedding,
                                  classifier, k, variant, alpha) configuration
    results/predictions/*.json   classification report for that notebook
    results/lengths/*.json       prompt / demonstration-count statistics

fol-reasoning/ablation/          controlled variations (class count, class
                                  imbalance, passage length) used in the
                                  appendix ablation study
fol-reasoning/generate_dataset.py, generate_dataset_ablation.py
                                  synthetic dataset construction

run.py             unified runner for executing the experiment notebooks
visualize.py        reproduces every figure from the paper
requirements.txt    pinned Python environment
figures/             output of visualize.py
```

Each notebook is named:

```
{dataset}-{model}-{method}-{embedding}-{classifier}-2.0k-samples-{k}-shots-{variant}-{alpha}-α.ipynb
```

For example, `yahoo-llama-3.1-8b-cicle-contriever-lr-2.0k-samples-2-shots-pc-0.05-α.ipynb`
is CICLe on Yahoo Answers, with Llama-3.1-8B, Contriever embeddings, a
logistic-regression base classifier, k=2 Per-Class demonstrations, at
α=0.05. Zero-shot and standard few-shot notebooks omit whichever fields
don't apply (no embedding/classifier/α for zero-shot; no classifier/α for
few-shot).

## Setup

```bash
pip install -r requirements.txt
```

This installs PyTorch with CUDA 12.8 support directly (see the comment at
the top of `requirements.txt` if your machine needs a different CUDA
build), along with `transformers`, `crepes`, `scikit-learn`, and everything
else needed to run or regenerate the notebooks.

## Running the experiments

`run.py` discovers any dataset directory containing both notebooks and a
`results/predictions/` folder, and executes them in place, skipping
anything that already has a saved result:

```bash
python run.py --list                                          # see what's available
python run.py --datasets yahoo-answers                        # run one dataset
python run.py --datasets yahoo-answers,sst                    # run several
python run.py --datasets go-emotions --filter cicle --gpu 0   # only CICLe notebooks, on GPU 0
python run.py --datasets sst --dry-run                        # preview without executing
python run.py --datasets sst --filter zeroshot --force        # re-run even if results exist
```

Each notebook writes its own `results/predictions/<name>.json`
(classification report) and `results/lengths/<name>.json` (prompt and
demonstration-count statistics) once it finishes.

## Reproducing the figures

`visualize.py` reads directly from every dataset's `results/predictions/`
and `results/lengths/`, and regenerates every figure from the paper into
`figures/`:

```bash
python visualize.py
```

| Figure | Output file(s) |
|---|---|
| CICLe vs. standard few-shot prompting | `comparability_yahoo_answers.png`, `comparability_sst.png`, `comparability_semeval_18.png`, `comparability_go_emotions.png` |
| Class imbalance vs. CICLe's benefit | `imbalance_vs_benefit.png` |
| Which CICLe components matter | `components_by_embedding_and_classifier.png` |
| Accuracy vs. the conformal miscoverage level α | `accuracy_vs_alpha.png` |
| Small model + CICLe vs. zero-shot with a larger model | `small_model_vs_large.png` |
| FOL-Reasoning comparability | `fol_reasoning_comparability.png` |
