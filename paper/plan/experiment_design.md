# Experiment design and the reasoning behind each choice

Written 2026-10-08 for the authors and for the Setup / Appendix text. Every experiment
below is run with the same protocol: 2,000 labelled examples per dataset split into
1,600 for training the base classifier and retrieval, and 400 for conformal calibration
(and epoch selection of the fine-tuned baseline); 1,000 stratified test instances;
three data seeds (42, 43, 44) that change the subsample, the calibration split and the
conformal tie-breaking; macro-F1 over the classes present in the test sample; paired
bootstrap over test instances. Unless stated, every experiment covers all five datasets
and all six small LLMs.

## Fixed choices and why

| Choice | Value | Reason |
|---|---|---|
| Labelled pool | 2,000 | Enough for a usable base classifier on 5–28 classes, small enough that every LLM configuration can be run on 1,000 test instances for six models and three seeds (about 1 GPU-minute per Fixed configuration, up to 17 for Per-Class at k=8). The pool-size sweep (250 to 2,000) shows how results move with this choice. |
| Calibration split | 20% (400) | The original CICLe split. The pool-size sweep also varies it (50 to 400). |
| Test sample | 1,000 stratified | One test sample shared by every method on a seed, which is what makes paired tests over instances possible. Bootstrap intervals quantify its noise (about 1–2 points of macro-F1 for one run). |
| Seeds | 3 | Three subsamples give a seed-level sanity check on top of the instance-level test; more seeds would cost a third of the budget each. |
| Reference setting | MiniLM, logistic regression, alpha = 0.05 | The original CICLe choices. Each is varied in an ablation. |
| k | {1, 2, 4, 8} main grid; {1, 4} elsewhere | The main grid shows the shape of the curve; the gains are visible at k = 1 and 4 and the Per-Class k = 8 runs cost half of all compute, so follow-up experiments use {1, 4}. |
| Variants | Fixed and Per-Class | Per-Class is the original CICLe retrieval rule; Fixed isolates narrowing from the example budget. |
| Ohsumed | Fixed variant; Per-Class only at k = 1 | Abstracts average about 190 words, so a Per-Class prompt at k = 4 would hold about 90 abstracts (about 28k tokens). k = 1 Per-Class (23 abstracts) is run with three seeds. |
| LLMs | six open 3B–8B models from three families; 12B and 32B for the size comparison | Covers families and sizes that fit one 24 GB GPU; the 12B/32B runs answer "does a bigger model change the picture" with three seeds like everything else. The 70B model does not fit the available GPUs. |

## Experiments

| Experiment | Scope | Purpose (the claim it supports) |
|---|---|---|
| Main grid: zero-shot, few-shot, CICLe × Fixed/Per-Class × k | 5 datasets, 6 models, 3 seeds | Claim 2: narrowing gives a small, significant gain and shorter prompts. |
| Narrowing rules: top-k (at the conformal mean set size), probability mass (0.95), marginal conformal, oracle | 5 datasets + every variant below, 6 models, 3 seeds, k ∈ {1,4} | Claim 3: which part of CICLe matters. Top-k holds the budget fixed, mass is adaptive but uncalibrated, marginal removes only the per-class calibration, oracle gives the ceiling at that set size. |
| Controlled imbalance 10× and 100× | Yahoo and SST-5 (the two balanced datasets), all methods, 3 seeds | Claim 3: imbalance of the labelled pool is the manipulated variable; everything else, including the test sample, stays fixed, so results are paired with the 1× runs. Only balanced datasets can be resampled without changing their natural skew; GoEmotions, SemEval-18 and Ohsumed provide the naturally imbalanced cases. |
| Label renaming (nonsense words) | 5 datasets, all narrowing rules and k = 0, 3 seeds | Claim 3: how much of the gain depends on label semantics (the natural version of the old synthetic benchmark, as Reviewer 1 suggested). |
| k = 0 (candidate set, no examples) | 5 datasets and the renamed variants, 3 seeds | Separates the two things CICLe does: shrinking the label list and restricting retrieval. |
| Supervised baselines: MiniLM+LR, TF-IDF, RoBERTa-base, RoBERTa-large | every dataset and variant, 3 seeds | Claim 4: when an LLM pipeline is worth it at all. Two encoder sizes so the comparison is not against a weak baseline only. |
| Pool size 250 / 500 / 1,000 | Yahoo and SST-5, all pipelines and baselines, 3 seeds | Claim 4: the crossover point between fine-tuning and in-context pipelines. Restricted to the two datasets whose pools are balanced, so pool size is the only factor that changes. |
| Alpha ∈ {0.01, 0.05, 0.10, 0.20} | 5 datasets, 6 models, 3 seeds, k ∈ {1,4} | Claim 4: how to set the miscoverage level given the classifier's strength; also where the LLM is skipped. |
| Embedding (TF-IDF, MiniLM, Contriever) and classifier (LR, SVM) | 5 datasets, 6 models, 3 seeds, k ∈ {1,4} | Which components of CICLe matter (Reviewer 3). |
| Larger models 12B and 32B | 5 datasets, zero-shot / few-shot / CICLe, 3 seeds, k ∈ {1,4} | Fair size comparison: both sizes with and without CICLe (Reviewer 3). |
| Prompt robustness: second template, random label order, most similar example last | 5 datasets, 6 models, seed 42, k ∈ {1,4} | Whether the gain depends on one prompt format; the example-order check answers Reviewer 4's recency question. One seed because this is a robustness check of a direction, not an estimate of its size. |
| Random instead of retrieved examples | 5 datasets, 6 models, seed 42, k ∈ {1,4} | How much of few-shot and CICLe comes from retrieval rather than narrowing. One seed, same reason. |
| Legacy protocol (original prompts, 5-token cap, exact matching) | 5 datasets, 6 models, seed 42, Fixed k ∈ {1,4} | Claim 1: the invalid-output artefact and the false model heterogeneity, measured inside the same framework. One seed because the artefact is of the order of 20–60 points of invalid outputs. |

## Not run, and why

- Per-Class on Ohsumed beyond k = 1, and the 70B model: compute (see above).
- Controlled imbalance and pool size on the naturally imbalanced datasets: their pools cannot be rebalanced without changing the task; the natural datasets serve as the uncontrolled cases.
- Multi-label classification: CICLe is defined for single-label prediction; stated in Limitations.
- Configuration selection on held-out data for the LLM pipelines: the paper reports a fixed configuration (Per-Class, k = 4) as the main number and the best configuration as secondary, and says so.
