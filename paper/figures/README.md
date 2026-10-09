# Figures and tables

Everything in this directory is produced by `make_figures.py` from the per-instance result
files `results/<variant>/seed-<s>/*.json`. Nothing here is edited by hand except this README.

## Running

```
.venv/bin/python paper/figures/make_figures.py --all              # everything, 5000 bootstrap resamples (~8 min)
.venv/bin/python paper/figures/make_figures.py --all --quick      # 500 resamples, for development
.venv/bin/python paper/figures/make_figures.py --only fig2,tab2   # a subset
.venv/bin/python paper/figures/make_figures.py --list             # the step names
```

Outputs: PDFs in `paper/figures/`, LaTeX table bodies (a `tabular` only, booktabs rules, no
`table` environment or caption) in `paper/figures/tables/`, PNG previews in
`cache/figures/preview/`, and `paper/figures/NUMBERS.md` with every number the figures and
tables show (regenerated on every run from a store in `cache/figures/numbers.json`, so a
`--only` run updates only its own sections).

The first run reads all result files (about 25 s) and caches a compact form of each run in
`cache/figures/<variant>.pkl` (gold and prediction codes, gold-in-set flags, candidate sets as a
packed bit matrix, prompt tokens, metrics, unparseable raw outputs). Later runs re-read only files
that are new or changed (mtime/size), so pending experiments are picked up automatically.
Dataset metadata that needs `experiment.Data` (class counts under imbalance, the nonsense-word
map, the test texts for the qualitative table) is cached in `cache/figures/data_meta.json`;
it is computed on CPU from the local Hugging Face cache with `HF_HUB_OFFLINE=1` and never loads
an LLM.

Missing runs never enter a number: a (models x seeds[x k]) grid with missing cells is treated as
pending, the slot is left empty (`--` or `[pending]` in tables, no mark in figures) and listed on
stdout and at the top of `NUMBERS.md`. The same command therefore works now and after the
remaining runs land. Still-empty slots at the time of writing (see the top of `NUMBERS.md` for
the live list): marginal CP on SemEval-18 and on the renamed variants (Figure 2/C1 and Tables
C1/C2 fill in per variant), oracle on the renamed variants, k = 0 on SST-5/SemEval-18 renamed,
random retrieval on SST-5, SemEval-18 and Ohsumed (Table R), the 12B/32B models on Ohsumed
(Table F1).

## Statistics

`figdata.py` holds the data layer and the statistics. Macro-F1 is `analyze.macro_f1` on integer
codes (over the classes present in the (resampled) test sample, invalid outputs coded as an extra
class), identical to the stored `metrics.macro_f1` to 1e-13. The paired bootstrap follows
`analyze.paired_test`: one resample of the 1,000 test indices per seed, applied to every cell
(model, k, variant) that shares that seed and to both methods; the statistic is the mean over
cells of the macro-F1 difference; 95% CI = 2.5/97.5 percentiles; two-sided p = share of
resamples on the other side of zero. The implementation is vectorised and the per-run bootstrap
vectors are computed once per process under a fixed resample matrix per (seed, B), so every test
in a build uses the same resamples (the CI of a given comparison is reproducible across runs).
Figure 1's error bars use the same resamples with the unpaired statistic (mean over 18 runs).
`--quick` uses B = 500; everything in the paper should be built with the default B = 5000.

Checked against `paper/plan/analysis_6models_permodel_bootstrap200.txt` (A6, 200 resamples):
all means agree exactly; CIs differ in the second decimal as expected from 200 vs 5000
resamples (e.g. Yahoo PC +1.40 [+1.04, +1.76] here vs [+1.03, +1.76] in A6).

## Mapping and aggregation

Reference setting everywhere unless stated: MiniLM embeddings, logistic regression, alpha = 0.05,
six small models (`SMALL`: Llama-3.2-3B, Ministral-3B, Qwen2.5-3B, Mistral-7B, Qwen2.5-7B,
Llama-3.1-8B), seeds 42/43/44. "All k" = {1, 2, 4, 8} on Yahoo, SST-5, SemEval-18, GoEmotions
and {1, 4} elsewhere; Ohsumed Per-Class is k = 1 only. Top-k, probability mass, marginal and
oracle runs exist at k in {1, 4} only.

| Step | Function | Output | Aggregation |
|---|---|---|---|
| Table 2 | `tab2` | `tables/main_results.tex` | Rows 1-2: mean over 3 seeds of `base-minilm-lr` / `finetuned-roberta-base`. Rows 3-7: mean macro-F1 and mean prompt tokens over 6 models x 3 seeds at k = 4 (Ohsumed PC: k = 1, marked). Δ rows: paired Δ CICLe − few-shot pooled over all k: 72 pairs; Ohsumed Fixed 36, Ohsumed PC 18. k = 0 row appears when `cicle` runs with k = 0 exist. |
| Figure 1 | `fig1` | `f1_vs_tokens.pdf` | One point per (method, variant, k): mean macro-F1 and mean prompt tokens over 18 runs; vertical bar = 95% bootstrap CI of the 18-run mean over test instances (unpaired). Hollow = Fixed, filled = Per-Class, star = zero-shot. Dotted = MiniLM+LR, dash-dot = RoBERTa (3-seed means). `k` annotated at the ends of the CICLe PC line. |
| Figure 2 | `fig2` | `narrowing_imbalance.pdf` | Per-Class, rows Yahoo / SST-5, x = pool imbalance 1/10/100. (a) paired Δ CICLe − {few-shot, top-m, prob. mass, marginal CP} over k in {1, 4}: 36 pairs per point (the 1x point is recomputed on k in {1, 4}, not all k); dashed black = oracle − few-shot, the ceiling for any narrowing rule at the conformal budget. (b) coverage and mean set size of the candidate sets of CICLe, top-m, mass, marginal CP and oracle, one run per seed (sets are identical across models, k and variants; verified), mean over 3 seeds; marker size grows with imbalance, 1x and 100x annotated on CICLe. (c) per-class coverage at 100x by class-frequency rank in the labelled pool (rank from `Data(ds, seed, imbalance=100).y_train`, aggregated by rank because the class order is a seed-dependent permutation), mean over 3 seeds. (d) per-class F1 by rank, PC k = 4, all six methods, mean over 6 models x 3 seeds (the "inset" of the spec, drawn as a fourth column). |
| Table 3 | `tab3` | `tables/relabel.tex` | Five datasets (Ohsumed Fixed only) x {original, nonsense words} x k in {0, 1, 4}: mean over 18 runs; k = 0 is CICLe with the candidate set and no examples (CICLe Fixed column). Δ columns: CICLe − few-shot paired over k in {1, 4}, 36 pairs, for the original and the renamed labels on the same k grid (uses `\multirow`). |
| Table 4 | `tab4` | `tables/baselines.tex` | Columns Yahoo 1x/10x/100x, SST-5 1x/10x/100x, SemEval-18, GoEmotions, Ohsumed. Supervised rows (MiniLM+LR, RoBERTa-base, RoBERTa-large) 3-seed means (per-seed values in NUMBERS.md); zero-shot 1x only; few-shot / CICLe PC k = 4 (Ohsumed: Fixed, marked F) 18-run means; best pipeline = argmax over (method, variant, k) of the 18-run mean, name in small type. Bold = best per column. Pool sizes are in Table P. |
| Figure 3 | `fig3` | `alpha.pdf` | Llama-3.1-8B + Llama-3.2-3B, both variants, k in {1, 4}, 3 seeds = 24 cells per point. Top: paired Δ CICLe(alpha) − few-shot with CI (one test pooling Fixed and PC). Middle: mean empirical coverage of the 24 CICLe runs with the 1 − alpha line. Bottom: share of test instances answered without an LLM call (1 − `llm_call_rate`). Dataset colours/markers are specific to this figure. |
| Tables B | `tabB` | `tables/grid_<variant>.tex` (14) | Method x variant x k grid: runs, macro-F1, accuracy, invalid %, mean shots, mean prompt tokens; means over runs of the six small models (base row = MiniLM+LR only). Rows with fewer than 18 runs are incomplete configurations. |
| Table C1 | `tabC1` | `tables/narrowing_all.tex` | 14 variants (incl. the five renamed sets) x {Fixed, PC} x comparator {few-shot, top-m, prob. mass, marginal CP, oracle}: paired Δ, CI, (n). vs few-shot pools all k, the other comparators k in {1, 4}. |
| Table C2 | `tabC2` | `tables/set_stats.tex` | Per variant and narrowing method: coverage, mean set size, singleton rate, and min over classes of per-class coverage, from one run per seed, mean over seeds. |
| Figure C1 | `figC1` | `narrowing_imbalance_fixed.pdf` | Figure 2 with the Fixed variant (hollow markers). |
| Table D1 | `tabD1` | `tables/per_model.tex` | Per model: paired Δ CICLe − few-shot over all k (12 pairs; Ohsumed 6) per dataset x variant. |
| Table E1 | `tabE1` | `tables/ablation.tex` | Llama-3.1-8B + Llama-3.2-3B, both variants, k in {1, 4}, 3 seeds (24 cells): rows contriever/LR, minilm/LR, minilm/SVM, tfidf/LR at alpha 0.05 and minilm/LR at alpha 0.01/0.10/0.20; CICLe mean, few-shot mean (same embedding), paired Δ with CI, coverage, set size, no-LLM-call rate. |
| Table F1 | `tabF1` | `tables/large_models.tex` | Mistral-Nemo-12B and Qwen2.5-32B on the five datasets: zero-shot, few-shot / CICLe at Fixed and PC, k in {1, 4}, mean over the seeds listed in the row; Δ per variant over k in {1, 4} paired within that model (n cells shown; partial seed sets allowed here). Plus best 3B + CICLe PC k = 4 vs Qwen2.5-32B zero-shot on seed 42. |
| Table G1 | `tabG1` | `tables/fixes_breaks.tex` | CICLe k = 4 vs few-shot k = 4 (same variant), 18,000 instance pairs: % fixed / broken / both right / both wrong; split by gold-in-set and by whether the few-shot answer was inside the conformal set (share of pairs, then fixed/broken within the subset). |
| Table G2 | `tabG2` | `tables/invalid.tex`, `tables/invalid_raw.tex` | Per model x dataset: mean (max) invalid-output rate over (variant, k, seed) runs of zero-shot / few-shot / CICLe; plus the eight most frequent unparseable raw outputs per model (pooled over the five main datasets; contain emoji, so compile with a Unicode-capable engine or replace). |
| Table G3 | `tabG3` | `tables/examples.tex` | Yahoo and GoEmotions, Llama-3.1-8B, seed 42, PC k = 4: three instances per case ("fixed: few-shot answered outside the set", "broken: gold outside the set", "broken: switched inside the set") sampled with `default_rng(0)`; texts truncated to 140 characters. |
| Table H1 | `tabH1` | `tables/imbalance_counts.tex`, `tables/relabel_words.tex` | Class order and training counts per seed under 10x and 100x (Yahoo, SST-5), and the nonsense-word map. |
| Table P | `tabP` | `tables/pool_size.tex` | Yahoo and SST-5, labelled pool 250/500/1,000/2,000 (results/<ds>-n<N>, 2,000 = the main runs): MiniLM+LR, RoBERTa-base, RoBERTa-large (3-seed means, per-seed in NUMBERS.md), few-shot PC k = 4, CICLe PC k = 4 (18-run means), best pipeline (k in {1,4} below 2,000). Bold = best per row. |
| Figure P | `figP` | `pool_size.pdf` | The same five series against pool size (log x), one panel per dataset, zero-shot as a dotted reference line. Column width. |
| Table L | `tabL` | `tables/legacy.tex` | Per dataset x small model, seed 42, Fixed retrieval, k in {1, 4}: invalid-output rate and macro-F1 (zero-shot / few-shot / CICLe, FS and CICLe averaged over k) under the legacy protocol (`results_legacy/`, `legacy_prompt: true` files, loaded through the `legacy:<dataset>` variant tag) and the corrected one (`results/`, same configs), and the per-model paired Δ CICLe − few-shot under both (2 cells x 1,000 instances). |
| Table R | `tabR` | `tables/robustness.tex` | Per dataset, conditions default prompt (results/<ds>), alternative prompt (results/<ds>-altprompt) and random retrieval (results/<ds>-random), all seed 42, six small models, k in {1, 4}: few-shot and CICLe means and the paired Δ (12 cells) for Fixed and Per-Class. Missing conditions print `[pending]`. |

Table 1 (datasets) is not generated here.

## Style

Matplotlib only, PDF with embedded TrueType (`pdf.fonttype 42`), DejaVu Sans, no titles inside
the figures beyond panel letters and dataset names, fonts >= 7 pt, full width 6.5 in or column
width 3.3 in. One colour per method in every figure, from the dataviz reference palette and
checked for colour-vision separation (OKLab Delta E under protan/deutan/tritan simulation):
few-shot grey `#6b6a66`, CICLe blue `#2a78d6`, top-m orange `#eb6834`, probability mass aqua
`#1baf7a`, marginal CP violet `#4a3aa7` (dashed, triangle), oracle black dashed with x marks,
zero-shot black star, supervised baselines black dotted (MiniLM+LR) and dash-dot (RoBERTa).
Fixed = hollow markers, Per-Class = filled. Error bars are always bootstrap CIs over test
instances, never the spread over hyperparameters.
