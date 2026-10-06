# Figures and tables — specification

Conventions
- Reference setting unless stated: MiniLM embeddings, logistic-regression base classifier,
  alpha = 0.05, six small LLMs (Llama-3.2-3B, Ministral-3B, Qwen-2.5-3B, Mistral-7B-v0.3,
  Qwen-2.5-7B, Llama-3.1-8B), seeds 42/43/44, test sample of 1,000 instances per seed.
- "Paired Δ" = CICLe minus alternative, averaged over all (model, seed, k) cells that share the
  variant; 95% CI and p from the paired bootstrap of `analyze.py` (resamples test instances; same
  resample applied to both methods). Camera-ready: `--bootstrap 5000`; the plan numbers use 200.
- Macro-F1 in percentage points, over the classes present in the test sample (GoEmotions has 27 of
  28 classes present, Ohsumed 23).
- Sources: **A6** `paper/plan/analysis_6models_permodel_bootstrap200.txt`; **A8**
  `paper/reference/analysis_2026-10-06_morning.txt` (only for imb10/imb100/relabel, where only the
  six small models exist); **R** per-instance files `results/<variant>/seed-*/*.json`.
- Significance marks in tables: bold Δ when the 95% CI excludes zero; "n.s." otherwise. No stars.
- Plot style: one colour per method (few-shot grey, CICLe/CP dark, top-k and mass two lighter hues,
  marginal dashed), Fixed = hollow markers, Per-Class = filled. Error bars are always the paired
  bootstrap CI, never the spread over hyperparameters (this is the reversal of the DS2026 figures;
  say so in the first caption).

---

## Main text

### Table 1 — Datasets (Sec. 4.1, column width, 6 rows)
Columns: Dataset | Domain | Classes | Imbalance (max/min class in the full training set) | Mean
input length (tokens) | Variants run.
Rows: SST-5 (sentiment, 5, 2.1x, 24 tok), Yahoo Answers (topic, 10, 1.0x, 51), SemEval-18 emoji
(affect, 20, 8.7x, 21), GoEmotions (emotion, 28, 184.7x, 17), Ohsumed (medical abstracts, 23,
~65x, ~190 words -> recompute in tokens from R `prompt_tokens` of zero-shot runs minus template).
"Variants run" column: Yahoo and SST: 1x/10x/100x controlled imbalance of the labelled pool,
label renaming (Yahoo), [PENDING pool size]; GoEmotions: renaming; Ohsumed: Fixed only.
Footnote: subsample 1,600 train / 400 calibration / 1,000 test, stratified, three seeds;
controlled imbalance uses a geometric class-size schedule with seed-dependent class order and
leaves the test sample untouched (answers R3 "why a subsample, how sampled").
Source: old Table 1 for the first four rows (lengths from the old pipeline — verify), Ohsumed from F
and R.

### Table 2 — Main results at k = 4 (Sec. 5.2, full width)
One block of columns per dataset (Yahoo, SST-5, SemEval-18, GoEmotions, Ohsumed).
Rows:
1. MiniLM + LR (no LLM)            65.4 | 35.5 | 9.7 | 11.2 | 45.2
2. RoBERTa-base fine-tuned          61.8 | 51.2 | 24.5 | 37.0 | 55.4
3. Zero-shot                        55.3 | 40.4 | 12.2 | 24.9 | 35.9
4. Few-shot Fixed                   58.4 | 44.1 | 15.3 | 26.5 | 50.4
5. CICLe Fixed                      58.2 | 45.8 | 15.9 | 26.8 | 51.7
6. Few-shot Per-Class               61.7 | 46.6 | 15.4 | 24.8 | —
7. CICLe Per-Class                  62.7 | 47.5 | 15.7 | 26.0 | [PENDING k=1 only]
Then two Δ rows spanning all k (the headline statistic):
8. Δ Fixed  (CICLe - few-shot, all k, 72 pairs; Ohsumed 36)  -0.19 n.s. | **+1.77** | **+0.52** | **+0.59** | **+1.15**
9. Δ Per-Class                                                 **+1.40** | **+1.07** | **+0.46** | **+0.92** | —
   with the CI in small type under each Δ: [-0.59,+0.17] | [+1.04,+2.49] | [+0.31,+0.71] |
   [+0.24,+1.07] | [+0.60,+1.86] (Fixed) and [+1.03,+1.76] | [+0.36,+1.69] | [+0.27,+0.68] |
   [+0.47,+1.37] (PC).
Each LLM cell carries a second small number: mean prompt tokens (e.g. Yahoo PC k=4: 2,166 few-shot
vs 1,378 CICLe). Rows 3-7 are means over 18 runs (6 models x 3 seeds); rows 1-2 over 3 seeds.
Source: A6 tables (k=4 rows), A6 paired tests. Caption states: same prompt wording, same 1,600
example pool, seeded conformal sets, paired bootstrap over test instances.
Why k=4 and not the mean over k: a single k keeps the token column meaningful; the full k grid is
Appendix B. Note in the caption that Δ rows pool all k.

### Figure 1 — Macro-F1 against prompt length (Sec. 5.2, full width, 5 panels in one row, ~1.5 q)
- x: mean prompt tokens per LLM call (log scale); y: macro-F1 (pp). One panel per dataset.
- Points: every (method, variant, k) at the reference setting, averaged over 6 models x 3 seeds:
  zero-shot (1 point), few-shot Fixed k=1,2,4,8 (hollow grey, connected), CICLe Fixed (hollow dark),
  few-shot PC (filled grey), CICLe PC (filled dark). Ohsumed: Fixed only, k=1,4.
- Horizontal reference lines: MiniLM+LR (dotted) and RoBERTa (dash-dot), labelled at the right
  edge. These lines carry claim 4 visually without a separate figure.
- Error bars: vertical, 95% bootstrap CI of the mean macro-F1 across the 18 runs' test instances
  (same resampling as the paired test but unpaired). Horizontal: none (token means are tight).
- Reading: CICLe PC sits up-and-left of few-shot PC on every dataset (same or better F1 at 11-38%
  fewer tokens); on GoEmotions Fixed beats PC at a sixth of the tokens (26.6 vs 25.9 mean over k);
  on Yahoo the supervised line is above every LLM point.
- Source: A6 tables (macro-F1 and tokens columns); CI needs R (per-run `records`).
- Replaces old Figure 1 (F1 vs k with bar insets) — the k curve is still readable along each
  connected line because tokens grow monotonically with k.

### Figure 2 — How you narrow matters under a long-tailed labelled pool (Sec. 5.3, full width,
2 rows x 3 panels, ~2.5 q)
Rows: Yahoo Answers (top), SST-5 (bottom). Variant: Per-Class (Fixed goes to Appendix Fig. C1).
- **Panel (a) Δ vs imbalance.** x: imbalance ratio of the labelled pool {1, 10, 100} (categorical,
  equally spaced). y: paired Δ (pp) of CICLe minus {few-shot, top-k, probability mass,
  [PENDING marginal CP]}, one line per comparator, 95% CI error bars. n = 36 pairs per point
  (6 models x 3 seeds x k in {1,4}; the 1x point must be recomputed on k in {1,4} only so n matches:
  Yahoo PC few-shot +1.32, top-k +0.76, mass +1.33; see `figure_scripts_todo.md`).
  Values (A8, PC): Yahoo few-shot +1.32/+1.09/+0.89; top-k +0.76/+1.74/+7.44; mass
  +1.33/+1.37/+9.44. SST few-shot +1.14/+1.82/+1.16; top-k +0.32/+2.44/+4.21; mass
  +0.71/+1.85/+10.70. [PENDING oracle: dashed horizontal ceiling per imbalance level.]
- **Panel (b) coverage vs mean set size.** x: mean candidate-set size; y: empirical coverage (%)
  of the gold label; one marker per (method, imbalance), method by colour, imbalance by marker
  size or annotation (1/10/100). Horizontal line at 95% (1 - alpha). Values (A8 candidate-set
  lines): Yahoo CP 95.0/5.67, 94.9/6.21, 96.7/7.73; top-k 93.7/5.67, 90.5/6.00, 81.6/7.67; mass
  98.4/7.84, 96.2/7.45, 74.2/5.88; SST CP 95.4/3.86, 96.5/4.04, 96.0/4.35; top-k 95.5/4.00,
  89.4/4.00, 88.4/4.33; mass 99.2/4.50, 92.2/4.13, 60.0/2.85. [PENDING marginal: 62.9/4.1 at
  Yahoo 100x from F.] Message: only CP stays on the 95% line as the pool skews.
- **Panel (c) coverage by class frequency at 100x.** x: classes ordered by their count in the
  labelled pool (most to least frequent; counts range from ~2000x100/sum to 5 per class); y:
  per-class coverage (%) of the candidate set, one line per method, averaged over 3 seeds (the
  class order differs per seed, so aggregate by frequency *rank*, not by class name). Values exist
  only for seed 42 / Llama-8B so far (R, Yahoo 100x PC k=4: CP 93-100% on every class; top-k 22%
  on the rarest class and 41% on the second rarest; mass 2% / 11% / 62% on the three rarest).
  Second y-axis or small inset: per-class macro-F1 of the LLM pipeline by rank (CICLe vs top-k vs
  mass vs few-shot) — in the seed-42 run the F1 loss is confined to the rare classes (Culture 38.2
  CP vs 27.2 top-k vs 27.5 mass; Family 49.9 vs 38.9 vs 36.8; Politics 55.5 vs 46.7 vs 38.0),
  while the frequent classes are within 1-2 points.
- Source: A8 (a, b); R + `Data(dataset, seed, imbalance=100).y_train` class counts (c).
- Caption must state that the test sample is identical across imbalance levels (only the labelled
  pool is skewed) so the three points of a line are paired.

### Table 3 — Label renaming (Sec. 5.4, column width, 8 data rows)
Columns: Dataset | Labels | k | Few-shot Fixed | CICLe Fixed | Few-shot PC | CICLe PC.
Rows: Yahoo original (k=1: 54.8/54.3/57.9/59.5; k=4: 58.4/58.2/61.7/62.7 — A6) and Yahoo renamed
(k=1: 25.3/34.3/34.1/41.1; k=4: 40.6/44.6/47.9/52.0 — A8); GoEmotions original (k=1:
24.6/25.5/25.0/26.0; k=4: 26.5/26.8/24.8/26.0) and renamed (k=1: 8.0/8.8/10.6/11.7; k=4:
11.8/12.3/14.4/15.3). Final row: paired Δ CICLe - few-shot on renamed labels, 36 pairs: Yahoo
**+6.52** [+5.89,+7.08] Fixed / **+5.62** [+5.02,+6.25] PC; GoEmotions **+0.66** [+0.31,+0.98] /
**+0.98** [+0.65,+1.39].
Caption: every class name replaced by a nonsense word (same mapping for all seeds); the base
classifier is unaffected, so the whole effect is on the LLM's use of the candidate list.
Source: A6 (original), A8 (renamed).
If space forces it into the appendix, keep the Δ row in the prose of 5.4.

### Table 4 — Supervised baselines vs LLM pipelines when the labelled pool is skewed [or small]
(Sec. 5.5, column width; may need full width if pool-size columns arrive)
Columns: Yahoo 1x | 10x | 100x | SST 1x | 10x | 100x | [PENDING n=250 | 500 | 1,000 for both]
| SemEval | GoEmotions | Ohsumed.
Rows (macro-F1, mean over seeds; LLM rows mean over 6 models x 3 seeds):
1. MiniLM + LR: 65.4 | 50.4 | 31.4 | 35.5 | 20.5 | 12.6 | 9.7 | 11.2 | 45.2
2. RoBERTa-base: 61.8 | 58.7 | 41.2 | 51.2 | 44.5 | 31.9 | 24.5 | 37.0 | 55.4
3. Zero-shot: 55.3 | — | — | 40.4 | — | — | 12.2 | 24.9 | 35.9  (zero-shot does not depend on the
   pool; repeat the 1x value in grey or leave a dash — caption says so)
4. Few-shot PC, k=4: 61.7 | 59.4 | 54.6 | 46.6 | 44.6 | 40.8 | 15.4 | 24.8 | (Fixed 50.4)
5. CICLe PC, k=4: 62.7 | 60.3 | 55.4 | 47.5 | 46.5 | 41.9 | 15.7 | 26.0 | (Fixed 51.7)
6. Best LLM pipeline (which): 63.8 CICLe PC k=8 | 60.3 CICLe PC k=4 | 56.0 CICLe PC k=1 | 47.6
   CICLe PC k=2 | 46.5 | 43.4 CICLe PC k=1 | 16.1 CICLe Fixed k=8 | 27.3 CICLe Fixed k=8 | 51.7
Bold the best per column. Source: A6 (1x), A8 (10x/100x), RoBERTa per-seed values from R
(`*-finetuned-roberta-base.json`: Yahoo 10x 58.5/62.5/55.1; 100x 38.9/44.0/40.8; SST 10x
46.9/44.3/42.2; 100x 30.5/29.6/35.6).
Caption: RoBERTa fine-tuned on the same 1,600 examples, epoch picked on the 400 calibration
examples; LR trained on the same 1,600. The test sample is the same at every imbalance level.

### Figure 3 — Choosing alpha (Sec. 5.6, column width, two stacked panels sharing x, ~1.0 q)
- x: alpha in {0.01, 0.05, 0.10, 0.20} (log-spaced ticks).
- Top panel y: paired Δ CICLe - few-shot (pp), one line per dataset (Yahoo, SST, SemEval,
  GoEmotions), Llama-3.1-8B + Llama-3.2-3B, 24 cells each (2 models x 3 seeds x 2 variants x k in
  {1,4}). Values (A6/A8 ablation blocks): Yahoo -0.07/-0.37/+0.93/+3.57; SST +1.05/+2.65/+2.75/
  +2.52; SemEval +0.12/+0.47/+1.04/+1.36; GoEmotions +0.49/+0.92/+1.45/+3.02. Error bars:
  bootstrap CI (must be computed; analyze.py prints only means for the ablation — see
  `figure_scripts_todo.md`).
- Bottom panel y: share of test instances answered without the LLM (singleton set), same lines:
  Yahoo 0/3.4/9.8/35.4%; SST 0/0/0.3/3.0%; SemEval and GoEmotions 0% throughout. Secondary
  annotation: empirical coverage at each alpha (Yahoo 99.1/95.0/90.1/79.3; SST 99.2/95.4/90.5/80.3;
  SemEval 98.8/94.4/88.5/77.8; GoEmotions 98.8/95.4/90.9/81.0).
- Message: alpha trades coverage for set size; the trade pays where the base classifier is strong
  (Yahoo: LR alone = 65.4) and where sets are otherwise huge (GoEmotions 22 of 28 classes at 0.05).
- Source: A6/A8 "ablation over 24 shared cells" blocks.

---

## Appendix

### Table B1-B5 — Full grids per dataset
Exactly the `analyze.py` method x variant x k table for each of the five datasets (six models):
macro-F1, accuracy, invalid %, mean shots, mean prompt tokens. Add the k=0 row [PENDING]. Source
A6. Also B6-B10 for the imb10/imb100/relabel variants (A8).

### Table C1 — Narrowing comparisons, all variants
Rows: 11 dataset variants (Yahoo 1x/10x/100x/relabel, SST 1x/10x/100x, SemEval, GoEmotions
(+relabel), Ohsumed). Columns: variant (Fixed/PC) x comparator (few-shot, top-k, mass, [marginal],
[oracle]) with Δ, CI, n. Source A6/A8 paired-test lines. Mark n.s.

### Table C2 — Candidate-set statistics
Per dataset variant and narrowing method: coverage, mean set size, singleton (LLM-skipped) rate,
and per-class minimum coverage (new, from R: min over classes of mean `gold_in_set`). Source A6/A8
candidate-set lines + R.

### Figure C1 — Figure 2 for the Fixed variant
Same layout as Figure 2 with Fixed values (A8: Yahoo few-shot -0.35(k1,4)/-0.45/+0.21; top-k
+0.37/+0.58/+6.52; mass +0.36/+0.29/+9.08; SST few-shot +2.05/+2.33/+1.51; top-k -0.14/+1.90/
+3.35; mass +1.07/+1.69/+8.54).

### Table D1 — Per-model CICLe minus few-shot
Rows: six models; column groups: dataset x variant (Yahoo F/PC, SST F/PC, SemEval F/PC,
GoEmotions F/PC, Ohsumed F). Each cell Δ with CI, 12 pairs (Ohsumed 6). Source A6 per-model lines,
e.g. Yahoo Fixed: Llama-8B +0.03 n.s., Llama-3B **-2.33** [-3.12,-1.60], Ministral -0.17 n.s.,
Mistral-7B **-0.69** [-1.21,-0.17], Qwen-3B **+1.99**, Qwen-7B +0.05 n.s.; Yahoo PC: +0.74,
+0.82, +3.40, +0.08 n.s., +2.14, +1.24. Caption contrasts with the DS2026 table (Ministral -2.07,
Qwen-7B -2.95 on Yahoo Fixed) and attributes the change to listing the labels in the prompt.

### Table E1 — Embedding / classifier ablation
For each dataset: rows contriever-LR, minilm-LR, minilm-SVM, tfidf-LR at alpha=0.05 with CICLe,
few-shot (same embedding for retrieval), Δ, coverage, set size, skip rate. Source A6/A8 ablation
blocks (24 cells, Llama-8B + Llama-3B). The alpha rows of the same blocks feed Figure 3.

### Table F1 — Larger models
Mistral-Nemo-12B (Yahoo, SST: 3 seeds; SemEval: 2 seeds) and Qwen-2.5-32B (seed 42) [PENDING
seeds 43/44]: zero-shot, few-shot and CICLe, Fixed and PC, k in {1,4}, macro-F1 and Δ. Values
from R (see `figure_scripts_todo.md` for the extraction): e.g. Qwen-32B Yahoo zero-shot 60.7,
few-shot PC k=4 63.1, CICLe PC k=4 64.2; SST 48.8 / 56.0 / 56.5; Nemo SST Fixed k=4 few-shot
41.6 vs CICLe 45.1 (mean of 3 seeds). Also list the best 3B + CICLe PC k=4 on seed 42 (Yahoo
61.5 Llama-3.2-3B; SST 51.0 Ministral-3B) beside the 32B zero-shot, which is the honest form of
the old Figure 5.

### Table G1 — Fixes and breaks relative to few-shot
CICLe PC k=4 vs few-shot PC k=4, 18,000 instance pairs per dataset (6 models x 3 seeds x 1,000):
% fixed (CICLe right, few-shot wrong), % broken, % both right, % both wrong; then the same split
by gold-in-set. Values from R: Yahoo 4.7 / 3.4 / 58.0 / 33.8; gold outside the set (5.0% of
instances): fixed 0.0, broken 4.1. GoEmotions 4.6 / 3.2 / 22.6 / 69.6; outside (4.6%): broken
8.3. Yahoo 100x 3.7 / 2.6 / 52.3 / 41.4; outside (3.3%): broken 9.9. Fixed variant rows likewise
(Yahoo 4.2/4.3; GoEmotions 4.1/3.0; Yahoo 100x 3.5/3.0). Extend to SST, SemEval, Ohsumed.

### Table G2 — Invalid outputs
Per model x method (zero-shot, few-shot, CICLe) x dataset: mean invalid rate and the maximum over
k/variant. From R metrics: maxima Qwen-2.5-3B 15.7% (Yahoo), 24.5% (SemEval), 23.7% (Ohsumed);
Mistral-7B 10.8/8.9/10.5%; Qwen-7B 11.4/13.4/7.8%; Llama-8B <= 1.4% everywhere; SST 0% for all.
Plus a short table of the most frequent raw outputs that failed to parse (from R `raw`).

### Table G3 — Qualitative examples (R4)
Six to eight instances: text (truncated), gold, candidate set, few-shot output, CICLe output; half
"fixed" (gold in set, few-shot picked a label outside the set) and half "broken" (gold outside
the set, or inside but CICLe switched). Chosen from R by the script in `figure_scripts_todo.md`.

### Table H1 — Controlled-imbalance class counts
Per seed, the class order and training counts under 10x and 100x for Yahoo and SST (from
`Data(...)`), and the nonsense-word mapping used for renaming.
