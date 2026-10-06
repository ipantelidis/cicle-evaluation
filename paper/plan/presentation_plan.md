# Presentation plan — CICLe resubmission (COLING 2027 via ARR, deadline 2026-10-12)

Format: ACL long paper, 8 pages main text + unlimited references/appendix + mandatory Limitations
(not counted). Budget below is in quarter pages (32 quarters = 8 pages) and *includes* the floats
that sit in each section. Figures/tables are specified in `figures_and_tables.md`; the Results
prose is drafted in `results_section_draft.md`.

Sources for numbers (abbreviations used throughout the plan):
- **A6** = `paper/plan/analysis_6models_permodel_bootstrap200.txt` — fresh `analyze.py --per-model`
  run over the six small LLMs only (72 pairs for the main grid). **Use these for the paper.**
- **A8** = `paper/reference/analysis_2026-10-06_morning.txt` — the stored run. Its Yahoo/SST/SemEval
  tables and paired tests silently include Mistral-Nemo-12B and Qwen-32B at k in {1,4}
  (80/80/74 pairs). Used only for the variants A6 does not cover (imb10/imb100/relabel), where
  only the six small models exist anyway.
- **F** = `paper/reference/findings_2026-10-06.md`.
- **R** = per-instance `results/<dataset>/seed-*/*.json` (new analyses, no GPU).

## 0. Storyline (fixed)

1. Earlier evaluations of this kind are fragile: unlisted labels, unequal example pools and
   unseeded conformal sets changed conclusions; we give a corrected protocol.
2. Narrowing the label set gives a small, significant gain over few-shot prompting and shortens
   prompts.
3. How you narrow matters: class-conditional CP is robust under class imbalance, where top-k,
   probability-mass and (pending) marginal CP lose several points; the gain is largest when label
   names are uninformative.
4. A plain classifier or fine-tuned encoder is enough on balanced data with ~1,600 labels; LLM
   pipelines pay off when the labelled pool is skewed (and, pending, small); alpha should be chosen
   by how strong the classifier is.

Honest framing to keep everywhere: gains over few-shot are 0.5-2 points; supervised baselines
often win on balanced data; the big effects (+4 to +11 points) are *between narrowing methods*
under imbalance and under label renaming, not between CICLe and few-shot.

## 1. Section outline and page budget

| # | Section | Quarters | Content / floats |
|---|---------|---------:|------------------|
| 1 | Introduction | 3.0 | Problem, the four storyline claims as contributions, one sentence on the protocol fixes. No figure. |
| 2 | Related Work | 2.0 | Two-stage classifier+LLM pipelines (R1), example selection, CP in NLP, LLM vs fine-tuning for classification. |
| 3 | Background: label narrowing with conformal prediction | 2.0 | Intuition for class-conditional CP (R1: "give an intuition of CP"); CICLe pipeline; definitions of the four narrowing rules (CP, marginal CP, top-k, probability mass) and of Fixed vs Per-Class retrieval. Half-column schematic optional (only if space). |
| 4 | Experimental Setup | 5.0 | 4.1 Datasets + **Table 1** (1.0 q); 4.2 Models (0.5); 4.3 Protocol and the five fixes (1.5); 4.4 Baselines and comparison methods (1.0); 4.5 Metrics and statistics (1.0). |
| 5 | Results | 16.0 | Prose 14.0 q (~2,200 words incl. the protocol/baseline parts that live in Sec. 4) + floats: **Table 2** (2.0, full width), **Figure 1** (1.5, full width), **Figure 2** (2.5, full width), **Table 3** (0.75, column), **Table 4** (1.25, column), **Figure 3** (1.0, column). Sum of floats = 9.0 q; see note below. |
| 6 | Conclusion | 1.5 | Rules of thumb for practitioners; one sentence per claim. |
| — | Limitations (uncounted) | — | Single-label only (R1 multi-label); 1,000-instance test samples; six models 3-8B plus partial 12B/32B; English only; alpha chosen on calibration coverage, not tuned on test; CP guarantee is marginal per class, not conditional on the input; renaming is an extreme stress test. |

Total = 29.5 quarters of budget assigned to 1-6 if the Results floats are counted inside the
16.0; actual float area (9.0 q) + Results prose (14.0 q) = 23 q, so the honest arithmetic is:
3.0 + 2.0 + 2.0 + 5.0 + 23.0 + 1.5 = **36.5 quarters = 9.1 pages**. We are ~1.1 pages over.
Resolution, in order of preference:
(a) move Table 3 (renaming) and Figure 3 (alpha) to the appendix and state their key numbers in
    prose (saves 1.75 q);
(b) cut Results prose to ~1,900 words (saves 2 q);
(c) shrink Figure 2 to two rows x three panels at 0.9 text width (saves 0.5 q).
Doing (a)+(b) lands at 32.75 q; (c) brings it to 32.25; a tighter Related Work (1.5 q) closes it.
The draft keeps Table 3 and Figure 3 in the main text so the decision can be taken once the
pending results are in; if the pool-size result is strong it earns its space in Table 4 and
Figure 3 goes to the appendix.

## 2. Results subsections: claim, evidence, reviewer point

### 5.1 The protocol matters (1.5 q prose)
- **Claim.** Three choices in the earlier protocol changed conclusions: (i) prompts that do not
  list the allowed labels produced 20-35% unparseable outputs for some models (F), which was
  misread as "CICLe hurts Qwen and Ministral"; (ii) few-shot and CICLe drew examples from pools of
  different size; (iii) conformal sets were not seeded, so CICLe runs were not paired with their
  few-shot counterparts. With the corrected protocol invalid outputs are 0-6% on average per
  dataset (A6 tables: zero-shot 6.1% Yahoo, 5.1% SemEval, 5.2% Ohsumed, 3.0% GoEmotions, 0.0%
  SST; CICLe 1.2/4.5/1.5/2.2/0.0%) and the per-model heterogeneity reverses (Appendix D).
- **Evidence.** Prose numbers + Appendix Table D1 (per-model deltas with CIs) + Appendix Table G1
  (invalid rates per model/method). Optional Appendix I: re-run of the legacy prompt with
  `--legacy-prompt` on one dataset/model to show the artefact directly — only if such results
  exist or can be produced on CPU-free time; **not** required for the storyline.
- **Answers.** R4 (three models benefit, three do not; wants mechanism) and R3 (unclear
  conclusions; significance).

### 5.2 Narrowing the label set: small, significant gains and shorter prompts (3.0 q prose)
- **Claim.** With the same wording, pool and seeds, CICLe beats few-shot in 8 of 9
  dataset x variant cells by +0.5 to +1.8 points (A6: Yahoo PC +1.40 [+1.03,+1.76]; SST Fixed
  +1.77 [+1.04,+2.49], PC +1.07 [+0.36,+1.69]; SemEval +0.52/+0.46; GoEmotions +0.59/+0.92;
  Ohsumed Fixed +1.15 [+0.60,+1.86]); the exception is Yahoo Fixed (-0.19, n.s.). Per-Class CICLe
  uses 11-38% fewer prompt tokens than Per-Class few-shot at equal k (R, computed from
  `mean_prompt_tokens`; A6 token columns). Gains do not vanish at k=8 (R3) and do not grow with k.
- **Evidence.** **Table 2** (main results at k=4 with paired Δ and 95% CI; full k grid in Appendix
  B), **Figure 1** (macro-F1 vs mean prompt tokens, all methods/variants/k, five panels).
- **Answers.** R3 (are differences significant at k=8; relation of k to number of classes: PC
  budget = k x |candidate set|, visible on Figure 1's x-axis), R1 (longer documents: Ohsumed).

### 5.3 How you narrow matters: class-conditional CP under imbalance (3.5 q prose)
- **Claim.** At equal mean set size, CP = top-k = probability mass on balanced data (SST, SemEval:
  all |Δ| < 0.35, n.s.; A6). Under a long-tailed labelled pool the alternatives lose coverage on
  rare classes and the LLM cannot recover it: Yahoo 100x CP - top-k = +6.5 (Fixed) / +7.4 (PC),
  CP - mass = +9.1 / +9.4; SST 100x +3.4/+4.2 and +8.5/+10.7 (A8, 36 pairs each). Coverage at
  100x: CP 96.7%, top-k 81.6%, mass 74.2% on Yahoo; 96.0 / 88.4 / 60.0% on SST (A8). On the
  naturally skewed sets: GoEmotions CP - top-k +2.5, CP - mass +1.9 to +2.3; Ohsumed +1.1 / +1.5
  (A6). Imbalance does **not** widen the CP-vs-few-shot gap (Yahoo PC: +1.32 at 1x on matched
  k in {1,4}, +1.09 at 10x, +0.89 at 100x) — this explicitly retracts the old paper's claim.
  [PENDING: marginal CP LLM runs; sets-only coverage 62.9% at size 4.1 on Yahoo 100x (F) says it
  will fall between top-k and mass.] [PENDING: oracle narrowing gives the ceiling at that budget.]
- **Evidence.** **Figure 2** (rows Yahoo/SST; columns: Δ vs imbalance for each narrowing rule;
  coverage vs set size; per-class coverage or F1 by class training frequency). Appendix Table C1
  (all CP - alternative deltas, every variant). Appendix G (fixes vs breaks: when the gold label is
  outside the set CICLe can only break, 4-10% of such instances; when inside, fixes exceed breaks
  4.9% vs 3.4% on Yahoo PC k=4, R).
- **Answers.** R1 (justify CP vs top-k / thresholding), R4 (mechanism: coverage on rare classes),
  R3 (dataset properties: controlled imbalance on two datasets isolates the variable).

### 5.4 Uninformative label names (1.5 q prose)
- **Claim.** Renaming every label to a nonsense word (same task, same texts) is where narrowing
  helps most: Yahoo CICLe - few-shot +6.5 (Fixed) / +5.6 (PC), GoEmotions +0.7 / +1.0 (A8, 36
  pairs). With one Fixed example per prompt, few-shot drops to 25.3 macro-F1 on renamed Yahoo while
  CICLe keeps 34.3 (A8). Replaces the dropped synthetic benchmark (R1's own suggestion).
- **Evidence.** **Table 3** (column width): original vs renamed labels, few-shot vs CICLe, Fixed
  and PC, k=1 and 4, two datasets.
- **Answers.** R1 (synthetic benchmark unconvincing; "rename labels while keeping the task").

### 5.5 When is an LLM pipeline worth it? Supervised baselines (2.5 q prose)
- **Claim.** With 1,600 balanced labels a fine-tuned RoBERTa-base beats every LLM pipeline on
  SST (51.2 vs 47.6), SemEval (24.5 vs 16.1), GoEmotions (37.0 vs 27.3) and Ohsumed (55.4 vs 51.7);
  on Yahoo even MiniLM+LR (65.4) beats the best pipeline (63.8) and RoBERTa (61.8) (A6). Under a
  100x long-tailed pool the order flips: RoBERTa 41.2 (Yahoo) / 31.9 (SST) and MiniLM+LR 31.4 /
  12.6 vs best LLM pipeline 56.0 / 43.4 (A8 base rows; RoBERTa from R files). [PENDING: pool size
  250/500/1,000 — expected to show the same flip for small pools.]
- **Evidence.** **Table 4** (column width): rows = MiniLM+LR, RoBERTa, zero-shot, few-shot PC,
  CICLe PC; columns = Yahoo 1x/10x/100x, SST 1x/10x/100x, [PENDING pool sizes], plus SemEval,
  GoEmotions, Ohsumed.
- **Answers.** R3 (unclear conclusions -> explicit decision rule), R1 (position vs classifier-only
  approaches), honest framing requirement.

### 5.6 Choosing alpha; robustness across models and model sizes (2.0 q prose)
- **Claim.** Loosening alpha shrinks sets and lets the base classifier answer alone; it helps where
  the classifier is strong (Yahoo alpha=0.2: +3.6 over few-shot with 35% of instances never sent to
  the LLM) and on the many-class sets (GoEmotions +3.0, SemEval +1.4), while SST is flat (+2.5 to
  +2.8 for all alpha >= 0.05) (A6/A8 ablation blocks, Llama-3.1-8B + Llama-3.2-3B, 24 cells).
  Contriever > MiniLM on SST/GoEmotions, < on Yahoo; TF-IDF weakest; SVM = LR. With the fixed
  protocol no model is significantly hurt by Per-Class CICLe; Fixed CICLe hurts Llama-3.2-3B
  (-2.3) and Mistral-7B (-0.7) on Yahoo only (A6 per-model). CICLe also adds +0.1 to +4.4 points to
  Mistral-Nemo-12B and Qwen-32B (R; Appendix F), and a 3B model with CICLe PC matches Qwen-32B
  zero-shot on Yahoo and SST (seed 42: 61.5/51.0 vs 60.7/48.8).
- **Evidence.** **Figure 3** (column width): Δ vs few-shot and LLM-call rate against alpha, one
  line per dataset. Appendix E (embedding/classifier table), D (per-model), F (large models).
- **Answers.** R3 ("alpha shows little change"; "Section 4.4 unfair": both large models are now
  run with and without CICLe, same reference configuration, no selection on the test set), R4
  (heterogeneity revisited with CIs).

## 3. Appendix (unlimited)

A. Prompt template (`build_prompt`), output normalisation (`parse_output`), generation budget.
B. Full result tables per dataset: method x variant x k with macro-F1, accuracy, invalid rate,
   mean shots, mean prompt tokens (direct from `analyze.py` tables, six models).
C. Narrowing comparisons: CICLe minus {few-shot, top-k, mass, [marginal], [oracle]} for every
   dataset variant (11 rows) and both retrieval variants, with CIs and n. Coverage / set size /
   skip rate per method and variant (Table C2).
D. Model heterogeneity: per-model CICLe - few-shot, both variants, five datasets, 12 (or 6) pairs
   each, CIs (from A6 `--per-model`). Contrast with the DS2026 Table 1.
E. Ablations: embedding x classifier x alpha blocks for all four datasets (A6/A8 ablation tables).
F. Larger models: Mistral-Nemo-12B (3 seeds Yahoo/SST, 2 SemEval) and Qwen-32B (seed 42) zero-shot,
   few-shot, CICLe, k in {1,4}, both variants; [PENDING 32B seeds 43/44].
G. Per-instance analyses: fixes vs breaks relative to few-shot split by gold-in-set; invalid
   output rates per model and method; qualitative examples of fixed and broken instances (raw
   outputs from R) — answers R4's request for failure-case analysis.
H. Dataset details: class lists, imbalance construction (geometric schedule, seed-dependent
   order), relabel word list, subsample sizes, controlled-imbalance class counts per seed.
I. (optional) Legacy-prompt reproduction showing the invalid-output artefact.

## 4. Dropped, and why

- **Synthetic FOL benchmark** (old Sec. 5, App. B): chance-level accuracy, unconvincing to R1 and
  R3 (TF-IDF too weak); replaced by label renaming on real data.
- **Image classification mention** (R1).
- **Llama-3.1-70B**: not run under the fixed protocol; no claim depends on it.
- **"Imbalance predicts benefit" figure** (old Fig. 2): contradicted by the controlled experiment
  (Yahoo PC +1.32 -> +1.09 -> +0.89). Retracted explicitly in 5.3.
- **"Small model + CICLe vs zero-shot large model" figure** (old Fig. 5): R3 called it unfair.
  Replaced by Appendix F where the large models are run with and without CICLe; one sentence in
  5.6.
- **Hyperparameter-spread error bars** (old Fig. 1): replaced by paired bootstrap over test
  instances, with hyperparameters held fixed at the reference setting.
- **Best-of-sweep configurations**: all main-text numbers are at the reference setting (MiniLM,
  LR, alpha=0.05); ablations are reported as ablations.

## 5. Pending results and where they land (due Wednesday evening)

| Pending run | Slot | If it does not arrive |
|-------------|------|-----------------------|
| Ohsumed PC k=1 | Table 2 Ohsumed PC cell; Appendix B | Report Ohsumed Fixed only, say PC omitted because 23 abstracts per prompt exceed context budgets (already the reason k=4 PC was not run). |
| Marginal CP LLM runs (Yahoo/SST 1x/10x/100x) | Figure 2 fourth line; Table C1 | Report sets-only coverage (62.9% at size 4.1 on Yahoo 100x, F) and say LLM runs were not completed. |
| k=0 (candidate set, no examples) | One sentence in 5.2 + Appendix B row | Omit. |
| Oracle narrowing | Figure 2 dashed ceiling; Table C1 | Omit. |
| Pool size 250/500/1,000 with baselines | Table 4 extra columns; one paragraph in 5.5 | Keep claim 4 to "skewed" only; move "small pool" to future work. |
| 32B seeds 43/44 | Appendix F | Keep seed 42, state n=1. |
