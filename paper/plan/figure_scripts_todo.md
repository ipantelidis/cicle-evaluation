# Figure and table scripts — to do

All scripts read `results/<variant>/seed-<s>/<file>.json` (fields: `config`, `metrics`,
`records[]` with `idx, gold, pred, raw, llm_called, conformal_set, gold_in_set, shots,
prompt_tokens`) and, where noted, call `analyze.py` functions or `experiment.Data`. Run with
`.venv/bin/python`; nothing needs a GPU. Suggested home: `paper/figures/make_figures.py` with one
function per figure, writing PDFs to `paper/figures/`. Reuse `analyze.py`'s loader and its paired
bootstrap rather than re-implementing them (import the module; its helpers for reading runs and
resampling test instances are what the Δ/CI numbers below come from).

File-name patterns (reference setting):
```
{ds}-{model}-zeroshot.json
{ds}-{model}-fewshot-{emb}-{k}-shots-{variant}.json
{ds}-{model}-{cicle|topk|mass|marginal|oracle}-{emb}-{clf}-{k}-shots-{variant}-{alpha}-alpha.json
{ds}-base-{emb}-{clf}.json          {ds}-finetuned-roberta-base.json
```
`ds` in {yahoo-answers, yahoo-answers-imb10, yahoo-answers-imb100, yahoo-answers-relabel, sst,
sst-imb10, sst-imb100, semeval-18, go-emotions, go-emotions-relabel, ohsumed}; alpha formatted
as `0.05`, `0.10`, `0.20`, `0.01`.

Shared helpers
```
SMALL = [llama-3.2-3b, ministral-3b, qwen-2.5-3b, mistral-7b-v0.3, qwen-2.5-7b, llama-3.1-8b]
load(ds, model, method, variant, k, seed, emb=minilm, clf=lr, alpha=0.05) -> dict
macro_f1(records) -> f1_score(gold, pred or "INVALID", labels=sorted(set(gold)), average=macro) * 100
paired_delta(ds, method_a, method_b, variant, ks, models=SMALL, seeds=[42,43,44], B=5000):
    for each (model, seed, k): records_a, records_b aligned on idx
    for b in 1..B: one resample of test indices per seed, applied to every cell of that seed
    delta_b = mean over cells of (macro_f1(a[idx]) - macro_f1(b[idx]))
    return mean, CI (2.5, 97.5 percentiles), p (two-sided, share of delta_b crossing 0), n_cells
```
(This is what `analyze.py` already does; the point is to expose it for subsets such as k in
{1,4} on the 1x variants, which the CLI does not offer.)

**Needs a new analysis** = value not printed by `analyze.py` and not in the findings file.

---

## Table 2 — main results at k = 4
```
for ds in [yahoo-answers, sst, semeval-18, go-emotions, ohsumed]:
    rows base/finetuned: mean over seeds of metrics.macro_f1 (base-minilm-lr, finetuned-roberta-base)
    rows zeroshot, fewshot/cicle x fixed/pc at k=4: mean over SMALL x seeds of macro_f1 and mean_prompt_tokens
    delta rows: paired_delta(ds, cicle, fewshot, variant, ks=[1,2,4,8] (ohsumed [1,4]))
```
Everything is in A6 already (k=4 rows; paired lines); the script only re-formats. Add the
[PENDING] Ohsumed PC k=1 cell and k=0 row when files appear.

## Figure 1 — macro-F1 vs prompt tokens
```
for ds: for (method, variant, ks): x = mean_prompt_tokens over SMALL x seeds, y = mean macro_f1
    zeroshot: single point; fewshot/cicle fixed/pc: connected in k order
    error bar: bootstrap over test instances of the 18-run mean (unpaired version of paired_delta)  <- needs a new analysis (CI only)
    hlines: base-minilm-lr mean, finetuned-roberta mean
log x axis; 5 panels in one row; shared legend
```

## Figure 2 — narrowing under imbalance (PC; Figure C1 = Fixed)
```
(a) for ds in [yahoo-answers, sst], imb in [1, 10, 100], comparator in [fewshot, topk, mass, marginal*]:
        variant_ds = ds if imb==1 else f"{ds}-imb{imb}"
        paired_delta(variant_ds, cicle, comparator, pc, ks=[1,4])      <- the 1x point with ks=[1,4] needs a new analysis (A6 pools all k)
    oracle*: mean macro_f1 of oracle runs minus cicle, dashed line     <- PENDING runs
(b) for each (variant_ds, method in [cicle, topk, mass, marginal*]):
        coverage = mean over runs of metrics.coverage; size = metrics.mean_set_size   (in A8; recompute from records for the per-seed spread)
(c) class frequency rank at 100x:
        for seed: data = experiment.Data(ds, seed, imbalance=100)   # needs HF cache; HF_HUB_OFFLINE=1
                  counts = Counter(data.y_train); rank[label] = position in descending counts
        for method, model in SMALL, seed: per-class coverage = mean(gold_in_set | gold == c) from any k=4 PC run (sets are identical across models/k; use one model to save time but verify)
        aggregate by rank over seeds -> line per method
        inset: per-class F1 by rank, methods [fewshot, cicle, topk, mass], mean over SMALL x seeds  <- needs a new analysis
```
Caveat for (c): the class order is a seed-dependent permutation (`experiment.Data`, `rng =
default_rng(seed)`), so never aggregate by class name across seeds. Caveat for (a): the 100x pool
has 5 examples in the rarest class, so with 20% calibration one calibration example per class;
mention in the caption.

## Table 3 — label renaming
Straight from A6 (original) and A8 (relabel) tables plus the two paired lines; no new analysis.
Optional: add the per-model deltas on relabel (run `analyze.py --per-model --datasets
yahoo-answers-relabel,go-emotions-relabel`).

## Table 4 — supervised vs pipelines
```
for col in [yahoo 1/10/100, sst 1/10/100, (pool sizes PENDING: results/<ds>-n{250,500,1000}/), semeval, go-emotions, ohsumed]:
    base-minilm-lr, finetuned-roberta: mean over seeds (print per-seed in the appendix version)
    zeroshot (1x only), fewshot pc k=4, cicle pc k=4: mean over SMALL x seeds
    best pipeline: argmax over (method in [fewshot,cicle], variant, k) of the 18-run mean; print its name
```
All inputs exist except the pool-size columns.

## Figure 3 — alpha
```
models = [llama-3.1-8b, llama-3.2-3b]; ks=[1,4]; variants=[fixed,pc]
for ds in four short-text sets, alpha in [0.01,0.05,0.10,0.20]:
    top: paired_delta(ds, cicle(alpha), fewshot, both variants, ks, models) -> mean + CI   <- CI needs a new analysis (A6 prints means only)
    bottom: mean metrics.singleton_rate (LLM-skipped) and metrics.coverage over the same 24 runs
```

---

## Appendix

### Tables B1-B10
`analyze.py --datasets <ds> --bootstrap 5000` for each variant; paste the method x variant x k
table. Six models only (default). Add accuracy column (already printed).

### Table C1 / C2
`analyze.py --bootstrap 5000 --per-model` over all 11 variants gives every CICLe - alternative
line and the candidate-set lines. C2's "per-class minimum coverage" column:
```
for variant_ds, method: for seed: from one cicle/topk/mass run: min over classes of mean(gold_in_set | gold==c); average over seeds   <- new analysis
```

### Table D1 — per-model
Already in A6 (`--per-model`); redo with 5000 bootstraps. Script only formats
(model x [dataset x variant], bold where CI excludes 0).

### Table E1 — embedding / classifier
From the "ablation over 24 shared cells" blocks in A6/A8 (rows contriever-lr, minilm-lr, minilm-svm,
tfidf-lr at alpha 0.05). Add CIs via paired_delta restricted to the two Llama models <- new
analysis (means exist, CIs do not).

### Table F1 — larger models
```
for ds in [yahoo-answers, sst, semeval-18], model in [mistral-nemo-2407, qwen-2.5-32b]:
    for method/variant/k in zeroshot, fewshot & cicle x fixed/pc x {1,4}: mean over available seeds of macro_f1 (nemo: 3 seeds yahoo/sst, 2 semeval; 32b: seed 42; semeval 32b cicle pc k=4 missing)
    delta = cicle - fewshot per cell; paired_delta with models=[that model] for the CI   <- new analysis
    also: best 3B+CICLe PC k=4 on seed 42 (yahoo: llama-3.2-3b 61.5; sst: ministral-3b 51.0) vs 32B zero-shot seed 42 (60.7 / 48.8)
```
Known values (seed 42, 32B): Yahoo ZS 60.7; FS fixed 62.6/63.2, PC 61.6/63.1; CICLe fixed
63.4/64.2, PC 63.1/64.2. SST ZS 48.8; FS 52.8/54.5, PC 56.4/56.0; CICLe 54.3/56.5, PC 57.2/56.5.
SemEval ZS 15.3; FS 16.8/18.6, PC 18.9/19.6; CICLe fixed 16.9/19.6, PC k=1 19.6. Nemo-12B
3-seed means: Yahoo ZS 57.8, FS fixed 58.9 vs CICLe 59.3, PC 60.5 vs 61.3; SST ZS 27.5, FS fixed
39.6 vs CICLe 43.8, PC 41.9 vs 46.2.

### Table G1 — fixes vs breaks
```
for ds, variant in [pc, fixed], k=4: for model in SMALL, seed: align cicle and fewshot records on idx
    fix = cicle right & fewshot wrong; brk = reverse; both_right; both_wrong; split by gold_in_set
    report % of all 18,000 pairs and % within the gold-outside subset
```
Already computed for Yahoo, GoEmotions, Yahoo-100x (values in `figures_and_tables.md`); extend to
SST, SemEval, Ohsumed (Fixed), SST-100x. Also split by whether the few-shot prediction was *in* the
conformal set (if few-shot's answer was outside the set and wrong, narrowing could only help).
<- new analysis.

### Table G2 — invalid outputs
```
for ds, model, method: mean and max over (variant, k, seed) of metrics.invalid_rate
top-10 raw outputs with pred == None, per model (Counter over records.raw where pred is None)   <- new analysis
```

### Table G3 — qualitative examples
```
for ds in [yahoo-answers, go-emotions], model = llama-3.1-8b, seed 42, cicle pc k=4 vs fewshot pc k=4:
    fixed_ex  = records where cicle right, fewshot wrong, fewshot.pred not in conformal_set   (narrowing removed the distractor)
    broken_ex = records where gold_in_set is False (CICLe could not answer) and fewshot right
    broken_ex2 = gold_in_set True, fewshot right, cicle wrong (switched inside the set)
    sample 3 of each with a fixed rng; print text (needs Data(ds, seed).X_test[idx] -> HF cache), gold, set, both raw outputs
```
<- new analysis; text retrieval requires `experiment.Data`, i.e. the HF datasets cache
(`HF_HUB_OFFLINE=1`).

### Table H1 — imbalance class counts and relabel words
```
for ds in [yahoo-answers, sst], seed, imb in [10,100]: Counter(Data(ds, seed, imbalance=imb).y_train) in class order
relabel: Data(ds, 42, relabel=True).label_map
```
<- needs `experiment.Data` (CPU; dataset loaders via HF cache).

---

## Order of work
1. `analyze.py --bootstrap 5000 --per-model` over all variants -> A-final (replaces A6/A8 as the
   number source; ~several minutes).
2. `paired_delta` wrapper (import from analyze.py) + Figure 2a with ks=[1,4] and Figure 3 CIs.
3. Figure 1, Figure 2b, Table 2/3/4 formatting (pure re-formatting of A-final).
4. Figure 2c + Table H1 (needs `experiment.Data` offline; test once that the HF cache resolves).
5. Appendix G (G1 extension, G2 raw outputs, G3 examples).
6. Slot in pending runs as they land (Ohsumed PC k=1, marginal, oracle, k=0, pool sizes, 32B
   seeds); each has a marked slot in `figures_and_tables.md`.
