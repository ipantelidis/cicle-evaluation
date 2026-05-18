# Group G — pc vs. fixed Variant

**Research goal:** Group G compares the two sampling strategies used in CICLe
and few-shot experiments:

- **pc** (per-class) — k shots *per class*, producing a class-balanced set of
  in-context examples regardless of how many classes the dataset has
- **fixed** (total) — k shots *in total*, drawn without class balancing

The group asks whether this choice has a meaningful impact on macro-F1, whether
the effect is consistent across models, embeddings, and shot counts, and which
variant should be preferred in practice.

## Model families

| Family  | Large model       | Small model     |
|---------|-------------------|-----------------|
| Llama   | Llama 3.1-8B      | Llama 3.2-3B    |
| Mistral | Mistral 7B v0.3   | Ministral 3B    |
| Qwen    | Qwen 2.5-7B       | Qwen 2.5-3B     |

## Datasets

| Dataset       | Classes | Short name    |
|---------------|---------|---------------|
| Yahoo Answers | 10      | yahoo-answers |
| Go Emotions   | 28      | go-emotions   |
| SemEval-18    | 20      | semeval-18    |

## Matched-pair convention

All G plots are built from **matched pairs**: for every configuration key
(dataset, model, embedding, shots, and for CICLe also alpha × classifier), the
pc and fixed scores are paired directly. This ensures that any observed
difference is attributable solely to the variant choice.

A tie is defined as |fixed − pc| ≤ 0.001 macro-F1 (≈ 0.1 percentage points).

## Plot inventory

### G1 — `G1_scatter_pc_vs_fixed`  (6 plots)
One plot per **dataset × method** (CICLe, few-shot).

Axes:
- **X-axis:** pc variant macro-F1
- **Y-axis:** fixed variant macro-F1

Each point represents one matched configuration. Points below the diagonal
mean pc wins; points above mean fixed wins. Win counts and percentages are
annotated in the upper-left corner.

**Reading:** The overall picture of which variant dominates. A dense cloud
below the diagonal with nearly all points on one side indicates a strong,
consistent preference for pc over fixed (or vice versa).

---

### G2 — `G2_delta_by_shots`  (6 plots)
One plot per **dataset × method** (CICLe, few-shot).

Boxplot of **(fixed − pc)** delta in percentage points, one box per shots
value (1, 2, 4, 8). A horizontal dashed line marks zero (no difference).

**Reading:** Shows whether the variant gap changes with shot count. If the
boxes shift upward with more shots, fixed becomes relatively better as more
examples are available. Consistent negative medians across all shot values
confirm that pc is robustly better regardless of shot count.

---

### G3 — `G3_delta_by_model`  (6 plots)
One plot per **dataset × method** (CICLe, few-shot).

Same (fixed − pc) delta boxplot, but one box per model (all 6 models).

**Reading:** Reveals whether the variant preference is universal or
model-specific. A model with a positive median benefits from fixed; one with
a strongly negative median should always use pc.

---

### G4 — `G4_delta_by_embedding`  (6 plots)
One plot per **dataset × method** (CICLe, few-shot).

Same (fixed − pc) delta boxplot, but one box per embedding (Contriever,
MiniLM, TF-IDF).

**Reading:** Shows whether the variant advantage depends on the retrieval
method. If one embedding interacts strongly with variant choice (large spread
or a different-sign median), it suggests the two choices are not independent
design decisions.

---

### G5 — `G5_winrate_pc_vs_fixed`  (3 plots)
One plot per **dataset**.

Horizontal stacked bar chart. One bar per **(model × method)** combination
(12 bars total: 6 models × 2 methods). Each bar shows the percentage of
matched configurations where:
- **PC wins** (blue)
- **Tie** ±0.1 pp (yellow)
- **Fixed wins** (red)

Percentages are annotated inside each segment when large enough to display.

**Reading:** The most compact summary. A bar that is almost entirely blue
means pc is the right choice for that model and method combination across
virtually all embedding × shots settings. Comparing CICLe and few-shot rows
for the same model shows whether CICLe changes the variant preference.
