# Group H — Classifier: LR vs. SVM (CICLe Only)

**Research goal:** Group H analyses whether the choice of linear classifier —
Logistic Regression (LR) or Support Vector Machine (SVM) — has a meaningful
impact on CICLe's macro-F1, and whether any preference is consistent across
models, embeddings, alpha values, and shot counts. All plots use **CICLe
results only**.

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

All plots are built from **matched pairs**: for every configuration key
(dataset, model, embedding, shots, alpha, variant), the LR and SVM scores
are compared directly. A tie is defined as |SVM − LR| ≤ 0.001 macro-F1
(≈ 0.1 percentage points).

## Plot inventory

### H1 — `H1_scatter_lr_vs_svm`  (3 plots)
One plot per **dataset**.

Axes:
- **X-axis:** LR macro-F1
- **Y-axis:** SVM macro-F1

Each point is one matched (model × embedding × shots × alpha × variant)
configuration. Points below the diagonal = LR wins; points above = SVM wins.
Win counts and percentages annotated in the upper-left corner.

**Reading:** The global picture of classifier dominance across all configs for
that dataset. A dense cloud below the diagonal with the majority of points
indicates LR is consistently better than SVM regardless of other settings.

---

### H2 — `H2_delta_by_model`  (3 plots)
One plot per **dataset**.

Boxplot of **(SVM − LR)** in percentage points, one box per model (all 6).
A horizontal dashed line marks zero (parity). Positive values = SVM wins.

**Reading:** Shows whether LR vs. SVM preference is model-specific. A model
with a consistently negative median should always use LR; one with a near-zero
median is classifier-agnostic.

---

### H3 — `H3_delta_by_embedding`  (3 plots)
One plot per **dataset**.

Same (SVM − LR) delta boxplot, one box per embedding (Contriever, MiniLM,
TF-IDF).

**Reading:** Reveals whether the classifier preference depends on the retrieval
method. If one embedding shows a very different distribution than the others,
the embedding and classifier are interacting — they should not be tuned
independently.

---

### H4 — `H4_delta_by_alpha`  (3 plots)
One plot per **dataset**.

Same (SVM − LR) delta boxplot, one box per alpha value (0.01, 0.05, 0.10,
0.20).

**Reading:** Shows whether the CICLe significance threshold interacts with
classifier choice. If the delta distribution shifts sign or spread across alpha
values, the best classifier depends on how strictly CICLe filters examples.

---

### H5 — `H5_delta_by_shots`  (3 plots)
One plot per **dataset**.

Same (SVM − LR) delta boxplot, one box per shots value (1, 2, 4, 8).

**Reading:** Shows whether the classifier gap changes with the number of
in-context examples. Widening or sign-changing boxes suggest that LR and SVM
have different data efficiency profiles.

---

### H6 — `H6_alpha_shots_heatmap_svm_minus_lr`  (18 plots)
One plot per **dataset × model** (3 × 6).

Diverging heatmap:
- **Rows:** alpha (0.01, 0.05, 0.10, 0.20)
- **Columns:** shots (1, 2, 4, 8)
- **Cell value:** mean (SVM − LR) in percentage points, averaged over
  embeddings and variants

Green cells = SVM wins; red cells = LR wins; near-white = parity.
Cell values annotated with sign.

**Reading:** The interaction map for a specific model. Consistent red across
all cells means LR is always preferred for that model regardless of alpha and
shots. A mixed pattern means the optimal classifier depends jointly on the
alpha × shots configuration.
