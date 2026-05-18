# Group E — Alpha Sensitivity (CICLe Only)

**Research goal:** Group E analyses how sensitive CICLe is to the significance
threshold alpha (α ∈ {0.01, 0.05, 0.10, 0.20}), which controls how strictly
CICLe filters in-context examples via its statistical test. The group asks
whether there is a single robust optimal alpha, whether the optimal value shifts
with shot count or embedding, and whether small and large models respond
differently to alpha changes.

All plots in this group use **CICLe results only**.

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

## Plot inventory

### E1 — `E1_alpha_curves_per_model_per_embedding`  (54 plots)
One plot per **dataset × model × embedding** (3 × 6 × 3).

Axes:
- **X-axis:** alpha (0.01, 0.05, 0.10, 0.20)
- **Y-axis:** macro-F1

Lines: one per shots value (1, 2, 4, 8), coloured by shots.

How values are computed:
- each point is the **mean macro-F1 over variant × classifier** for the given
  (dataset, model, embedding, shots, alpha)

**Reading:** The most granular view of alpha sensitivity. Flat lines mean the
model is robust to alpha for that embedding and shots count; steep lines
indicate that alpha must be tuned carefully. Consistent peak positions across
lines mean a single alpha works for all shot counts.

---

### E2 — `E2_alpha_shots_heatmap_per_model`  (18 plots)
One plot per **dataset × model** (3 × 6).

Heatmap structure:
- **Rows:** alpha (0.01, 0.05, 0.10, 0.20)
- **Columns:** shots (1, 2, 4, 8)
- **Cell value:** mean macro-F1 over **embeddings × classifiers × variants**

Sequential green palette (darker = higher F1). Cell values annotated.

**Reading:** Reveals whether the optimal alpha row is stable across shot
columns. If the same row is always the darkest, that alpha generalises across
shot counts. If the darkest cell shifts between rows across columns, the
optimal alpha depends on how many shots are used.

---

### E3 — `E3_alpha_shots_heatmap_per_model_per_embedding`  (54 plots)
One plot per **dataset × model × embedding** (3 × 6 × 3).

Same structure as E2 (rows = alpha, cols = shots), but each cell contains the
**mean macro-F1 over classifiers and variants only** — embeddings are not
averaged. This exposes whether the alpha × shots interaction is consistent
across retrieval methods.

**Reading:** Comparing E3 plots across embeddings for the same model reveals
whether a particular embedding changes the optimal alpha, or whether alpha
sensitivity is an intrinsic property of the model.

---

### E4 — `E4_optimal_alpha_distribution`  (3 plots)
One plot per **dataset**.

For every (model × embedding × shots × classifier × variant) configuration,
the alpha that yields the highest macro-F1 is recorded as the winner. The bar
chart shows how many configurations each alpha wins.

**Reading:** A single-number summary of alpha dominance across all possible
combinations. A strongly skewed distribution (one alpha wins most of the time)
means the hyperparameter is easy to set; a flat distribution means it must be
tuned per configuration.

---

### E5 — `E5_alpha_sensitivity_by_model_size`  (3 plots)
One plot per **dataset**, with two side-by-side panels (large models 7–8B |
small models 3B).

Each panel shows one line per model family. X = alpha, Y = macro-F1 averaged
over **shots × embeddings × classifiers × variants**.

**Reading:** Directly compares how much performance degrades as alpha increases
for large vs. small models. Steeper slopes in the small-model panel indicate
that smaller models are more sensitive to alpha and require tighter tuning.
Crossing lines between families reveal which family is most affected.

---

### E6 — `E6_lr_vs_svm_alpha_sensitivity`  (18 plots)
One plot per **dataset × model** (3 × 6).

Each plot has three side-by-side subplots, one per embedding. Within each
subplot: two lines (LR in blue, SVM in red). X = alpha, Y = mean macro-F1
over **shots and variants**.

**Reading:** Shows whether the classifier choice (LR vs. SVM) interacts with
alpha. Parallel lines mean the gap between LR and SVM is constant regardless
of alpha; crossing lines mean the better classifier depends on the alpha value
chosen.
