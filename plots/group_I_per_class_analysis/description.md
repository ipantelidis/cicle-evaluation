# Group I — Per-Class Analysis

**Research goal:** Group I moves beyond macro-F1 to examine individual class
behaviour. It identifies which classes drive the small vs. large model
difference, whether class frequency (support) predicts the benefit of CICLe,
and whether the two models make structurally similar errors as revealed by
their confusion matrices.

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

## Best-config convention

Throughout this group, **best config** means the single result file that
achieves the highest macro-F1 for a given (dataset, model, method) combination,
optimised over all other hyperparameters (embeddings, shots, variant, alpha,
classifier). The per-class F1 values and confusion matrix are taken from that
same file.

## Plot inventory

### I1 — `I1_per_class_f1_heatmap`  (9 plots)
One plot per **dataset × family** (3 × 3).

Three-column heatmap:
- **Column 1:** small model CICLe (best config) per-class F1
- **Column 2:** large model few-shot (best config) per-class F1
- **Column 3:** large model zero-shot per-class F1

Rows = class labels. Sequential green palette (darker = higher F1). Values
annotated in each cell.

**Reading:** Immediately reveals which classes are harder (pale across all
columns) and whether the small model's best CICLe config matches the large
model class by class. Rows where the small model column is as dark as or
darker than the large columns indicate classes where CICLe fully compensates
for the size difference.

---

### I2 — `I2_per_class_gain_vs_zeroshot`  (9 plots)
One plot per **dataset × family** (3 × 3).

Horizontal bar chart of **(small CICLe best − large zero-shot)** per-class F1
delta in percentage points, sorted from largest gain to largest loss.

- **Green bars:** small CICLe beats the large zero-shot baseline for this class
- **Red bars:** large zero-shot is still ahead

**Reading:** Identifies the classes that benefit most and least from using the
small model with CICLe instead of the large model's zero-shot baseline. Classes
with large positive bars are the primary beneficiaries.

---

### I3 — `I3_class_win_rate`  (3 plots)
One plot per **dataset**.

For each class, computes the fraction of matched configuration pairs —
across all families × embeddings × shots × variants — where the small
model's CICLe per-class F1 exceeds the large model's few-shot per-class F1.
The best small CICLe config (over alpha × classifier) is compared against the
matched large few-shot score for each (family, embedding, shots, variant) slice.

Sorted horizontal bar chart, green = win rate above 50%, red = below 50%.
A vertical dashed line marks 50% (parity).

**Reading:** Shows which classes are consistently easier or harder for the
small model relative to the large one across all configurations. A class with
a high win rate is reliably handled better by the small CICLe model; a class
near 0% is one where the large model almost always wins.

---

### I4 — `I4_confusion_matrix_comparison`  (9 plots)
One plot per **dataset × family** (3 × 3).

Side-by-side row-normalised confusion matrices:
- **Left:** small model CICLe (best config)
- **Right:** large model few-shot (best config)

Row normalisation means each cell shows the recall for that true class
(i.e., what fraction of true-class examples are predicted as each class).
Blue intensity encodes recall; the diagonal represents correct predictions.

**Reading:** Structural comparison of the two models' error patterns. If both
matrices have the same off-diagonal hotspots, the models make the same
systematic mistakes — they are confused by the same class boundaries. Cells
that are dark in one matrix but not the other reveal class-specific
differences in error behaviour.

---

### I5 — `I5_per_class_delta_vs_frequency`  (3 plots)
One plot per **dataset**.

Scatter plot:
- **X-axis:** class support (test-set count, from the large few-shot record)
- **Y-axis:** per-class F1 delta (small CICLe best − large few-shot best) in
  percentage points
- **Colour:** model family

Each point represents one class from one family. A horizontal dashed line
marks zero (parity). Points above zero = small CICLe wins for that class;
below = large few-shot wins.

**Reading:** Tests whether rare classes benefit more or less from CICLe. A
positive slope (rare classes above the line) would suggest CICLe particularly
helps the long tail. A negative slope would mean the large model's advantage
concentrates on rare classes that need more examples to classify correctly.
