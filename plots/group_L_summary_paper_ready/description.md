# Group L — Summary / Paper-Ready Aggregates

**Research goal:** Group L provides the high-level, paper-ready summary views
of the entire evaluation. Each plot is designed to give a complete picture of
results in a single panel — suitable for inclusion in a paper's main body,
appendix, or supplementary material. Rather than drilling into specific
dimensions, these plots aggregate across all configurations to answer: *what
is the overall ranking, how much does CICLe improve over few-shot, and where
do the best configurations sit on the efficiency frontier?*

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

### L1 — `L1_grand_summary_heatmap`  (3 plots)
One plot per **dataset**.

A heatmap with:
- **Rows:** method (Baseline LR, Baseline SVM, Zero-shot, Few-shot, CICLe)
- **Columns:** all 6 models (Llama 3.1-8B, Llama 3.2-3B, Mistral 7B,
  Ministral 3B, Qwen 2.5-7B, Qwen 2.5-3B)
- **Cell value:** best macro-F1 across all configs for that (method, model)
  slice; baseline rows repeat the same value across all model columns since
  TF-IDF baselines are model-agnostic

Sequential green palette (darker = higher F1). Cell values annotated.

**Reading:** The single-panel reference view of the entire experiment. At a
glance it shows which method × model combinations reach the highest F1, how
much the methods differ within each model, and which models are consistently
strong across methods.

---

### L2 — `L2_method_family_boxplots`  (3 plots)
One plot per **dataset**.

Box plots of the full macro-F1 distribution across all configurations,
grouped by method on the x-axis (Baseline, Zero-shot, Few-shot, CICLe),
coloured by model family (Llama = blue, Mistral = purple, Qwen = orange).
Outliers hidden for clarity.

**Reading:** Reveals the spread of performance within each method — a narrow
box means the method is robust to config choice; a wide box means results vary
greatly. Comparing family colours within each method group shows whether one
family is consistently better than the others under that method.

---

### L3 — `L3_cicle_improvement_over_fewshot`  (3 plots)
One plot per **dataset**.

For each model: **CICLe best − few-shot best** in percentage points, matched
on (embedding × shots) so the comparison is fair. Bars sorted by mean delta
(highest gain on the left). Error bars show the standard deviation of deltas
across matched configurations.

- **Green bars:** CICLe improves over few-shot for this model
- **Red bars:** few-shot is still better for this model

**Reading:** The most direct answer to "does CICLe help beyond few-shot
prompting?" A large positive bar means the model benefits substantially from
CICLe's retrieval-based example selection. Models with negative bars get no
lift from CICLe and are best served by plain few-shot prompting.

---

### L4 — `L4_top10_configs_table`  (3 plots)
One plot per **dataset**.

A visual table (rendered as a matplotlib figure) listing the 10 best
configurations ranked by macro-F1. Columns:

| # | Model | Method | Embedding | Shots | Variant | Alpha | CLF | Macro-F1 |

The rank-1 row is highlighted in green. The table header uses a dark
background with white text. Alternating row shading aids readability.

**Reading:** Identifies the exact configuration that produced the best result
on each dataset, along with the full hyperparameter metadata. Useful for
reproducing the top result and for understanding which configuration choices
co-occur in the best runs.

---

### L5 — `L5_pareto_frontier`  (9 plots)
One plot per **dataset × family** (3 × 3).

Scatter of all (small model, large model) configurations in the
(mean prompt length, macro-F1) space. Background points show all
configurations; the solid line connects the **Pareto-optimal** points —
configurations that are not dominated by any other (i.e., no other config
achieves both shorter prompt and higher F1 simultaneously).

- **Dark colour:** large model
- **Warm colour:** small model

A point (x₁, y₁) dominates (x₂, y₂) if x₁ ≤ x₂ and y₁ ≥ y₂ with at
least one strict inequality (strictly shorter prompt or strictly higher F1).

**Reading:** The efficiency frontier plot — the most policy-relevant summary
for deployment. Configs on the Pareto frontier represent the best achievable
F1 at each budget level. If the small model's frontier overlaps with or
exceeds the large model's frontier at shorter prompt lengths, CICLe with the
small model achieves competitive quality at lower inference cost.
