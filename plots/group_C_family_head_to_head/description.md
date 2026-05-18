# Group C — Model Family: Small vs. Large Head-to-Head

**Research goal:** Group C compares the small and large models *within the same
family* under matched settings. The central question is not whether the small
model beats a generic baseline, but how much of the large model's performance
it can recover when we compare Llama against Llama, Mistral against Mistral,
and Qwen against Qwen.

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

### C1 — `C1_cicle_small_vs_large`  (9 plots)
One plot per dataset × family.  
Each plot is a **2×2 grid**, one subplot per shots value (1, 2, 4, 8).

Axes:
- **X-axis:** embedding (`Contriever`, `MiniLM`, `TF-IDF`)
- **Y-axis:** macro-F1

Bars:
- **Large CICLe**
- **Small CICLe**

How values are computed:
- for a fixed `(dataset, family, model, embedding, shots)`, the bar height is
  the best CICLe macro-F1 after optimising over **variant × alpha × classifier**

**Reading:** Shows how much performance is lost or retained when moving from
the large model to the small model within the same family, while keeping the
method fixed to CICLe.

---

### C2 — `C2_fewshot_small_vs_large`  (9 plots)
One plot per dataset × family.  
Each plot is a **2×2 grid**, one subplot per shots value (1, 2, 4, 8).

Axes:
- **X-axis:** embedding (`Contriever`, `MiniLM`, `TF-IDF`)
- **Y-axis:** macro-F1

Bars:
- **Large few-shot**
- **Small few-shot**

How values are computed:
- for a fixed `(dataset, family, model, embedding, shots)`, the bar height is
  the best few-shot macro-F1 after optimising over **variant**

**Reading:** The few-shot analogue of C1. It reveals whether the large-vs-small
gap is already present in plain few-shot prompting before CICLe is applied.

---

### C3 — `C3_delta_heatmap_cicle`  (9 plots)
One plot per dataset × family.  
Rows: embeddings.  
Columns: shots.

Cell value:
- **small-model CICLe best**
- minus
- **large-model CICLe best**

How values are computed:
- each cell fixes `(dataset, family, embedding, shots)`
- both small and large CICLe scores are optimised over **variant × alpha ×
  classifier**

Values are shown in **percentage points**.

**Reading:** Green cells mean the small model matches or beats the large model
under CICLe; red cells show where the large model still holds an advantage.

---

### C4 — `C4_delta_heatmap_small_cicle_vs_large_fewshot`  (9 plots)
One plot per dataset × family.  
Rows: embeddings.  
Columns: shots.

Cell value:
- **small-model CICLe best**
- minus
- **large-model few-shot best**

How values are computed:
- each cell fixes `(dataset, family, embedding, shots)`
- the small-model CICLe score is optimised over **variant × alpha × classifier**
- the large-model few-shot score is optimised over **variant**

Values are shown in **percentage points**.

**Reading:** This is the most policy-relevant heatmap in Group C: can the small
model with CICLe close the gap to the large model running plain few-shot?

---

### C5 — `C5_delta_distribution_small_cicle_vs_large_fewshot`  (9 plots)
One plot per dataset × family.  
X-axis: delta macro-F1.  
Y-axis: count.

Delta definition:
- **small-model CICLe**
- minus
- **large-model few-shot**

How deltas are formed:
- each histogram entry corresponds to one matched
  `(embedding, shots, variant)` configuration
- for the small-model CICLe score, the value is the best over
  **alpha × classifier**
- for the large-model few-shot score, the value is the exact matched few-shot
  value for that `(embedding, shots, variant)` setting

The vertical line at zero marks parity.

**Reading:** Shows not just the average gap, but the full distribution: whether
the small model is usually close, usually behind, or occasionally ahead by a
large margin.

---

### C6 — `C6_all_families_side_by_side`  (3 plots)
One plot per dataset.  
X-axis: model family (`Llama`, `Mistral`, `Qwen`).  
Y-axis: macro-F1.

Bars:
- **Small CICLe (best)**
- **Large few-shot (best)**
- **Large zero-shot**

How values are computed:
- **Small CICLe (best):** best small-model CICLe score over all
  **embedding × shots × variant × alpha × classifier**
- **Large few-shot (best):** best large-model few-shot score over all
  **embedding × shots × variant**
- **Large zero-shot:** the single zero-shot score for the large model

**Reading:** A compact family-level summary. It shows, for each dataset, how
close the best small-model CICLe run gets to the best large-model few-shot
reference and how both compare with the large model's zero-shot baseline.

