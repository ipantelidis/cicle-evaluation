# Group D — Embedding Type Analysis

**Research goal:** Group D asks which retrieval embedding — Contriever, MiniLM,
or TF-IDF — produces the best results, and whether that ranking is stable across
models, datasets, shot counts, and sampling variants. The embedding is the
primary dimension of analysis; all other axes (model, shots, variant, alpha,
classifier) are marginalised over by taking the best or the mean as appropriate.

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

### D1 — `D1_embedding_ranking_per_model`  (6 plots)
One plot per **dataset × method** (CICLe, few-shot).

Axes:
- **X-axis:** model (all 6 models, small and large)
- **Y-axis:** macro-F1

Bars: one bar group per model, three bars = three embeddings (Contriever,
MiniLM, TF-IDF). Values annotated above each bar.

How values are computed:
- for CICLe: best macro-F1 over all **shots × variant × alpha × classifier**
  for the given (dataset, model, embedding)
- for few-shot: best macro-F1 over all **shots × variant** for the given
  (dataset, model, embedding)

**Reading:** Shows which embedding dominates within each model, and whether
the ranking is consistent across models.

---

### D2 — `D2_embedding_shots_heatmap_per_model`  (18 plots)
One plot per **dataset × model** (3 datasets × 6 models).

Each plot contains two side-by-side heatmaps (CICLe | few-shot).

Heatmap structure:
- **Rows:** embedding (Contriever, MiniLM, TF-IDF)
- **Columns:** shots (1, 2, 4, 8)
- **Cell value:** best macro-F1 for that (dataset, model, embedding, shots),
  optimised over **variant × alpha × classifier** for CICLe and over
  **variant** for few-shot

Sequential green palette (darker = higher F1). Cell values annotated.

**Reading:** Reveals how embedding choice interacts with the number of shots
for a specific model. A consistent row ranking across all columns means the
embedding winner is stable regardless of shot count.

---

### D3 — `D3_embedding_shots_heatmap_per_method`  (6 plots)
One plot per **dataset × method** (CICLe, few-shot).

Same structure as D2 (rows = embeddings, columns = shots), but each cell
contains the **mean macro-F1 averaged over all 6 models** for that
(dataset, method, embedding, shots) combination.

**Reading:** The dataset-level aggregate picture. Smooths out model-specific
idiosyncrasies to show the overall trend of embedding × shots interaction.

---

### D4 — `D4_embedding_consistency_parallel`  (3 plots)
One plot per **dataset**, with two side-by-side panels (CICLe | few-shot).

Each panel is a parallel-coordinates chart:
- **Axes (x positions):** the three embeddings (Contriever, MiniLM, TF-IDF)
- **Y-axis:** macro-F1
- **Lines:** one per model (6 total); solid = large model, dashed = small model;
  colour encodes the model family (Llama = blue, Mistral = purple, Qwen = orange)

How values are computed:
- each point on a line is the best macro-F1 for that model on that embedding,
  optimised over all **shots × variant × alpha × classifier** (CICLe) or
  **shots × variant** (few-shot)

**Reading:** If all lines have the same shape (same ups and downs), the
embedding ranking is consistent across models. Crossing lines indicate that
the best embedding varies by model.

---

### D5 — `D5_embedding_win_counts`  (3 plots)
One plot per **dataset**.

For each embedding, counts how many (model × shots × variant) configurations
it produces the highest macro-F1 among the three embeddings. Grouped bars
show the win count separately for CICLe and few-shot.

How wins are counted:
- for each (dataset, model, shots, variant) combination, the embedding with
  the highest best macro-F1 is declared the winner
- ties go to the first embedding in alphabetical order

**Reading:** A single-number summary of embedding dominance. An embedding that
wins across many configurations is robustly the best choice.

---

### D6 — `D6_embedding_variant_interaction`  (6 plots)
One plot per **dataset × method** (CICLe, few-shot).

A 3×2 heatmap:
- **Rows:** embedding (Contriever, MiniLM, TF-IDF)
- **Columns:** sampling variant (PC — per-class, Fixed — total)
- **Cell value:** mean macro-F1 averaged over all **models × shots**

Sequential green palette. Cell values annotated.

**Reading:** Shows whether the advantage of one embedding depends on the
variant. If a row is uniformly the best regardless of column, the embedding
wins independently of variant; if the best cell is concentrated in one
(embedding, variant) corner, the two choices interact.
