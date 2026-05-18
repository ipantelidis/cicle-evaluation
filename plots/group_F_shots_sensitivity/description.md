# Group F — Shots Sensitivity

**Research goal:** Group F analyses how the number of in-context examples
(shots ∈ {1, 2, 4, 8}) affects macro-F1, whether returns diminish with more
shots, which models extract the most value per added example, and how quickly
small and large models saturate. F5 isolates the cold-start scenario (1-shot
only) to highlight the hardest practical setting.

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

### F1 — `F1_return_on_shots`  (18 plots)
One plot per **dataset × model** (3 × 6).

Axes:
- **X-axis:** shot transition label (1→2, 2→4, 4→8)
- **Y-axis:** ΔMacro-F1 in percentage points (gain over the previous shots value)

Lines:
- **CICLe** (solid red): best CICLe config optimised over all
  embeddings × variant × alpha × classifier
- **Few-shot** (dashed blue): best few-shot config optimised over all
  embeddings × variant

A horizontal dotted line at Δ=0 marks the break-even point (no gain from
adding more shots).

**Reading:** Steep early drops indicate rapid diminishing returns — most of the
gain is already captured at 1 or 2 shots. A flat curve from 2→4 to 4→8 means
the model has saturated. Negative values mean adding shots actually hurts the
best-config performance.

---

### F2 — `F2_shot_efficiency`  (18 plots)
One plot per **dataset × model** (3 × 6).

Axes:
- **X-axis:** number of shots (1, 2, 4, 8)
- **Y-axis:** Macro-F1 ÷ shots (efficiency — F1 earned per example provided)

Lines:
- **CICLe** (solid red) and **Few-shot** (dashed blue), each using the best
  config at that shots value.

**Reading:** A model that scores 60% with 2 shots is more efficient than one
that needs 4 shots for the same score. Declining curves show that later shots
yield proportionally less value. Comparing models on this metric reveals which
architecture makes the best use of limited annotation budget.

---

### F3 — `F3_shots_embedding_heatmap_per_model`  (18 plots)
One plot per **dataset × model** (3 × 6).

Two side-by-side heatmaps per plot: **CICLe** (left) and **Few-shot** (right).

Heatmap structure:
- **Rows:** embedding (Contriever, MiniLM, TF-IDF)
- **Columns:** shots (1, 2, 4, 8)
- **Cell value:** best macro-F1 for that (dataset, model, embedding, shots),
  optimised over variant × alpha × classifier for CICLe and over variant for
  few-shot

Sequential green palette (darker = higher F1). Cell values annotated.

**Reading:** Reveals whether the best embedding changes with shot count (rows
change ranking across columns) and how quickly performance grows along the
shots axis for each embedding. Comparing the two heatmaps shows whether CICLe
changes the shots-by-embedding interaction relative to plain few-shot.

---

### F4 — `F4_shots_saturation_small_vs_large`  (9 plots)
One plot per **dataset × family** (3 × 3).

Axes:
- **X-axis:** number of shots (1, 2, 4, 8)
- **Y-axis:** macro-F1 (best CICLe config)

Lines:
- **Large model** (solid dark): best CICLe F1 optimised over all
  embeddings × variant × alpha × classifier
- **Small model** (dashed warm): same

Values annotated directly above each point.

**Reading:** Shows the saturation profile within each family. If the small
model's curve flattens earlier than the large model's, it saturates sooner.
If the gap between the two lines narrows with more shots, additional examples
help the small model close the performance deficit.

---

### F5 — `F5_one_shot_cold_start`  (3 plots)
One plot per **dataset**.

Axes:
- **X-axis:** model (all 6 models, small and large)
- **Y-axis:** macro-F1

Three grouped bars per model:
- **Zero-shot** (grey): the model's single zero-shot score (no examples)
- **Few-shot 1-shot** (blue): best few-shot macro-F1 at 1 shot, optimised
  over embeddings × variant
- **CICLe 1-shot** (red): best CICLe macro-F1 at 1 shot, optimised over
  embeddings × variant × alpha × classifier

Values annotated above each bar.

**Reading:** The cold-start panel — the hardest practical scenario where only
a single labelled example is available. Shows whether CICLe and few-shot
already improve over zero-shot at 1 shot, and which models benefit most from
that single example.
