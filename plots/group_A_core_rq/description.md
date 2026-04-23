# Group A — Core Research Question

**Research question:** Can CICLe with a small model (3B) achieve performance
comparable to a large model (7–8B) operating in zero-shot or few-shot mode?

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

### A1 — `A1_gap_cicle_vs_zeroshot_best`  (9 plots)
One plot per dataset × family.  
X-axis: number of shots (1, 2, 4, 8).  
Elements:
- **Best-config line** (solid, coloured): oracle upper bound — for each shots
  value independently, the maximum macro-F1 across all (embedding × variant ×
  alpha × classifier) combinations. Values annotated directly above each point.
- **Large zero-shot** (dashed, dark): flat reference line for the large model.
- **Small zero-shot** (dotted, grey): flat reference for the small model itself.
- **Green shading**: region where the best-config curve already exceeds the
  large zero-shot line.

**Reading:** Shows the ceiling of what small CICLe can achieve. Once the
coloured curve crosses the dashed line, the small model has matched the large
one at its optimal configuration.

---

### A2 — `A2_gap_cicle_vs_zeroshot_mean`  (9 plots)
One plot per dataset × family. Same axes and references as A1, but shows the
**distribution of performance** across all configurations instead of the peak.  
Elements:
- **Mean line** (solid, coloured): average macro-F1 across all (embedding ×
  variant × alpha × classifier) combinations at each shots value. Values
  annotated directly above each point.
- **± std band** (shaded, same colour): spread of the distribution, showing
  how consistent performance is across configs.
- **Large zero-shot** (dashed, dark): flat reference line for the large model.
- **Small zero-shot** (dotted, grey): flat reference for the small model itself.

**Reading:** Complements A1 — where A1 shows the ceiling, A2 shows the
realistic typical performance. A narrow band means the result is robust to
config choice; a wide band means it is sensitive.

---

### A3 — `A3_gap_with_fewshot/`  (9 + 9 + 9 = 27 plots)
One plot per dataset × family.  
Three lines per plot:
- **Large few-shot best** (dotted, grey, squares): best few-shot macro-F1 for
  the large model across all embeddings and variants at each shots value.
- **Small few-shot best** (dashed, family colour, triangles, α 0.65): same for
  the small model.
- **Small CICLe best** (solid, family colour, circles): best CICLe macro-F1 for
  the small model across all configs (embedding × variant × alpha × clf).
  Values annotated directly above each point.

The 27 plots are split across three sub-folders:

#### `A3_gap_with_fewshot/` — all variants combined  (9 plots)
Best config optimised jointly over **pc and fixed** variants. Gives the overall
ceiling comparison.

#### `A3_gap_with_fewshot/A3_1_gap_with_fewshot_pc/`  (9 plots)
Same setting but restricted to the **PC (per-class)** variant only, where
*k* shots means **k examples per class**. Shows whether the denser,
class-balanced sampling strategy alone is enough for the small model to match
the large baseline.

#### `A3_gap_with_fewshot/A3_2_gap_with_fewshot_fixed/`  (9 plots)
Same setting but restricted to the **Fixed** variant only, where *k* shots
means **k examples in total** regardless of class. Useful for comparing the
two shot-counting strategies head-to-head when read alongside A3_1.

**Reading:** A3 shows the overall ceiling; A3_1 and A3_2 reveal whether the
advantage (or gap) is driven by the per-class or the fixed shot-counting
strategy — and whether one calibration variant dominates the other.

---

### A4 — `A4_breakeven_shots`  (3 plots, one per dataset)
For each model family: the minimum number of shots *k* at which the best
small-model CICLe configuration first matches (≥) two references:

- **Green bar — vs. large zero-shot**: the large model's zero-shot macro-F1
  (a single fixed score, independent of *k*).
- **Blue bar — vs. large few-shot (same k)**: the large model's best few-shot
  macro-F1 **at the same shot count *k*** as small CICLe — a head-to-head
  comparison where both models see the same number of examples.

Families for which small CICLe never reaches parity across all tested shot
counts (k = 1, 2, 4, 8) are annotated **N/A** in the bar's colour (no bar is
drawn, to avoid a misleading zero-height bar).

**Reading:** The most compact summary of how many labelled examples are needed
for the small model to become competitive. A lower bar is better; N/A means
the small model does not catch up within 8 shots.

---

### A5 — `A5_scatter_vs_zeroshot/`  (3 + 3 = 6 plots)
These plots compare **small-model CICLe** against the **large model's
zero-shot baseline**.

#### `A5_scatter_vs_zeroshot/A5_1_scatter_vs_zeroshot_per_family/`  (3 plots)
One plot per **model family**.  
Axes:
- **X-axis:** the large model's zero-shot macro-F1
- **Y-axis:** the small model's CICLe macro-F1

How points are formed:
- Each point is one **(dataset × shots × embedding × variant)** combination.
- Point colour encodes the **number of shots**.
- For the small-model CICLe score, the plot uses the **best value over
  alpha × classifier** for the given (dataset, shots, embedding, variant).
- For the large-model zero-shot reference, there is **one fixed score per
  dataset**.

Visual guides:
- The diagonal `y = x` marks parity.
- Points **above** the diagonal mean the small model with CICLe beats the
  large model's zero-shot score.
- The win count in the corner reports how many plotted configs are above the
  diagonal.

**Reading:** A5_1 asks: *within one family, across all three datasets, how
often does small CICLe clear the large model's zero-shot bar, and at what
shot counts?*

---

#### `A5_scatter_vs_zeroshot/A5_2_scatter_vs_zeroshot_all_families/`  (3 plots)
One plot per **dataset**.  
Axes:
- **X-axis:** the large model's zero-shot macro-F1
- **Y-axis:** the small model's CICLe macro-F1

How points are formed:
- Each point is one **(family × shots × embedding × variant)** combination.
- Point colour encodes the **model family**.
- As in A5_1, the small-model CICLe score is the **best over alpha ×
  classifier** for the given (family, shots, embedding, variant).
- The large-model zero-shot score is the **single fixed zero-shot baseline**
  for that family on that dataset.

Visual guides:
- The diagonal `y = x` marks parity.
- Points above the diagonal are wins for the small model.
- The corner annotation reports the overall small-model win count.

**Reading:** A5_2 asks: *within one dataset, which families cluster above or
below large-model zero-shot, and how often does small CICLe win overall?*

---

### A6 — `A6_scatter_vs_fewshot/`  (9 + 3 = 12 plots)
These plots compare **small-model CICLe** against the **large model's
few-shot performance**.

#### `A6_scatter_vs_fewshot/A6_1_scatter_vs_fewshot_per_family/`  (9 plots)
One plot per **dataset × family**.  
Axes:
- **X-axis:** the large model's few-shot macro-F1
- **Y-axis:** the small model's CICLe macro-F1

How points are formed:
- Each point is one **(shots × embedding × variant)** combination.
- Point colour encodes the **number of shots**.
- The large-model few-shot score and small-model CICLe score are **matched on
  dataset, family, shots, embedding, and variant**.
- For the small-model CICLe score, the value is the **best over alpha ×
  classifier** within that matched (shots, embedding, variant) slice.

Visual guides:
- The diagonal `y = x` marks parity.
- Points above the diagonal mean small CICLe beats large few-shot for that
  matched configuration.
- The win annotation reports the count and percentage of those wins.

**Reading:** A6_1 is the most direct head-to-head scatter: *for a fixed
dataset and family, where does small CICLe sit relative to large few-shot
across matched settings?*

---

#### `A6_scatter_vs_fewshot/A6_2_scatter_vs_fewshot_all_families/`  (3 plots)
One plot per **dataset** with all families overlaid.  
Axes are the same as A6_1:
- **X-axis:** large model few-shot macro-F1
- **Y-axis:** small model CICLe macro-F1

How points are formed:
- Each point is one **(family × shots × embedding × variant)** combination.
- Point colour encodes the **model family**.
- Scores are still matched on family, shots, embedding, and variant, with
  small-model CICLe taking the **best value over alpha × classifier**.

**Reading:** A6_2 gives the dataset-level picture: *across all families at
once, how often and by how much does small CICLe beat large few-shot?*

---

### A7 — `A7_grouped_bar_bestconfig`  (9 plots)
One plot per dataset × family.  
Seven grouped bars per plot: Baseline LR, Baseline SVM, Large zero-shot,
Large few-shot (best), Small few-shot (best), **Small CICLe (best)**,
Large CICLe (best). Values annotated on each bar. Small CICLe bar is
highlighted with a coloured border.

How the bar heights are computed:
- **Baseline LR / SVM:** one fixed TF-IDF baseline score per dataset.
- **Large zero-shot:** one fixed zero-shot score for the large model.
- **Large few-shot (best):** the best large-model few-shot score over all
  tested shot counts, embeddings, and variants.
- **Small few-shot (best):** the best small-model few-shot score over all
  tested shot counts, embeddings, and variants.
- **Small CICLe (best):** the best small-model CICLe score over all tested
  shot counts, embeddings, variants, alphas, and classifiers.
- **Large CICLe (best):** the best large-model CICLe score over the same CICLe
  configuration space.

**Reading:** The full ranking in one panel — where exactly does small CICLe
sit relative to every other method?

---

### A8 — `A8_heatmap_gap_zeroshot`  (12 plots)
One plot per dataset × shots value (3 datasets × 4 shots = 12 plots).  
Rows: embedding × variant combinations (6 rows).  
Columns: model families (3 columns).  
Cell value:
- **small-model CICLe best** for that `(dataset, family, shots, embedding, variant)`,
  where "best" means optimised over **alpha × classifier**
- minus
- the **single large-model zero-shot score** for that `(dataset, family)`

Values are shown in **percentage points** (not raw 0–1 F1 units).  
Diverging palette:
- **green** = small CICLe beats large zero-shot
- **red** = small CICLe is below large zero-shot

Each cell is annotated with its exact signed gap (for example `+1.8` means the
small model is 1.8 percentage points above the large zero-shot baseline).  
**Reading:** Reveals which (embedding, variant, family) combinations drive the
advantage and how the gap evolves with more shots.

---

### A9 — `A9_winrate`  (3 plots, one per dataset)
One plot per dataset.  
One horizontal bar per **family × shots** combination.

For each row, the script compares matched configuration pairs across:
- all **embeddings** (`contriever`, `minilm`, `tfidf`)
- both **variants** (`pc`, `fixed`)

Comparison rule for each matched pair:
- **Large few-shot:** exact score for that `(dataset, family, shots, embedding, variant)`
- **Small CICLe:** best score for the same `(dataset, family, shots, embedding, variant)`
  after optimising over **alpha × classifier**

Each matched pair is classified as:
- **Win:** small CICLe > large few-shot
- **Tie:** absolute difference ≤ 0.001 macro-F1 (about ±0.1 percentage points)
- **Loss:** small CICLe < large few-shot

The stacked bar shows the percentage of wins / ties / losses across those
matched pairs, with percentages annotated inside the coloured segments when
large enough.

**Reading:** Summarises robustness — is the small model competitive across the
board, or only in cherry-picked configurations?
