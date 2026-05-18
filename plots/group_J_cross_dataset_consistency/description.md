# Group J — Cross-Dataset Consistency

**Research goal:** Group J examines whether findings from one dataset
generalise to the others. It asks whether model and embedding rankings are
stable across datasets, whether the small-model advantage holds consistently,
and which dataset is most sensitive to the choice of method. All three
datasets — Yahoo Answers, Go Emotions, and SemEval-18 — differ substantially
in number of classes (10, 28, 20), domain, and difficulty, making
cross-dataset consistency a meaningful test of robustness.

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

### J1 — `J1_spearman_rank_correlation`  (1 plot)
A 3×3 Spearman ρ correlation matrix.

For each dataset, the six models are ranked by their best macro-F1 across all
methods and configurations. The Spearman ρ between two datasets' ranking
vectors measures how consistently models rank relative to each other.

Diverging palette: green = high positive correlation (consistent rankings),
red = negative (reversed rankings), white = no correlation.

**Reading:** A value close to +1 means a model that is best on one dataset is
also best on the other. Values well below 1 indicate that performance on one
dataset is a poor predictor of performance on another — the ranking is
dataset-dependent.

---

### J2 — `J2_model_rankings_per_dataset`  (1 plot)
A 6×3 grid: rows = models (sorted by mean rank across datasets), columns =
datasets.

Each cell contains the model's rank on that dataset (1 = best macro-F1 across
all methods and configs). Cell colour encodes rank on a red-to-green scale
(green = top rank, red = bottom rank).

**Reading:** A model with consistent colour across all three columns has a
stable rank regardless of dataset. Cells that switch from green to red across
columns indicate a model whose relative performance is dataset-dependent.

---

### J3 — `J3_embedding_rankings_per_dataset`  (1 plot)
Same structure as J2 but for the three embeddings (Contriever, MiniLM,
TF-IDF), ranked by their mean best CICLe macro-F1 averaged over all models.

**Reading:** Shows whether the embedding ranking (e.g. MiniLM > Contriever >
TF-IDF) holds across all datasets or whether the best embedding depends on the
specific classification task.

---

### J4 — `J4_small_vs_large_gap_consistency`  (3 plots)
One plot per **family** (Llama, Mistral, Qwen).

Bar chart with one bar per dataset. Bar height = (small model CICLe best −
large model few-shot best) in percentage points. Green = small model wins,
red = large model still ahead. Values annotated above/below each bar.

**Reading:** Shows whether the small-model advantage (or deficit) is consistent
across datasets for each family. A pattern where some bars are green and others
red indicates the advantage is dataset-dependent. A uniformly green or red
pattern indicates a family-level finding that holds regardless of task.

---

### J5 — `J5_dataset_difficulty_profile`  (1 plot)
Two-panel figure.

**Left panel:** Mean best macro-F1 per dataset × method (averaged over all
models). Three grouped bars per dataset: Zero-shot (grey), Few-shot (blue),
CICLe (red).

**Right panel:** Standard deviation of the mean macro-F1 across datasets,
one bar per method. A higher standard deviation means the method's performance
varies more across datasets — the method is more dataset-sensitive.

**Reading:** The left panel shows the absolute difficulty ordering of datasets
within each method. The right panel answers: which method's performance is most
affected by dataset choice? A method with high std is powerful on easy datasets
but struggles on hard ones; a method with low std is more uniformly applicable.
