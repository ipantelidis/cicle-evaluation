# Group K — Prompt Length / Computational Efficiency

**Research goal:** Group K examines the computational cost side of CICLe and
few-shot prompting, using prompt length (in characters) as a proxy for
inference cost. It asks whether longer prompts yield better macro-F1, which
(model × method) combination is most efficient, how context length grows with
shots and embedding choice, and whether small and large models build prompts
of comparable length for the same shot count.

Prompt length data is loaded from the `/results/lengths/` directory, which
mirrors the naming of the `/results/predictions/` directory. Each length file
contains `prompt_lengths.mean` (mean over the 2k test samples).

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

### K1 — `K1_f1_vs_prompt_length`  (18 plots)
One plot per **dataset × model** (3 × 6).

Axes:
- **X-axis:** mean prompt length (characters)
- **Y-axis:** macro-F1

Points coloured by shots value. Marker shape distinguishes method: circle (○)
for CICLe, square (□) for few-shot. A linear regression fit line is drawn per
method (solid red for CICLe, dashed blue for few-shot).

**Reading:** A positive slope means longer prompts tend to yield higher F1 for
that model and method. A flat or negative slope means adding context (longer
prompts) does not improve — or even hurts — performance on average. The
relative position of CICLe vs. few-shot points at the same x-value shows
whether CICLe achieves higher F1 for a given prompt budget.

---

### K2 — `K2_efficiency_f1_per_prompt_length`  (3 plots)
One plot per **dataset**.

For each (model × method) combination, the efficiency metric is:

    efficiency = (best macro-F1 / mean prompt length of best config) × 1000

Bar chart sorted in descending order of efficiency. Bars coloured by method
(red = CICLe, blue = few-shot).

**Reading:** A higher bar means the model achieves more F1 per character of
prompt context. A model-method pair that is both high-performing and
short-prompted will dominate. This is the single most compact summary of
which combination gives the best performance-per-cost trade-off.

---

### K3 — `K3_prompt_length_by_shots_and_variant`  (18 plots)
One plot per **dataset × model** (3 × 6).

Violin plots grouped by shots value (1, 2, 4, 8). Within each shots group,
two violins are shown side by side: PC variant (blue) and Fixed variant (red).

Each violin shows the distribution of mean prompt lengths across all
(embedding [× alpha × classifier]) configurations for that (shots, variant)
slice. The white line marks the median.

**Reading:** Shows how the choice of variant (per-class vs. fixed total)
affects prompt length at each shot count. If the PC violin is consistently
higher than Fixed, the per-class strategy produces longer prompts (because it
retrieves k examples per class, so total examples = k × n_classes). The shape
of each violin reveals how variable prompt length is across configurations.

---

### K4 — `K4_prompt_length_vs_shots`  (18 plots)
One plot per **dataset × model** (3 × 6).

Two side-by-side panels: CICLe (left) and few-shot (right). Each panel shows
three lines, one per embedding (Contriever, MiniLM, TF-IDF). X = shots,
Y = mean prompt length averaged over variants (and alpha × classifier for
CICLe).

**Reading:** Quantifies how rapidly context grows with shots for each
embedding. A steeper slope means each additional shot adds more context.
TF-IDF typically produces longer prompts because it retrieves by TF-IDF
similarity which can select long documents. Comparing the two panels shows
whether CICLe's statistical filtering changes the growth rate relative to
plain few-shot.

---

### K5 — `K5_small_vs_large_prompt_length`  (9 plots)
One plot per **dataset × family** (3 × 3).

Side-by-side box plots for large (dark) and small (warm) models. One pair of
boxes per shots value (1, 2, 4, 8). Each box shows the distribution of mean
prompt lengths across all (method × embedding × variant [× alpha × clf])
configurations for that model and shots value. Outliers hidden for clarity.

**Reading:** Tests whether small and large models in the same family build
prompts of comparable length for the same shot count. If the boxes overlap
substantially, the prompt length is similar and any performance difference is
not due to context size. Systematic offset between the two boxes would suggest
the models retrieve different amounts of context.
