# Group B — Per-Dataset × Per-Model Deep Dives

**Research goal:** Once Group A establishes the headline comparison, Group B
opens the box and looks inside each dataset × model combination. These plots
help explain *why* a model is strong or weak by breaking performance down by
embedding, alpha, classifier, variant, shots, and prompt length.

## Models

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

### B1 — `B1_macro_f1_vs_shots_per_embedding`  (18 plots)
One plot per dataset × model.  
X-axis: number of shots (1, 2, 4, 8).  
Y-axis: macro-F1.

Lines:
- **CICLe** lines are solid
- **Few-shot** lines are dashed
- Colour encodes the embedding (`Contriever`, `MiniLM`, `TF-IDF`)

How values are computed:
- **CICLe:** for a fixed `(dataset, model, embedding, shots)`, the plotted
  value is the best macro-F1 after optimising over **variant × alpha ×
  classifier**
- **Few-shot:** for a fixed `(dataset, model, embedding, shots)`, the plotted
  value is the best macro-F1 after optimising over **variant**

**Reading:** The cleanest first diagnostic. It shows how performance scales
with shots for each embedding, and whether CICLe consistently sits above
few-shot for the same model.

---

### B2 — `B2_cicle_alpha_sensitivity`  (18 plots)
One plot per dataset × model.  
X-axis: alpha (`0.01`, `0.05`, `0.10`, `0.20`).  
Y-axis: macro-F1.

Lines:
- one line per embedding

How values are computed:
- the current implementation is restricted to **2 shots only**
- for a fixed `(dataset, model, embedding, alpha)`, the plotted value is the
  best CICLe macro-F1 after optimising over **variant × classifier**

**Reading:** Shows how sensitive each model is to alpha at the 2-shot setting,
and whether different embeddings prefer different alpha values. A flatter line
means alpha matters less; a sharper peak means tuning alpha is more important.

---

### B3 — `B3_lr_vs_svm_per_embedding`  (18 plots)
One plot per dataset × model.  
X-axis: embedding × shots combinations.  
Y-axis: macro-F1.

Bars:
- **LR**
- **SVM**

How values are computed:
- for a fixed `(dataset, model, embedding, shots, classifier)`, the plotted
  value is the best CICLe macro-F1 after optimising over **alpha × variant**

**Reading:** Highlights whether classifier choice matters in practice, and
whether one classifier is only better for certain embeddings or shot counts.

---

### B4 — `B4_pc_vs_fixed_variant`  (18 plots)
One plot per dataset × model.  
X-axis: embedding × shots combinations, sorted by `(fixed - pc)` delta.  
Y-axis: macro-F1.

Bars:
- **PC**
- **Fixed**

How values are computed:
- for a fixed `(dataset, model, embedding, shots, variant)`, the plotted value
  is the best CICLe macro-F1 after optimising over **alpha × classifier**

**Reading:** Shows where the choice of variant matters most. Sorting the
categories by delta makes the largest `fixed` advantages appear on one side and
the largest `pc` advantages on the other.

---

### B5 — `B5_radar_cicle_vs_fewshot`  (18 plots)
One plot per dataset × model.

Axes:
- one radar axis per `(embedding × shots)` combination

Polygons:
- **CICLe**
- **Few-shot**

How values are computed:
- **CICLe:** for each `(dataset, model, embedding, shots)` axis, the plotted
  radius is the best macro-F1 after optimising over **variant × alpha ×
  classifier**
- **Few-shot:** for the same axis, the plotted radius is the best macro-F1
  after optimising over **variant**

**Reading:** Gives a compact performance fingerprint for one model. It is less
precise than B1 numerically, but useful for spotting broad strengths and
weaknesses across embeddings and shot counts.

---

### B6 — `B6_macro_f1_vs_prompt_length`  (18 plots)
One plot per dataset × model.  
X-axis: mean prompt length.  
Y-axis: macro-F1.

Points:
- colour encodes the method (`Zeroshot`, `Few-shot`, `CICLe`)
- point size encodes the number of shots

How points are formed:
- each point is one prediction file with a matching prompt-length file
- macro-F1 comes from the matching file under `results/predictions/`
- mean prompt length comes from the matching file under `results/lengths/`
- only records with a usable prompt-length value are plotted

Notes:
- in practice this includes **zeroshot**, **few-shot**, and **CICLe**
- the TF-IDF baselines are not shown because they do not have matching prompt
  length files

**Reading:** This is the efficiency view. It shows whether better performance
comes with substantially longer prompts, and whether CICLe sits above few-shot
at comparable prompt lengths.

## Draft status

This Group B package is currently a **draft scaffold**:
- the plots are wired up and generate real figures from the current results
- the reduction rules are intentionally simple and consistent
- some plots are narrower than the long-term design spec
- the next step is to refine styling, annotations, and any aggregation choices
  that should be made more paper-ready
