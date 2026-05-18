"""
Group J — Cross-Dataset Consistency
=====================================
Are rankings of models and embeddings stable across datasets? Is the
small-model advantage consistent, and which dataset is most sensitive to
method choice?

Run:
    python generate_group_J.py

Outputs are written to subdirectories J1 … J5 relative to this file.
"""

import glob
import json
import os
import re
import warnings
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "/home/v25/ippa6201/cicle-evaluation"

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = ["yahoo-answers", "go-emotions", "semeval-18"]
DATASET_LABELS = {
    "yahoo-answers": "Yahoo Answers",
    "go-emotions":   "Go Emotions",
    "semeval-18":    "SemEval-18",
}
DATASET_PREFIX = {
    "yahoo-answers": "yahoo",
    "go-emotions":   "go-emotions",
    "semeval-18":    "semeval-18",
}

# ── Models ────────────────────────────────────────────────────────────────────
FAMILIES = {
    "Llama":   {"large": "llama-3.1-8b",   "small": "llama-3.2-3b"},
    "Mistral": {"large": "mistral-7b-v0.3", "small": "ministral-3b"},
    "Qwen":    {"large": "qwen-2.5-7b",     "small": "qwen-2.5-3b"},
}
ALL_MODELS = [m for pair in FAMILIES.values() for m in pair.values()]
MODEL_LABELS = {
    "llama-3.1-8b":    "Llama 3.1-8B",
    "llama-3.2-3b":    "Llama 3.2-3B",
    "mistral-7b-v0.3": "Mistral 7B",
    "ministral-3b":    "Ministral 3B",
    "qwen-2.5-7b":     "Qwen 2.5-7B",
    "qwen-2.5-3b":     "Qwen 2.5-3B",
}
FAMILY_COLORS = {"Llama": "#4361ee", "Mistral": "#7209b7", "Qwen": "#e07c00"}

SHOTS      = [1, 2, 4, 8]
EMBEDDINGS = ["contriever", "minilm", "tfidf"]
EMB_LABELS = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
VARIANTS   = ["pc", "fixed"]

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.axisbelow":    True,
})


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _f1(payload):
    return payload["classification_report"]["macro avg"]["f1-score"]


def load_records():
    records = []
    target_models = set(ALL_MODELS)

    for ds in DATASETS:
        pfx      = DATASET_PREFIX[ds]
        pred_dir = f"{BASE_DIR}/{ds}/results/predictions"

        for path in glob.glob(f"{pred_dir}/*.json"):
            stem = os.path.basename(path).replace(".json", "")
            if not stem.startswith(f"{pfx}-"):
                continue
            name = stem[len(pfx) + 1:]

            m = re.match(r"^(.+)-zeroshot-2\.0k-samples$", name)
            if m and m.group(1) in target_models:
                records.append({"dataset": ds, "llm": m.group(1),
                                "method": "zeroshot", "embedding": None,
                                "shots": 0, "variant": None,
                                "macro_f1": _f1(_load_json(path))})
                continue

            m = re.match(
                r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?$",
                name)
            if m and m.group(1) in target_models:
                records.append({"dataset": ds, "llm": m.group(1),
                                "method": "fewshot", "embedding": m.group(2),
                                "shots": int(m.group(3)),
                                "variant": m.group(4) or "none",
                                "macro_f1": _f1(_load_json(path))})
                continue

            m = re.match(
                r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots"
                r"(?:-(pc|fixed))?-([\d.]+)-α$", name)
            if m and m.group(1) in target_models:
                records.append({"dataset": ds, "llm": m.group(1),
                                "method": "cicle", "embedding": m.group(2),
                                "shots": int(m.group(4)),
                                "variant": m.group(5) or "none",
                                "macro_f1": _f1(_load_json(path))})

    return records


# ── Reduction helpers ─────────────────────────────────────────────────────────
def _max(vals):
    clean = [v for v in vals if not np.isnan(v)]
    return max(clean) if clean else np.nan


def best_overall(ds, llm):
    """Best macro-F1 across all methods and configs for a (dataset, model)."""
    sub = [r for r in RECORDS if r["dataset"] == ds and r["llm"] == llm]
    return _max([r["macro_f1"] for r in sub])


def best_cicle(ds, llm):
    sub = [r for r in RECORDS if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "cicle"]
    return _max([r["macro_f1"] for r in sub])


def best_fewshot(ds, llm):
    sub = [r for r in RECORDS if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "fewshot"]
    return _max([r["macro_f1"] for r in sub])


def zeroshot_val(ds, llm):
    sub = [r for r in RECORDS if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "zeroshot"]
    return sub[0]["macro_f1"] if sub else np.nan


def best_cicle_emb(ds, emb):
    """Best CICLe F1 for this embedding, averaged over all models."""
    sub = [r for r in RECORDS if r["dataset"] == ds and r["method"] == "cicle"
           and r["embedding"] == emb]
    vals = [r["macro_f1"] for r in sub]
    clean = [v for v in vals if not np.isnan(v)]
    return float(np.mean(clean)) if clean else np.nan


def best_method_ds(ds, method):
    """Best macro-F1 for a (dataset, method), maximised over all configs."""
    sub = [r for r in RECORDS if r["dataset"] == ds and r["method"] == method]
    return _max([r["macro_f1"] for r in sub])


# ── Shared helpers ────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_fname(s):
    return s.replace("-", "_").replace(".", "")


def _spearman(x, y):
    valid = [(xi, yi) for xi, yi in zip(x, y)
             if not np.isnan(xi) and not np.isnan(yi)]
    if len(valid) < 3:
        return np.nan
    xs, ys = zip(*valid)
    rho, _ = stats.spearmanr(xs, ys)
    return rho


# ── J1: Spearman ρ heatmap — dataset × dataset ───────────────────────────────
def plot_J1():
    """
    1 plot. 3×3 Spearman ρ matrix.
    For each dataset pair: correlate the vector of per-model best macro-F1
    scores to measure how consistently models rank across datasets.
    """
    out = ensure_dir("J1_spearman_rank_correlation")

    # One score vector per dataset: best macro-F1 per model
    scores = {
        ds: np.array([best_overall(ds, llm) for llm in ALL_MODELS])
        for ds in DATASETS
    }

    n   = len(DATASETS)
    mat = np.full((n, n), np.nan)
    for i, ds_i in enumerate(DATASETS):
        for j, ds_j in enumerate(DATASETS):
            mat[i, j] = _spearman(scores[ds_i], scores[ds_j])

    ds_labels = [DATASET_LABELS[ds] for ds in DATASETS]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(ds_labels, rotation=20, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(ds_labels)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Spearman ρ", fontsize=9)

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not np.isnan(v):
                txt_color = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_color)

    ax.set_title(
        "Cross-dataset rank correlation  (Spearman ρ)\n"
        "Model rankings: how consistent are they across datasets?",
        fontweight="bold",
    )

    fname = "J1_spearman_rank_correlation.png"
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  J1 saved: {fname}")


# ── J2: Model ranking grid per dataset ───────────────────────────────────────
def plot_J2():
    """
    1 plot. Columns = datasets, rows = models sorted by rank within each
    dataset. Cell colour = rank position (1 = best, 6 = worst).
    """
    out = ensure_dir("J2_model_rankings_per_dataset")

    # Rank models per dataset (lower rank = better)
    ranks = {}
    for ds in DATASETS:
        scores_ds = [(llm, best_overall(ds, llm)) for llm in ALL_MODELS]
        scores_ds.sort(key=lambda x: x[1], reverse=True)
        ranks[ds] = {llm: i + 1 for i, (llm, _) in enumerate(scores_ds)}

    n_models  = len(ALL_MODELS)
    n_datasets = len(DATASETS)

    # Build matrix: rows = models (sorted by mean rank), cols = datasets
    mean_rank = {llm: np.mean([ranks[ds][llm] for ds in DATASETS])
                 for llm in ALL_MODELS}
    sorted_models = sorted(ALL_MODELS, key=lambda m: mean_rank[m])

    mat = np.array([[ranks[ds][llm] for ds in DATASETS]
                    for llm in sorted_models], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=1, vmax=n_models, aspect="auto")

    ax.set_xticks(range(n_datasets))
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS],
                       rotation=18, ha="right")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([MODEL_LABELS[m] for m in sorted_models])
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Rank  (1 = best)", fontsize=9)
    cbar.set_ticks(range(1, n_models + 1))

    for ri in range(n_models):
        for ci in range(n_datasets):
            v = mat[ri, ci]
            txt_color = "white" if v <= 2 or v >= n_models - 1 else "black"
            ax.text(ci, ri, f"{int(v)}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=txt_color)

    ax.set_title(
        "Model rankings per dataset\n"
        "(sorted by mean rank; 1 = best macro-F1 across all methods)",
        fontweight="bold",
    )

    fname = "J2_model_rankings_per_dataset.png"
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  J2 saved: {fname}")


# ── J3: Embedding ranking grid per dataset ────────────────────────────────────
def plot_J3():
    """
    1 plot. Same structure as J2 but for embeddings (3 rows, 3 cols).
    Score = best CICLe macro-F1 for that embedding, averaged over models.
    """
    out = ensure_dir("J3_embedding_rankings_per_dataset")

    ranks = {}
    for ds in DATASETS:
        scores_ds = [(emb, best_cicle_emb(ds, emb)) for emb in EMBEDDINGS]
        scores_ds.sort(key=lambda x: x[1], reverse=True)
        ranks[ds] = {emb: i + 1 for i, (emb, _) in enumerate(scores_ds)}

    mean_rank = {emb: np.mean([ranks[ds][emb] for ds in DATASETS])
                 for emb in EMBEDDINGS}
    sorted_embs = sorted(EMBEDDINGS, key=lambda e: mean_rank[e])

    mat = np.array([[ranks[ds][emb] for ds in DATASETS]
                    for emb in sorted_embs], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=1, vmax=len(EMBEDDINGS),
                   aspect="auto")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS],
                       rotation=18, ha="right")
    ax.set_yticks(range(len(EMBEDDINGS)))
    ax.set_yticklabels([EMB_LABELS[e] for e in sorted_embs])
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Rank  (1 = best)", fontsize=9)
    cbar.set_ticks([1, 2, 3])

    for ri in range(len(EMBEDDINGS)):
        for ci in range(len(DATASETS)):
            v = mat[ri, ci]
            txt_color = "white" if v == 1 or v == len(EMBEDDINGS) else "black"
            ax.text(ci, ri, f"{int(v)}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_color)

    ax.set_title(
        "Embedding rankings per dataset  (CICLe)\n"
        "(sorted by mean rank; 1 = best mean macro-F1 across models)",
        fontweight="bold",
    )

    fname = "J3_embedding_rankings_per_dataset.png"
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  J3 saved: {fname}")


# ── J4: Small-vs-large gap consistency ───────────────────────────────────────
def plot_J4():
    """
    3 plots (one per family).
    X = dataset. Y = (small CICLe best − large few-shot best) in pp.
    Shows whether the small-model advantage is consistent across datasets.
    """
    out = ensure_dir("J4_small_vs_large_gap_consistency")

    for fam, models in FAMILIES.items():
        large, small = models["large"], models["small"]

        gaps = [
            (best_cicle(ds, small) - best_fewshot(ds, large)) * 100
            for ds in DATASETS
        ]
        colors = ["#2dc653" if g >= 0 else "#e63946" for g in gaps]
        ds_labels = [DATASET_LABELS[ds] for ds in DATASETS]

        fig, ax = plt.subplots(figsize=(6, 4.5))

        bars = ax.bar(ds_labels, gaps, color=colors, edgecolor="white",
                      linewidth=1.2, width=0.5)
        for bar, v in zip(bars, gaps):
            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                offset = 0.15 if v >= 0 else -0.15
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + offset, f"{v:+.1f} pp",
                        ha="center", va=va, fontsize=10, fontweight="bold")

        ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_ylabel("Small CICLe − Large few-shot  (pp)")
        ax.set_title(
            f"{fam} family\n"
            f"Small-model advantage across datasets  "
            f"({MODEL_LABELS[small]} CICLe best vs. {MODEL_LABELS[large]} few-shot best)",
            fontweight="bold",
        )

        fname = f"J4_{fam.lower()}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  J4 saved: {fname}")


# ── J5: Dataset difficulty profile ───────────────────────────────────────────
def plot_J5():
    """
    1 plot. For each method (zeroshot, fewshot, cicle): show the mean
    best macro-F1 per dataset, and the standard deviation across datasets.
    A high std means performance is dataset-sensitive for that method.
    """
    out = ensure_dir("J5_dataset_difficulty_profile")

    methods      = ["zeroshot", "fewshot", "cicle"]
    method_labels = {"zeroshot": "Zero-shot", "fewshot": "Few-shot",
                     "cicle": "CICLe"}
    method_colors = {"zeroshot": "#6c757d", "fewshot": "#4361ee",
                     "cicle": "#e63946"}

    # For each method × dataset: mean best macro-F1 over all models
    means = {}
    for method in methods:
        means[method] = []
        for ds in DATASETS:
            if method == "zeroshot":
                vals = [zeroshot_val(ds, llm) for llm in ALL_MODELS]
            elif method == "fewshot":
                vals = [best_fewshot(ds, llm) for llm in ALL_MODELS]
            else:
                vals = [best_cicle(ds, llm) for llm in ALL_MODELS]
            clean = [v for v in vals if not np.isnan(v)]
            means[method].append(float(np.mean(clean)) if clean else np.nan)

    ds_labels = [DATASET_LABELS[ds] for ds in DATASETS]
    x = np.arange(len(DATASETS))
    w = 0.26
    offsets = [-w, 0, w]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: mean best macro-F1 per dataset × method
    ax = axes[0]
    for method, offset in zip(methods, offsets):
        vals = means[method]
        bars = ax.bar(x + offset, vals, w,
                      color=method_colors[method],
                      label=method_labels[method],
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("Mean best Macro-F1  (averaged over models)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
    ax.legend(framealpha=0.9)
    ax.set_title("Mean best Macro-F1 per dataset and method",
                 fontweight="bold")

    # Right panel: std across datasets per method (dataset sensitivity)
    ax = axes[1]
    stds = {method: float(np.nanstd(means[method])) * 100
            for method in methods}
    bars = ax.bar(
        [method_labels[m] for m in methods],
        [stds[m] for m in methods],
        color=[method_colors[m] for m in methods],
        edgecolor="white", linewidth=1.2, width=0.5,
    )
    for bar, v in zip(bars, [stds[m] for m in methods]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{v:.2f} pp", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Std of mean Macro-F1 across datasets  (pp)")
    ax.set_title("Dataset sensitivity per method\n"
                 "(higher = more variable across datasets)",
                 fontweight="bold")

    fig.suptitle(
        "Dataset difficulty profile  —  how much does performance vary across datasets?",
        fontweight="bold", fontsize=12,
    )

    fname = "J5_dataset_difficulty_profile.png"
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  J5 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group J data …")
    RECORDS = load_records()
    print(f"  Records loaded: {len(RECORDS)}")

    print("\nGenerating Group J plots …")
    print("J1 — Spearman ρ heatmap: dataset × dataset rank correlation  (1 plot)")
    plot_J1()
    print("J2 — Model ranking grid per dataset  (1 plot)")
    plot_J2()
    print("J3 — Embedding ranking grid per dataset  (1 plot)")
    plot_J3()
    print("J4 — Small-vs-large gap consistency across datasets  (3 plots)")
    plot_J4()
    print("J5 — Dataset difficulty profile  (1 plot)")
    plot_J5()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
