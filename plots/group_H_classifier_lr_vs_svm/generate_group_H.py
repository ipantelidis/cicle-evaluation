"""
Group H — Classifier: LR vs. SVM (CICLe Only)
===============================================
Does the choice of linear classifier (Logistic Regression vs. Support Vector
Machine) matter for CICLe? Does the preference interact with alpha, shots,
embedding, or model?

Run:
    python generate_group_H.py

Outputs are written to subdirectories H1 … H6 relative to this file.
"""

import glob
import json
import os
import re
import warnings

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
    "yahoo-answers": "Yahoo Answers (10 classes)",
    "go-emotions":   "Go Emotions (28 classes)",
    "semeval-18":    "SemEval-18 (20 classes)",
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

SHOTS      = [1, 2, 4, 8]
EMBEDDINGS = ["contriever", "minilm", "tfidf"]
EMB_LABELS = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
ALPHAS     = [0.01, 0.05, 0.10, 0.20]
ALPHA_LABELS = ["0.01", "0.05", "0.10", "0.20"]
VARIANTS   = ["fixed"]

CLF_COLORS = {"lr": "#4361ee", "svm": "#e63946"}
EPSILON    = 0.001   # tie threshold

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
    """Load CICLe-only records for all target models."""
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

            m = re.match(
                r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots"
                r"(?:-(pc|fixed))?-([\d.]+)-α$",
                name,
            )
            if m and m.group(1) in target_models:
                records.append({
                    "dataset":   ds,
                    "llm":       m.group(1),
                    "embedding": m.group(2),
                    "clf":       m.group(3),
                    "shots":     int(m.group(4)),
                    "variant":   m.group(5) or "none",
                    "alpha":     float(m.group(6)),
                    "macro_f1":  _f1(_load_json(path)),
                })

    return records


# ── Matched-pair helper ───────────────────────────────────────────────────────
def matched_pairs(ds):
    """
    Yield (lr_f1, svm_f1, llm, emb, shots, alpha, variant) for every
    matched CICLe config (same dataset, llm, emb, shots, alpha, variant).
    """
    subset = [r for r in RECORDS if r["dataset"] == ds and r["variant"] == "fixed"]
    lookup = {
        (r["llm"], r["embedding"], r["shots"], r["alpha"], r["variant"], r["clf"]): r["macro_f1"]
        for r in subset
    }
    seen = set()
    for r in subset:
        key = (r["llm"], r["embedding"], r["shots"], r["alpha"], r["variant"])
        if key in seen:
            continue
        seen.add(key)
        lr_f1  = lookup.get((*key, "lr"),  np.nan)
        svm_f1 = lookup.get((*key, "svm"), np.nan)
        if not np.isnan(lr_f1) and not np.isnan(svm_f1):
            yield lr_f1, svm_f1, r["llm"], r["embedding"], r["shots"], r["alpha"], r["variant"]


# ── Shared helpers ────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_fname(s):
    return s.replace("-", "_").replace(".", "")


def _boxplot_style(ax, bp, color):
    for element in ["boxes", "whiskers", "caps", "medians"]:
        plt.setp(bp[element], color=color, linewidth=1.6)
    plt.setp(bp["fliers"], marker="o", markerfacecolor=color,
             markersize=4, alpha=0.5, markeredgewidth=0)
    plt.setp(bp["medians"], color="white", linewidth=2)


def _delta_boxplot(ax, groups, group_labels, title, color="#6c3483"):
    positions = np.arange(1, len(groups) + 1)
    bp = ax.boxplot(
        [g if g else [np.nan] for g in groups],
        positions=positions, widths=0.5,
        patch_artist=True, notch=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    _boxplot_style(ax, bp, color)
    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, rotation=15, ha="right")
    ax.set_ylabel("SVM − LR  (pp)")
    ax.set_title(title, fontweight="bold")


# ── H1: Scatter — LR vs. SVM macro-F1 ────────────────────────────────────────
def plot_H1():
    """
    3 plots (one per dataset).
    Each point = one matched (model × embedding × shots × alpha × variant) config.
    X = LR macro-F1, Y = SVM macro-F1. Diagonal = parity.
    """
    out = ensure_dir("H1_scatter_lr_vs_svm")

    for ds in DATASETS:
        pairs = list(matched_pairs(ds))
        if not pairs:
            continue

        lr_vals  = [p[0] for p in pairs]
        svm_vals = [p[1] for p in pairs]

        lo = min(min(lr_vals), min(svm_vals)) - 0.02
        hi = max(max(lr_vals), max(svm_vals)) + 0.02

        svm_wins = sum(1 for lr, sv in zip(lr_vals, svm_vals) if sv - lr >  EPSILON)
        lr_wins  = sum(1 for lr, sv in zip(lr_vals, svm_vals) if lr - sv >  EPSILON)
        total    = len(pairs)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(lr_vals, svm_vals, color="#7209b7", s=30,
                   alpha=0.55, edgecolors="white", linewidths=0.3, zorder=3)
        ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.2,
                linestyle="--", zorder=2, label="y = x  (parity)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        ax.text(0.04, 0.96,
                f"SVM wins: {svm_wins}/{total} ({100*svm_wins/total:.0f}%)",
                transform=ax.transAxes, fontsize=9,
                va="top", color=CLF_COLORS["svm"], fontweight="bold")
        ax.text(0.04, 0.88,
                f"LR wins:  {lr_wins}/{total} ({100*lr_wins/total:.0f}%)",
                transform=ax.transAxes, fontsize=9,
                va="top", color=CLF_COLORS["lr"], fontweight="bold")

        ax.set_xlabel("LR  (Macro-F1)")
        ax.set_ylabel("SVM  (Macro-F1)")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"LR vs. SVM  (each point = one matched CICLe config)",
            fontweight="bold",
        )
        ax.legend(loc="lower right", framealpha=0.9)

        fname = f"H1_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  H1 saved: {fname}")


# ── H2: Delta boxplot — grouped by model ─────────────────────────────────────
def plot_H2():
    """3 plots (one per dataset). Boxplot of (SVM − LR) grouped by model."""
    out = ensure_dir("H2_delta_by_model")

    for ds in DATASETS:
        pairs = list(matched_pairs(ds))
        if not pairs:
            continue

        groups = [
            [(sv - lr) * 100 for lr, sv, llm, emb, s, a, v in pairs if llm == m]
            for m in ALL_MODELS
        ]
        labels = [MODEL_LABELS[m] for m in ALL_MODELS]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        _delta_boxplot(ax, groups, labels,
                       title=(f"{DATASET_LABELS[ds]}\n"
                              f"SVM − LR delta  (pp), grouped by model"))
        fname = f"H2_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  H2 saved: {fname}")


# ── H3: Delta boxplot — grouped by embedding ─────────────────────────────────
def plot_H3():
    """3 plots (one per dataset). Boxplot of (SVM − LR) grouped by embedding."""
    out = ensure_dir("H3_delta_by_embedding")

    for ds in DATASETS:
        pairs = list(matched_pairs(ds))
        if not pairs:
            continue

        groups = [
            [(sv - lr) * 100 for lr, sv, llm, emb, s, a, v in pairs if emb == e]
            for e in EMBEDDINGS
        ]
        labels = [EMB_LABELS[e] for e in EMBEDDINGS]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _delta_boxplot(ax, groups, labels,
                       title=(f"{DATASET_LABELS[ds]}\n"
                              f"SVM − LR delta  (pp), grouped by embedding"))
        ax.set_xlabel("Embedding")
        fname = f"H3_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  H3 saved: {fname}")


# ── H4: Delta boxplot — grouped by alpha ─────────────────────────────────────
def plot_H4():
    """3 plots (one per dataset). Boxplot of (SVM − LR) grouped by alpha."""
    out = ensure_dir("H4_delta_by_alpha")

    for ds in DATASETS:
        pairs = list(matched_pairs(ds))
        if not pairs:
            continue

        groups = [
            [(sv - lr) * 100 for lr, sv, llm, emb, s, a, v in pairs if a == alpha]
            for alpha in ALPHAS
        ]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _delta_boxplot(ax, groups, ALPHA_LABELS,
                       title=(f"{DATASET_LABELS[ds]}\n"
                              f"SVM − LR delta  (pp), grouped by alpha"))
        ax.set_xlabel("Alpha (α)")
        fname = f"H4_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  H4 saved: {fname}")


# ── H5: Delta boxplot — grouped by shots ─────────────────────────────────────
def plot_H5():
    """3 plots (one per dataset). Boxplot of (SVM − LR) grouped by shots."""
    out = ensure_dir("H5_delta_by_shots")

    for ds in DATASETS:
        pairs = list(matched_pairs(ds))
        if not pairs:
            continue

        groups = [
            [(sv - lr) * 100 for lr, sv, llm, emb, s, a, v in pairs if s == shots]
            for shots in SHOTS
        ]
        labels = [f"{s} shot{'s' if s > 1 else ''}" for s in SHOTS]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        _delta_boxplot(ax, groups, labels,
                       title=(f"{DATASET_LABELS[ds]}\n"
                              f"SVM − LR delta  (pp), grouped by shots"))
        ax.set_xlabel("Number of shots")
        fname = f"H5_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  H5 saved: {fname}")


# ── H6: Alpha × shots heatmap of SVM − LR per model ─────────────────────────
def plot_H6():
    """
    18 plots (3 datasets × 6 models).
    Rows = alpha, cols = shots. Cell = mean(SVM − LR) over embeddings × variants.
    Diverging palette: green = SVM wins, red = LR wins.
    """
    out = ensure_dir("H6_alpha_shots_heatmap_svm_minus_lr")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            pairs = [
                p for p in matched_pairs(ds) if p[2] == llm
            ]
            if not pairs:
                continue

            mat = np.full((len(ALPHAS), len(SHOTS)), np.nan)
            for ri, alpha in enumerate(ALPHAS):
                for ci, shots in enumerate(SHOTS):
                    deltas = [
                        (sv - lr) * 100
                        for lr, sv, lm, emb, s, a, v in pairs
                        if s == shots and a == alpha
                    ]
                    if deltas:
                        mat[ri, ci] = float(np.mean(deltas))

            if np.all(np.isnan(mat)):
                continue

            fig, ax = plt.subplots(figsize=(6.5, 5))
            vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.1
            im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                           aspect="auto")
            ax.set_xticks(range(len(SHOTS)))
            ax.set_xticklabels([str(s) for s in SHOTS])
            ax.set_yticks(range(len(ALPHAS)))
            ax.set_yticklabels(ALPHA_LABELS)
            ax.set_xlabel("Shots")
            ax.set_ylabel("Alpha (α)")
            ax.grid(False)
            cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
            cbar.set_label("SVM − LR  (pp)", fontsize=9)

            for ri in range(mat.shape[0]):
                for ci in range(mat.shape[1]):
                    v = mat[ri, ci]
                    if not np.isnan(v):
                        txt_color = "white" if abs(v) > vmax * 0.6 else "black"
                        ax.text(ci, ri, f"{v:+.2f}",
                                ha="center", va="center",
                                fontsize=9, fontweight="bold", color=txt_color)

            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"SVM − LR  (pp)  ·  alpha × shots  "
                f"(mean over embeddings and variants)",
                fontweight="bold",
            )

            fname = f"H6_{_safe_fname(ds)}_{_safe_fname(llm)}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  H6 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group H data …")
    RECORDS = load_records()
    print(f"  CICLe records loaded: {len(RECORDS)}")

    print("\nGenerating Group H plots …")
    print("H1 — Scatter: LR vs. SVM macro-F1  (3 plots)")
    plot_H1()
    print("H2 — Delta boxplot by model  (3 plots)")
    plot_H2()
    print("H3 — Delta boxplot by embedding  (3 plots)")
    plot_H3()
    print("H4 — Delta boxplot by alpha  (3 plots)")
    plot_H4()
    print("H5 — Delta boxplot by shots  (3 plots)")
    plot_H5()
    print("H6 — Alpha × shots heatmap of SVM − LR per model  (18 plots)")
    plot_H6()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
