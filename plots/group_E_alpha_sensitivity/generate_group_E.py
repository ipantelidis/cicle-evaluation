"""
Group E — Alpha Sensitivity (CICLe Only)
=========================================
How sensitive is CICLe to the alpha significance threshold (0.01, 0.05, 0.10,
0.20), does the optimal value shift with shot count, and do small vs. large
models respond differently?

Run:
    python generate_group_E.py

Outputs are written to subdirectories E1 … E6 relative to this file.
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
FAMILY_COLORS = {"Llama": "#4361ee", "Mistral": "#7209b7", "Qwen": "#e07c00"}

ALPHAS     = [0.01, 0.05, 0.10, 0.20]
ALPHA_LABELS = ["0.01", "0.05", "0.10", "0.20"]
SHOTS      = [1, 2, 4, 8]
SHOTS_COLORS = {1: "#4cc9f0", 2: "#4361ee", 4: "#7209b7", 8: "#f72585"}
EMBEDDINGS = ["contriever", "minilm", "tfidf"]
EMB_LABELS = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
EMB_COLORS = {"contriever": "#4361ee", "minilm": "#e07c00", "tfidf": "#2dc653"}
VARIANTS   = ["fixed"]
CLASSIFIERS = ["lr", "svm"]
CLF_COLORS  = {"lr": "#4361ee", "svm": "#e63946"}
CLF_LABELS  = {"lr": "LR", "svm": "SVM"}

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
    """Load only CICLe records for the target models."""
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
            if not m:
                continue
            llm = m.group(1)
            if llm not in target_models:
                continue
            records.append({
                "dataset":   ds,
                "llm":       llm,
                "embedding": m.group(2),
                "clf":       m.group(3),
                "shots":     int(m.group(4)),
                "variant":   m.group(5) or "none",
                "alpha":     float(m.group(6)),
                "macro_f1":  _f1(_load_json(path)),
            })
    return records


# ── Reduction helpers ─────────────────────────────────────────────────────────
def _max(vals):
    clean = [v for v in vals if not np.isnan(v)]
    return max(clean) if clean else np.nan


def _mean(vals):
    clean = [v for v in vals if not np.isnan(v)]
    return float(np.mean(clean)) if clean else np.nan


def f1_alpha(ds, llm, emb, shots, alpha, clf=None, variant="fixed"):
    """Mean F1 over matching records — fixed variant by default."""
    sub = [
        r for r in RECORDS
        if r["dataset"] == ds and r["llm"] == llm
        and r["embedding"] == emb and r["shots"] == shots
        and r["alpha"] == alpha
        and r["variant"] == variant
        and (clf is None or r["clf"] == clf)
    ]
    return _mean([r["macro_f1"] for r in sub])


def best_alpha_for(ds, llm, emb, shots, clf=None, variant=None):
    """Return the alpha that maximises mean F1 for the given slice."""
    vals = {a: f1_alpha(ds, llm, emb, shots, a, clf, variant) for a in ALPHAS}
    valid = {a: v for a, v in vals.items() if not np.isnan(v)}
    return max(valid, key=valid.get) if valid else np.nan


# ── Shared helpers ────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_fname(s):
    return s.replace("-", "_").replace(".", "")


# ── E1: Alpha curves per model × embedding ────────────────────────────────────
def plot_E1():
    """
    54 plots (3 datasets × 6 models × 3 embeddings).
    X = alpha, one line per shots value. Shows optimal alpha per shots.
    """
    out = ensure_dir("E1_alpha_curves_per_model_per_embedding")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            for emb in EMBEDDINGS:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                has_data = False

                for shots in SHOTS:
                    vals = [f1_alpha(ds, llm, emb, shots, a) for a in ALPHAS]
                    if all(np.isnan(v) for v in vals):
                        continue
                    ax.plot(
                        ALPHAS, vals,
                        color=SHOTS_COLORS[shots], marker="o",
                        linewidth=2, markersize=7,
                        label=f"{shots} shot{'s' if shots > 1 else ''}",
                    )
                    has_data = True

                if not has_data:
                    plt.close(fig)
                    continue

                ax.set_xticks(ALPHAS)
                ax.set_xticklabels(ALPHA_LABELS)
                ax.set_xlabel("Alpha (α)")
                ax.set_ylabel("Macro-F1")
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
                ax.legend(framealpha=0.9)
                ax.set_title(
                    f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}  ·  {EMB_LABELS[emb]}\n"
                    f"Macro-F1 vs. alpha  (mean over variant × classifier)",
                    fontweight="bold",
                )

                fname = (f"E1_{_safe_fname(ds)}_{_safe_fname(llm)}"
                         f"_{emb}.png")
                fig.tight_layout()
                fig.savefig(os.path.join(out, fname), bbox_inches="tight")
                plt.close(fig)
                print(f"  E1 saved: {fname}")


# ── E2: Alpha × shots heatmap per model (averaged over embeddings & clf) ──────
def plot_E2():
    """
    18 plots (3 datasets × 6 models).
    4×4 heatmap: rows = alpha, cols = shots.
    """
    out = ensure_dir("E2_alpha_shots_heatmap_per_model")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            mat = np.full((len(ALPHAS), len(SHOTS)), np.nan)

            for ri, alpha in enumerate(ALPHAS):
                for ci, shots in enumerate(SHOTS):
                    vals = [
                        f1_alpha(ds, llm, emb, shots, alpha)
                        for emb in EMBEDDINGS
                    ]
                    mat[ri, ci] = _mean(vals)

            if np.all(np.isnan(mat)):
                continue

            fig, ax = plt.subplots(figsize=(6.5, 5))
            vmax = np.nanmax(mat)
            vmin = np.nanmin(mat)
            im = ax.imshow(mat, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(SHOTS)))
            ax.set_xticklabels([str(s) for s in SHOTS])
            ax.set_yticks(range(len(ALPHAS)))
            ax.set_yticklabels(ALPHA_LABELS)
            ax.set_xlabel("Shots")
            ax.set_ylabel("Alpha (α)")
            ax.grid(False)
            cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
            cbar.set_label("Mean Macro-F1", fontsize=9)
            cbar.ax.yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))

            for ri in range(mat.shape[0]):
                for ci in range(mat.shape[1]):
                    v = mat[ri, ci]
                    if not np.isnan(v):
                        ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                                fontsize=9, fontweight="bold", color="black")

            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"Alpha × shots  (mean Macro-F1 over embeddings and classifiers)",
                fontweight="bold",
            )

            fname = f"E2_{_safe_fname(ds)}_{_safe_fname(llm)}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  E2 saved: {fname}")


# ── E3: Alpha × shots heatmap per model × embedding ───────────────────────────
def plot_E3():
    """
    54 plots (3 datasets × 6 models × 3 embeddings).
    Same as E2 but not averaged over embeddings.
    """
    out = ensure_dir("E3_alpha_shots_heatmap_per_model_per_embedding")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            for emb in EMBEDDINGS:
                mat = np.full((len(ALPHAS), len(SHOTS)), np.nan)

                for ri, alpha in enumerate(ALPHAS):
                    for ci, shots in enumerate(SHOTS):
                        mat[ri, ci] = f1_alpha(ds, llm, emb, shots, alpha)

                if np.all(np.isnan(mat)):
                    continue

                fig, ax = plt.subplots(figsize=(6.5, 5))
                vmax = np.nanmax(mat)
                vmin = np.nanmin(mat)
                im = ax.imshow(mat, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_xticks(range(len(SHOTS)))
                ax.set_xticklabels([str(s) for s in SHOTS])
                ax.set_yticks(range(len(ALPHAS)))
                ax.set_yticklabels(ALPHA_LABELS)
                ax.set_xlabel("Shots")
                ax.set_ylabel("Alpha (α)")
                ax.grid(False)
                cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
                cbar.set_label("Mean Macro-F1", fontsize=9)
                cbar.ax.yaxis.set_major_formatter(
                    mticker.PercentFormatter(xmax=1, decimals=0))

                for ri in range(mat.shape[0]):
                    for ci in range(mat.shape[1]):
                        v = mat[ri, ci]
                        if not np.isnan(v):
                            ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                                    fontsize=9, fontweight="bold", color="black")

                ax.set_title(
                    f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}  ·  {EMB_LABELS[emb]}\n"
                    f"Alpha × shots  (mean Macro-F1 over classifiers and variants)",
                    fontweight="bold",
                )

                fname = f"E3_{_safe_fname(ds)}_{_safe_fname(llm)}_{emb}.png"
                fig.tight_layout()
                fig.savefig(os.path.join(out, fname), bbox_inches="tight")
                plt.close(fig)
                print(f"  E3 saved: {fname}")


# ── E4: Optimal alpha distribution ────────────────────────────────────────────
def plot_E4():
    """
    3 plots (one per dataset).
    For every (model × embedding × shots × clf × variant) config, find the
    best alpha. Bar chart of how often each alpha wins.
    """
    out = ensure_dir("E4_optimal_alpha_distribution")

    for ds in DATASETS:
        alpha_counts = {a: 0 for a in ALPHAS}

        for llm in ALL_MODELS:
            for emb in EMBEDDINGS:
                for shots in SHOTS:
                    for clf in CLASSIFIERS:
                        for variant in VARIANTS:
                            vals = {
                                a: f1_alpha(ds, llm, emb, shots, a, clf, variant)
                                for a in ALPHAS
                            }
                            valid = {a: v for a, v in vals.items()
                                     if not np.isnan(v)}
                            if valid:
                                winner = max(valid, key=valid.get)
                                alpha_counts[winner] += 1

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        colors = ["#4cc9f0", "#4361ee", "#7209b7", "#f72585"]
        bars = ax.bar(
            ALPHA_LABELS,
            [alpha_counts[a] for a in ALPHAS],
            color=colors, edgecolor="white", linewidth=1.2,
        )
        for bar, cnt in zip(bars, [alpha_counts[a] for a in ALPHAS]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(cnt), ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.set_xlabel("Alpha (α)")
        ax.set_ylabel("Win count")
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Optimal alpha distribution  "
            f"(configs where this alpha yields the highest Macro-F1)",
            fontweight="bold",
        )

        fname = f"E4_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  E4 saved: {fname}")


# ── E5: Alpha sensitivity by model size ───────────────────────────────────────
def plot_E5():
    """
    3 plots (one per dataset).
    Two panels (small / large). One line per family. X = alpha,
    Y = macro-F1 averaged over shots, embeddings, classifiers, variants.
    """
    out = ensure_dir("E5_alpha_sensitivity_by_model_size")

    for ds in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        for ax, size, title_suffix in zip(
            axes, ["large", "small"], ["Large models (7–8B)", "Small models (3B)"]
        ):
            for fam, models in FAMILIES.items():
                llm = models[size]
                vals = []
                for alpha in ALPHAS:
                    sub_vals = [
                        f1_alpha(ds, llm, emb, shots, alpha)
                        for emb in EMBEDDINGS
                        for shots in SHOTS
                    ]
                    vals.append(_mean(sub_vals))

                if all(np.isnan(v) for v in vals):
                    continue

                ax.plot(
                    ALPHAS, vals,
                    color=FAMILY_COLORS[fam], marker="o",
                    linewidth=2.2, markersize=8,
                    label=fam,
                )

            ax.set_xticks(ALPHAS)
            ax.set_xticklabels(ALPHA_LABELS)
            ax.set_xlabel("Alpha (α)")
            ax.set_ylabel("Macro-F1")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_title(title_suffix, fontweight="bold")
            ax.legend(framealpha=0.9)

        fig.suptitle(
            f"{DATASET_LABELS[ds]}\n"
            f"Alpha sensitivity by model size  "
            f"(mean over shots, embeddings, classifiers, variants)",
            fontweight="bold",
        )

        fname = f"E5_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  E5 saved: {fname}")


# ── E6: LR vs SVM alpha sensitivity per model ─────────────────────────────────
def plot_E6():
    """
    18 plots (3 datasets × 6 models).
    One subplot per embedding. Two lines per subplot: LR and SVM.
    X = alpha, Y = mean Macro-F1 over shots and variants.
    """
    out = ensure_dir("E6_lr_vs_svm_alpha_sensitivity")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
            has_data = False

            for ax, emb in zip(axes, EMBEDDINGS):
                for clf in CLASSIFIERS:
                    vals = []
                    for alpha in ALPHAS:
                        sub = [
                            f1_alpha(ds, llm, emb, shots, alpha, clf=clf)
                            for shots in SHOTS
                        ]
                        vals.append(_mean(sub))

                    if all(np.isnan(v) for v in vals):
                        continue

                    ax.plot(
                        ALPHAS, vals,
                        color=CLF_COLORS[clf], marker="o",
                        linewidth=2, markersize=7,
                        label=CLF_LABELS[clf],
                    )
                    has_data = True

                ax.set_xticks(ALPHAS)
                ax.set_xticklabels(ALPHA_LABELS)
                ax.set_xlabel("Alpha (α)")
                ax.set_title(EMB_LABELS[emb], fontweight="bold")
                ax.legend(framealpha=0.9)

            if not has_data:
                plt.close(fig)
                continue

            axes[0].set_ylabel("Macro-F1")
            axes[0].yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))

            fig.suptitle(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"LR vs SVM alpha sensitivity  (mean over shots and variants)",
                fontweight="bold",
            )

            fname = f"E6_{_safe_fname(ds)}_{_safe_fname(llm)}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  E6 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group E data …")
    RECORDS = load_records()
    print(f"  CICLe records loaded: {len(RECORDS)}")

    print("\nGenerating Group E plots …")
    print("E1 — Alpha curves per model × embedding  (54 plots)")
    plot_E1()
    print("E2 — Alpha × shots heatmap per model  (18 plots)")
    plot_E2()
    print("E3 — Alpha × shots heatmap per model × embedding  (54 plots)")
    plot_E3()
    print("E4 — Optimal alpha distribution  (3 plots)")
    plot_E4()
    print("E5 — Alpha sensitivity by model size  (3 plots)")
    plot_E5()
    print("E6 — LR vs SVM alpha sensitivity  (18 plots)")
    plot_E6()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
