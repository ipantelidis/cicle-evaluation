"""
Group G — pc vs. fixed Variant
================================
Compares the two sampling strategies used in CICLe and few-shot:
  pc    — k shots per class (class-balanced)
  fixed — k shots total (class-agnostic)

Run:
    python generate_group_G.py

Outputs are written to subdirectories G1 … G5 relative to this file.
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
CLASSIFIERS = ["lr", "svm"]

VARIANT_COLORS = {"pc": "#4361ee", "fixed": "#e63946"}
METHOD_COLORS  = {"cicle": "#e63946", "fewshot": "#4361ee"}
EPSILON        = 0.001   # tie threshold (≈ 0.1 pp)

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
    """Load CICLe and few-shot records for all target models."""
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

            # few-shot
            m = re.match(
                r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?$",
                name,
            )
            if m and m.group(1) in target_models:
                records.append({
                    "dataset": ds, "llm": m.group(1), "method": "fewshot",
                    "embedding": m.group(2), "shots": int(m.group(3)),
                    "variant": m.group(4) or "none",
                    "alpha": None, "clf": None,
                    "macro_f1": _f1(_load_json(path)),
                })
                continue

            # CICLe
            m = re.match(
                r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots"
                r"(?:-(pc|fixed))?-([\d.]+)-α$",
                name,
            )
            if m and m.group(1) in target_models:
                records.append({
                    "dataset": ds, "llm": m.group(1), "method": "cicle",
                    "embedding": m.group(2), "shots": int(m.group(4)),
                    "variant": m.group(5) or "none",
                    "alpha": float(m.group(6)), "clf": m.group(3),
                    "macro_f1": _f1(_load_json(path)),
                })

    return records


# ── Matched-pair helpers ──────────────────────────────────────────────────────
def cicle_pairs(ds):
    """
    Yield (pc_f1, fixed_f1, llm, emb, shots) for every matched CICLe config
    (dataset, llm, emb, shots, alpha, clf).
    """
    cicle = [r for r in RECORDS if r["dataset"] == ds and r["method"] == "cicle"]
    lookup = {
        (r["llm"], r["embedding"], r["shots"], r["alpha"], r["clf"], r["variant"]): r["macro_f1"]
        for r in cicle
    }
    seen = set()
    for r in cicle:
        key_base = (r["llm"], r["embedding"], r["shots"], r["alpha"], r["clf"])
        if key_base in seen:
            continue
        seen.add(key_base)
        pc_f1    = lookup.get((*key_base, "pc"),    np.nan)
        fixed_f1 = lookup.get((*key_base, "fixed"), np.nan)
        if not np.isnan(pc_f1) and not np.isnan(fixed_f1):
            yield pc_f1, fixed_f1, r["llm"], r["embedding"], r["shots"]


def fewshot_pairs(ds):
    """
    Yield (pc_f1, fixed_f1, llm, emb, shots) for every matched few-shot config
    (dataset, llm, emb, shots).
    """
    fs = [r for r in RECORDS if r["dataset"] == ds and r["method"] == "fewshot"]
    lookup = {
        (r["llm"], r["embedding"], r["shots"], r["variant"]): r["macro_f1"]
        for r in fs
    }
    seen = set()
    for r in fs:
        key_base = (r["llm"], r["embedding"], r["shots"])
        if key_base in seen:
            continue
        seen.add(key_base)
        pc_f1    = lookup.get((*key_base, "pc"),    np.nan)
        fixed_f1 = lookup.get((*key_base, "fixed"), np.nan)
        if not np.isnan(pc_f1) and not np.isnan(fixed_f1):
            yield pc_f1, fixed_f1, r["llm"], r["embedding"], r["shots"]


# ── Shared helpers ────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_fname(s):
    return s.replace("-", "_").replace(".", "")


def _boxplot_style(ax, bp, color):
    """Apply consistent styling to a boxplot."""
    for element in ["boxes", "whiskers", "caps", "medians"]:
        plt.setp(bp[element], color=color, linewidth=1.6)
    plt.setp(bp["fliers"], marker="o", markerfacecolor=color,
             markersize=4, alpha=0.5, markeredgewidth=0)
    plt.setp(bp["medians"], color="white", linewidth=2)


def _delta_boxplot(ax, groups, group_labels, title, method):
    """
    Draw a boxplot of (fixed − pc) deltas.
    groups: list of lists of delta values, one per group.
    """
    color = METHOD_COLORS[method]
    positions = np.arange(1, len(groups) + 1)
    bp = ax.boxplot(
        [g if g else [np.nan] for g in groups],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        notch=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    _boxplot_style(ax, bp, color)

    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, rotation=15, ha="right")
    ax.set_ylabel("Fixed − PC  (pp)")
    ax.set_title(title, fontweight="bold")


# ── G1: Scatter — pc vs. fixed macro-F1 ──────────────────────────────────────
def plot_G1():
    """
    6 plots (3 datasets × 2 methods).
    Each point = one matched (model × embedding × shots [× alpha × clf]) config.
    X = pc macro-F1, Y = fixed macro-F1. Diagonal = parity.
    """
    out = ensure_dir("G1_scatter_pc_vs_fixed")

    for ds in DATASETS:
        for method, pair_fn in [("cicle", cicle_pairs), ("fewshot", fewshot_pairs)]:
            pairs = list(pair_fn(ds))
            if not pairs:
                continue

            pc_vals    = [p[0] for p in pairs]
            fixed_vals = [p[1] for p in pairs]

            lo = min(min(pc_vals), min(fixed_vals)) - 0.02
            hi = max(max(pc_vals), max(fixed_vals)) + 0.02

            fixed_wins = sum(1 for pc, fx in zip(pc_vals, fixed_vals)
                             if fx - pc > EPSILON)
            pc_wins    = sum(1 for pc, fx in zip(pc_vals, fixed_vals)
                             if pc - fx > EPSILON)
            total      = len(pairs)

            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            color = METHOD_COLORS[method]
            ax.scatter(pc_vals, fixed_vals, color=color, s=40,
                       alpha=0.65, edgecolors="white", linewidths=0.4,
                       zorder=3)
            ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.2,
                    linestyle="--", zorder=2, label="y = x  (parity)")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")

            ax.text(0.04, 0.96,
                    f"Fixed wins: {fixed_wins}/{total} "
                    f"({100*fixed_wins/total:.0f}%)",
                    transform=ax.transAxes, fontsize=9,
                    va="top", color="#e63946", fontweight="bold")
            ax.text(0.04, 0.88,
                    f"PC wins: {pc_wins}/{total} "
                    f"({100*pc_wins/total:.0f}%)",
                    transform=ax.transAxes, fontsize=9,
                    va="top", color="#4361ee", fontweight="bold")

            ax.set_xlabel("PC variant  (Macro-F1)")
            ax.set_ylabel("Fixed variant  (Macro-F1)")
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            method_label = "CICLe" if method == "cicle" else "Few-shot"
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {method_label}\n"
                f"PC vs. Fixed  (each point = one matched config)",
                fontweight="bold",
            )
            ax.legend(loc="lower right", framealpha=0.9)

            fname = f"G1_{_safe_fname(ds)}_{method}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  G1 saved: {fname}")


# ── G2: Delta boxplot — grouped by shots ─────────────────────────────────────
def plot_G2():
    """
    6 plots (3 datasets × 2 methods).
    Boxplot of (fixed − pc) delta, one box per shots value.
    """
    out = ensure_dir("G2_delta_by_shots")

    for ds in DATASETS:
        for method, pair_fn in [("cicle", cicle_pairs), ("fewshot", fewshot_pairs)]:
            pairs = list(pair_fn(ds))
            if not pairs:
                continue

            groups = [
                [(fx - pc) * 100 for pc, fx, llm, emb, s in pairs if s == shots]
                for shots in SHOTS
            ]
            labels = [f"{s} shot{'s' if s > 1 else ''}" for s in SHOTS]

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            method_label = "CICLe" if method == "cicle" else "Few-shot"
            _delta_boxplot(
                ax, groups, labels,
                title=(f"{DATASET_LABELS[ds]}  ·  {method_label}\n"
                       f"Fixed − PC delta  (pp), grouped by shots"),
                method=method,
            )
            ax.set_xlabel("Number of shots")

            fname = f"G2_{_safe_fname(ds)}_{method}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  G2 saved: {fname}")


# ── G3: Delta boxplot — grouped by model ─────────────────────────────────────
def plot_G3():
    """
    6 plots (3 datasets × 2 methods).
    Boxplot of (fixed − pc) delta, one box per model.
    """
    out = ensure_dir("G3_delta_by_model")

    for ds in DATASETS:
        for method, pair_fn in [("cicle", cicle_pairs), ("fewshot", fewshot_pairs)]:
            pairs = list(pair_fn(ds))
            if not pairs:
                continue

            groups = [
                [(fx - pc) * 100 for pc, fx, llm, emb, s in pairs if llm == m]
                for m in ALL_MODELS
            ]
            labels = [MODEL_LABELS[m] for m in ALL_MODELS]

            fig, ax = plt.subplots(figsize=(9, 4.5))
            method_label = "CICLe" if method == "cicle" else "Few-shot"
            _delta_boxplot(
                ax, groups, labels,
                title=(f"{DATASET_LABELS[ds]}  ·  {method_label}\n"
                       f"Fixed − PC delta  (pp), grouped by model"),
                method=method,
            )

            fname = f"G3_{_safe_fname(ds)}_{method}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  G3 saved: {fname}")


# ── G4: Delta boxplot — grouped by embedding ─────────────────────────────────
def plot_G4():
    """
    6 plots (3 datasets × 2 methods).
    Boxplot of (fixed − pc) delta, one box per embedding.
    """
    out = ensure_dir("G4_delta_by_embedding")

    for ds in DATASETS:
        for method, pair_fn in [("cicle", cicle_pairs), ("fewshot", fewshot_pairs)]:
            pairs = list(pair_fn(ds))
            if not pairs:
                continue

            groups = [
                [(fx - pc) * 100 for pc, fx, llm, emb, s in pairs if emb == e]
                for e in EMBEDDINGS
            ]
            labels = [EMB_LABELS[e] for e in EMBEDDINGS]

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            method_label = "CICLe" if method == "cicle" else "Few-shot"
            _delta_boxplot(
                ax, groups, labels,
                title=(f"{DATASET_LABELS[ds]}  ·  {method_label}\n"
                       f"Fixed − PC delta  (pp), grouped by embedding"),
                method=method,
            )
            ax.set_xlabel("Embedding")

            fname = f"G4_{_safe_fname(ds)}_{method}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  G4 saved: {fname}")


# ── G5: Win-rate summary — pc vs. fixed ──────────────────────────────────────
def plot_G5():
    """
    3 plots (one per dataset).
    Horizontal stacked bars: pc wins / tie / fixed wins.
    One bar per (model × method) combination.
    """
    out = ensure_dir("G5_winrate_pc_vs_fixed")

    for ds in DATASETS:
        row_labels = []
        pc_pcts, tie_pcts, fixed_pcts = [], [], []

        for method, pair_fn in [("cicle", cicle_pairs), ("fewshot", fewshot_pairs)]:
            method_label = "CICLe" if method == "cicle" else "Few-shot"
            pairs = list(pair_fn(ds))

            for llm in ALL_MODELS:
                sub = [(pc, fx) for pc, fx, lm, emb, s in pairs if lm == llm]
                if not sub:
                    continue
                total      = len(sub)
                fixed_wins = sum(1 for pc, fx in sub if fx - pc >  EPSILON)
                pc_wins    = sum(1 for pc, fx in sub if pc - fx >  EPSILON)
                ties       = total - fixed_wins - pc_wins

                row_labels.append(f"{MODEL_LABELS[llm]}\n({method_label})")
                pc_pcts.append(100 * pc_wins    / total)
                tie_pcts.append(100 * ties       / total)
                fixed_pcts.append(100 * fixed_wins / total)

        if not row_labels:
            continue

        n = len(row_labels)
        fig, ax = plt.subplots(figsize=(8, 0.55 * n + 1.8))
        y = np.arange(n)

        ax.barh(y, pc_pcts,    color="#4361ee", label="PC wins")
        ax.barh(y, tie_pcts,   left=pc_pcts,
                color="#ffd166", label=f"Tie (±{EPSILON*100:.1f} pp)")
        ax.barh(y, fixed_pcts,
                left=[p + t for p, t in zip(pc_pcts, tie_pcts)],
                color="#e63946", label="Fixed wins")

        for i, (p, t, f) in enumerate(zip(pc_pcts, tie_pcts, fixed_pcts)):
            if p > 6:
                ax.text(p / 2, i, f"{p:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            if f > 6:
                ax.text(p + t + f / 2, i, f"{f:.0f}%",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel("Percentage of matched configs (%)")
        ax.set_xlim(0, 100)
        ax.axvline(50, color="#333333", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Win-rate: PC vs. Fixed variant  "
            f"(each config = one matched embedding × shots [× alpha × clf])",
            fontweight="bold",
        )
        ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
        ax.grid(axis="x", alpha=0.3)
        ax.yaxis.grid(False)

        fname = f"G5_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  G5 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group G data …")
    RECORDS = load_records()
    print(f"  Records loaded: {len(RECORDS)}")

    print("\nGenerating Group G plots …")
    print("G1 — Scatter: pc vs. fixed macro-F1  (6 plots)")
    plot_G1()
    print("G2 — Delta boxplot by shots  (6 plots)")
    plot_G2()
    print("G3 — Delta boxplot by model  (6 plots)")
    plot_G3()
    print("G4 — Delta boxplot by embedding  (6 plots)")
    plot_G4()
    print("G5 — Win-rate summary: pc vs. fixed  (3 plots)")
    plot_G5()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
