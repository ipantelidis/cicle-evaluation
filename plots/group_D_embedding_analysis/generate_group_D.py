"""
Group D — Embedding Type Analysis
==================================
Which retrieval embedding (Contriever, MiniLM, TF-IDF) works best,
and is that ranking consistent across models, datasets, and shot counts?

Run:
    python generate_group_D.py

Outputs are written to subdirectories D1 … D6 relative to this file.
"""

import glob
import json
import os
import re
import warnings
from collections import defaultdict

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

# ── Model families ────────────────────────────────────────────────────────────
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
EMB_COLORS = {"contriever": "#4361ee", "minilm": "#e07c00", "tfidf": "#2dc653"}
VARIANTS   = ["pc", "fixed"]
VARIANT_LABELS = {"pc": "PC", "fixed": "Fixed"}

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "axes.axisbelow":   True,
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

            # few-shot
            m = re.match(
                r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?$",
                name,
            )
            if m:
                llm = m.group(1)
                if llm not in target_models:
                    continue
                records.append({
                    "dataset": ds, "llm": llm, "method": "fewshot",
                    "embedding": m.group(2), "variant": m.group(4) or "none",
                    "shots": int(m.group(3)), "alpha": None, "clf": None,
                    "macro_f1": _f1(_load_json(path)),
                })
                continue

            # CICLe
            m = re.match(
                r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots"
                r"(?:-(pc|fixed))?-([\d.]+)-α$",
                name,
            )
            if m:
                llm = m.group(1)
                if llm not in target_models:
                    continue
                records.append({
                    "dataset": ds, "llm": llm, "method": "cicle",
                    "embedding": m.group(2), "variant": m.group(5) or "none",
                    "shots": int(m.group(4)), "alpha": float(m.group(6)),
                    "clf": m.group(3),
                    "macro_f1": _f1(_load_json(path)),
                })

    return records


# ── Reduction helpers ─────────────────────────────────────────────────────────
def _max(vals):
    clean = [v for v in vals if not np.isnan(v)]
    return max(clean) if clean else np.nan


def _mean(vals):
    clean = [v for v in vals if not np.isnan(v)]
    return float(np.mean(clean)) if clean else np.nan


def best_cicle(ds, llm, emb, shots=None):
    """Best CICLe F1 over variant × alpha × clf (and optionally shots)."""
    sub = [r for r in RECORDS
           if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "cicle" and r["embedding"] == emb
           and (shots is None or r["shots"] == shots)]
    return _max([r["macro_f1"] for r in sub])


def best_fewshot(ds, llm, emb, shots=None):
    """Best few-shot F1 over variant (and optionally shots)."""
    sub = [r for r in RECORDS
           if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "fewshot" and r["embedding"] == emb
           and (shots is None or r["shots"] == shots)]
    return _max([r["macro_f1"] for r in sub])


def best_cicle_emb_var(ds, llm, emb, shots, variant):
    sub = [r for r in RECORDS
           if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "cicle" and r["embedding"] == emb
           and r["shots"] == shots and r["variant"] == variant]
    return _max([r["macro_f1"] for r in sub])


def best_fewshot_emb_var(ds, llm, emb, shots, variant):
    sub = [r for r in RECORDS
           if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "fewshot" and r["embedding"] == emb
           and r["shots"] == shots and r["variant"] == variant]
    return _max([r["macro_f1"] for r in sub])


# ── Shared helpers ────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def _heatmap(ax, mat, row_labels, col_labels, title, fmt=".1f", cbar_label="Macro-F1"):
    vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.1
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontweight="bold")
    ax.grid(False)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            if not np.isnan(v):
                txt_color = "white" if abs(v) > vmax * 0.6 else "black"
                ax.text(ci, ri, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=txt_color)
    return im


# ── D1: Embedding ranking per model ──────────────────────────────────────────
def plot_D1():
    """
    6 plots (3 datasets × 2 methods).
    X = model, grouped bars = embeddings. Shows which embedding dominates.
    """
    out = ensure_dir("D1_embedding_ranking_per_model")

    for ds in DATASETS:
        for method, fn in [("cicle", best_cicle), ("fewshot", best_fewshot)]:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(ALL_MODELS))
            w = 0.26
            offsets = [-w, 0, w]

            for emb, offset in zip(EMBEDDINGS, offsets):
                vals = [fn(ds, llm, emb) for llm in ALL_MODELS]
                bars = ax.bar(x + offset, vals, w,
                              color=EMB_COLORS[emb], label=EMB_LABELS[emb],
                              edgecolor="white", linewidth=0.8)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.003,
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=7, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS],
                               rotation=18, ha="right")
            ax.set_ylabel("Macro-F1")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
            ax.legend(title="Embedding", framealpha=0.9)
            method_label = "CICLe" if method == "cicle" else "Few-shot"
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {method_label}\n"
                f"Embedding ranking across all models (best config per embedding)",
                fontweight="bold",
            )

            fname = f"D1_{ds.replace('-', '_')}_{method}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  D1 saved: {fname}")


# ── D2: Embedding × shots heatmap per model ───────────────────────────────────
def plot_D2():
    """
    18 plots (3 datasets × 6 models).
    3×4 heatmap: rows = embeddings, cols = shots.
    Two heatmaps side by side (CICLe | few-shot).
    """
    out = ensure_dir("D2_embedding_shots_heatmap_per_model")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            mat_cicle   = np.full((len(EMBEDDINGS), len(SHOTS)), np.nan)
            mat_fewshot = np.full((len(EMBEDDINGS), len(SHOTS)), np.nan)

            for ri, emb in enumerate(EMBEDDINGS):
                for ci, shots in enumerate(SHOTS):
                    mat_cicle[ri, ci]   = best_cicle(ds, llm, emb, shots)
                    mat_fewshot[ri, ci] = best_fewshot(ds, llm, emb, shots)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            row_labels = [EMB_LABELS[e] for e in EMBEDDINGS]
            col_labels = [str(s) for s in SHOTS]

            for ax, mat, title_suffix in zip(
                axes,
                [mat_cicle, mat_fewshot],
                ["CICLe", "Few-shot"],
            ):
                vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 0.1
                vmin = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0.0
                im = ax.imshow(mat, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels)
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels)
                ax.set_xlabel("Shots")
                ax.set_title(title_suffix, fontweight="bold")
                ax.grid(False)
                cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
                cbar.set_label("Macro-F1", fontsize=9)
                cbar.ax.yaxis.set_major_formatter(
                    mticker.PercentFormatter(xmax=1, decimals=0))
                for ri in range(mat.shape[0]):
                    for ci in range(mat.shape[1]):
                        v = mat[ri, ci]
                        if not np.isnan(v):
                            ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                                    fontsize=8.5, fontweight="bold", color="black")

            fig.suptitle(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"Embedding × shots heatmap (best config per cell)",
                fontweight="bold",
            )

            fname = f"D2_{ds.replace('-', '_')}_{llm.replace('-', '_').replace('.', '')}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  D2 saved: {fname}")


# ── D3: Embedding × shots heatmap per method (averaged over models) ───────────
def plot_D3():
    """
    6 plots (3 datasets × 2 methods).
    Same structure as D2 but averaged over all 6 models.
    """
    out = ensure_dir("D3_embedding_shots_heatmap_per_method")

    for ds in DATASETS:
        for method, fn in [("cicle", best_cicle), ("fewshot", best_fewshot)]:
            mat = np.full((len(EMBEDDINGS), len(SHOTS)), np.nan)

            for ri, emb in enumerate(EMBEDDINGS):
                for ci, shots in enumerate(SHOTS):
                    vals = [fn(ds, llm, emb, shots) for llm in ALL_MODELS]
                    mat[ri, ci] = _mean(vals)

            row_labels = [EMB_LABELS[e] for e in EMBEDDINGS]
            col_labels = [str(s) for s in SHOTS]
            method_label = "CICLe" if method == "cicle" else "Few-shot"

            fig, ax = plt.subplots(figsize=(7, 4))
            vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 0.1
            vmin = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0.0
            im = ax.imshow(mat, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
            ax.set_xlabel("Shots")
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {method_label}\n"
                f"Embedding × shots (mean Macro-F1 across all models)",
                fontweight="bold",
            )
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

            fname = f"D3_{ds.replace('-', '_')}_{method}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  D3 saved: {fname}")


# ── D4: Embedding consistency across models (parallel coordinates) ────────────
def plot_D4():
    """
    3 plots (one per dataset).
    Parallel coordinates: axes = embeddings, one line per model.
    Two panels side by side: CICLe | few-shot.
    """
    out = ensure_dir("D4_embedding_consistency_parallel")

    for ds in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

        for ax, method, fn, title_suffix in zip(
            axes,
            ["cicle", "fewshot"],
            [best_cicle, best_fewshot],
            ["CICLe (best over all shots)", "Few-shot (best over all shots)"],
        ):
            x_pos = np.arange(len(EMBEDDINGS))

            for fam, models in FAMILIES.items():
                for size, llm in models.items():
                    vals = [fn(ds, llm, emb) for emb in EMBEDDINGS]
                    if all(np.isnan(v) for v in vals):
                        continue
                    ls = "-" if size == "large" else "--"
                    lw = 2.0 if size == "large" else 1.5
                    color = {
                        "Llama": "#4361ee", "Mistral": "#7209b7", "Qwen": "#e07c00"
                    }[fam]
                    ax.plot(x_pos, vals, color=color, linewidth=lw,
                            linestyle=ls, marker="o", markersize=6, alpha=0.85,
                            label=MODEL_LABELS[llm])

            ax.set_xticks(x_pos)
            ax.set_xticklabels([EMB_LABELS[e] for e in EMBEDDINGS])
            ax.set_ylabel("Macro-F1")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_title(title_suffix, fontweight="bold")
            ax.legend(fontsize=8, framealpha=0.9, loc="lower right")

        fig.suptitle(
            f"{DATASET_LABELS[ds]}\n"
            f"Embedding consistency across models  (solid = large, dashed = small)",
            fontweight="bold",
        )

        fname = f"D4_{ds.replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  D4 saved: {fname}")


# ── D5: Embedding win counts ──────────────────────────────────────────────────
def plot_D5():
    """
    3 plots (one per dataset).
    For each embedding: how many (model × shots × variant) configs does it
    produce the best macro-F1? Grouped bars per method (CICLe | few-shot).
    """
    out = ensure_dir("D5_embedding_win_counts")

    for ds in DATASETS:
        cicle_wins   = {e: 0 for e in EMBEDDINGS}
        fewshot_wins = {e: 0 for e in EMBEDDINGS}

        # CICLe wins: for each (model, shots, variant), which embedding is best?
        for llm in ALL_MODELS:
            for shots in SHOTS:
                for variant in VARIANTS:
                    vals = {
                        emb: best_cicle_emb_var(ds, llm, emb, shots, variant)
                        for emb in EMBEDDINGS
                    }
                    valid = {e: v for e, v in vals.items() if not np.isnan(v)}
                    if valid:
                        winner = max(valid, key=valid.get)
                        cicle_wins[winner] += 1

        # Few-shot wins: for each (model, shots, variant), which embedding is best?
        for llm in ALL_MODELS:
            for shots in SHOTS:
                for variant in VARIANTS:
                    vals = {
                        emb: best_fewshot_emb_var(ds, llm, emb, shots, variant)
                        for emb in EMBEDDINGS
                    }
                    valid = {e: v for e, v in vals.items() if not np.isnan(v)}
                    if valid:
                        winner = max(valid, key=valid.get)
                        fewshot_wins[winner] += 1

        x = np.arange(len(EMBEDDINGS))
        w = 0.35
        fig, ax = plt.subplots(figsize=(7, 4.5))

        bars1 = ax.bar(x - w / 2,
                       [cicle_wins[e] for e in EMBEDDINGS], w,
                       color="#e63946", label="CICLe", edgecolor="white")
        bars2 = ax.bar(x + w / 2,
                       [fewshot_wins[e] for e in EMBEDDINGS], w,
                       color="#4361ee", label="Few-shot", edgecolor="white")

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                            str(int(h)), ha="center", va="bottom",
                            fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([EMB_LABELS[e] for e in EMBEDDINGS])
        ax.set_ylabel("Win count")
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Embedding win counts  (configs where this embedding is best)",
            fontweight="bold",
        )
        ax.legend(framealpha=0.9)

        fname = f"D5_{ds.replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  D5 saved: {fname}")


# ── D6: Embedding × variant interaction ───────────────────────────────────────
def plot_D6():
    """
    6 plots (3 datasets × 2 methods).
    3×2 heatmap: rows = embedding, cols = variant (pc/fixed).
    Cell = mean macro-F1 across all models and shots.
    """
    out = ensure_dir("D6_embedding_variant_interaction")

    for ds in DATASETS:
        for method, fn in [("cicle", best_cicle_emb_var),
                           ("fewshot", best_fewshot_emb_var)]:
            mat = np.full((len(EMBEDDINGS), len(VARIANTS)), np.nan)

            for ri, emb in enumerate(EMBEDDINGS):
                for ci, variant in enumerate(VARIANTS):
                    vals = [
                        fn(ds, llm, emb, shots, variant)
                        for llm in ALL_MODELS
                        for shots in SHOTS
                    ]
                    mat[ri, ci] = _mean(vals)

            row_labels = [EMB_LABELS[e] for e in EMBEDDINGS]
            col_labels = [VARIANT_LABELS[v] for v in VARIANTS]
            method_label = "CICLe" if method == "cicle" else "Few-shot"

            fig, ax = plt.subplots(figsize=(5.5, 4))
            vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 0.1
            vmin = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0.0
            im = ax.imshow(mat, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
            ax.set_xlabel("Variant")
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {method_label}\n"
                f"Embedding × variant  (mean Macro-F1 across models and shots)",
                fontweight="bold",
            )
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
                                fontsize=10, fontweight="bold", color="black")

            fname = f"D6_{ds.replace('-', '_')}_{method}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  D6 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group D data …")
    RECORDS = load_records()
    print(f"  Records loaded: {len(RECORDS)}")

    print("\nGenerating Group D plots …")
    print("D1 — Embedding ranking per model")
    plot_D1()
    print("D2 — Embedding × shots heatmap per model")
    plot_D2()
    print("D3 — Embedding × shots heatmap per method (averaged over models)")
    plot_D3()
    print("D4 — Embedding consistency across models (parallel coordinates)")
    plot_D4()
    print("D5 — Embedding win counts")
    plot_D5()
    print("D6 — Embedding × variant interaction")
    plot_D6()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
