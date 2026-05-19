"""
Group L — Summary / Paper-Ready Aggregates
===========================================
High-level summary plots designed to give a complete overview of all results
in a compact, paper-ready form.

Run:
    python generate_group_L.py

Outputs are written to subdirectories L1 … L5 relative to this file.
"""

import glob
import json
import os
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
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
SIZE_COLORS   = {"large": "#264653", "small": "#e76f51"}

SHOTS      = [1, 2, 4, 8]
EMBEDDINGS = ["contriever", "minilm", "tfidf"]
EMB_LABELS = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
VARIANTS   = ["fixed"]

METHOD_COLORS = {
    "baseline": "#adb5bd",
    "zeroshot": "#6c757d",
    "fewshot":  "#4361ee",
    "cicle":    "#e63946",
}
METHOD_LABELS = {
    "baseline": "Baseline (TF-IDF)",
    "zeroshot": "Zero-shot",
    "fewshot":  "Few-shot",
    "cicle":    "CICLe",
}

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
        len_dir  = f"{BASE_DIR}/{ds}/results/lengths"

        for path in glob.glob(f"{pred_dir}/*.json"):
            stem = os.path.basename(path).replace(".json", "")
            if not stem.startswith(f"{pfx}-"):
                continue
            name = stem[len(pfx) + 1:]

            # Baseline
            m = re.match(r"^(tfidf)-(lr|svm)-2\.0k-samples$", name)
            if m:
                records.append({"dataset": ds, "llm": None,
                                "method": "baseline", "clf": m.group(2),
                                "embedding": None, "shots": None,
                                "variant": None, "alpha": None,
                                "macro_f1": _f1(_load_json(path)),
                                "mean_prompt": np.nan})
                continue

            # Zero-shot
            m = re.match(r"^(.+)-zeroshot-2\.0k-samples$", name)
            if m and m.group(1) in target_models:
                records.append({"dataset": ds, "llm": m.group(1),
                                "method": "zeroshot", "clf": None,
                                "embedding": None, "shots": 0,
                                "variant": None, "alpha": None,
                                "macro_f1": _f1(_load_json(path)),
                                "mean_prompt": np.nan})
                continue

            # Few-shot
            m = re.match(
                r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?$",
                name)
            if m and m.group(1) in target_models:
                len_path = os.path.join(len_dir, stem + ".json")
                mp = _load_json(len_path)["prompt_lengths"]["mean"] \
                     if os.path.exists(len_path) else np.nan
                records.append({"dataset": ds, "llm": m.group(1),
                                "method": "fewshot", "clf": None,
                                "embedding": m.group(2),
                                "shots": int(m.group(3)),
                                "variant": m.group(4) or "none",
                                "alpha": None,
                                "macro_f1": _f1(_load_json(path)),
                                "mean_prompt": mp})
                continue

            # CICLe
            m = re.match(
                r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots"
                r"(?:-(pc|fixed))?-([\d.]+)-α$", name)
            if m and m.group(1) in target_models:
                len_path = os.path.join(len_dir, stem + ".json")
                mp = _load_json(len_path)["prompt_lengths"]["mean"] \
                     if os.path.exists(len_path) else np.nan
                records.append({"dataset": ds, "llm": m.group(1),
                                "method": "cicle", "clf": m.group(3),
                                "embedding": m.group(2),
                                "shots": int(m.group(4)),
                                "variant": m.group(5) or "none",
                                "alpha": float(m.group(6)),
                                "macro_f1": _f1(_load_json(path)),
                                "mean_prompt": mp})

    return records


# ── Reduction helpers ─────────────────────────────────────────────────────────
def _max_f1(recs):
    vals = [r["macro_f1"] for r in recs if not np.isnan(r["macro_f1"])]
    return max(vals) if vals else np.nan


def _safe_fname(s):
    return s.replace("-", "_").replace(".", "")


def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def _pareto_frontier(xs, ys):
    """Return boolean mask of Pareto-optimal points (min x, max y)."""
    n    = len(xs)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                mask[i] = False
                break
    return mask


# ── L1: Grand summary heatmap ─────────────────────────────────────────────────
def plot_L1():
    """
    3 plots (one per dataset).
    Rows = methods × clf (baseline-LR, baseline-SVM, zero-shot, few-shot,
    CICLe). Cols = all 6 models (baseline rows use the same value everywhere).
    Cell = best macro-F1 across all configs for that (method, model) slice.
    """
    out = ensure_dir("L1_grand_summary_heatmap")

    row_defs = [
        ("Baseline LR",  lambda ds, llm: _max_f1([r for r in RECORDS if r["dataset"]==ds and r["method"]=="baseline" and r["clf"]=="lr"])),
        ("Baseline SVM", lambda ds, llm: _max_f1([r for r in RECORDS if r["dataset"]==ds and r["method"]=="baseline" and r["clf"]=="svm"])),
        ("Zero-shot",    lambda ds, llm: _max_f1([r for r in RECORDS if r["dataset"]==ds and r["llm"]==llm and r["method"]=="zeroshot"])),
        ("Few-shot",     lambda ds, llm: _max_f1([r for r in RECORDS if r["dataset"]==ds and r["llm"]==llm and r["method"]=="fewshot" and r["variant"]=="fixed"])),
        ("CICLe",        lambda ds, llm: _max_f1([r for r in RECORDS if r["dataset"]==ds and r["llm"]==llm and r["method"]=="cicle" and r["variant"]=="fixed"])),
    ]

    for ds in DATASETS:
        mat = np.full((len(row_defs), len(ALL_MODELS)), np.nan)
        for ri, (_, fn) in enumerate(row_defs):
            for ci, llm in enumerate(ALL_MODELS):
                mat[ri, ci] = fn(ds, llm)

        row_labels = [r[0] for r in row_defs]
        col_labels = [MODEL_LABELS[m] for m in ALL_MODELS]

        fig, ax = plt.subplots(figsize=(11, 5))
        vmin = np.nanmin(mat)
        vmax = np.nanmax(mat)
        im = ax.imshow(mat, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=18, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.grid(False)

        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label("Best Macro-F1", fontsize=9)
        cbar.ax.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=1, decimals=0))

        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                v = mat[ri, ci]
                if not np.isnan(v):
                    rel  = (v - vmin) / (vmax - vmin + 1e-9)
                    txt  = "white" if rel < 0.35 or rel > 0.75 else "black"
                    ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=txt)

        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Grand summary: best Macro-F1 per method × model",
            fontweight="bold",
        )

        fname = f"L1_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  L1 saved: {fname}")


# ── L2: Method × model family box plots ───────────────────────────────────────
def plot_L2():
    """
    3 plots (one per dataset).
    X = method. Distribution of all macro-F1 values, grouped by method,
    colored by model family. Shows spread across all configs.
    """
    out = ensure_dir("L2_method_family_boxplots")

    methods = ["baseline", "zeroshot", "fewshot", "cicle"]

    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(10, 5))
        positions, data, colors = [], [], []
        xtick_pos, xtick_labels = [], []

        n_fam = len(FAMILIES)
        w     = 0.18
        gap   = 0.9  # between method groups

        for mi, method in enumerate(methods):
            base = mi * gap
            xtick_pos.append(base + (n_fam - 1) * w / 2)
            xtick_labels.append(METHOD_LABELS[method])

            for fi, (fam, models) in enumerate(FAMILIES.items()):
                if method == "baseline":
                    pts = [r["macro_f1"] for r in RECORDS
                           if r["dataset"] == ds and r["method"] == "baseline"]
                else:
                    fam_llms = list(models.values())
                    pts = [r["macro_f1"] for r in RECORDS
                           if r["dataset"] == ds and r["method"] == method
                           and r["llm"] in fam_llms
                           and (r["variant"] is None or r["variant"] == "fixed")]

                if not pts:
                    continue
                pos = base + fi * w
                positions.append(pos)
                data.append(pts)
                colors.append(FAMILY_COLORS[fam])

        if not data:
            plt.close(fig)
            continue

        bp = ax.boxplot(data, positions=positions, widths=w * 0.82,
                        patch_artist=True, notch=False, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for element in ["whiskers", "caps"]:
            for line, color in zip(bp[element],
                                   [c for c in colors for _ in range(2)]):
                line.set_color(color)
                line.set_linewidth(1.4)
        for line, color in zip(bp["medians"], colors):
            line.set_color("white")
            line.set_linewidth(2)

        # Separator lines between method groups
        for mi in range(1, len(methods)):
            ax.axvline(mi * gap - gap / 2, color="#cccccc",
                       linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels)
        ax.set_ylabel("Macro-F1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        legend_handles = [mpatches.Patch(facecolor=FAMILY_COLORS[f],
                                         alpha=0.65, label=f)
                          for f in FAMILIES]
        ax.legend(handles=legend_handles, title="Family",
                  framealpha=0.9, loc="lower right")
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Macro-F1 distribution by method and model family  "
            f"(all configs, outliers hidden)",
            fontweight="bold",
        )

        fname = f"L2_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  L2 saved: {fname}")


# ── L3: CICLe improvement over matched few-shot ───────────────────────────────
def plot_L3():
    """
    3 plots (one per dataset).
    For each model: CICLe best − few-shot best, matched on (embedding × shots).
    Bar chart per model, sorted by mean delta.
    """
    out = ensure_dir("L3_cicle_improvement_over_fewshot")

    for ds in DATASETS:
        model_deltas = {}

        for llm in ALL_MODELS:
            deltas = []
            for emb in EMBEDDINGS:
                for shots in SHOTS:
                    cicle_best = _max_f1([
                        r for r in RECORDS
                        if r["dataset"] == ds and r["llm"] == llm
                        and r["method"] == "cicle" and r["variant"] == "fixed"
                        and r["embedding"] == emb and r["shots"] == shots
                    ])
                    fs_best = _max_f1([
                        r for r in RECORDS
                        if r["dataset"] == ds and r["llm"] == llm
                        and r["method"] == "fewshot" and r["variant"] == "fixed"
                        and r["embedding"] == emb and r["shots"] == shots
                    ])
                    if not np.isnan(cicle_best) and not np.isnan(fs_best):
                        deltas.append((cicle_best - fs_best) * 100)
            model_deltas[llm] = deltas

        # Sort by mean delta
        sorted_models = sorted(
            [m for m in ALL_MODELS if model_deltas[m]],
            key=lambda m: float(np.mean(model_deltas[m])),
            reverse=True,
        )
        if not sorted_models:
            continue

        means  = [float(np.mean(model_deltas[m])) for m in sorted_models]
        stds   = [float(np.std(model_deltas[m]))   for m in sorted_models]
        colors = ["#2dc653" if v >= 0 else "#e63946" for v in means]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(sorted_models))
        bars = ax.bar(x, means, color=colors, edgecolor="white",
                      linewidth=1.0, width=0.55)
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    color="#333333", capsize=4, linewidth=1.2)

        for bar, v in zip(bars, means):
            va     = "bottom" if v >= 0 else "top"
            offset = 0.1 if v >= 0 else -0.1
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + offset, f"{v:+.2f}",
                    ha="center", va=va, fontsize=9, fontweight="bold")

        ax.axhline(0, color="#333333", linewidth=1.0,
                   linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in sorted_models],
                           rotation=18, ha="right")
        ax.set_ylabel("CICLe − Few-shot  (pp, mean ± std)")
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"CICLe improvement over matched few-shot  "
            f"(matched on embedding × shots, sorted by mean gain)",
            fontweight="bold",
        )

        fname = f"L3_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  L3 saved: {fname}")


# ── L4: Visual table — top-10 configs per dataset ────────────────────────────
def plot_L4():
    """
    3 plots (one per dataset).
    A visual table showing the 10 best configurations ranked by macro-F1.
    Columns: Rank, Model, Method, Embedding, Shots, Variant, Alpha, CLF, F1.
    """
    out = ensure_dir("L4_top10_configs_table")

    col_keys    = ["rank", "model", "method", "embedding", "shots",
                   "variant", "alpha", "clf", "macro_f1"]
    col_headers = ["#", "Model", "Method", "Embedding", "Shots",
                   "Variant", "Alpha", "CLF", "Macro-F1"]
    col_widths  = [0.04, 0.18, 0.11, 0.12, 0.07, 0.09, 0.08, 0.07, 0.10]

    for ds in DATASETS:
        llm_recs = [r for r in RECORDS
                    if r["dataset"] == ds and r["llm"] is not None
                    and (r["variant"] is None or r["variant"] == "fixed")]
        top10 = sorted(llm_recs, key=lambda r: r["macro_f1"], reverse=True)[:10]
        if not top10:
            continue

        rows = []
        for i, r in enumerate(top10, 1):
            rows.append({
                "rank":      str(i),
                "model":     MODEL_LABELS.get(r["llm"], r["llm"] or "—"),
                "method":    METHOD_LABELS.get(r["method"], r["method"]),
                "embedding": EMB_LABELS.get(r["embedding"], r["embedding"] or "—"),
                "shots":     str(r["shots"]) if r["shots"] else "—",
                "variant":   r["variant"].upper() if r["variant"] and r["variant"] != "none" else "—",
                "alpha":     str(r["alpha"]) if r["alpha"] is not None else "—",
                "clf":       r["clf"].upper() if r["clf"] else "—",
                "macro_f1":  f"{r['macro_f1']:.4f}",
            })

        n_rows = len(rows)
        fig_h  = 0.45 * n_rows + 1.2
        fig, ax = plt.subplots(figsize=(13, fig_h))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_rows + 1)
        ax.axis("off")

        # Header row
        x_pos = 0.01
        for header, width in zip(col_headers, col_widths):
            ax.text(x_pos + width / 2, n_rows + 0.5, header,
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#264653", linewidth=0))
            x_pos += width

        # Data rows
        for ri, row in enumerate(rows):
            y = n_rows - ri - 0.5
            bg = "#f8f9fa" if ri % 2 == 0 else "white"
            ax.axhline(y + 0.5, color="#dee2e6", linewidth=0.5)

            x_pos = 0.01
            for key, width in zip(col_keys, col_widths):
                val = row[key]
                weight = "bold" if key == "macro_f1" else "normal"
                color  = "#2dc653" if key == "rank" and ri == 0 else "#212529"
                ax.text(x_pos + width / 2, y, val,
                        ha="center", va="center", fontsize=8.5,
                        fontweight=weight, color=color)
                x_pos += width

        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Top-10 configurations by Macro-F1",
            fontweight="bold", fontsize=12, pad=10,
        )

        fname = f"L4_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  L4 saved: {fname}")


# ── L5: Pareto frontier — F1 vs. prompt length ────────────────────────────────
def plot_L5():
    """
    9 plots (3 datasets × 3 families).
    X = mean prompt length, Y = macro-F1.
    Small and large model points plotted together.
    Pareto-optimal points (best F1 for given cost) highlighted and connected.
    """
    out = ensure_dir("L5_pareto_frontier")

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]

            fig, ax = plt.subplots(figsize=(7, 5))
            has_data = False

            for llm, size in [(large, "large"), (small, "small")]:
                sub = [r for r in RECORDS
                       if r["dataset"] == ds and r["llm"] == llm
                       and r["variant"] == "fixed"
                       and not np.isnan(r["mean_prompt"])]
                if not sub:
                    continue

                xs = np.array([r["mean_prompt"] for r in sub])
                ys = np.array([r["macro_f1"]    for r in sub])

                color = SIZE_COLORS[size]
                ax.scatter(xs, ys, color=color, s=18, alpha=0.35,
                           edgecolors="none", zorder=2)

                # Pareto frontier
                mask   = _pareto_frontier(xs, ys)
                px, py = xs[mask], ys[mask]
                order  = np.argsort(px)
                px, py = px[order], py[order]

                ax.plot(px, py, color=color, linewidth=2.0, zorder=4,
                        marker="o", markersize=7,
                        label=f"{MODEL_LABELS[llm]} ({size})",
                        markeredgecolor="white", markeredgewidth=0.5)
                has_data = True

            if not has_data:
                plt.close(fig)
                continue

            ax.set_xlabel("Mean prompt length  (characters)")
            ax.set_ylabel("Macro-F1")
            ax.yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))
            ax.legend(framealpha=0.9)
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {fam} family\n"
                f"Pareto frontier: Macro-F1 vs. prompt length  "
                f"(line = Pareto-optimal configs)",
                fontweight="bold",
            )

            fname = f"L5_{_safe_fname(ds)}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  L5 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group L data …")
    RECORDS = load_records()
    print(f"  Records loaded: {len(RECORDS)}")

    print("\nGenerating Group L plots …")
    print("L1 — Grand summary heatmap  (3 plots)")
    plot_L1()
    print("L2 — Method × model family box plots  (3 plots)")
    plot_L2()
    print("L3 — CICLe improvement over matched few-shot  (3 plots)")
    plot_L3()
    print("L4 — Visual table: top-10 configs per dataset  (3 plots)")
    plot_L4()
    print("L5 — Pareto frontier: F1 vs. prompt length  (9 plots)")
    plot_L5()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
