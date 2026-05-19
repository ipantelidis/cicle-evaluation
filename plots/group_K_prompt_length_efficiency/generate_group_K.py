"""
Group K — Prompt Length / Computational Efficiency
====================================================
How does prompt length relate to macro-F1? Which model and method are most
efficient (F1 per token)? How does context grow with shots, and do small vs.
large models differ in prompt length?

Run:
    python generate_group_K.py

Outputs are written to subdirectories K1 … K5 relative to this file.
"""

import glob
import json
import os
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats as scipy_stats

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

SHOTS        = [1, 2, 4, 8]
SHOTS_COLORS = {1: "#4cc9f0", 2: "#4361ee", 4: "#7209b7", 8: "#f72585"}
EMBEDDINGS   = ["contriever", "minilm", "tfidf"]
EMB_LABELS   = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
EMB_COLORS   = {"contriever": "#4361ee", "minilm": "#e07c00", "tfidf": "#2dc653"}
VARIANTS     = ["fixed"]
VARIANT_LABELS = {"pc": "PC", "fixed": "Fixed"}

METHOD_COLORS = {"cicle": "#e63946", "fewshot": "#4361ee"}

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


def _parse_name(pfx, name, target_models):
    """Parse a filename stem (without prefix) into metadata dict or None."""
    m = re.match(
        r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?$",
        name)
    if m and m.group(1) in target_models:
        return {"llm": m.group(1), "method": "fewshot",
                "embedding": m.group(2), "shots": int(m.group(3)),
                "variant": m.group(4) or "none",
                "alpha": None, "clf": None}

    m = re.match(
        r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots"
        r"(?:-(pc|fixed))?-([\d.]+)-α$", name)
    if m and m.group(1) in target_models:
        return {"llm": m.group(1), "method": "cicle",
                "embedding": m.group(2), "shots": int(m.group(4)),
                "variant": m.group(5) or "none",
                "alpha": float(m.group(6)), "clf": m.group(3)}

    return None


def load_records():
    """
    Load paired (macro_f1, mean_prompt_length) records.
    Only keeps configs that have both a prediction file and a length file.
    """
    records = []
    target_models = set(ALL_MODELS)

    for ds in DATASETS:
        pfx       = DATASET_PREFIX[ds]
        pred_dir  = f"{BASE_DIR}/{ds}/results/predictions"
        len_dir   = f"{BASE_DIR}/{ds}/results/lengths"

        for pred_path in glob.glob(f"{pred_dir}/*.json"):
            stem = os.path.basename(pred_path).replace(".json", "")
            if not stem.startswith(f"{pfx}-"):
                continue
            name = stem[len(pfx) + 1:]

            meta = _parse_name(pfx, name, target_models)
            if meta is None:
                continue

            len_path = os.path.join(len_dir, stem + ".json")
            if not os.path.exists(len_path):
                continue

            pred_data = _load_json(pred_path)
            len_data  = _load_json(len_path)

            records.append({
                "dataset":     ds,
                "macro_f1":    _f1(pred_data),
                "mean_prompt": len_data["prompt_lengths"]["mean"],
                "std_prompt":  len_data["prompt_lengths"]["std"],
                **meta,
            })

    return records


# ── Reduction helpers ─────────────────────────────────────────────────────────
def _max_f1(recs):
    vals = [r["macro_f1"] for r in recs if not np.isnan(r["macro_f1"])]
    return max(vals) if vals else np.nan


def _mean_prompt(recs):
    vals = [r["mean_prompt"] for r in recs if not np.isnan(r["mean_prompt"])]
    return float(np.mean(vals)) if vals else np.nan


# ── Shared helpers ────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_fname(s):
    return s.replace("-", "_").replace(".", "")


# ── K1: F1 vs. prompt length scatter per model ───────────────────────────────
def plot_K1():
    """
    18 plots (3 datasets × 6 models).
    X = mean prompt length, Y = macro-F1.
    Points coloured by shots. Regression line per method.
    """
    out = ensure_dir("K1_f1_vs_prompt_length")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            sub = [r for r in RECORDS
                   if r["dataset"] == ds and r["llm"] == llm
                   and r["variant"] == "fixed"]
            if not sub:
                continue

            fig, ax = plt.subplots(figsize=(6.5, 5))
            has_data = False

            for method in ["cicle", "fewshot"]:
                method_recs = [r for r in sub if r["method"] == method]
                if not method_recs:
                    continue

                for shots in SHOTS:
                    pts = [r for r in method_recs if r["shots"] == shots]
                    if not pts:
                        continue
                    xs = [r["mean_prompt"] for r in pts]
                    ys = [r["macro_f1"]    for r in pts]
                    marker = "o" if method == "cicle" else "s"
                    ax.scatter(xs, ys, color=SHOTS_COLORS[shots],
                               s=30, alpha=0.6, marker=marker,
                               edgecolors="white", linewidths=0.3, zorder=3)
                    has_data = True

                # Regression line for this method
                xs_all = [r["mean_prompt"] for r in method_recs]
                ys_all = [r["macro_f1"]    for r in method_recs]
                if len(xs_all) >= 3:
                    slope, intercept, *_ = scipy_stats.linregress(xs_all, ys_all)
                    x_line = np.linspace(min(xs_all), max(xs_all), 100)
                    y_line = slope * x_line + intercept
                    color  = METHOD_COLORS[method]
                    ls     = "-" if method == "cicle" else "--"
                    label  = "CICLe fit" if method == "cicle" else "Few-shot fit"
                    ax.plot(x_line, y_line, color=color, linewidth=1.8,
                            linestyle=ls, label=label, zorder=4)

            if not has_data:
                plt.close(fig)
                continue

            # Shot legend
            shot_handles = [
                plt.scatter([], [], color=SHOTS_COLORS[s], s=40,
                            label=f"{s} shot{'s' if s > 1 else ''}")
                for s in SHOTS
            ]
            first_legend = ax.legend(handles=shot_handles, loc="upper left",
                                     fontsize=8, framealpha=0.9,
                                     title="Shots", title_fontsize=8)
            ax.add_artist(first_legend)
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

            ax.set_xlabel("Mean prompt length  (characters)")
            ax.set_ylabel("Macro-F1")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"Macro-F1 vs. prompt length  "
                f"(○ = CICLe, □ = few-shot, coloured by shots)",
                fontweight="bold",
            )

            fname = f"K1_{_safe_fname(ds)}_{_safe_fname(llm)}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  K1 saved: {fname}")


# ── K2: Efficiency metric — F1 / prompt-length × 1000 ────────────────────────
def plot_K2():
    """
    3 plots (one per dataset).
    Bar chart: efficiency = (best macro-F1 / mean prompt length) × 1000,
    sorted descending. One bar per (model × method) combination.
    """
    out = ensure_dir("K2_efficiency_f1_per_prompt_length")

    for ds in DATASETS:
        entries = []
        for llm in ALL_MODELS:
            for method in ["cicle", "fewshot"]:
                sub = [r for r in RECORDS
                       if r["dataset"] == ds and r["llm"] == llm
                       and r["method"] == method and r["variant"] == "fixed"]
                if not sub:
                    continue
                best = max(sub, key=lambda r: r["macro_f1"])
                eff  = (best["macro_f1"] / best["mean_prompt"]) * 1000
                entries.append({
                    "label":  f"{MODEL_LABELS[llm]}\n({('CICLe' if method == 'cicle' else 'Few-shot')})",
                    "eff":    eff,
                    "color":  METHOD_COLORS[method],
                })

        if not entries:
            continue

        entries.sort(key=lambda e: e["eff"], reverse=True)
        labels = [e["label"] for e in entries]
        effs   = [e["eff"]   for e in entries]
        colors = [e["color"] for e in entries]

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(range(len(entries)), effs, color=colors,
                      edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars, effs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Macro-F1 / mean-prompt-length × 1000")
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Efficiency: Macro-F1 ÷ prompt length × 1000  (sorted descending)",
            fontweight="bold",
        )

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=METHOD_COLORS["cicle"],   label="CICLe"),
                           Patch(facecolor=METHOD_COLORS["fewshot"], label="Few-shot")]
        ax.legend(handles=legend_elements, framealpha=0.9)

        fname = f"K2_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  K2 saved: {fname}")


# ── K3: Prompt length distribution by shots and variant ───────────────────────
def plot_K3():
    """
    18 plots (3 datasets × 6 models).
    Violin plots: X = shots, two violins per shots (pc / fixed).
    Y = mean prompt length across configs for that (shots, variant) slice.
    """
    out = ensure_dir("K3_prompt_length_by_shots_and_variant")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            sub = [r for r in RECORDS
                   if r["dataset"] == ds and r["llm"] == llm]
            if not sub:
                continue

            # Collect mean prompt lengths per (shots, variant)
            groups = {}
            for shots in SHOTS:
                for variant in VARIANTS:
                    pts = [r["mean_prompt"] for r in sub
                           if r["shots"] == shots and r["variant"] == variant]
                    if pts:
                        groups[(shots, variant)] = pts

            if not groups:
                continue

            n_shots = len(SHOTS)
            fig, ax = plt.subplots(figsize=(8, 4.5))

            positions, data, colors, xtick_pos, xtick_labels = [], [], [], [], []
            w = 0.35
            for i, shots in enumerate(SHOTS):
                base = i * 1.0
                for j, variant in enumerate(VARIANTS):
                    pos  = base + (j - 0.5) * w
                    pts  = groups.get((shots, variant), [])
                    if not pts:
                        continue
                    positions.append(pos)
                    data.append(pts)
                    colors.append("#4361ee" if variant == "pc" else "#e63946")
                xtick_pos.append(base)
                xtick_labels.append(f"{shots} shot{'s' if shots > 1 else ''}")

            if not data:
                plt.close(fig)
                continue

            vp = ax.violinplot(data, positions=positions, widths=w * 0.9,
                               showmedians=True, showextrema=False)

            for body, color in zip(vp["bodies"], colors):
                body.set_facecolor(color)
                body.set_alpha(0.65)
            vp["cmedians"].set_color("white")
            vp["cmedians"].set_linewidth(2)

            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(xtick_labels)
            ax.set_ylabel("Mean prompt length  (characters)")
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"Prompt length distribution by shots and variant",
                fontweight="bold",
            )

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor="#4361ee", alpha=0.65, label="PC"),
                               Patch(facecolor="#e63946", alpha=0.65, label="Fixed")]
            ax.legend(handles=legend_elements, framealpha=0.9)

            fname = f"K3_{_safe_fname(ds)}_{_safe_fname(llm)}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  K3 saved: {fname}")


# ── K4: Prompt length vs. shots curves ───────────────────────────────────────
def plot_K4():
    """
    18 plots (3 datasets × 6 models).
    X = shots. One line per embedding. Y = mean prompt length averaged over
    variants (and alpha × clf for CICLe). Separate panels for CICLe / few-shot.
    """
    out = ensure_dir("K4_prompt_length_vs_shots")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            sub = [r for r in RECORDS
                   if r["dataset"] == ds and r["llm"] == llm
                   and r["variant"] == "fixed"]
            if not sub:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
            has_data = False

            for ax, method, title_suffix in zip(
                axes, ["cicle", "fewshot"], ["CICLe", "Few-shot"]
            ):
                method_recs = [r for r in sub if r["method"] == method]
                for emb in EMBEDDINGS:
                    emb_recs = [r for r in method_recs if r["embedding"] == emb]
                    curve = [_mean_prompt(
                                 [r for r in emb_recs if r["shots"] == s])
                             for s in SHOTS]
                    if all(np.isnan(v) for v in curve):
                        continue
                    ax.plot(SHOTS, curve, color=EMB_COLORS[emb],
                            marker="o", linewidth=2, markersize=7,
                            label=EMB_LABELS[emb])
                    has_data = True

                ax.set_xticks(SHOTS)
                ax.set_xlabel("Number of shots")
                ax.set_ylabel("Mean prompt length  (chars)")
                ax.set_title(title_suffix, fontweight="bold")
                ax.legend(framealpha=0.9)

            if not has_data:
                plt.close(fig)
                continue

            fig.suptitle(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"Mean prompt length vs. shots  (averaged over variants)",
                fontweight="bold",
            )

            fname = f"K4_{_safe_fname(ds)}_{_safe_fname(llm)}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  K4 saved: {fname}")


# ── K5: Small vs. large prompt length comparison ─────────────────────────────
def plot_K5():
    """
    9 plots (3 datasets × 3 families).
    Box plots side by side: large model vs. small model.
    One box per (model × shots) showing distribution of mean prompt lengths
    across all (embedding × variant [× alpha × clf]) configs.
    """
    out = ensure_dir("K5_small_vs_large_prompt_length")

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]

            large_recs = [r for r in RECORDS
                          if r["dataset"] == ds and r["llm"] == large
                          and r["variant"] == "fixed"]
            small_recs = [r for r in RECORDS
                          if r["dataset"] == ds and r["llm"] == small
                          and r["variant"] == "fixed"]

            if not large_recs and not small_recs:
                continue

            fig, ax = plt.subplots(figsize=(8, 4.5))
            positions, data, colors = [], [], []
            xtick_pos, xtick_labels = [], []
            w = 0.35

            for i, shots in enumerate(SHOTS):
                base = float(i)
                for j, (recs, color, size) in enumerate([
                    (large_recs, SIZE_COLORS["large"], "large"),
                    (small_recs, SIZE_COLORS["small"], "small"),
                ]):
                    pts = [r["mean_prompt"] for r in recs if r["shots"] == shots]
                    if not pts:
                        continue
                    pos = base + (j - 0.5) * w
                    positions.append(pos)
                    data.append(pts)
                    colors.append(color)
                xtick_pos.append(base)
                xtick_labels.append(f"{shots} shot{'s' if shots > 1 else ''}")

            if not data:
                plt.close(fig)
                continue

            bp = ax.boxplot(data, positions=positions, widths=w * 0.85,
                            patch_artist=True, notch=False,
                            showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for element in ["whiskers", "caps", "medians"]:
                for line, color in zip(bp[element],
                                       [c for c in colors for _ in range(2)]
                                       if element != "medians"
                                       else colors):
                    line.set_color(color)
                    line.set_linewidth(1.6)
            plt.setp(bp["medians"], color="white", linewidth=2)

            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(xtick_labels)
            ax.set_ylabel("Mean prompt length  (characters)")
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {fam} family\n"
                f"Prompt length: {MODEL_LABELS[large]} vs. {MODEL_LABELS[small]}",
                fontweight="bold",
            )

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=SIZE_COLORS["large"], alpha=0.7,
                      label=f"{MODEL_LABELS[large]}  (large)"),
                Patch(facecolor=SIZE_COLORS["small"], alpha=0.7,
                      label=f"{MODEL_LABELS[small]}  (small)"),
            ]
            ax.legend(handles=legend_elements, framealpha=0.9)

            fname = f"K5_{_safe_fname(ds)}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  K5 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group K data …")
    RECORDS = load_records()
    print(f"  Records loaded: {len(RECORDS)}")

    print("\nGenerating Group K plots …")
    print("K1 — F1 vs. prompt length scatter per model  (18 plots)")
    plot_K1()
    print("K2 — Efficiency: F1 / prompt-length × 1000  (3 plots)")
    plot_K2()
    print("K3 — Prompt length distribution by shots and variant  (18 plots)")
    plot_K3()
    print("K4 — Prompt length vs. shots curves per embedding  (18 plots)")
    plot_K4()
    print("K5 — Small vs. large prompt length comparison  (9 plots)")
    plot_K5()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
