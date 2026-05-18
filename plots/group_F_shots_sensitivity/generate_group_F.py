"""
Group F — Shots Sensitivity
============================
How does the number of in-context examples (1, 2, 4, 8) affect performance?
Do returns diminish? Which models extract the most value per added example,
and how quickly do small vs. large models saturate?

Run:
    python generate_group_F.py

Outputs are written to subdirectories F1 … F5 relative to this file.
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
SIZE_COLORS   = {"large": "#264653", "small": "#e76f51"}

SHOTS      = [1, 2, 4, 8]
EMBEDDINGS = ["contriever", "minilm", "tfidf"]
EMB_LABELS = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
VARIANTS   = ["pc", "fixed"]

METHOD_COLORS = {
    "cicle":   "#e63946",
    "fewshot": "#4361ee",
    "zeroshot": "#6c757d",
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
    """Load CICLe, few-shot, and zero-shot records for all target models."""
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

            # zero-shot
            m = re.match(r"^(.+)-zeroshot-2\.0k-samples$", name)
            if m and m.group(1) in target_models:
                records.append({
                    "dataset": ds, "llm": m.group(1), "method": "zeroshot",
                    "embedding": None, "variant": None,
                    "shots": 0, "alpha": None, "clf": None,
                    "macro_f1": _f1(_load_json(path)),
                })
                continue

            # few-shot
            m = re.match(
                r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?$",
                name,
            )
            if m and m.group(1) in target_models:
                records.append({
                    "dataset": ds, "llm": m.group(1), "method": "fewshot",
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
            if m and m.group(1) in target_models:
                records.append({
                    "dataset": ds, "llm": m.group(1), "method": "cicle",
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


def best_cicle(ds, llm, shots, emb=None):
    """Best CICLe F1 over variant × alpha × clf (optionally fixing embedding)."""
    sub = [r for r in RECORDS
           if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "cicle" and r["shots"] == shots
           and (emb is None or r["embedding"] == emb)]
    return _max([r["macro_f1"] for r in sub])


def best_fewshot(ds, llm, shots, emb=None):
    """Best few-shot F1 over variant (optionally fixing embedding)."""
    sub = [r for r in RECORDS
           if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "fewshot" and r["shots"] == shots
           and (emb is None or r["embedding"] == emb)]
    return _max([r["macro_f1"] for r in sub])


def zeroshot_value(ds, llm):
    sub = [r for r in RECORDS
           if r["dataset"] == ds and r["llm"] == llm
           and r["method"] == "zeroshot"]
    return sub[0]["macro_f1"] if sub else np.nan


# ── Shared helpers ────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_fname(s):
    return s.replace("-", "_").replace(".", "")


def _annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h) and h != 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")


# ── F1: Return-on-shots curve — marginal F1 gain per step ────────────────────
def plot_F1():
    """
    18 plots (3 datasets × 6 models).
    X = transition label (1→2, 2→4, 4→8).
    Y = ΔMacro-F1 (pp) from the previous shots value.
    Two lines: CICLe (solid) and few-shot (dashed).
    """
    out = ensure_dir("F1_return_on_shots")

    transitions  = [(1, 2), (2, 4), (4, 8)]
    trans_labels = ["1→2", "2→4", "4→8"]
    x_pos = np.arange(len(transitions))

    for ds in DATASETS:
        for llm in ALL_MODELS:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            has_data = False

            for method, fn, ls, lw in [
                ("CICLe",    best_cicle,   "-",  2.2),
                ("Few-shot", best_fewshot, "--", 1.8),
            ]:
                deltas = []
                for s_prev, s_curr in transitions:
                    v_prev = fn(ds, llm, s_prev)
                    v_curr = fn(ds, llm, s_curr)
                    if np.isnan(v_prev) or np.isnan(v_curr):
                        deltas.append(np.nan)
                    else:
                        deltas.append((v_curr - v_prev) * 100)

                if all(np.isnan(d) for d in deltas):
                    continue

                color = METHOD_COLORS["cicle"] if method == "CICLe" \
                    else METHOD_COLORS["fewshot"]
                ax.plot(x_pos, deltas, color=color, linestyle=ls,
                        linewidth=lw, marker="o", markersize=8,
                        label=method)
                has_data = True

            if not has_data:
                plt.close(fig)
                continue

            ax.axhline(0, color="#333333", linewidth=0.8,
                       linestyle=":", alpha=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(trans_labels)
            ax.set_xlabel("Shot transition")
            ax.set_ylabel("ΔMacro-F1 (pp)")
            ax.legend(framealpha=0.9)
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"Marginal gain per shot step  "
                f"(best config, optimised over all hyperparameters)",
                fontweight="bold",
            )

            fname = f"F1_{_safe_fname(ds)}_{_safe_fname(llm)}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  F1 saved: {fname}")


# ── F2: Shot efficiency — Macro-F1 / shots ───────────────────────────────────
def plot_F2():
    """
    18 plots (3 datasets × 6 models).
    X = shots. Y = Macro-F1 / shots (efficiency metric).
    Two lines: CICLe (solid) and few-shot (dashed).
    """
    out = ensure_dir("F2_shot_efficiency")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            has_data = False

            for method, fn, ls, lw in [
                ("CICLe",    best_cicle,   "-",  2.2),
                ("Few-shot", best_fewshot, "--", 1.8),
            ]:
                vals = [fn(ds, llm, s) for s in SHOTS]
                eff  = [v / s if not np.isnan(v) else np.nan
                        for v, s in zip(vals, SHOTS)]

                if all(np.isnan(e) for e in eff):
                    continue

                color = METHOD_COLORS["cicle"] if method == "CICLe" \
                    else METHOD_COLORS["fewshot"]
                ax.plot(SHOTS, eff, color=color, linestyle=ls,
                        linewidth=lw, marker="o", markersize=8,
                        label=method)
                has_data = True

            if not has_data:
                plt.close(fig)
                continue

            ax.set_xticks(SHOTS)
            ax.set_xlabel("Number of shots")
            ax.set_ylabel("Macro-F1 / shots")
            ax.legend(framealpha=0.9)
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"Shot efficiency  (Macro-F1 ÷ shots, best config)",
                fontweight="bold",
            )

            fname = f"F2_{_safe_fname(ds)}_{_safe_fname(llm)}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  F2 saved: {fname}")


# ── F3: Shots × embedding heatmap per model ───────────────────────────────────
def plot_F3():
    """
    18 plots (3 datasets × 6 models).
    Two side-by-side heatmaps (CICLe | few-shot).
    Rows = embeddings, cols = shots. Best config per cell.
    """
    out = ensure_dir("F3_shots_embedding_heatmap_per_model")

    for ds in DATASETS:
        for llm in ALL_MODELS:
            mat_cicle   = np.full((len(EMBEDDINGS), len(SHOTS)), np.nan)
            mat_fewshot = np.full((len(EMBEDDINGS), len(SHOTS)), np.nan)

            for ri, emb in enumerate(EMBEDDINGS):
                for ci, shots in enumerate(SHOTS):
                    mat_cicle[ri, ci]   = best_cicle(ds, llm, shots, emb)
                    mat_fewshot[ri, ci] = best_fewshot(ds, llm, shots, emb)

            if np.all(np.isnan(mat_cicle)) and np.all(np.isnan(mat_fewshot)):
                continue

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            row_labels = [EMB_LABELS[e] for e in EMBEDDINGS]
            col_labels = [str(s) for s in SHOTS]

            for ax, mat, title_suffix in zip(
                axes,
                [mat_cicle, mat_fewshot],
                ["CICLe", "Few-shot"],
            ):
                valid = mat[~np.isnan(mat)]
                vmin  = valid.min() if len(valid) else 0.0
                vmax  = valid.max() if len(valid) else 0.1
                im = ax.imshow(mat, cmap="YlGn", vmin=vmin, vmax=vmax,
                               aspect="auto")
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
                            ax.text(ci, ri, f"{v:.3f}",
                                    ha="center", va="center",
                                    fontsize=8.5, fontweight="bold",
                                    color="black")

            fig.suptitle(
                f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\n"
                f"Shots × embedding heatmap  (best config per cell)",
                fontweight="bold",
            )

            fname = f"F3_{_safe_fname(ds)}_{_safe_fname(llm)}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  F3 saved: {fname}")


# ── F4: Shots saturation — small vs. large CICLe ─────────────────────────────
def plot_F4():
    """
    9 plots (3 datasets × 3 families).
    X = shots. Two lines: large CICLe (solid, dark) and small CICLe (dashed,
    warm). Shows which model saturates faster with more shots.
    """
    out = ensure_dir("F4_shots_saturation_small_vs_large")

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large = models["large"]
            small = models["small"]

            large_vals = [best_cicle(ds, large, s) for s in SHOTS]
            small_vals = [best_cicle(ds, small, s) for s in SHOTS]

            if (all(np.isnan(v) for v in large_vals) and
                    all(np.isnan(v) for v in small_vals)):
                continue

            fig, ax = plt.subplots(figsize=(6, 4.5))

            ax.plot(SHOTS, large_vals, color=SIZE_COLORS["large"],
                    linestyle="-", linewidth=2.2, marker="o", markersize=8,
                    label=f"{MODEL_LABELS[large]}  (large)")
            ax.plot(SHOTS, small_vals, color=SIZE_COLORS["small"],
                    linestyle="--", linewidth=2.0, marker="s", markersize=8,
                    label=f"{MODEL_LABELS[small]}  (small)")

            for vals in [large_vals, small_vals]:
                color = (SIZE_COLORS["large"]
                         if vals is large_vals else SIZE_COLORS["small"])
                for s, v in zip(SHOTS, vals):
                    if not np.isnan(v):
                        ax.annotate(
                            f"{v:.3f}", (s, v),
                            textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8, color=color,
                            fontweight="bold",
                        )

            ax.set_xticks(SHOTS)
            ax.set_xlabel("Number of shots")
            ax.set_ylabel("Macro-F1  (best CICLe config)")
            ax.yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))
            ax.legend(framealpha=0.9)
            ax.set_ylim(bottom=0,
                        top=np.nanmax(large_vals + small_vals) * 1.18)
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {fam} family\n"
                f"CICLe saturation: small vs. large model",
                fontweight="bold",
            )

            fname = f"F4_{_safe_fname(ds)}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  F4 saved: {fname}")


# ── F5: 1-shot cold-start bar chart ──────────────────────────────────────────
def plot_F5():
    """
    3 plots (one per dataset).
    6 models on x-axis. Three grouped bars per model:
      - Large zero-shot (grey)
      - Few-shot at 1 shot, best config (blue)
      - CICLe at 1 shot, best config (red)
    Highlights the cold-start (1-shot) scenario.
    """
    out = ensure_dir("F5_one_shot_cold_start")

    bar_defs = [
        ("Zero-shot",      METHOD_COLORS["zeroshot"], lambda ds, llm: zeroshot_value(ds, llm)),
        ("Few-shot 1-shot",METHOD_COLORS["fewshot"],  lambda ds, llm: best_fewshot(ds, llm, 1)),
        ("CICLe 1-shot",   METHOD_COLORS["cicle"],    lambda ds, llm: best_cicle(ds, llm, 1)),
    ]

    for ds in DATASETS:
        x   = np.arange(len(ALL_MODELS))
        w   = 0.26
        offsets = [-w, 0, w]

        fig, ax = plt.subplots(figsize=(11, 5))

        for (label, color, fn), offset in zip(bar_defs, offsets):
            vals = [fn(ds, llm) for llm in ALL_MODELS]
            bars = ax.bar(x + offset, vals, w, color=color, label=label,
                          edgecolor="white", linewidth=0.8)
            _annotate_bars(ax, bars)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS],
                           rotation=18, ha="right")
        ax.set_ylabel("Macro-F1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
        ax.legend(framealpha=0.9)
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Cold-start (1-shot): zero-shot vs. few-shot vs. CICLe  "
            f"(best config per model)",
            fontweight="bold",
        )

        fname = f"F5_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  F5 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group F data …")
    RECORDS = load_records()
    print(f"  Records loaded: {len(RECORDS)}")

    print("\nGenerating Group F plots …")
    print("F1 — Return-on-shots: marginal gain per step  (18 plots)")
    plot_F1()
    print("F2 — Shot efficiency: Macro-F1 / shots  (18 plots)")
    plot_F2()
    print("F3 — Shots × embedding heatmap per model  (18 plots)")
    plot_F3()
    print("F4 — Saturation: small vs. large CICLe  (9 plots)")
    plot_F4()
    print("F5 — 1-shot cold-start bar chart  (3 plots)")
    plot_F5()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
