"""
Group A — Core Research Question
=================================
Does CICLe with a small model (3B) match a large model (7–8B)
in zero-shot or few-shot settings?

Run:
    python generate_group_A.py

Outputs are written to the Group A subdirectories relative to this file.
"""

import json, glob, os, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from collections import defaultdict

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
MODEL_LABELS = {
    "llama-3.1-8b":    "Llama 3.1-8B",
    "llama-3.2-3b":    "Llama 3.2-3B",
    "mistral-7b-v0.3": "Mistral 7B",
    "ministral-3b":    "Ministral 3B",
    "qwen-2.5-7b":     "Qwen 2.5-7B",
    "qwen-2.5-3b":     "Qwen 2.5-3B",
}
FAMILY_COLORS = {"Llama": "#4361ee", "Mistral": "#7209b7", "Qwen": "#e07c00"}
SHOTS = [1, 2, 4, 8]
SHOTS_COLORS = {1: "#4cc9f0", 2: "#4361ee", 4: "#7209b7", 8: "#f72585"}
EMBEDDINGS = ["contriever", "minilm", "tfidf"]
EMB_LABELS  = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
VARIANTS      = ["pc", "fixed"]
VARIANT_LABELS = {"pc": "PC (per-class)", "fixed": "Fixed (total)"}

# ── Global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
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

def _f1(d):
    return d["classification_report"]["macro avg"]["f1-score"]


def load_zeroshot():
    """Dict: (dataset, model_key) → macro_f1."""
    out = {}
    for ds in DATASETS:
        pfx  = DATASET_PREFIX[ds]
        pred = f"{BASE_DIR}/{ds}/results/predictions"
        for path in glob.glob(f"{pred}/*-zeroshot-*.json"):
            stem = os.path.basename(path).replace(".json", "")
            # e.g. "yahoo-llama-3.1-8b-zeroshot-2.0k-samples"
            m = re.match(rf"^{re.escape(pfx)}-(.+)-zeroshot-2\.0k-samples$", stem)
            if m:
                out[(ds, m.group(1))] = _f1(_load_json(path))
    return out


def load_cicle_fewshot():
    """
    Returns two dicts:
      cicle[dataset][llm][embedding][variant][shots][alpha][clf] = macro_f1
      fewshot[dataset][llm][embedding][variant][shots]           = macro_f1

    Filenames encode the configuration, so this function parses the stem of
    each JSON file and stores only the macro-F1 needed for plotting.
    """
    cicle   = defaultdict(lambda: defaultdict(lambda: defaultdict(
              lambda: defaultdict(lambda: defaultdict(
              lambda: defaultdict(lambda: defaultdict(float)))))))
    fewshot = defaultdict(lambda: defaultdict(lambda: defaultdict(
              lambda: defaultdict(lambda: defaultdict(float)))))

    for ds in DATASETS:
        pfx  = DATASET_PREFIX[ds]
        pred = f"{BASE_DIR}/{ds}/results/predictions"
        for path in glob.glob(f"{pred}/*.json"):
            stem = os.path.basename(path).replace(".json", "")
            name = stem[len(pfx) + 1:]

            # Fewshot
            m = re.match(r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?$", name)
            if m:
                llm, emb, s, v = m.group(1), m.group(2), int(m.group(3)), m.group(4) or "none"
                fewshot[ds][llm][emb][v][s] = _f1(_load_json(path))
                continue

            # CICLe
            m = re.match(r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?-([\d.]+)-α$", name)
            if m:
                llm, emb, clf = m.group(1), m.group(2), m.group(3)
                s, v, a = int(m.group(4)), m.group(5) or "none", float(m.group(6))
                cicle[ds][llm][emb][v][s][a][clf] = _f1(_load_json(path))

    return dict(cicle), dict(fewshot)


def load_baseline():
    """Dict: (dataset, clf) → macro_f1."""
    out = {}
    for ds in DATASETS:
        pfx  = DATASET_PREFIX[ds]
        pred = f"{BASE_DIR}/{ds}/results/predictions"
        for clf in ("lr", "svm"):
            path = f"{pred}/{pfx}-tfidf-{clf}-2.0k-samples.json"
            if os.path.exists(path):
                out[(ds, clf)] = _f1(_load_json(path))
    return out


def all_cicle_vals(ds, llm, shots):
    """All macro-F1 values for a (dataset, llm, shots) combo across all configs."""
    vals = []
    for emb in CICLE.get(ds, {}).get(llm, {}):
        for v in CICLE[ds][llm][emb]:
            for a in CICLE[ds][llm][emb][v].get(shots, {}):
                for clf in CICLE[ds][llm][emb][v][shots][a]:
                    vals.append(CICLE[ds][llm][emb][v][shots][a][clf])
    return vals


def best_cicle(ds, llm, shots):
    """Best macro-F1 for a (dataset, llm, shots) combo across all configs."""
    vals = all_cicle_vals(ds, llm, shots)
    return max(vals) if vals else np.nan


def mean_std_cicle(ds, llm, shots):
    """Mean and std of macro-F1 across all configs for (dataset, llm, shots)."""
    vals = all_cicle_vals(ds, llm, shots)
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))


def best_fewshot(ds, llm, shots):
    """Best macro-F1 for a (dataset, llm, shots) combo across all configs."""
    vals = []
    for emb in FEWSHOT.get(ds, {}).get(llm, {}):
        for v in FEWSHOT[ds][llm][emb]:
            vals.append(FEWSHOT[ds][llm][emb][v].get(shots, np.nan))
    return max([x for x in vals if not np.isnan(x)], default=np.nan)


def best_cicle_by_emb_var(ds, llm, shots, emb, var):
    """Best macro-F1 for fixed (emb, variant), optimising over alpha and clf."""
    vals = []
    for a in CICLE.get(ds, {}).get(llm, {}).get(emb, {}).get(var, {}).get(shots, {}):
        for clf in CICLE[ds][llm][emb][var][shots][a]:
            vals.append(CICLE[ds][llm][emb][var][shots][a][clf])
    return max(vals) if vals else np.nan


def fewshot_val(ds, llm, shots, emb, var):
    return FEWSHOT.get(ds, {}).get(llm, {}).get(emb, {}).get(var, {}).get(shots, np.nan)


def best_fewshot_by_var(ds, llm, shots, var):
    """Best macro-F1 for a given variant, optimising over embeddings."""
    vals = []
    for emb in FEWSHOT.get(ds, {}).get(llm, {}):
        v = FEWSHOT[ds][llm][emb].get(var, {}).get(shots, np.nan)
        if not np.isnan(v):
            vals.append(v)
    return max(vals) if vals else np.nan


def best_cicle_by_var(ds, llm, shots, var):
    """Best macro-F1 for a given variant, optimising over embeddings, alpha, clf."""
    vals = []
    for emb in CICLE.get(ds, {}).get(llm, {}):
        for a in CICLE[ds][llm][emb].get(var, {}).get(shots, {}):
            for clf in CICLE[ds][llm][emb][var][shots][a]:
                vals.append(CICLE[ds][llm][emb][var][shots][a][clf])
    return max(vals) if vals else np.nan


# ── A1: Gap plot — small CICLe best config vs. large zero-shot ────────────────
def plot_A1():
    out = os.path.join(HERE, "A1_gap_cicle_vs_zeroshot_best")
    os.makedirs(out, exist_ok=True)

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]
            lz = ZEROSHOT.get((ds, large), np.nan)
            sz = ZEROSHOT.get((ds, small), np.nan)

            best_vals = np.array([best_cicle(ds, small, s) for s in SHOTS])
            xs = np.array(SHOTS)

            fig, ax = plt.subplots(figsize=(6, 4.5))

            # Green shading where best-config exceeds large zero-shot
            if not np.isnan(lz):
                mask = best_vals >= lz
                ax.fill_between(xs, lz, np.where(mask, best_vals, lz),
                                alpha=0.10, color="#2dc653", zorder=1)

            # Large zero-shot reference
            if not np.isnan(lz):
                ax.axhline(lz, color="#333333", linestyle="--", linewidth=1.8,
                           zorder=2, label=f"{MODEL_LABELS[large]} zero-shot")

            # Small zero-shot reference
            if not np.isnan(sz):
                ax.axhline(sz, color="#999999", linestyle=":", linewidth=1.5,
                           zorder=2, label=f"{MODEL_LABELS[small]} zero-shot")

            # Best-config oracle line
            ax.plot(xs, best_vals, color=FAMILY_COLORS[fam],
                    marker="o", linewidth=2.2, markersize=8, zorder=3,
                    label=f"{MODEL_LABELS[small]} CICLe (best config)")

            # Annotate values with enough headroom from title
            for s, v in zip(SHOTS, best_vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.3f}", (s, v),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=8.5,
                                color=FAMILY_COLORS[fam], fontweight="bold")

            ax.set_xlabel("Number of shots")
            ax.set_ylabel("Macro-F1")
            ax.set_xticks(SHOTS)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_title(f"{DATASET_LABELS[ds]}  ·  {fam} family",
                         fontweight="bold", pad=14)
            ax.legend(loc="lower right", framealpha=0.9)

            all_vals = [v for v in list(best_vals) + [lz, sz] if not np.isnan(v)]
            ax.set_ylim(bottom=0, top=max(all_vals) * 1.18)

            fname = f"A1_{ds.replace('-', '_')}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  A1 saved: {fname}")


# ── A2: Mean ± std across all CICLe configs vs. large zero-shot ───────────────
def plot_A2():
    out = os.path.join(HERE, "A2_gap_cicle_vs_zeroshot_mean")
    os.makedirs(out, exist_ok=True)

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]
            lz = ZEROSHOT.get((ds, large), np.nan)
            sz = ZEROSHOT.get((ds, small), np.nan)

            means = np.array([mean_std_cicle(ds, small, s)[0] for s in SHOTS])
            stds  = np.array([mean_std_cicle(ds, small, s)[1] for s in SHOTS])
            xs    = np.array(SHOTS)

            fig, ax = plt.subplots(figsize=(6, 4.5))

            # Shaded std band
            valid = ~np.isnan(means)
            if valid.any():
                ax.fill_between(xs[valid],
                                (means - stds)[valid],
                                (means + stds)[valid],
                                alpha=0.22, color=FAMILY_COLORS[fam],
                                zorder=1,
                                label=f"{MODEL_LABELS[small]} CICLe (mean ± std)")
                ax.plot(xs[valid], means[valid], color=FAMILY_COLORS[fam],
                        marker="o", linewidth=2.2, markersize=8, zorder=3)

            # Annotate mean values
            for s, m in zip(SHOTS, means):
                if not np.isnan(m):
                    ax.annotate(f"{m:.3f}", (s, m),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=8.5,
                                color=FAMILY_COLORS[fam], fontweight="bold")

            # Large zero-shot reference
            if not np.isnan(lz):
                ax.axhline(lz, color="#333333", linestyle="--", linewidth=1.8,
                           zorder=2, label=f"{MODEL_LABELS[large]} zero-shot")

            # Small zero-shot reference
            if not np.isnan(sz):
                ax.axhline(sz, color="#999999", linestyle=":", linewidth=1.5,
                           zorder=2, label=f"{MODEL_LABELS[small]} zero-shot")

            ax.set_xlabel("Number of shots")
            ax.set_ylabel("Macro-F1")
            ax.set_xticks(SHOTS)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_title(f"{DATASET_LABELS[ds]}  ·  {fam} family",
                         fontweight="bold", pad=14)
            ax.legend(loc="lower right", framealpha=0.9)

            all_ref = [lz, sz] + list(means[valid] + stds[valid])
            top = max([v for v in all_ref if not np.isnan(v)], default=1.0)
            ax.set_ylim(bottom=0, top=top * 1.18)

            fname = f"A2_{ds.replace('-', '_')}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  A2 saved: {fname}")


# ── A3: Gap plot — adding large few-shot as reference ─────────────────────────
def plot_A3():
    out = os.path.join(HERE, "A3_gap_with_fewshot")
    os.makedirs(out, exist_ok=True)

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]

            large_fs    = [best_fewshot(ds, large, s) for s in SHOTS]
            small_cicle = [best_cicle(ds, small, s)   for s in SHOTS]
            small_fs    = [best_fewshot(ds, small, s)  for s in SHOTS]

            fig, ax = plt.subplots(figsize=(6, 4.5))

            ax.plot(SHOTS, large_fs, color="#666666", marker="s",
                    linewidth=1.8, markersize=7, linestyle=":", zorder=3,
                    label=f"{MODEL_LABELS[large]} few-shot (best)")

            ax.plot(SHOTS, small_fs, color=FAMILY_COLORS[fam], marker="^",
                    linewidth=1.6, markersize=7, linestyle="--",
                    alpha=0.65, zorder=3,
                    label=f"{MODEL_LABELS[small]} few-shot (best)")

            ax.plot(SHOTS, small_cicle, color=FAMILY_COLORS[fam], marker="o",
                    linewidth=2.2, markersize=8, zorder=4,
                    label=f"{MODEL_LABELS[small]} CICLe (best)")

            for s, v in zip(SHOTS, small_cicle):
                if not np.isnan(v):
                    ax.annotate(f"{v:.3f}", (s, v),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=8.5, color=FAMILY_COLORS[fam],
                                fontweight="bold")

            ax.set_xlabel("Number of shots")
            ax.set_ylabel("Macro-F1")
            ax.set_xticks(SHOTS)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_title(f"{DATASET_LABELS[ds]}  ·  {fam} family",
                         fontweight="bold", pad=14)
            ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

            all_vals = [v for v in list(small_cicle) + list(large_fs) + list(small_fs)
                        if v is not None and not np.isnan(v)]
            ax.set_ylim(bottom=0, top=max(all_vals) * 1.18 if all_vals else 1)

            fname = f"A3_{ds.replace('-', '_')}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  A3 saved: {fname}")


# ── A3_1 / A3_2: Same as A3 broken down by variant (pc / fixed) ───────────────
def _plot_A3_variant(var, label, subdir):
    out = os.path.join(HERE, "A3_gap_with_fewshot", subdir)
    os.makedirs(out, exist_ok=True)

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]

            large_fs    = [best_fewshot_by_var(ds, large, s, var) for s in SHOTS]
            small_cicle = [best_cicle_by_var(ds, small, s, var)   for s in SHOTS]
            small_fs    = [best_fewshot_by_var(ds, small, s, var)  for s in SHOTS]

            fig, ax = plt.subplots(figsize=(6, 4.5))

            ax.plot(SHOTS, large_fs, color="#666666", marker="s",
                    linewidth=1.8, markersize=7, linestyle=":", zorder=3,
                    label=f"{MODEL_LABELS[large]} few-shot (best)")

            ax.plot(SHOTS, small_fs, color=FAMILY_COLORS[fam], marker="^",
                    linewidth=1.6, markersize=7, linestyle="--",
                    alpha=0.65, zorder=3,
                    label=f"{MODEL_LABELS[small]} few-shot (best)")

            ax.plot(SHOTS, small_cicle, color=FAMILY_COLORS[fam], marker="o",
                    linewidth=2.2, markersize=8, zorder=4,
                    label=f"{MODEL_LABELS[small]} CICLe (best)")

            for s, v in zip(SHOTS, small_cicle):
                if not np.isnan(v):
                    ax.annotate(f"{v:.3f}", (s, v),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=8.5, color=FAMILY_COLORS[fam],
                                fontweight="bold")

            ax.set_xlabel("Number of shots")
            ax.set_ylabel("Macro-F1")
            ax.set_xticks(SHOTS)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_title(f"{DATASET_LABELS[ds]}  ·  {fam} family  [{label} variant]",
                         fontweight="bold", pad=14)
            ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

            all_vals = [v for v in list(small_cicle) + list(large_fs) + list(small_fs)
                        if v is not None and not np.isnan(v)]
            ax.set_ylim(bottom=0, top=max(all_vals) * 1.18 if all_vals else 1)

            prefix = "A3_1" if var == "pc" else "A3_2"
            fname = f"{prefix}_{ds.replace('-', '_')}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  {prefix} saved: {fname}")


def plot_A3_1():
    _plot_A3_variant("pc",    "PC — k shots per class",    "A3_1_gap_with_fewshot_pc")


def plot_A3_2():
    _plot_A3_variant("fixed", "Fixed — k shots total", "A3_2_gap_with_fewshot_fixed")


# ── A4: Break-even shots ───────────────────────────────────────────────────────
def plot_A4():
    out = os.path.join(HERE, "A4_breakeven_shots")
    os.makedirs(out, exist_ok=True)

    for ds in DATASETS:
        fam_names = list(FAMILIES.keys())
        be_zs, be_fs = [], []   # break-even vs large zero-shot / large few-shot

        for fam in fam_names:
            large, small = FAMILIES[fam]["large"], FAMILIES[fam]["small"]
            lz = ZEROSHOT.get((ds, large), np.nan)

            found_zs, found_fs = None, None
            for s in SHOTS:
                sc = best_cicle(ds, small, s)
                lf = best_fewshot(ds, large, s)
                if not np.isnan(sc):
                    if found_zs is None and not np.isnan(lz) and sc >= lz:
                        found_zs = s
                    if found_fs is None and not np.isnan(lf) and sc >= lf:
                        found_fs = s
            be_zs.append(found_zs)
            be_fs.append(found_fs)

        x = np.arange(len(fam_names))
        w = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))

        COLOR_ZS = "#2dc653"
        COLOR_FS = "#4361ee"
        NA_Y = 0.35  # fixed y for "N/A" annotation when no bar is drawn

        for i, (v_zs, v_fs) in enumerate(zip(be_zs, be_fs)):
            if v_zs is not None:
                ax.bar(x[i] - w/2, v_zs, w, color=COLOR_ZS, edgecolor="white")
                ax.text(x[i] - w/2, v_zs + 0.1, str(v_zs),
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color=COLOR_ZS)
            else:
                ax.text(x[i] - w/2, NA_Y, "N/A",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color=COLOR_ZS)

            if v_fs is not None:
                ax.bar(x[i] + w/2, v_fs, w, color=COLOR_FS, edgecolor="white")
                ax.text(x[i] + w/2, v_fs + 0.1, str(v_fs),
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color=COLOR_FS)
            else:
                ax.text(x[i] + w/2, NA_Y, "N/A",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color=COLOR_FS)

        legend_elements = [
            mpatches.Patch(facecolor=COLOR_ZS, label="vs. large zero-shot"),
            mpatches.Patch(facecolor=COLOR_FS, label="vs. large few-shot"),
        ]
        ax.legend(handles=legend_elements, framealpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(fam_names, fontsize=11)
        ax.set_yticks([1, 2, 4, 8])
        ax.set_yticklabels(["1", "2", "4", "8"])
        ax.set_ylabel("Min. shots to match baseline")
        ax.set_title(f"{DATASET_LABELS[ds]}  ·  Break-even shots\n"
                     f"(first k where small CICLe ≥ large reference; N/A = never within k≤8)",
                     fontweight="bold", fontsize=10)
        ax.set_ylim(0, 11)
        ax.grid(axis="y")

        fname = f"A4_{ds.replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  A4 saved: {fname}")


# ── Shared scatter helper ─────────────────────────────────────────────────────
def _scatter_ax(ax, all_x, all_y, xlabel, ylabel, title, win_label="Small wins",
                show_wins=True, wins_pos=(0.04, 0.96), wins_va="top"):
    """Apply the shared styling used by the scatter plots."""
    if not all_x:
        return
    lo = min(min(all_x), min(all_y)) - 0.02
    hi = max(max(all_x), max(all_y)) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.2,
            linestyle="--", zorder=2, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if show_wins:
        win   = sum(1 for y, x in zip(all_y, all_x) if y > x)
        lose  = sum(1 for y, x in zip(all_y, all_x) if y < x)
        total = win + lose
        pct   = f" ({100*win/total:.0f}%)" if total else ""
        ax.text(wins_pos[0], wins_pos[1], f"{win_label}: {win}/{total}{pct}",
                transform=ax.transAxes, fontsize=9,
                va=wins_va, color="#2dc653", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_aspect("equal")


# ── A5: Scatter — small CICLe vs. large model zero-shot ──────────────────────
def plot_A5():
    out   = os.path.join(HERE, "A5_scatter_vs_zeroshot")
    out_1 = os.path.join(out, "A5_1_scatter_vs_zeroshot_per_family")
    out_2 = os.path.join(out, "A5_2_scatter_vs_zeroshot_all_families")
    os.makedirs(out_1, exist_ok=True)
    os.makedirs(out_2, exist_ok=True)

    # ── A5_1: one plot per family — x = large model zero-shot F1 ───────────────
    # Each dataset gives a different x (large zero-shot F1 on that dataset).
    # Multiple y-values per dataset (one per CICLe config), coloured by shots.
    DS_SHORT = {
        "yahoo-answers": "Yahoo",
        "go-emotions":   "Go-Emotions",
        "semeval-18":    "SemEval-18",
    }

    for fam, models in FAMILIES.items():
        large, small = models["large"], models["small"]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        all_x, all_y = [], []

        for s in SHOTS:
            x_pts, y_pts = [], []
            for ds in DATASETS:
                lz = ZEROSHOT.get((ds, large), np.nan)
                if np.isnan(lz):
                    continue
                for emb in EMBEDDINGS:
                    for v in VARIANTS:
                        y_val = best_cicle_by_emb_var(ds, small, s, emb, v)
                        if not np.isnan(y_val):
                            x_pts.append(lz)
                            y_pts.append(y_val)
            if x_pts:
                ax.scatter(x_pts, y_pts, color=SHOTS_COLORS[s], s=55,
                           alpha=0.8, zorder=3,
                           label=f"{s} shot{'s' if s>1 else ''}",
                           edgecolors="white", linewidths=0.5)
            all_x.extend(x_pts)
            all_y.extend(y_pts)

        if not all_x:
            plt.close(fig)
            continue

        _scatter_ax(ax, all_x, all_y,
                    xlabel=f"{MODEL_LABELS[large]} zero-shot  (Macro-F1)",
                    ylabel=f"{MODEL_LABELS[small]} CICLe  (Macro-F1)",
                    title=f"{fam} family — small CICLe vs. large zero-shot\n(x varies by dataset, coloured by shots)",
                    wins_pos=(0.04, 0.55), wins_va="bottom")

        # Prominent dataset labels at top of each vertical reference line
        ylim = ax.get_ylim()
        label_y = ylim[0] + (ylim[1] - ylim[0]) * 0.97
        for ds in DATASETS:
            lz = ZEROSHOT.get((ds, large), np.nan)
            if not np.isnan(lz):
                ax.axvline(lz, color="#aaaaaa", linewidth=1.0, linestyle="--", zorder=1)
                ax.text(lz, label_y, DS_SHORT[ds],
                        fontsize=9, color="#333333",
                        ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                                  edgecolor="#aaaaaa", linewidth=0.8, alpha=0.9),
                        zorder=5)

        fname = f"A5_1_{fam.lower()}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out_1, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  A5_1 saved: {fname}")

    # ── A5_2: one plot per dataset — x = large model zero-shot F1 ───────────
    # Each family gives a different x (its large zero-shot F1 on that dataset).
    # Multiple y-values per family (one per CICLe config), coloured by family.
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        all_x, all_y = [], []

        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]
            lz = ZEROSHOT.get((ds, large), np.nan)
            if np.isnan(lz):
                continue
            x_pts, y_pts = [], []
            for s in SHOTS:
                for emb in EMBEDDINGS:
                    for v in VARIANTS:
                        y_val = best_cicle_by_emb_var(ds, small, s, emb, v)
                        if not np.isnan(y_val):
                            x_pts.append(lz)
                            y_pts.append(y_val)
            if x_pts:
                ax.scatter(x_pts, y_pts, color=FAMILY_COLORS[fam], s=45,
                           alpha=0.75, label=fam, zorder=3,
                           edgecolors="white", linewidths=0.4)
            all_x.extend(x_pts)
            all_y.extend(y_pts)

        if not all_x:
            plt.close(fig)
            continue

        _scatter_ax(ax, all_x, all_y,
                    xlabel=f"{MODEL_LABELS[large]} zero-shot  (Macro-F1)",
                    ylabel=f"{MODEL_LABELS[small]} CICLe  (Macro-F1)",
                    title=f"{DATASET_LABELS[ds]}\nAll families — small CICLe vs. large zero-shot\n(x varies by family)",
                    wins_pos=(0.04, 0.96), wins_va="top")

        fname = f"A5_2_{ds.replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out_2, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  A5_2 saved: {fname}")


# ── A6: Scatter — small CICLe vs. large few-shot (A6_1 per-family, A6_2 all) ──
def plot_A6():
    out   = os.path.join(HERE, "A6_scatter_vs_fewshot")
    out_1 = os.path.join(out, "A6_1_scatter_vs_fewshot_per_family")
    out_2 = os.path.join(out, "A6_2_scatter_vs_fewshot_all_families")
    os.makedirs(out_1, exist_ok=True)
    os.makedirs(out_2, exist_ok=True)

    # ── A6_1: one plot per dataset × family ──────────────────────────────────
    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]

            fig, ax = plt.subplots(figsize=(5, 5))
            all_x, all_y = [], []

            for s in SHOTS:
                x_pts, y_pts = [], []
                for emb in EMBEDDINGS:
                    for v in VARIANTS:
                        x_val = fewshot_val(ds, large, s, emb, v)
                        y_val = best_cicle_by_emb_var(ds, small, s, emb, v)
                        if not np.isnan(x_val) and not np.isnan(y_val):
                            x_pts.append(x_val)
                            y_pts.append(y_val)
                if x_pts:
                    ax.scatter(x_pts, y_pts, color=SHOTS_COLORS[s], s=55,
                               alpha=0.8, zorder=3,
                               label=f"{s} shot{'s' if s>1 else ''}",
                               edgecolors="white", linewidths=0.5)
                all_x.extend(x_pts)
                all_y.extend(y_pts)

            if not all_x:
                plt.close(fig)
                continue

            _scatter_ax(ax, all_x, all_y,
                        xlabel=f"{MODEL_LABELS[large]} few-shot  (Macro-F1)",
                        ylabel=f"{MODEL_LABELS[small]} CICLe  (Macro-F1)",
                        title=f"{DATASET_LABELS[ds]}  ·  {fam} family")

            fname = f"A6_1_{ds.replace('-', '_')}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out_1, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  A6_1 saved: {fname}")

    # ── A6_2: one plot per dataset, all families overlaid ────────────────────
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        all_x, all_y = [], []

        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]
            x_pts, y_pts = [], []
            for s in SHOTS:
                for emb in EMBEDDINGS:
                    for v in VARIANTS:
                        x_val = fewshot_val(ds, large, s, emb, v)
                        y_val = best_cicle_by_emb_var(ds, small, s, emb, v)
                        if not np.isnan(x_val) and not np.isnan(y_val):
                            x_pts.append(x_val)
                            y_pts.append(y_val)
            if x_pts:
                ax.scatter(x_pts, y_pts, color=FAMILY_COLORS[fam], s=45,
                           alpha=0.75, label=fam, zorder=3,
                           edgecolors="white", linewidths=0.4)
                all_x.extend(x_pts)
                all_y.extend(y_pts)

        if not all_x:
            plt.close(fig)
            continue

        _scatter_ax(ax, all_x, all_y,
                    xlabel="Large model few-shot  (Macro-F1)",
                    ylabel="Small model CICLe  (Macro-F1)",
                    title=f"{DATASET_LABELS[ds]}\nAll families: small CICLe vs. large few-shot")

        fname = f"A6_2_{ds.replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out_2, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  A6_2 saved: {fname}")


# ── A7: Grouped bar — full method comparison ──────────────────────────────────
def plot_A7():
    out = os.path.join(HERE, "A7_grouped_bar_bestconfig")
    os.makedirs(out, exist_ok=True)

    bar_defs = [
        ("Baseline LR",        "#adb5bd"),
        ("Baseline SVM",       "#6c757d"),
        ("Large zero-shot",    "#333333"),
        ("Large few-shot",     "#888888"),
        ("Small few-shot",     "#90caf9"),
        ("Small CICLe",        "#e63946"),
        ("Large CICLe",        "#4361ee"),
    ]

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]

            vals = [
                BASELINE.get((ds, "lr"),  np.nan),
                BASELINE.get((ds, "svm"), np.nan),
                ZEROSHOT.get((ds, large), np.nan),
                max((best_fewshot(ds, large, s) for s in SHOTS), default=np.nan),
                max((best_fewshot(ds, small, s) for s in SHOTS), default=np.nan),
                max((best_cicle(ds, small, s)   for s in SHOTS), default=np.nan),
                max((best_cicle(ds, large, s)   for s in SHOTS), default=np.nan),
            ]

            labels  = [b[0] for b in bar_defs]
            colors  = [b[1] for b in bar_defs]
            x       = np.arange(len(labels))

            fig, ax = plt.subplots(figsize=(9, 4.5))
            bars = ax.bar(x, vals, color=colors, width=0.6,
                          edgecolor="white", linewidth=1.2)

            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.004,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=9, fontweight="bold")

            # Highlight small CICLe bar
            idx = labels.index("Small CICLe")
            bars[idx].set_linewidth(2.5)
            bars[idx].set_edgecolor("#b00020")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
            ax.set_ylabel("Macro-F1")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.set_ylim(0, min(1.0, np.nanmax(vals) * 1.22))
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {fam} family\n"
                f"({MODEL_LABELS[large]}  vs.  {MODEL_LABELS[small]})",
                fontweight="bold")

            fname = f"A7_{ds.replace('-', '_')}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  A7 saved: {fname}")


# ── A8: Heatmap — gap from large zero-shot ────────────────────────────────────
def plot_A8():
    out = os.path.join(HERE, "A8_heatmap_gap_zeroshot")
    os.makedirs(out, exist_ok=True)

    row_keys   = [(e, v) for e in EMBEDDINGS for v in VARIANTS]
    row_labels = [f"{EMB_LABELS[e]} / {VARIANT_LABELS[v]}" for e, v in row_keys]
    col_keys   = list(FAMILIES.keys())

    for ds in DATASETS:
        for shots in SHOTS:
            mat = np.full((len(row_keys), len(col_keys)), np.nan)

            for ci, fam in enumerate(col_keys):
                large, small = FAMILIES[fam]["large"], FAMILIES[fam]["small"]
                lz = ZEROSHOT.get((ds, large), np.nan)
                if np.isnan(lz):
                    continue
                for ri, (emb, v) in enumerate(row_keys):
                    sc = best_cicle_by_emb_var(ds, small, shots, emb, v)
                    if not np.isnan(sc):
                        mat[ri, ci] = (sc - lz) * 100  # in percentage points

            fig, ax = plt.subplots(figsize=(5.5, 5))

            vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 0.1
            im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

            cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
            cbar.set_label("ΔMacro-F1 (pp)", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

            ax.set_xticks(range(len(col_keys)))
            ax.set_xticklabels(col_keys, fontsize=10)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=9)
            ax.set_xlabel("Model family")

            for ri in range(len(row_keys)):
                for ci in range(len(col_keys)):
                    v = mat[ri, ci]
                    if not np.isnan(v):
                        txt_color = "white" if abs(v) > vmax * 0.6 else "black"
                        ax.text(ci, ri, f"{v:+.1f}", ha="center", va="center",
                                fontsize=9, fontweight="bold", color=txt_color)

            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {shots} shot{'s' if shots>1 else ''}\n"
                f"Small CICLe minus Large zero-shot  (percentage points)",
                fontweight="bold")

            # Remove grid for heatmaps
            ax.grid(False)

            fname = f"A8_{ds.replace('-', '_')}_{shots}shots.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  A8 saved: {fname}")


# ── A9: Win-rate summary ───────────────────────────────────────────────────────
def plot_A9():
    out = os.path.join(HERE, "A9_winrate")
    os.makedirs(out, exist_ok=True)

    EPSILON = 0.001  # tolerance for "tie"

    for ds in DATASETS:
        # y-axis: one row per (family, shots)
        row_labels, wins, ties, losses = [], [], [], []

        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]
            for shots in SHOTS:
                w, t, l = 0, 0, 0
                for emb in EMBEDDINGS:
                    for v in VARIANTS:
                        lf = fewshot_val(ds, large, shots, emb, v)
                        sc = best_cicle_by_emb_var(ds, small, shots, emb, v)
                        if np.isnan(lf) or np.isnan(sc):
                            continue
                        diff = sc - lf
                        if diff > EPSILON:       w += 1
                        elif diff < -EPSILON:    l += 1
                        else:                    t += 1

                row_labels.append(f"{fam}  {shots}s")
                wins.append(w)
                ties.append(t)
                losses.append(l)

        n = len(row_labels)
        if n == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 0.55 * n + 1.5))
        y = np.arange(n)

        totals = [w + t + l for w, t, l in zip(wins, ties, losses)]
        w_pct  = [w/tot*100 if tot else 0 for w, tot in zip(wins,   totals)]
        t_pct  = [t/tot*100 if tot else 0 for t, tot in zip(ties,   totals)]
        l_pct  = [l/tot*100 if tot else 0 for l, tot in zip(losses, totals)]

        ax.barh(y, w_pct, color="#2dc653", label="Small CICLe wins")
        ax.barh(y, t_pct, left=w_pct, color="#ffd166", label="Tie (±0.1 pp)")
        ax.barh(y, l_pct, left=[w+t for w, t in zip(w_pct, t_pct)],
                color="#ef233c", label="Small CICLe loses")

        for i, (w, t, l) in enumerate(zip(w_pct, t_pct, l_pct)):
            if w > 5:
                ax.text(w/2, i, f"{w:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            if l > 5:
                ax.text(w + t + l/2, i, f"{l:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel("Percentage of configurations (%)")
        ax.set_xlim(0, 100)
        ax.axvline(50, color="#333333", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(
            f"{DATASET_LABELS[ds]}\nWin-rate: small CICLe vs. large few-shot\n"
            f"(each config pair = one embedding × variant combination)",
            fontweight="bold")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
        ax.grid(axis="x", alpha=0.3)
        ax.yaxis.grid(False)

        fname = f"A9_{ds.replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  A9 saved: {fname}")


def count_saved_plots():
    """Return the number of PNG files saved under the Group A output folders."""
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    """Load all metrics once, then generate the full Group A figure set."""
    global ZEROSHOT, CICLE, FEWSHOT, BASELINE

    print("Loading data …")
    ZEROSHOT = load_zeroshot()
    CICLE, FEWSHOT = load_cicle_fewshot()
    BASELINE = load_baseline()
    print(f"  Zeroshot entries : {len(ZEROSHOT)}")
    print(f"  CICLe datasets   : {len(CICLE)}")
    print(f"  Fewshot datasets : {len(FEWSHOT)}")

    print("\nGenerating Group A plots …")
    print("A1 — Gap plot: small CICLe best config vs. large zero-shot")
    plot_A1()
    print("A2 — Mean ± std across all CICLe configs vs. large zero-shot")
    plot_A2()
    print("A3 — Gap plot: adding large few-shot reference")
    plot_A3()
    print("A3_1 — Same as A3, PC variant only")
    plot_A3_1()
    print("A3_2 — Same as A3, Fixed variant only")
    plot_A3_2()
    print("A4 — Break-even shots")
    plot_A4()
    print("A5 — Scatter: small CICLe vs. large zero-shot (A5_1 per-family, A5_2 all families)")
    plot_A5()
    print("A6 — Scatter: small CICLe vs. large few-shot (A6_1 per-family, A6_2 all families)")
    plot_A6()
    print("A7 — Grouped bar: full method comparison")
    plot_A7()
    print("A8 — Heatmap: gap from large zero-shot")
    plot_A8()
    print("A9 — Win-rate summary")
    plot_A9()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
