#!/usr/bin/env python3
"""
Reproduces every figure in "Evaluating Conformal In-Context Learning
Across Tasks and Models", reading directly from each dataset's
results/predictions and results/lengths JSON files and saving the
output into figures/.

Each paper figure has its own function below, named after what it
shows rather than its number, so the mapping survives any future
renumbering:

    Fig. 1 -> figure_comparability_main()         comparability_yahoo_answers.png
                                                   comparability_sst.png
                                                   comparability_semeval_18.png
                                                   comparability_go_emotions.png
    Fig. 2 -> figure_imbalance_vs_benefit()       imbalance_vs_benefit.png
    Fig. 3 -> figure_components()                 components_by_embedding_and_classifier.png
    Fig. 4 -> figure_alpha_sensitivity()           accuracy_vs_alpha.png
    Fig. 5 -> figure_small_vs_large()              small_model_vs_large.png
    Fig. 6 -> figure_fol_reasoning_comparability() fol_reasoning_comparability.png

Usage:
    python visualize.py
"""
import os
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

CORE_DATASETS = ["yahoo-answers", "sst", "semeval-18", "go-emotions"]

DATASET_PREFIX = {
    "yahoo-answers": "yahoo",
    "sst":           "sst",
    "semeval-18":    "semeval-18",
    "go-emotions":   "go-emotions",
    "fol-reasoning": "fol-reasoning",
}

DATASET_LABELS = {
    "yahoo-answers": "Yahoo Answers",
    "sst":           "SST-5",
    "semeval-18":    "SemEval-18 Emoji",
    "go-emotions":   "GoEmotions",
    "fol-reasoning": "FOL-Reasoning",
}

SHORT_LABELS = {
    "yahoo-answers": "Yahoo",
    "sst":           "SST-5",
    "semeval-18":    "SemEval-18",
    "go-emotions":   "GoEmotions",
}

# class count and imbalance ratio (max/min class size in the full training
# set), as reported in Table 1 of the paper
DATASET_META = {
    "yahoo-answers": {"classes": 10, "imbalance": 1.0},
    "sst":           {"classes": 5,  "imbalance": 2.1},
    "semeval-18":    {"classes": 20, "imbalance": 8.7},
    "go-emotions":   {"classes": 28, "imbalance": 184.7},
    "fol-reasoning": {"classes": 12, "imbalance": 1.2},
}

ALL_MODELS = [
    "llama-3.1-8b", "llama-3.2-3b",
    "mistral-7b-v0.3", "ministral-3b",
    "qwen-2.5-7b", "qwen-2.5-3b",
]
LARGE_MODELS = ["llama-3.1-70b", "qwen-2.5-32b", "mistral-nemo-2407"]
MODEL_SIZES_B = {
    "llama-3.2-3b": 3, "qwen-2.5-3b": 3, "ministral-3b": 3,
    "mistral-7b-v0.3": 7, "qwen-2.5-7b": 7, "llama-3.1-8b": 8,
    "mistral-nemo-2407": 12, "qwen-2.5-32b": 32, "llama-3.1-70b": 70,
}

EMBEDDINGS = ["contriever", "minilm", "tfidf"]
CLASSIFIERS = ["lr", "svm"]
SHOTS = [1, 2, 4, 8]
ALPHAS = [0.01, 0.05, 0.10, 0.20]

# shared colour palette, used consistently across every figure
DS_COLORS = {
    "yahoo-answers": "#1d3557",
    "sst":           "#457b9d",
    "semeval-18":    "#e76f51",
    "go-emotions":   "#2a9d8f",
}
EMB_COLORS = {"contriever": "#1d3557", "minilm": "#457b9d", "tfidf": "#e76f51"}
EMB_LABELS = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
CLF_COLORS = {"lr": "#2a9d8f", "svm": "#9b5de5"}
CLF_LABELS = {"lr": "Logistic Regression", "svm": "SVM"}
ZS_COLOR = "#1d4e89"
CICLE_COLOR = "#d62828"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11.5,
    "axes.titlesize":    13.5,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10.5,
    "figure.dpi":        140,
    "axes.spines.top":   False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "axes.axisbelow":    True,
})


# ---------------------------------------------------------------------------
# Data loading: everything reads straight from results/predictions and
# results/lengths, so figures always reflect what's currently on disk.
# ---------------------------------------------------------------------------
import json
import re


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_f1(path):
    d = _load_json(path)
    return None if d is None else d["classification_report"]["macro avg"]["f1-score"]


def _load_mean_shots(lengths_path):
    d = _load_json(lengths_path)
    if d is None or "num_shots" not in d:
        return None
    ns = d["num_shots"]
    return ns["mean"] if isinstance(ns, dict) else float(np.mean(ns))


def collect_records(ds):
    """All few-shot + CICLe records for a dataset (excludes zero-shot and
    the standalone TF-IDF baseline, which collect_zeroshot/collect_tfidf
    handle separately)."""
    pfx = DATASET_PREFIX[ds]
    pred_dir = os.path.join(BASE_DIR, ds, "results", "predictions")
    len_dir = os.path.join(BASE_DIR, ds, "results", "lengths")
    records = []
    if not os.path.isdir(pred_dir):
        return records

    fewshot_re = re.compile(rf"^{pfx}-(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots-(fixed|pc)\.json$")
    cicle_re = re.compile(rf"^{pfx}-(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots-(fixed|pc)-([\d.]+)-α\.json$")

    for fname in os.listdir(pred_dir):
        if not fname.endswith(".json"):
            continue

        m = fewshot_re.match(fname)
        if m:
            llm, emb, shots, variant = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            if llm not in ALL_MODELS:
                continue
            f1 = _load_f1(os.path.join(pred_dir, fname))
            if f1 is None:
                continue
            n_classes = DATASET_META[ds]["classes"]
            mean_examples = shots if variant == "fixed" else shots * n_classes
            records.append({
                "method": "fewshot", "variant": variant, "llm": llm, "emb": emb,
                "shots": shots, "alpha": None, "clf": None, "f1": f1,
                "mean_examples": mean_examples,
            })
            continue

        m = cicle_re.match(fname)
        if m:
            llm, emb, clf = m.group(1), m.group(2), m.group(3)
            shots, variant, alpha = int(m.group(4)), m.group(5), float(m.group(6))
            if llm not in ALL_MODELS:
                continue
            f1 = _load_f1(os.path.join(pred_dir, fname))
            if f1 is None:
                continue
            if variant == "fixed":
                mean_examples = shots
            else:
                mean_examples = _load_mean_shots(os.path.join(len_dir, fname))
            if mean_examples is None:
                continue
            records.append({
                "method": "cicle", "variant": variant, "llm": llm, "emb": emb,
                "shots": shots, "alpha": alpha, "clf": clf, "f1": f1,
                "mean_examples": mean_examples,
            })

    return records


def collect_zeroshot(ds):
    """model -> macro-F1 for zero-shot prompting, across small and large models."""
    pfx = DATASET_PREFIX[ds]
    pred_dir = os.path.join(BASE_DIR, ds, "results", "predictions")
    out = {}
    for model in ALL_MODELS + LARGE_MODELS:
        f1 = _load_f1(os.path.join(pred_dir, f"{pfx}-{model}-zeroshot-2.0k-samples.json"))
        if f1 is not None:
            out[model] = f1
    return out


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def aggregate_by_k(records, method, variant):
    """Group records by shots (k). Returns ks, mean_f1, sem_f1, mean_examples."""
    sub = [r for r in records if r["method"] == method and r["variant"] == variant]
    if not sub:
        return [], [], [], []
    buckets_f1, buckets_ex = defaultdict(list), defaultdict(list)
    for r in sub:
        buckets_f1[r["shots"]].append(r["f1"])
        buckets_ex[r["shots"]].append(r["mean_examples"])
    ks = sorted(buckets_f1)
    means, sems, exs = [], [], []
    for k in ks:
        m, s = mean_sem(buckets_f1[k])
        means.append(m)
        sems.append(s)
        exs.append(np.mean(buckets_ex[k]))
    return ks, means, sems, exs


def paired_delta(ds):
    """CICLe Fixed minus few-shot Fixed, averaged over every matched
    (model, embedding, k) configuration the two methods share. This is
    the statistic behind every '+X.XXpp' delta reported in the paper."""
    records = collect_records(ds)
    fewshot_by_key = defaultdict(list)
    for r in records:
        if r["method"] == "fewshot" and r["variant"] == "fixed":
            fewshot_by_key[(r["llm"], r["emb"], r["shots"])].append(r["f1"])

    deltas = []
    for r in records:
        if r["method"] != "cicle" or r["variant"] != "fixed":
            continue
        key = (r["llm"], r["emb"], r["shots"])
        if key in fewshot_by_key:
            deltas.append(r["f1"] - np.mean(fewshot_by_key[key]))

    deltas = np.array(deltas)
    mean, sem = mean_sem(deltas)
    return mean, sem, len(deltas)


# ---------------------------------------------------------------------------
# Shared plotting building block: one comparability panel (Macro-F1 vs. k,
# four lines, Per-Class demonstration-budget bars on a secondary axis).
# Used by both Figure 1 (the four core benchmarks) and Figure 6
# (FOL-Reasoning).
# ---------------------------------------------------------------------------
LINES_CFG = [
    ("fewshot", "fixed", "#4361ee", "o", "-",  2.0, "Few-shot (Fixed)",     0.94),
    ("cicle",   "fixed", "#e63946", "o", "-",  2.2, "CICLe (Fixed)",        0.98),
    ("cicle",   "pc",    "#f4a261", "^", "--", 1.8, "CICLe (Per-Class)",    1.02),
    ("fewshot", "pc",    "#6c757d", "s", ":",  1.6, "Few-shot (Per-Class)", 1.06),
]
BUDGET_BARS_CFG = [
    ("cicle",   "pc", "#f4a261", -1, "CICLe (Per-Class) budget"),
    ("fewshot", "pc", "#6c757d", +1, "Few-shot (Per-Class) budget"),
]


def _plot_comparability_panel(ax1, ds):
    """Draws one dataset's Macro-F1-vs-k panel onto ax1 (with a secondary
    axis for the demonstration budget), returning the legend handles so
    the caller can place a single shared legend wherever it likes."""
    records = collect_records(ds)
    ax2 = ax1.twinx()

    handles_labels = []
    for method, variant, color, marker, ls, lw, label, dodge in LINES_CFG:
        ks, means, sems, _ = aggregate_by_k(records, method, variant)
        if not ks:
            continue
        ks_dodged = np.array(ks) * dodge
        line = ax1.errorbar(
            ks_dodged, np.array(means) * 100, yerr=1.96 * np.array(sems) * 100,
            color=color, marker=marker, ls=ls, lw=lw, markersize=7,
            capsize=3.5, capthick=1.2, elinewidth=1.2, alpha=0.95, zorder=4,
        )
        handles_labels.append((line, label))

    width_frac = 0.11
    for method, variant, color, side, label in BUDGET_BARS_CFG:
        ks, _, _, exs = aggregate_by_k(records, method, variant)
        if not ks:
            continue
        ks = np.array(ks, dtype=float)
        widths = ks * width_frac
        offsets = ks * width_frac * 0.6 * side
        ax2.bar(ks + offsets, exs, width=widths, color=color, alpha=0.30,
                zorder=1, edgecolor="none")
        from matplotlib.patches import Patch
        handles_labels.append((Patch(facecolor=color, alpha=0.30), label))

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(SHOTS)
    ax1.set_xticklabels([str(k) for k in SHOTS])
    ax1.set_xlim(SHOTS[0] * 0.78, SHOTS[-1] * 1.30)
    ax1.set_xlabel("k (number of shots)")
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=8, integer=True))
    ax2.tick_params(axis="y", colors="#555555")
    ax2.grid(False)
    ax1.set_title(f"{DATASET_LABELS[ds]} ({DATASET_META[ds]['classes']} classes)",
                   fontweight="bold")
    return handles_labels, ax2


# ---------------------------------------------------------------------------
# Figure 1: CICLe vs. standard few-shot prompting, all four core benchmarks.
# Saved as four separate files, matching how the paper's LaTeX actually
# places them (two \includegraphics per row) rather than one combined
# image. CORE_DATASETS is ordered to match that 2x2 layout: Yahoo
# Answers (top-left), SST-5 (top-right), SemEval-18 (bottom-left),
# GoEmotions (bottom-right) -- the legend appears only on the
# bottom-right panel.
# ---------------------------------------------------------------------------
def figure_comparability_main():
    for ds in CORE_DATASETS:
        fig, ax1 = plt.subplots(figsize=(6.0, 4.5))
        handles_labels, ax2 = _plot_comparability_panel(ax1, ds)
        ax1.set_ylabel("Mean Macro-F1 (%)")
        ax2.set_ylabel("Example budget (mean # examples)", color="#555555")

        if ds == "go-emotions":  # bottom-right panel in the paper's layout
            handles = [h for h, _ in handles_labels]
            labels = [l for _, l in handles_labels]
            ax1.legend(handles, labels, framealpha=0.95, loc="lower right")

        fig.tight_layout()
        _save(fig, f"comparability_{ds.replace('-', '_')}.png")


# ---------------------------------------------------------------------------
# Figure 2: class imbalance predicts CICLe's benefit
# ---------------------------------------------------------------------------
def figure_imbalance_vs_benefit():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    xs, ys, errs, labels = [], [], [], []
    for ds in CORE_DATASETS:
        mean, sem, _ = paired_delta(ds)
        xs.append(DATASET_META[ds]["imbalance"])
        ys.append(mean * 100)
        errs.append(1.96 * sem * 100)
        labels.append(SHORT_LABELS[ds])
    xs, ys, errs = np.array(xs), np.array(ys), np.array(errs)

    ax.axhline(0, color="#999999", lw=1.0, zorder=1)
    ax.plot(xs, ys, color="#a8b8c8", lw=2.2, zorder=2)
    ax.errorbar(xs, ys, yerr=errs, fmt="o", color="#1d3557", markersize=9,
                markeredgecolor="white", markeredgewidth=1.3, elinewidth=1.5,
                capsize=4, zorder=3)

    pad = 0.10
    for x, y, e, lbl in zip(xs, ys, errs, labels):
        ax.annotate(lbl, xy=(x, y), xytext=(x, y - e - pad), textcoords="data",
                    ha="center", va="top", fontsize=11, color="#1d3557")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.1f}$\\times$" for x in xs])
    ax.set_xlabel("Class imbalance")
    ax.set_ylabel("CICLe's Macro-F1 advantage (pp)")
    ax.set_title("CICLe's benefit grows with class imbalance", fontweight="bold")
    ax.set_ylim(min(ys - errs) - 0.35, max(ys + errs) + 0.15)

    fig.tight_layout()
    _save(fig, "imbalance_vs_benefit.png")


# ---------------------------------------------------------------------------
# Figure 3: which CICLe components matter (embedding, base classifier)
# ---------------------------------------------------------------------------
def figure_components():
    fig, (ax_emb, ax_clf) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(CORE_DATASETS))

    width = 0.25
    for i, emb in enumerate(EMBEDDINGS):
        means, sems = [], []
        for ds in CORE_DATASETS:
            f1s = [r["f1"] * 100 for r in collect_records(ds)
                   if r["method"] == "cicle" and r["variant"] == "fixed" and r["emb"] == emb]
            m, s = mean_sem(f1s)
            means.append(m)
            sems.append(1.96 * s)
        ax_emb.bar(x + (i - 1) * width, means, width, yerr=sems, capsize=3,
                   color=EMB_COLORS[emb], label=EMB_LABELS[emb],
                   edgecolor="white", linewidth=0.6)
    ax_emb.set_xticks(x)
    ax_emb.set_xticklabels([SHORT_LABELS[ds] for ds in CORE_DATASETS])
    ax_emb.set_ylabel("Mean Macro-F1 (%)")
    ax_emb.set_title("(a) By embedding", fontweight="bold", fontsize=12)
    ax_emb.legend(framealpha=0.95)

    width = 0.35
    for i, clf in enumerate(CLASSIFIERS):
        means, sems = [], []
        for ds in CORE_DATASETS:
            f1s = [r["f1"] * 100 for r in collect_records(ds)
                   if r["method"] == "cicle" and r["variant"] == "fixed" and r["clf"] == clf]
            m, s = mean_sem(f1s)
            means.append(m)
            sems.append(1.96 * s)
        ax_clf.bar(x + (i - 0.5) * width, means, width, yerr=sems, capsize=3,
                   color=CLF_COLORS[clf], label=CLF_LABELS[clf],
                   edgecolor="white", linewidth=0.6)
    ax_clf.set_xticks(x)
    ax_clf.set_xticklabels([SHORT_LABELS[ds] for ds in CORE_DATASETS])
    ax_clf.set_title("(b) By base classifier", fontweight="bold", fontsize=12)
    ax_clf.legend(framealpha=0.95)

    fig.tight_layout()
    _save(fig, "components_by_embedding_and_classifier.png")


# ---------------------------------------------------------------------------
# Figure 4: accuracy vs. the conformal miscoverage level alpha
# ---------------------------------------------------------------------------
def figure_alpha_sensitivity():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for ds in CORE_DATASETS:
        records = collect_records(ds)
        means = []
        for a in ALPHAS:
            f1s = [r["f1"] for r in records if r["method"] == "cicle"
                   and r["variant"] == "fixed" and r["shots"] == 2 and r["alpha"] == a]
            means.append(np.mean(f1s) * 100)
        ax.plot(ALPHAS, means, marker="o", lw=2.2, markersize=7,
                color=DS_COLORS[ds], label=SHORT_LABELS[ds])

    ax.set_xticks(ALPHAS)
    ax.set_xlabel(r"Miscoverage level $\alpha$")
    ax.set_ylabel("Mean Macro-F1 (%)")
    ax.set_title(r"Accuracy vs. $\alpha$ (CICLe Fixed, $k$=2)", fontweight="bold")
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    _save(fig, "accuracy_vs_alpha.png")


# ---------------------------------------------------------------------------
# Figure 5: small model + CICLe vs. zero-shot with a larger model
# ---------------------------------------------------------------------------
def figure_small_vs_large():
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.2))

    # 7B and 8B models are merged into one "7-8B" bucket on the x-axis,
    # since their tick labels collide at this resolution on a log scale.
    def bucket(size_b):
        return 7.5 if size_b in (7, 8) else size_b
    bucket_label = {3: "3B", 7.5: "7-8B", 12: "12B", 32: "32B", 70: "70B"}

    for ax, ds in zip(axes.flat, CORE_DATASETS):
        zeroshot = collect_zeroshot(ds)
        size_to_f1 = defaultdict(list)
        for model, f1 in zeroshot.items():
            size_to_f1[bucket(MODEL_SIZES_B[model])].append(f1 * 100)
        size_to_f1 = {b: np.mean(vals) for b, vals in size_to_f1.items()}
        xs = sorted(size_to_f1)
        ys = [size_to_f1[x] for x in xs]
        ax.plot(xs, ys, color=ZS_COLOR, marker="o", lw=2, markersize=7,
                label="Zero-shot", zorder=3)

        small_models = [m for m in ALL_MODELS if MODEL_SIZES_B[m] == 3]
        cicle_3b = [r["f1"] * 100 for r in collect_records(ds)
                    if r["method"] == "cicle" and r["variant"] == "fixed" and r["llm"] in small_models]
        best_cicle_3b = max(cicle_3b)
        ax.axhline(best_cicle_3b, color=CICLE_COLOR, ls="--", lw=1.8, zorder=2)
        ax.scatter([3], [best_cicle_3b], color=CICLE_COLOR, s=140, marker="*",
                   zorder=4, label="Best 3B + CICLe", edgecolor="white", linewidth=0.8)

        # Shade only the contiguous runs of model sizes where 3B+CICLe
        # matches or beats zero-shot (computed per-run, in log-space, so a
        # non-monotonic dip-then-rise in the zero-shot curve, as on SST-5,
        # doesn't get bridged over by the shading).
        ys_arr = np.array(ys)
        mask = ys_arr <= best_cicle_3b
        log_xs = np.log(xs)
        left_edge = np.empty_like(log_xs)
        right_edge = np.empty_like(log_xs)
        left_edge[0] = log_xs[0] - (log_xs[1] - log_xs[0]) / 2 if len(log_xs) > 1 else log_xs[0] - 0.5
        right_edge[-1] = log_xs[-1] + (log_xs[-1] - log_xs[-2]) / 2 if len(log_xs) > 1 else log_xs[-1] + 0.5
        for i in range(1, len(log_xs)):
            mid = (log_xs[i - 1] + log_xs[i]) / 2
            right_edge[i - 1] = mid
            left_edge[i] = mid
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                ax.axvspan(np.exp(left_edge[i]), np.exp(right_edge[j]),
                           color=CICLE_COLOR, alpha=0.07, zorder=1)
                i = j + 1
            else:
                i += 1

        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([bucket_label[x] for x in xs])
        ax.set_title(DATASET_LABELS[ds], fontweight="bold")

    for ax in axes[1, :]:
        ax.set_xlabel("Zero-shot model size")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1 (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.025), framealpha=0.95)

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save(fig, "small_model_vs_large.png")


# ---------------------------------------------------------------------------
# Figure 6: FOL-Reasoning comparability plot
# ---------------------------------------------------------------------------
def figure_fol_reasoning_comparability():
    fig, ax1 = plt.subplots(figsize=(6.8, 4.5))
    handles_labels, ax2 = _plot_comparability_panel(ax1, "fol-reasoning")

    ax1.set_ylabel("Mean Macro-F1 (%)")
    ax2.set_ylabel("Example budget (mean # examples)", color="#555555")

    # FOL-Reasoning's data fills essentially the whole inside-axes area,
    # so the legend goes outside, below the plot, instead of competing
    # for space with any of the curves.
    handles = [h for h, _ in handles_labels]
    labels = [l for _, l in handles_labels]
    ax1.legend(handles, labels, framealpha=0.95, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5)

    fig.tight_layout()
    _save(fig, "fol_reasoning_comparability.png")


# ---------------------------------------------------------------------------
def _save(fig, fname):
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{fname}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Figure 1: CICLe vs. standard few-shot prompting")
    figure_comparability_main()
    print("Figure 2: class imbalance predicts CICLe's benefit")
    figure_imbalance_vs_benefit()
    print("Figure 3: which CICLe components matter")
    figure_components()
    print("Figure 4: accuracy vs. the miscoverage level alpha")
    figure_alpha_sensitivity()
    print("Figure 5: small model + CICLe vs. zero-shot with a larger model")
    figure_small_vs_large()
    print("Figure 6: FOL-Reasoning comparability")
    figure_fol_reasoning_comparability()
    print("\nDone.")


if __name__ == "__main__":
    main()
