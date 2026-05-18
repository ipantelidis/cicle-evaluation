"""
Group I — Per-Class Analysis
==============================
Goes beyond macro-F1 to examine which individual classes drive the small vs.
large model difference, whether rare classes benefit more or less, and whether
the two models make structurally similar errors.

Run:
    python generate_group_I.py

Outputs are written to subdirectories I1 … I5 relative to this file.
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
VARIANTS   = ["pc", "fixed"]
EPSILON    = 0.001

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
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


def _parse_record(ds, path):
    """Parse one result file into a record dict including per-class data."""
    pfx  = DATASET_PREFIX[ds]
    stem = os.path.basename(path).replace(".json", "")
    if not stem.startswith(f"{pfx}-"):
        return None
    name = stem[len(pfx) + 1:]

    m = re.match(r"^(.+)-zeroshot-2\.0k-samples$", name)
    if m and m.group(1) in target_models:
        meta = {"llm": m.group(1), "method": "zeroshot",
                "embedding": None, "shots": 0,
                "variant": None, "alpha": None, "clf": None}
    else:
        m = re.match(
            r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?$", name)
        if m and m.group(1) in target_models:
            meta = {"llm": m.group(1), "method": "fewshot",
                    "embedding": m.group(2), "shots": int(m.group(3)),
                    "variant": m.group(4) or "none",
                    "alpha": None, "clf": None}
        else:
            m = re.match(
                r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots"
                r"(?:-(pc|fixed))?-([\d.]+)-α$", name)
            if m and m.group(1) in target_models:
                meta = {"llm": m.group(1), "method": "cicle",
                        "embedding": m.group(2), "shots": int(m.group(4)),
                        "variant": m.group(5) or "none",
                        "alpha": float(m.group(6)), "clf": m.group(3)}
            else:
                return None

    payload = _load_json(path)
    cr      = payload["classification_report"]
    labels  = payload.get("labels", [])
    cm      = payload.get("confusion_matrix", [])

    skip = {"accuracy", "macro avg", "micro avg", "weighted avg"}
    per_class_f1  = {k: cr[k]["f1-score"] for k in cr if k not in skip}
    per_class_sup = {k: cr[k]["support"]  for k in cr if k not in skip}
    macro_f1      = cr["macro avg"]["f1-score"]

    return {
        "dataset": ds, "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_sup": per_class_sup,
        "labels": labels, "confusion_matrix": cm,
        **meta,
    }


def load_records():
    records = []
    for ds in DATASETS:
        pred_dir = f"{BASE_DIR}/{ds}/results/predictions"
        for path in glob.glob(f"{pred_dir}/*.json"):
            rec = _parse_record(ds, path)
            if rec:
                records.append(rec)
    return records


# ── Lookup helpers ────────────────────────────────────────────────────────────
def best_record(ds, llm, method, emb=None, shots=None):
    """Return the record with the highest macro-F1 matching the filters."""
    sub = [r for r in RECORDS
           if r["dataset"] == ds and r["llm"] == llm and r["method"] == method
           and (emb   is None or r["embedding"] == emb)
           and (shots is None or r["shots"]     == shots)]
    return max(sub, key=lambda r: r["macro_f1"]) if sub else None


def zeroshot_record(ds, llm):
    sub = [r for r in RECORDS
           if r["dataset"] == ds and r["llm"] == llm and r["method"] == "zeroshot"]
    return sub[0] if sub else None


# ── Shared helpers ────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_fname(s):
    return s.replace("-", "_").replace(".", "")


def _short_label(label, max_len=18):
    return label if len(label) <= max_len else label[:max_len - 1] + "…"


# ── I1: Per-class F1 heatmap ──────────────────────────────────────────────────
def plot_I1():
    """
    9 plots (3 datasets × 3 families).
    Rows = class labels, cols = (small CICLe best, large few-shot best,
    large zero-shot). Color intensity = per-class F1.
    """
    out = ensure_dir("I1_per_class_f1_heatmap")

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]

            rec_sc = best_record(ds, small, "cicle")
            rec_lf = best_record(ds, large, "fewshot")
            rec_lz = zeroshot_record(ds, large)

            if not rec_sc or not rec_lf or not rec_lz:
                continue

            classes = sorted(rec_sc["per_class_f1"].keys())
            if not classes:
                continue

            col_labels = [
                f"{MODEL_LABELS[small]}\nCICLe (best)",
                f"{MODEL_LABELS[large]}\nFew-shot (best)",
                f"{MODEL_LABELS[large]}\nZero-shot",
            ]
            mat = np.array([
                [rec_sc["per_class_f1"].get(c, np.nan) for c in classes],
                [rec_lf["per_class_f1"].get(c, np.nan) for c in classes],
                [rec_lz["per_class_f1"].get(c, np.nan) for c in classes],
            ]).T   # shape: (n_classes, 3)

            n = len(classes)
            fig_h = max(5, n * 0.38 + 1.5)
            fig, ax = plt.subplots(figsize=(7, fig_h))
            im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=1, aspect="auto")

            ax.set_xticks(range(3))
            ax.set_xticklabels(col_labels, fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels([_short_label(c) for c in classes], fontsize=7)
            ax.grid(False)

            for ri in range(n):
                for ci in range(3):
                    v = mat[ri, ci]
                    if not np.isnan(v):
                        txt_color = "white" if v < 0.35 else "black"
                        ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                                fontsize=6.5, color=txt_color)

            cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.6)
            cbar.set_label("Per-class F1", fontsize=8)
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {fam} family\n"
                f"Per-class F1: small CICLe vs. large references",
                fontweight="bold",
            )

            fname = f"I1_{_safe_fname(ds)}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  I1 saved: {fname}")


# ── I2: Per-class gain bar chart ──────────────────────────────────────────────
def plot_I2():
    """
    9 plots (3 datasets × 3 families).
    Bar chart of (small CICLe − large zero-shot) per class, sorted by delta.
    """
    out = ensure_dir("I2_per_class_gain_vs_zeroshot")

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]

            rec_sc = best_record(ds, small, "cicle")
            rec_lz = zeroshot_record(ds, large)

            if not rec_sc or not rec_lz:
                continue

            classes = sorted(rec_sc["per_class_f1"].keys())
            deltas  = [
                (rec_sc["per_class_f1"].get(c, np.nan)
                 - rec_lz["per_class_f1"].get(c, np.nan)) * 100
                for c in classes
            ]
            order   = np.argsort(deltas)[::-1]
            classes = [classes[i] for i in order]
            deltas  = [deltas[i]  for i in order]

            colors = ["#2dc653" if d >= 0 else "#e63946" for d in deltas]
            n      = len(classes)
            fig_h  = max(5, n * 0.35 + 1.5)
            fig, ax = plt.subplots(figsize=(7, fig_h))

            ax.barh(range(n), deltas, color=colors, edgecolor="white",
                    linewidth=0.6)
            ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--",
                       alpha=0.7)
            ax.set_yticks(range(n))
            ax.set_yticklabels([_short_label(c) for c in classes], fontsize=7)
            ax.set_xlabel("Small CICLe − Large zero-shot  (pp)")
            ax.set_title(
                f"{DATASET_LABELS[ds]}  ·  {fam} family\n"
                f"Per-class gain: small CICLe vs. large zero-shot  "
                f"(sorted by delta)",
                fontweight="bold",
            )
            ax.invert_yaxis()

            fname = f"I2_{_safe_fname(ds)}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  I2 saved: {fname}")


# ── I3: Class-level win rate ──────────────────────────────────────────────────
def plot_I3():
    """
    3 plots (one per dataset).
    For each class: fraction of (family × emb × shots × variant) config pairs
    where small CICLe per-class F1 > large few-shot per-class F1.
    Sorted bar chart.
    """
    out = ensure_dir("I3_class_win_rate")

    for ds in DATASETS:
        # Collect all class names from zero-shot records
        sample = next((r for r in RECORDS
                       if r["dataset"] == ds and r["method"] == "zeroshot"), None)
        if not sample:
            continue
        all_classes = sorted(sample["per_class_f1"].keys())

        win_counts = {c: 0 for c in all_classes}
        total_counts = {c: 0 for c in all_classes}

        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]
            for emb in EMBEDDINGS:
                for shots in SHOTS:
                    for variant in VARIANTS:
                        # best small CICLe for this slice
                        sc_sub = [r for r in RECORDS
                                  if r["dataset"] == ds and r["llm"] == small
                                  and r["method"] == "cicle"
                                  and r["embedding"] == emb
                                  and r["shots"] == shots
                                  and r["variant"] == variant]
                        # large fewshot for this slice
                        lf_sub = [r for r in RECORDS
                                  if r["dataset"] == ds and r["llm"] == large
                                  and r["method"] == "fewshot"
                                  and r["embedding"] == emb
                                  and r["shots"] == shots
                                  and r["variant"] == variant]

                        if not sc_sub or not lf_sub:
                            continue

                        rec_sc = max(sc_sub, key=lambda r: r["macro_f1"])
                        rec_lf = lf_sub[0]

                        for c in all_classes:
                            sc_f1 = rec_sc["per_class_f1"].get(c, np.nan)
                            lf_f1 = rec_lf["per_class_f1"].get(c, np.nan)
                            if np.isnan(sc_f1) or np.isnan(lf_f1):
                                continue
                            total_counts[c] += 1
                            if sc_f1 - lf_f1 > EPSILON:
                                win_counts[c] += 1

        win_fracs = {
            c: win_counts[c] / total_counts[c]
            if total_counts[c] > 0 else np.nan
            for c in all_classes
        }
        valid = [(c, win_fracs[c]) for c in all_classes
                 if not np.isnan(win_fracs[c])]
        valid.sort(key=lambda x: x[1], reverse=True)

        classes = [v[0] for v in valid]
        fracs   = [v[1] for v in valid]
        colors  = ["#2dc653" if f >= 0.5 else "#e63946" for f in fracs]

        n      = len(classes)
        fig_h  = max(5, n * 0.35 + 1.5)
        fig, ax = plt.subplots(figsize=(7, fig_h))

        ax.barh(range(n), [f * 100 for f in fracs],
                color=colors, edgecolor="white", linewidth=0.6)
        ax.axvline(50, color="#333333", linewidth=1.0, linestyle="--",
                   alpha=0.7, label="50% (parity)")
        ax.set_yticks(range(n))
        ax.set_yticklabels([_short_label(c) for c in classes], fontsize=7)
        ax.set_xlabel("% of configs where small CICLe wins per-class F1")
        ax.set_xlim(0, 100)
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Class-level win rate: small CICLe vs. large few-shot  "
            f"(sorted by win %)",
            fontweight="bold",
        )
        ax.invert_yaxis()
        ax.legend(framealpha=0.9)

        fname = f"I3_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  I3 saved: {fname}")


# ── I4: Side-by-side confusion matrices ───────────────────────────────────────
def plot_I4():
    """
    9 plots (3 datasets × 3 families).
    Left: confusion matrix for best small CICLe config (normalised by row).
    Right: confusion matrix for best large few-shot config (normalised by row).
    """
    out = ensure_dir("I4_confusion_matrix_comparison")

    for ds in DATASETS:
        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]

            rec_sc = best_record(ds, small, "cicle")
            rec_lf = best_record(ds, large, "fewshot")

            if not rec_sc or not rec_lf:
                continue
            if not rec_sc["confusion_matrix"] or not rec_lf["confusion_matrix"]:
                continue

            labels_sc = rec_sc["labels"]
            labels_lf = rec_lf["labels"]
            if not labels_sc or not labels_lf:
                continue

            cm_sc = np.array(rec_sc["confusion_matrix"], dtype=float)
            cm_lf = np.array(rec_lf["confusion_matrix"], dtype=float)

            # Normalise rows to sum to 1 (recall per class)
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_sc = np.where(cm_sc.sum(axis=1, keepdims=True) > 0,
                                 cm_sc / cm_sc.sum(axis=1, keepdims=True), 0)
                cm_lf = np.where(cm_lf.sum(axis=1, keepdims=True) > 0,
                                 cm_lf / cm_lf.sum(axis=1, keepdims=True), 0)

            n     = len(labels_sc)
            fig_s = max(5, n * 0.4 + 1)
            fig, axes = plt.subplots(1, 2, figsize=(fig_s * 2 + 1, fig_s))

            short_sc = [_short_label(l, 12) for l in labels_sc]
            short_lf = [_short_label(l, 12) for l in labels_lf]

            for ax, cm, short_labels, title_suffix in zip(
                axes,
                [cm_sc, cm_lf],
                [short_sc, short_lf],
                [f"{MODEL_LABELS[small]} CICLe (best)",
                 f"{MODEL_LABELS[large]} Few-shot (best)"],
            ):
                im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(n))
                ax.set_xticklabels(short_labels, rotation=45, ha="right",
                                   fontsize=6)
                ax.set_yticks(range(n))
                ax.set_yticklabels(short_labels, fontsize=6)
                ax.set_xlabel("Predicted", fontsize=8)
                ax.set_ylabel("True", fontsize=8)
                ax.set_title(title_suffix, fontweight="bold", fontsize=9)
                ax.grid(False)
                fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8,
                             label="Recall")

            fig.suptitle(
                f"{DATASET_LABELS[ds]}  ·  {fam} family\n"
                f"Confusion matrices  (row-normalised = per-class recall)",
                fontweight="bold",
            )

            fname = f"I4_{_safe_fname(ds)}_{fam.lower()}.png"
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), bbox_inches="tight")
            plt.close(fig)
            print(f"  I4 saved: {fname}")


# ── I5: Per-class F1 delta vs. class frequency ────────────────────────────────
def plot_I5():
    """
    3 plots (one per dataset).
    X = class support (test-set frequency from best large few-shot record).
    Y = per-class F1 delta (small CICLe best − large few-shot best).
    One point per class, aggregated across all families.
    """
    out = ensure_dir("I5_per_class_delta_vs_frequency")

    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(7, 5))
        has_data = False

        for fam, models in FAMILIES.items():
            large, small = models["large"], models["small"]
            rec_sc = best_record(ds, small, "cicle")
            rec_lf = best_record(ds, large, "fewshot")

            if not rec_sc or not rec_lf:
                continue

            classes = sorted(rec_sc["per_class_f1"].keys())
            x_vals, y_vals = [], []
            for c in classes:
                sup   = rec_lf["per_class_sup"].get(c, np.nan)
                sc_f1 = rec_sc["per_class_f1"].get(c, np.nan)
                lf_f1 = rec_lf["per_class_f1"].get(c, np.nan)
                if np.isnan(sup) or np.isnan(sc_f1) or np.isnan(lf_f1):
                    continue
                x_vals.append(sup)
                y_vals.append((sc_f1 - lf_f1) * 100)

            if not x_vals:
                continue

            color = {"Llama": "#4361ee", "Mistral": "#7209b7",
                     "Qwen": "#e07c00"}[fam]
            ax.scatter(x_vals, y_vals, color=color, s=45, alpha=0.75,
                       edgecolors="white", linewidths=0.4,
                       label=fam, zorder=3)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--",
                   alpha=0.7)
        ax.set_xlabel("Class support (test-set count)")
        ax.set_ylabel("Small CICLe − Large few-shot  (pp)")
        ax.legend(framealpha=0.9)
        ax.set_title(
            f"{DATASET_LABELS[ds]}\n"
            f"Per-class F1 delta vs. class frequency  "
            f"(each point = one class, coloured by family)",
            fontweight="bold",
        )

        fname = f"I5_{_safe_fname(ds)}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  I5 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS, target_models

    target_models = set(ALL_MODELS)

    print("Loading Group I data …")
    RECORDS = load_records()
    print(f"  Records loaded: {len(RECORDS)}")

    print("\nGenerating Group I plots …")
    print("I1 — Per-class F1 heatmap  (9 plots)")
    plot_I1()
    print("I2 — Per-class gain bar chart vs. large zero-shot  (9 plots)")
    plot_I2()
    print("I3 — Class-level win rate: small CICLe vs. large few-shot  (3 plots)")
    plot_I3()
    print("I4 — Side-by-side confusion matrices  (9 plots)")
    plot_I4()
    print("I5 — Per-class F1 delta vs. class frequency  (3 plots)")
    plot_I5()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
