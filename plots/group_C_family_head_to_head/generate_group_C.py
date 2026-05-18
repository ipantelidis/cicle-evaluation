"""
Group C — Model Family: Small vs. Large Head-to-Head
====================================================
Compare the small and large models within each family
under matched CICLe and few-shot settings.

Run:
    python generate_group_C.py

Outputs are written to the Group C subdirectories relative to this file.
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
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "/home/v25/ippa6201/cicle-evaluation"

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = ["yahoo-answers", "go-emotions", "semeval-18"]
DATASET_LABELS = {
    "yahoo-answers": "Yahoo Answers (10 classes)",
    "go-emotions": "Go Emotions (28 classes)",
    "semeval-18": "SemEval-18 (20 classes)",
}
DATASET_PREFIX = {
    "yahoo-answers": "yahoo",
    "go-emotions": "go-emotions",
    "semeval-18": "semeval-18",
}

# ── Model families ────────────────────────────────────────────────────────────
FAMILIES = {
    "Llama": {"large": "llama-3.1-8b", "small": "llama-3.2-3b"},
    "Mistral": {"large": "mistral-7b-v0.3", "small": "ministral-3b"},
    "Qwen": {"large": "qwen-2.5-7b", "small": "qwen-2.5-3b"},
}
MODEL_LABELS = {
    "llama-3.1-8b": "Llama 3.1-8B",
    "llama-3.2-3b": "Llama 3.2-3B",
    "mistral-7b-v0.3": "Mistral 7B",
    "ministral-3b": "Ministral 3B",
    "qwen-2.5-7b": "Qwen 2.5-7B",
    "qwen-2.5-3b": "Qwen 2.5-3B",
}
SHOTS = [1, 2, 4, 8]
EMBEDDINGS = ["contriever", "minilm", "tfidf"]
EMB_LABELS = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
VARIANTS = ["pc", "fixed"]

FAMILY_COLORS = {"Llama": "#4361ee", "Mistral": "#7209b7", "Qwen": "#e07c00"}
SIZE_COLORS = {"large": "#264653", "small": "#e76f51"}
METHOD_COLORS = {
    "small_cicle": "#e63946",
    "large_fewshot": "#4361ee",
    "large_zeroshot": "#6c757d",
}

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _f1(payload):
    return payload["classification_report"]["macro avg"]["f1-score"]


# ── Result parsing and loading ────────────────────────────────────────────────
def parse_result_name(dataset, stem):
    """Parse one result filename into metadata used by the plots."""
    pfx = DATASET_PREFIX[dataset]
    if not stem.startswith(f"{pfx}-"):
        return None
    name = stem[len(pfx) + 1:]

    m = re.match(r"^(.+)-zeroshot-2\.0k-samples$", name)
    if m:
        return {
            "llm": m.group(1),
            "method": "zeroshot",
            "embedding": None,
            "variant": None,
            "shots": 0,
            "alpha": None,
            "clf": None,
        }

    m = re.match(
        r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?$",
        name,
    )
    if m:
        return {
            "llm": m.group(1),
            "method": "fewshot",
            "embedding": m.group(2),
            "variant": m.group(4) or "none",
            "shots": int(m.group(3)),
            "alpha": None,
            "clf": None,
        }

    m = re.match(
        r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots(?:-(pc|fixed))?-([\d.]+)-α$",
        name,
    )
    if m:
        return {
            "llm": m.group(1),
            "method": "cicle",
            "embedding": m.group(2),
            "variant": m.group(5) or "none",
            "shots": int(m.group(4)),
            "alpha": float(m.group(6)),
            "clf": m.group(3),
        }

    m = re.match(r"^(tfidf)-(lr|svm)-2\.0k-samples$", name)
    if m:
        return {
            "llm": None,
            "method": "baseline",
            "embedding": m.group(1),
            "variant": None,
            "shots": None,
            "alpha": None,
            "clf": m.group(2),
        }

    return None


def load_records():
    """Load the target-model results into a flat record table."""
    records = []
    target_models = {
        pair["large"] for pair in FAMILIES.values()
    } | {
        pair["small"] for pair in FAMILIES.values()
    }

    for ds in DATASETS:
        pred_dir = f"{BASE_DIR}/{ds}/results/predictions"
        for pred_path in glob.glob(f"{pred_dir}/*.json"):
            stem = os.path.basename(pred_path).replace(".json", "")
            meta = parse_result_name(ds, stem)
            if meta is None:
                continue
            if meta["llm"] is not None and meta["llm"] not in target_models:
                continue

            records.append({
                "dataset": ds,
                "macro_f1": _f1(_load_json(pred_path)),
                **meta,
            })

    return records


# ── Record helpers ────────────────────────────────────────────────────────────
def records_for(dataset, llm=None, method=None):
    return [
        r for r in RECORDS
        if r["dataset"] == dataset
        and (llm is None or r["llm"] == llm)
        and (method is None or r["method"] == method)
    ]


def max_f1(records):
    vals = [r["macro_f1"] for r in records if not np.isnan(r["macro_f1"])]
    return max(vals) if vals else np.nan


# ── Plot-specific reductions ─────────────────────────────────────────────────
def zeroshot_value(dataset, llm):
    subset = records_for(dataset, llm, "zeroshot")
    return subset[0]["macro_f1"] if subset else np.nan


def best_cicle(dataset, llm):
    return max_f1(records_for(dataset, llm, "cicle"))


def best_fewshot(dataset, llm):
    return max_f1(records_for(dataset, llm, "fewshot"))


def best_cicle_for_embedding(dataset, llm, embedding, shots):
    subset = [
        r for r in records_for(dataset, llm, "cicle")
        if r["embedding"] == embedding and r["shots"] == shots
    ]
    return max_f1(subset)


def best_fewshot_for_embedding(dataset, llm, embedding, shots):
    subset = [
        r for r in records_for(dataset, llm, "fewshot")
        if r["embedding"] == embedding and r["shots"] == shots
    ]
    return max_f1(subset)


def best_cicle_for_emb_var(dataset, llm, embedding, shots, variant):
    subset = [
        r for r in records_for(dataset, llm, "cicle")
        if r["embedding"] == embedding and r["shots"] == shots and r["variant"] == variant
    ]
    return max_f1(subset)


def fewshot_for_emb_var(dataset, llm, embedding, shots, variant):
    subset = [
        r for r in records_for(dataset, llm, "fewshot")
        if r["embedding"] == embedding and r["shots"] == shots and r["variant"] == variant
    ]
    return subset[0]["macro_f1"] if subset else np.nan


# ── Plot plumbing ─────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def dataset_family_pairs():
    for ds in DATASETS:
        for fam in FAMILIES:
            yield ds, fam


def _heatmap(ax, mat, row_labels, col_labels, title):
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
                ax.text(ci, ri, f"{v:+.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=txt_color)

    return im


# ── C1: CICLe — small vs large across embeddings ─────────────────────────────
def plot_C1():
    out = ensure_dir("C1_cicle_small_vs_large")

    for ds, fam in dataset_family_pairs():
        large = FAMILIES[fam]["large"]
        small = FAMILIES[fam]["small"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharey=True)
        axes = axes.flatten()

        for ax, shots in zip(axes, SHOTS):
            large_vals = [best_cicle_for_embedding(ds, large, emb, shots) for emb in EMBEDDINGS]
            small_vals = [best_cicle_for_embedding(ds, small, emb, shots) for emb in EMBEDDINGS]
            x = np.arange(len(EMBEDDINGS))
            w = 0.36

            ax.bar(x - w/2, large_vals, w, color=SIZE_COLORS["large"], label=MODEL_LABELS[large])
            ax.bar(x + w/2, small_vals, w, color=SIZE_COLORS["small"], label=MODEL_LABELS[small])
            ax.set_xticks(x)
            ax.set_xticklabels([EMB_LABELS[e] for e in EMBEDDINGS])
            ax.set_title(f"{shots} shot{'s' if shots > 1 else ''}")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        axes[0].set_ylabel("Macro-F1")
        axes[2].set_ylabel("Macro-F1")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"{DATASET_LABELS[ds]}  ·  {fam} family\nCICLe: small vs. large", fontweight="bold")

        fname = f"C1_{ds.replace('-', '_')}_{fam.lower()}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  C1 saved: {fname}")


# ── C2: Few-shot — small vs large across embeddings ──────────────────────────
def plot_C2():
    out = ensure_dir("C2_fewshot_small_vs_large")

    for ds, fam in dataset_family_pairs():
        large = FAMILIES[fam]["large"]
        small = FAMILIES[fam]["small"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharey=True)
        axes = axes.flatten()

        for ax, shots in zip(axes, SHOTS):
            large_vals = [best_fewshot_for_embedding(ds, large, emb, shots) for emb in EMBEDDINGS]
            small_vals = [best_fewshot_for_embedding(ds, small, emb, shots) for emb in EMBEDDINGS]
            x = np.arange(len(EMBEDDINGS))
            w = 0.36

            ax.bar(x - w/2, large_vals, w, color=SIZE_COLORS["large"], label=MODEL_LABELS[large])
            ax.bar(x + w/2, small_vals, w, color=SIZE_COLORS["small"], label=MODEL_LABELS[small])
            ax.set_xticks(x)
            ax.set_xticklabels([EMB_LABELS[e] for e in EMBEDDINGS])
            ax.set_title(f"{shots} shot{'s' if shots > 1 else ''}")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        axes[0].set_ylabel("Macro-F1")
        axes[2].set_ylabel("Macro-F1")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"{DATASET_LABELS[ds]}  ·  {fam} family\nFew-shot: small vs. large", fontweight="bold")

        fname = f"C2_{ds.replace('-', '_')}_{fam.lower()}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  C2 saved: {fname}")


# ── C3: Delta heatmap — small CICLe minus large CICLe ────────────────────────
def plot_C3():
    out = ensure_dir("C3_delta_heatmap_cicle")

    for ds, fam in dataset_family_pairs():
        large = FAMILIES[fam]["large"]
        small = FAMILIES[fam]["small"]
        mat = np.full((len(EMBEDDINGS), len(SHOTS)), np.nan)

        for ri, emb in enumerate(EMBEDDINGS):
            for ci, shots in enumerate(SHOTS):
                s_val = best_cicle_for_embedding(ds, small, emb, shots)
                l_val = best_cicle_for_embedding(ds, large, emb, shots)
                if not np.isnan(s_val) and not np.isnan(l_val):
                    mat[ri, ci] = (s_val - l_val) * 100

        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        im = _heatmap(
            ax,
            mat,
            [EMB_LABELS[e] for e in EMBEDDINGS],
            [str(s) for s in SHOTS],
            f"{DATASET_LABELS[ds]}  ·  {fam} family\nSmall CICLe minus Large CICLe (pp)",
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.92)
        cbar.set_label("ΔMacro-F1 (pp)", fontsize=9)
        ax.set_xlabel("Shots")

        fname = f"C3_{ds.replace('-', '_')}_{fam.lower()}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  C3 saved: {fname}")


# ── C4: Delta heatmap — small CICLe minus large few-shot ─────────────────────
def plot_C4():
    out = ensure_dir("C4_delta_heatmap_small_cicle_vs_large_fewshot")

    for ds, fam in dataset_family_pairs():
        large = FAMILIES[fam]["large"]
        small = FAMILIES[fam]["small"]
        mat = np.full((len(EMBEDDINGS), len(SHOTS)), np.nan)

        for ri, emb in enumerate(EMBEDDINGS):
            for ci, shots in enumerate(SHOTS):
                s_val = best_cicle_for_embedding(ds, small, emb, shots)
                l_val = best_fewshot_for_embedding(ds, large, emb, shots)
                if not np.isnan(s_val) and not np.isnan(l_val):
                    mat[ri, ci] = (s_val - l_val) * 100

        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        im = _heatmap(
            ax,
            mat,
            [EMB_LABELS[e] for e in EMBEDDINGS],
            [str(s) for s in SHOTS],
            f"{DATASET_LABELS[ds]}  ·  {fam} family\nSmall CICLe minus Large few-shot (pp)",
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.92)
        cbar.set_label("ΔMacro-F1 (pp)", fontsize=9)
        ax.set_xlabel("Shots")

        fname = f"C4_{ds.replace('-', '_')}_{fam.lower()}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  C4 saved: {fname}")


# ── C5: Distribution of deltas — small CICLe minus large few-shot ────────────
def plot_C5():
    out = ensure_dir("C5_delta_distribution_small_cicle_vs_large_fewshot")

    for ds, fam in dataset_family_pairs():
        large = FAMILIES[fam]["large"]
        small = FAMILIES[fam]["small"]
        deltas = []

        for shots in SHOTS:
            for emb in EMBEDDINGS:
                for variant in VARIANTS:
                    s_val = best_cicle_for_emb_var(ds, small, emb, shots, variant)
                    l_val = fewshot_for_emb_var(ds, large, emb, shots, variant)
                    if not np.isnan(s_val) and not np.isnan(l_val):
                        deltas.append((s_val - l_val) * 100)

        fig, ax = plt.subplots(figsize=(6.6, 4.8))
        if deltas:
            ax.hist(deltas, bins=12, color=FAMILY_COLORS[fam], alpha=0.78, edgecolor="white")
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.4)
        ax.set_xlabel("Small CICLe minus Large few-shot (pp)")
        ax.set_ylabel("Count")
        ax.set_title(f"{DATASET_LABELS[ds]}  ·  {fam} family\nDelta distribution across matched configs",
                     fontweight="bold")

        fname = f"C5_{ds.replace('-', '_')}_{fam.lower()}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  C5 saved: {fname}")


# ── C6: All families side by side ─────────────────────────────────────────────
def plot_C6():
    out = ensure_dir("C6_all_families_side_by_side")

    for ds in DATASETS:
        fam_names = list(FAMILIES.keys())
        small_cicle_vals = []
        large_fewshot_vals = []
        large_zeroshot_vals = []

        for fam in fam_names:
            large = FAMILIES[fam]["large"]
            small = FAMILIES[fam]["small"]
            small_cicle_vals.append(best_cicle(ds, small))
            large_fewshot_vals.append(best_fewshot(ds, large))
            large_zeroshot_vals.append(zeroshot_value(ds, large))

        x = np.arange(len(fam_names))
        w = 0.24
        fig, ax = plt.subplots(figsize=(7.6, 4.8))

        bars1 = ax.bar(x - w, small_cicle_vals, w, color=METHOD_COLORS["small_cicle"], label="Small CICLe (best)")
        bars2 = ax.bar(x, large_fewshot_vals, w, color=METHOD_COLORS["large_fewshot"], label="Large few-shot (best)")
        bars3 = ax.bar(x + w, large_zeroshot_vals, w, color=METHOD_COLORS["large_zeroshot"], label="Large zero-shot")

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.004, f"{h:.3f}",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(fam_names)
        ax.set_ylabel("Macro-F1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"{DATASET_LABELS[ds]}\nAll families side by side", fontweight="bold")
        ax.legend(loc="lower right", framealpha=0.9)

        fname = f"C6_{ds.replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  C6 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group C data …")
    RECORDS = load_records()
    print(f"  Records loaded: {len(RECORDS)}")

    print("\nGenerating Group C plots …")
    print("C1 — CICLe: small vs large across embeddings")
    plot_C1()
    print("C2 — Few-shot: small vs large across embeddings")
    plot_C2()
    print("C3 — Delta heatmap: small CICLe minus large CICLe")
    plot_C3()
    print("C4 — Delta heatmap: small CICLe minus large few-shot")
    plot_C4()
    print("C5 — Distribution of deltas: small CICLe minus large few-shot")
    plot_C5()
    print("C6 — All families side by side")
    plot_C6()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
