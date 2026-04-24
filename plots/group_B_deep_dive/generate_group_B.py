"""
Group B — Per-Dataset × Per-Model Deep Dives
============================================
Draft generator for the Group B diagnostic plots.

Run:
    python generate_group_B.py

Outputs are written to the Group B subdirectories relative to this file.
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

# ── Datasets and models ───────────────────────────────────────────────────────
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
TARGET_MODELS = [
    FAMILIES["Llama"]["large"],
    FAMILIES["Llama"]["small"],
    FAMILIES["Mistral"]["large"],
    FAMILIES["Mistral"]["small"],
    FAMILIES["Qwen"]["large"],
    FAMILIES["Qwen"]["small"],
]
MODEL_TO_FAMILY = {
    llm: fam
    for fam, pair in FAMILIES.items()
    for llm in (pair["large"], pair["small"])
}
MODEL_SIZE = {
    pair["large"]: "large"
    for pair in FAMILIES.values()
}
MODEL_SIZE.update({
    pair["small"]: "small"
    for pair in FAMILIES.values()
})

SHOTS = [1, 2, 4, 8]
ALPHAS = [0.01, 0.05, 0.10, 0.20]
EMBEDDINGS = ["contriever", "minilm", "tfidf"]
EMB_LABELS = {"contriever": "Contriever", "minilm": "MiniLM", "tfidf": "TF-IDF"}
VARIANTS = ["pc", "fixed"]
VARIANT_LABELS = {"pc": "PC", "fixed": "Fixed"}
CLASSIFIERS = ["lr", "svm"]
CLF_LABELS = {"lr": "LR", "svm": "SVM"}

METHOD_COLORS = {"cicle": "#e63946", "fewshot": "#4361ee", "zeroshot": "#6c757d"}
EMBED_COLORS = {"contriever": "#4361ee", "minilm": "#2a9d8f", "tfidf": "#e07c00"}
SHOT_COLORS = {1: "#4cc9f0", 2: "#4361ee", 4: "#7209b7", 8: "#f72585"}
FAMILY_COLORS = {"Llama": "#4361ee", "Mistral": "#7209b7", "Qwen": "#e07c00"}

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


def _mean_prompt_length(length_payload):
    prompt_lengths = length_payload.get("prompt_lengths", [])
    if not prompt_lengths:
        return np.nan
    if isinstance(prompt_lengths, dict):
        if "mean" in prompt_lengths:
            return float(prompt_lengths["mean"])
        prompt_lengths = prompt_lengths.get("values", [])
    if not prompt_lengths:
        return np.nan
    return float(np.mean(prompt_lengths))


# ── Result parsing and loading ────────────────────────────────────────────────
def parse_result_name(dataset, stem):
    """Parse one result filename into plot-friendly metadata."""
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

    return None


def load_records():
    """
    Load predictions and prompt lengths into a flat record table.

    The draft plots work from records instead of nested dicts because Group B
    slices the data many different ways.
    """
    records = []
    for ds in DATASETS:
        pred_dir = f"{BASE_DIR}/{ds}/results/predictions"
        len_dir = f"{BASE_DIR}/{ds}/results/lengths"
        for pred_path in glob.glob(f"{pred_dir}/*.json"):
            stem = os.path.basename(pred_path).replace(".json", "")
            meta = parse_result_name(ds, stem)
            if meta is None or meta["llm"] not in TARGET_MODELS:
                continue

            length_path = os.path.join(len_dir, os.path.basename(pred_path))
            prompt_mean = np.nan
            if os.path.exists(length_path):
                prompt_mean = _mean_prompt_length(_load_json(length_path))

            records.append({
                "dataset": ds,
                "family": MODEL_TO_FAMILY[meta["llm"]],
                "size": MODEL_SIZE[meta["llm"]],
                "prompt_mean": prompt_mean,
                "macro_f1": _f1(_load_json(pred_path)),
                **meta,
            })
    return records


# ── Record helpers ────────────────────────────────────────────────────────────
def records_for(dataset, llm, method=None):
    return [
        r for r in RECORDS
        if r["dataset"] == dataset and r["llm"] == llm and (method is None or r["method"] == method)
    ]


def max_f1(records):
    vals = [r["macro_f1"] for r in records if not np.isnan(r["macro_f1"])]
    return max(vals) if vals else np.nan


def mean_f1(records):
    vals = [r["macro_f1"] for r in records if not np.isnan(r["macro_f1"])]
    return float(np.mean(vals)) if vals else np.nan


# ── Plot-specific reductions ─────────────────────────────────────────────────
def best_cicle_for_embedding(dataset, llm, embedding, shots):
    """Best CICLe score for one embedding at one shots value."""
    subset = [
        r for r in records_for(dataset, llm, "cicle")
        if r["embedding"] == embedding and r["shots"] == shots
    ]
    return max_f1(subset)


def best_fewshot_for_embedding(dataset, llm, embedding, shots):
    """Best few-shot score for one embedding at one shots value."""
    subset = [
        r for r in records_for(dataset, llm, "fewshot")
        if r["embedding"] == embedding and r["shots"] == shots
    ]
    return max_f1(subset)


def best_cicle_for_alpha(dataset, llm, embedding, shots, alpha):
    """Best CICLe score at a fixed alpha, optimizing over variant and classifier."""
    subset = [
        r for r in records_for(dataset, llm, "cicle")
        if r["embedding"] == embedding and r["shots"] == shots and r["alpha"] == alpha
    ]
    return max_f1(subset)


def best_cicle_for_clf(dataset, llm, embedding, shots, clf):
    """Best CICLe score for one classifier, optimizing over alpha and variant."""
    subset = [
        r for r in records_for(dataset, llm, "cicle")
        if r["embedding"] == embedding and r["shots"] == shots and r["clf"] == clf
    ]
    return max_f1(subset)


def best_cicle_for_variant(dataset, llm, embedding, shots, variant):
    """Best CICLe score for one variant, optimizing over alpha and classifier."""
    subset = [
        r for r in records_for(dataset, llm, "cicle")
        if r["embedding"] == embedding and r["shots"] == shots and r["variant"] == variant
    ]
    return max_f1(subset)


def prompt_length_records(dataset, llm):
    """All model-specific records with a usable prompt length."""
    return [
        r for r in records_for(dataset, llm)
        if not np.isnan(r["prompt_mean"])
    ]


# ── Plot plumbing ─────────────────────────────────────────────────────────────
def ensure_dir(name):
    path = os.path.join(HERE, name)
    os.makedirs(path, exist_ok=True)
    return path


def model_pairs():
    for ds in DATASETS:
        for llm in TARGET_MODELS:
            yield ds, llm


# ── B1: Macro-F1 vs shots, per embedding ─────────────────────────────────────
def plot_B1():
    out = ensure_dir("B1_macro_f1_vs_shots_per_embedding")

    for ds, llm in model_pairs():
        fig, ax = plt.subplots(figsize=(7, 4.8))

        for emb in EMBEDDINGS:
            cicle_vals = [best_cicle_for_embedding(ds, llm, emb, s) for s in SHOTS]
            fewshot_vals = [best_fewshot_for_embedding(ds, llm, emb, s) for s in SHOTS]

            if not np.all(np.isnan(cicle_vals)):
                ax.plot(
                    SHOTS,
                    cicle_vals,
                    color=EMBED_COLORS[emb],
                    marker="o",
                    linewidth=2.0,
                    label=f"CICLe - {EMB_LABELS[emb]}",
                )
            if not np.all(np.isnan(fewshot_vals)):
                ax.plot(
                    SHOTS,
                    fewshot_vals,
                    color=EMBED_COLORS[emb],
                    marker="o",
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.75,
                    label=f"Few-shot - {EMB_LABELS[emb]}",
                )

        ax.set_xticks(SHOTS)
        ax.set_xlabel("Number of shots")
        ax.set_ylabel("Macro-F1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}", fontweight="bold")
        ax.legend(loc="best", ncol=2, framealpha=0.9)

        fname = f"B1_{ds.replace('-', '_')}_{llm.replace('.', '_').replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  B1 saved: {fname}")


# ── B2: CICLe alpha sensitivity at 2 shots ───────────────────────────────────
def plot_B2():
    out = ensure_dir("B2_cicle_alpha_sensitivity")

    for ds, llm in model_pairs():
        shots = 2
        fig, ax = plt.subplots(figsize=(7, 4.8))

        for emb in EMBEDDINGS:
            vals = [best_cicle_for_alpha(ds, llm, emb, shots, alpha) for alpha in ALPHAS]
            if not np.all(np.isnan(vals)):
                ax.plot(
                    ALPHAS,
                    vals,
                    color=EMBED_COLORS[emb],
                    marker="o",
                    linewidth=2.0,
                    label=EMB_LABELS[emb],
                )

        ax.set_xlabel("Alpha")
        ax.set_ylabel("Macro-F1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(
            f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}\nAlpha sensitivity at 2 shots",
            fontweight="bold",
        )
        ax.legend(loc="lower right", ncol=1, framealpha=0.9)

        fname = f"B2_{ds.replace('-', '_')}_{llm.replace('.', '_').replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  B2 saved: {fname}")


# ── B3: LR vs SVM by embedding and shots ─────────────────────────────────────
def plot_B3():
    out = ensure_dir("B3_lr_vs_svm_per_embedding")

    for ds, llm in model_pairs():
        labels = [f"{EMB_LABELS[emb]}\n{s}s" for s in SHOTS for emb in EMBEDDINGS]
        lr_vals, svm_vals = [], []

        for s in SHOTS:
            for emb in EMBEDDINGS:
                lr_vals.append(best_cicle_for_clf(ds, llm, emb, s, "lr"))
                svm_vals.append(best_cicle_for_clf(ds, llm, emb, s, "svm"))

        x = np.arange(len(labels))
        w = 0.38
        fig, ax = plt.subplots(figsize=(11, 4.8))

        ax.bar(x - w / 2, lr_vals, w, color="#4cc9f0", label="LR")
        ax.bar(x + w / 2, svm_vals, w, color="#7209b7", label="SVM")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Macro-F1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}", fontweight="bold")
        ax.legend(framealpha=0.9)

        fname = f"B3_{ds.replace('-', '_')}_{llm.replace('.', '_').replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  B3 saved: {fname}")


# ── B4: PC vs Fixed variant comparison ───────────────────────────────────────
def plot_B4():
    out = ensure_dir("B4_pc_vs_fixed_variant")

    for ds, llm in model_pairs():
        rows = []
        for s in SHOTS:
            for emb in EMBEDDINGS:
                pc = best_cicle_for_variant(ds, llm, emb, s, "pc")
                fixed = best_cicle_for_variant(ds, llm, emb, s, "fixed")
                delta = fixed - pc if not np.isnan(pc) and not np.isnan(fixed) else -np.inf
                rows.append((f"{EMB_LABELS[emb]} · {s}s", pc, fixed, delta))

        rows.sort(key=lambda item: item[3], reverse=True)
        labels = [r[0] for r in rows]
        pc_vals = [r[1] for r in rows]
        fixed_vals = [r[2] for r in rows]

        x = np.arange(len(labels))
        w = 0.38
        fig, ax = plt.subplots(figsize=(10.5, 4.8))

        ax.bar(x - w / 2, pc_vals, w, color="#4361ee", label="PC")
        ax.bar(x + w / 2, fixed_vals, w, color="#f72585", label="Fixed")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Macro-F1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}", fontweight="bold")
        ax.legend(framealpha=0.9)

        fname = f"B4_{ds.replace('-', '_')}_{llm.replace('.', '_').replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  B4 saved: {fname}")


# ── B5: Radar chart — CICLe vs few-shot ──────────────────────────────────────
def plot_B5():
    out = ensure_dir("B5_radar_cicle_vs_fewshot")

    for ds, llm in model_pairs():
        axes_labels = [f"{EMB_LABELS[emb]}\n{s}s" for s in SHOTS for emb in EMBEDDINGS]
        cicle_vals = []
        fewshot_vals = []
        for s in SHOTS:
            for emb in EMBEDDINGS:
                cicle_vals.append(best_cicle_for_embedding(ds, llm, emb, s))
                fewshot_vals.append(best_fewshot_for_embedding(ds, llm, emb, s))

        if np.all(np.isnan(cicle_vals)) and np.all(np.isnan(fewshot_vals)):
            continue

        num_axes = len(axes_labels)
        angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
        angles += angles[:1]
        cicle_plot = [0 if np.isnan(v) else v for v in cicle_vals] + [0 if np.isnan(cicle_vals[0]) else cicle_vals[0]]
        fewshot_plot = [0 if np.isnan(v) else v for v in fewshot_vals] + [0 if np.isnan(fewshot_vals[0]) else fewshot_vals[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.plot(angles, cicle_plot, color="#e63946", linewidth=2.0, label="CICLe")
        ax.fill(angles, cicle_plot, color="#e63946", alpha=0.14)
        ax.plot(angles, fewshot_plot, color="#4361ee", linewidth=2.0, label="Few-shot")
        ax.fill(angles, fewshot_plot, color="#4361ee", alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes_labels, fontsize=8)
        max_val = np.nanmax([v for v in cicle_vals + fewshot_vals if not np.isnan(v)]) if not (
            np.all(np.isnan(cicle_vals)) and np.all(np.isnan(fewshot_vals))
        ) else 0.1
        tick_top = max(0.1, np.ceil(max_val * 20) / 20)
        rticks = np.linspace(tick_top / 4, tick_top, 4)
        ax.set_ylim(0, tick_top)
        ax.set_yticks(rticks)
        ax.set_yticklabels([f"{t:.0%}" for t in rticks], fontsize=8, color="#666666")
        ax.set_rlabel_position(18)
        ax.yaxis.grid(True, color="#cfcfcf", alpha=0.6, linestyle="--")
        ax.xaxis.grid(True, color="#d9d9d9", alpha=0.5, linestyle=":")
        ax.spines["polar"].set_color("#bbbbbb")
        ax.spines["polar"].set_linewidth(1.0)
        ax.set_title(f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}", fontweight="bold", pad=24)
        ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), framealpha=0.9)

        fname = f"B5_{ds.replace('-', '_')}_{llm.replace('.', '_').replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  B5 saved: {fname}")


# ── B6: Macro-F1 vs prompt length ────────────────────────────────────────────
def plot_B6():
    out = ensure_dir("B6_macro_f1_vs_prompt_length")

    size_map = {0: 50, 1: 55, 2: 65, 4: 80, 8: 95}

    for ds, llm in model_pairs():
        recs = prompt_length_records(ds, llm)
        if not recs:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.8))
        for method in ["zeroshot", "fewshot", "cicle"]:
            subset = [r for r in recs if r["method"] == method]
            if not subset:
                continue
            ax.scatter(
                [r["prompt_mean"] for r in subset],
                [r["macro_f1"] for r in subset],
                color=METHOD_COLORS[method],
                s=[size_map.get(r["shots"], 55) for r in subset],
                alpha=0.72,
                edgecolors="white",
                linewidths=0.4,
                label=method.capitalize(),
            )

        ax.set_xlabel("Mean prompt length")
        ax.set_ylabel("Macro-F1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"{DATASET_LABELS[ds]}  ·  {MODEL_LABELS[llm]}", fontweight="bold")
        ax.legend(framealpha=0.9)

        fname = f"B6_{ds.replace('-', '_')}_{llm.replace('.', '_').replace('-', '_')}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  B6 saved: {fname}")


# ── Script entrypoint ─────────────────────────────────────────────────────────
def count_saved_plots():
    return sum(
        len([f for f in os.listdir(os.path.join(HERE, d)) if f.endswith(".png")])
        for d in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, d))
    )


def main():
    global RECORDS

    print("Loading Group B data …")
    RECORDS = load_records()
    print(f"  Records loaded: {len(RECORDS)}")

    print("\nGenerating Group B plots …")
    print("B1 — Macro-F1 vs shots, per embedding")
    plot_B1()
    print("B2 — CICLe alpha sensitivity at 2 shots")
    plot_B2()
    print("B3 — LR vs SVM per embedding per shots")
    plot_B3()
    print("B4 — PC vs Fixed variant comparison")
    plot_B4()
    print("B5 — Radar chart: CICLe vs few-shot")
    plot_B5()
    print("B6 — Macro-F1 vs prompt length")
    plot_B6()

    print(f"\nDone. {count_saved_plots()} plots saved under {HERE}/")


if __name__ == "__main__":
    main()
