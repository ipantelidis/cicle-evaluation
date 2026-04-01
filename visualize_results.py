import json
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR  = "/home/v25/ippa6201/cicle-evaluation"
OUT_DIR   = f"{BASE_DIR}/plots"
DATASETS  = ["yahoo-answers", "go-emotions", "semeval-18"]
DATASET_LABELS = {
    "yahoo-answers": "Yahoo Answers\n(10 classes)",
    "go-emotions":   "Go Emotions\n(28 classes)",
    "semeval-18":    "SemEval-18\n(20 classes)",
}
# Some datasets use a shorter prefix in their filenames
DATASET_PREFIX = {
    "yahoo-answers": "yahoo",
    "go-emotions":   "go-emotions",
    "semeval-18":    "semeval-18",
}
LLM_LABELS = {
    "llama-3.1-8b":    "Llama 3.1-8B",
    "mistral-7b-v0.3": "Mistral 7B",
    "qwen-2.5-7b":     "Qwen 2.5-7B",
}
EMB_LABELS = {
    "contriever": "Contriever",
    "minilm":     "MiniLM",
    "tfidf":      "TF-IDF",
}
CLF_LABELS = {"lr": "LR", "svm": "SVM"}

PALETTE_METHOD = {
    "Baseline":      "#6c757d",
    "Few-shot":      "#0077b6",
    "CICLe":         "#e63946",
}
PALETTE_EMB = {
    "Contriever": "#2a9d8f",
    "MiniLM":     "#e9c46a",
    "TF-IDF":     "#e76f51",
}
PALETTE_LLM = {
    "Llama 3.1-8B": "#4361ee",
    "Mistral 7B":   "#7209b7",
    "Qwen 2.5-7B":  "#f72585",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────
def parse_filename(stem, dataset):
    """Return a dict of parsed fields from a result filename stem."""
    prefix = DATASET_PREFIX[dataset] + "-"
    if not stem.startswith(prefix):
        return None
    name = stem[len(prefix):]  # strip dataset prefix

    # Baseline: tfidf-lr or tfidf-svm
    m = re.match(r"^(tfidf)-(lr|svm)-2\.0k-samples$", name)
    if m:
        return dict(method="baseline", embedding=m.group(1), classifier=m.group(2),
                    llm=None, shots=None, alpha=None)

    # Fewshot: {llm}-fewshot-{emb}-2.0k-samples-{shots}-shots
    m = re.match(r"^(.+)-fewshot-(\w+)-2\.0k-samples-(\d+)-shots$", name)
    if m:
        return dict(method="fewshot", llm=m.group(1), embedding=m.group(2),
                    shots=int(m.group(3)), alpha=None, classifier=None)

    # CICLe: {llm}-cicle-{emb}-{clf}-2.0k-samples-{shots}-shots-{alpha}-α
    m = re.match(r"^(.+)-cicle-(\w+)-(lr|svm)-2\.0k-samples-(\d+)-shots-([\d.]+)-α$", name)
    if m:
        return dict(method="cicle", llm=m.group(1), embedding=m.group(2),
                    classifier=m.group(3), shots=int(m.group(4)),
                    alpha=float(m.group(5)))

    return None


def load_all():
    records = []
    for dataset in DATASETS:
        pred_dir = f"{BASE_DIR}/{dataset}/results/predictions"
        for path in glob.glob(f"{pred_dir}/*.json"):
            stem = os.path.basename(path).replace(".json", "")
            meta = parse_filename(stem, dataset)
            if meta is None:
                print(f"  [warn] could not parse: {stem}")
                continue
            with open(path) as f:
                d = json.load(f)
            meta["dataset"]  = dataset
            meta["accuracy"] = d["accuracy"]
            meta["macro_f1"] = d["classification_report"]["macro avg"]["f1-score"]
            records.append(meta)
    return records


records = load_all()
print(f"Loaded {len(records)} result files")

# Helper: filter records
def R(method=None, dataset=None, llm=None, embedding=None,
      classifier=None, shots=None, alpha=None):
    out = records
    if method    is not None: out = [r for r in out if r["method"]    == method]
    if dataset   is not None: out = [r for r in out if r["dataset"]   == dataset]
    if llm       is not None: out = [r for r in out if r["llm"]       == llm]
    if embedding is not None: out = [r for r in out if r["embedding"] == embedding]
    if classifier is not None: out = [r for r in out if r["classifier"] == classifier]
    if shots     is not None: out = [r for r in out if r["shots"]     == shots]
    if alpha     is not None: out = [r for r in out if r["alpha"]     == alpha]
    return out

def best(recs, metric="macro_f1"):
    return max(recs, key=lambda r: r[metric]) if recs else None


# ── PLOT 1: Overview — best macro-F1 per method type per dataset ──────────────
def plot1_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    fig.suptitle("Best Macro-F1 per Method Type across Datasets", fontsize=14, fontweight="bold")

    for ax, dataset in zip(axes, DATASETS):
        groups = {
            "Baseline":  R(method="baseline", dataset=dataset),
            "Few-shot":  R(method="fewshot",  dataset=dataset),
            "CICLe":     R(method="cicle",    dataset=dataset),
        }
        names, vals, colors = [], [], []
        for label, recs in groups.items():
            b = best(recs)
            if b:
                names.append(label)
                vals.append(b["macro_f1"])
                colors.append(PALETTE_METHOD[label])

        bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10, fontweight="bold")
        ax.set_title(DATASET_LABELS[dataset], fontsize=11)
        ax.set_ylim(0, min(1.0, max(vals) * 1.25))
        ax.set_ylabel("Macro-F1" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/1_overview_best_macrof1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot 1")


# ── PLOT 2: Macro-F1 vs #shots (fewshot & cicle, α=0.05, averaged over emb/clf) ─
def plot2_shots():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Macro-F1 vs. Number of Shots (α=0.05 for CICLe)", fontsize=14, fontweight="bold")
    shot_vals = [1, 2, 4, 8]

    for ax, dataset in zip(axes, DATASETS):
        for llm_key, llm_label in LLM_LABELS.items():
            # fewshot: average over embeddings
            fs_vals = []
            for s in shot_vals:
                recs = R(method="fewshot", dataset=dataset, llm=llm_key, shots=s)
                fs_vals.append(np.mean([r["macro_f1"] for r in recs]) if recs else np.nan)
            ax.plot(shot_vals, fs_vals, marker="o", linestyle="--",
                    label=f"Few-shot {llm_label}", alpha=0.8)

            # cicle: alpha=0.05, average over embeddings and classifiers
            cl_vals = []
            for s in shot_vals:
                recs = R(method="cicle", dataset=dataset, llm=llm_key, shots=s, alpha=0.05)
                cl_vals.append(np.mean([r["macro_f1"] for r in recs]) if recs else np.nan)
            ax.plot(shot_vals, cl_vals, marker="s", linestyle="-",
                    label=f"CICLe {llm_label}", alpha=0.8)

        # baseline reference lines
        for clf, style in [("lr", ":"), ("svm", "-.")]:
            recs = R(method="baseline", dataset=dataset, classifier=clf)
            if recs:
                ax.axhline(recs[0]["macro_f1"], linestyle=style, color="gray",
                           alpha=0.6, label=f"Baseline {clf.upper()}")

        ax.set_title(DATASET_LABELS[dataset], fontsize=11)
        ax.set_xlabel("Shots")
        ax.set_ylabel("Macro-F1" if ax == axes[0] else "")
        ax.set_xticks(shot_vals)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    # single legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/2_macrof1_vs_shots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot 2")


# ── PLOT 3: Embedding comparison (best per embedding, averaged over LLMs) ─────
def plot3_embeddings():
    methods = [("fewshot", "Few-shot"), ("cicle", "CICLe")]
    embeddings = ["contriever", "minilm", "tfidf"]
    x = np.arange(len(DATASETS))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Macro-F1 by Embedding Method", fontsize=14, fontweight="bold")

    for ax, (method_key, method_label) in zip(axes, methods):
        for i, emb in enumerate(embeddings):
            vals = []
            for dataset in DATASETS:
                recs = R(method=method_key, dataset=dataset, embedding=emb)
                vals.append(np.mean([r["macro_f1"] for r in recs]) if recs else 0)
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=EMB_LABELS[emb],
                          color=PALETTE_EMB[EMB_LABELS[emb]], edgecolor="white")
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)

        ax.set_title(method_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=9)
        ax.set_ylabel("Macro-F1 (mean over LLMs & configs)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/3_embedding_comparison_macrof1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot 3")


# ── PLOT 4: LLM comparison heatmaps (macro-F1, best config per cell) ──────────
def plot4_llm_heatmap():
    llms = list(LLM_LABELS.keys())
    methods = [("fewshot", "Few-shot"), ("cicle", "CICLe")]

    fig, axes = plt.subplots(len(methods), len(DATASETS),
                             figsize=(14, 7), squeeze=False)
    fig.suptitle("Macro-F1 by LLM and Dataset\n(best configuration per cell)",
                 fontsize=14, fontweight="bold")

    for row, (method_key, method_label) in enumerate(methods):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row][col]
            mat = []
            for llm in llms:
                recs = R(method=method_key, dataset=dataset, llm=llm)
                b = best(recs, metric="macro_f1")
                mat.append([b["macro_f1"] if b else np.nan])

            mat = np.array(mat)
            sns.heatmap(mat, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                        vmin=0, vmax=1, cbar=(col == len(DATASETS) - 1),
                        xticklabels=[method_label if row == 0 else ""],
                        yticklabels=[LLM_LABELS[l] for l in llms] if col == 0 else False)
            if row == 0:
                ax.set_title(DATASET_LABELS[dataset], fontsize=10)
            ax.set_xlabel("")
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/4_llm_heatmap_macrof1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot 4")


# ── PLOT 5: CICLe — effect of alpha (significance level) ──────────────────────
def plot5_alpha():
    alphas = [0.01, 0.05, 0.10, 0.20]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("CICLe: Macro-F1 vs. Significance Level α (2 shots, averaged over LLMs & embeddings)",
                 fontsize=13, fontweight="bold")

    for ax, dataset in zip(axes, DATASETS):
        for clf, style in [("lr", "-"), ("svm", "--")]:
            f1_vals = []
            for a in alphas:
                recs = R(method="cicle", dataset=dataset, classifier=clf,
                         shots=2, alpha=a)
                f1_vals.append(np.mean([r["macro_f1"] for r in recs]) if recs else np.nan)
            ax.plot(alphas, f1_vals, marker="o", linestyle=style,
                    label=f"Classifier: {clf.upper()}")

        ax.set_title(DATASET_LABELS[dataset], fontsize=11)
        ax.set_xlabel("α (significance level)")
        ax.set_ylabel("Macro-F1" if ax == axes[0] else "")
        ax.set_xticks(alphas)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/5_cicle_alpha_macrof1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot 5")


# ── PLOT 6: Full comparison — best per (dataset × method × LLM) grouped bar ───
def plot6_full_comparison():
    llms = list(LLM_LABELS.keys())
    n_datasets = len(DATASETS)
    n_llms = len(llms)
    x = np.arange(n_datasets)
    total_groups = n_llms + 1  # LLMs + baseline
    width = 0.8 / total_groups

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Best Macro-F1: CICLe vs. Few-shot vs. Baseline per LLM",
                 fontsize=14, fontweight="bold")

    for ax, (method_key, method_label) in zip(axes, [("cicle", "CICLe"), ("fewshot", "Few-shot")]):
        for i, llm in enumerate(llms):
            vals = []
            for dataset in DATASETS:
                recs = R(method=method_key, dataset=dataset, llm=llm)
                b = best(recs)
                vals.append(b["macro_f1"] if b else 0)
            offset = (i - (total_groups - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=LLM_LABELS[llm],
                   color=list(PALETTE_LLM.values())[i], edgecolor="white")

        # baseline
        bl_lr, bl_svm = [], []
        for dataset in DATASETS:
            lr  = best(R(method="baseline", dataset=dataset, classifier="lr"))
            svm = best(R(method="baseline", dataset=dataset, classifier="svm"))
            bl_lr.append(lr["macro_f1"]  if lr  else 0)
            bl_svm.append(svm["macro_f1"] if svm else 0)

        offset = (n_llms - (total_groups - 1) / 2) * width
        ax.bar(x + offset, bl_lr,  width * 0.5, label="Baseline LR",  color="#adb5bd", edgecolor="white")
        ax.bar(x + offset + width * 0.5, bl_svm, width * 0.5, label="Baseline SVM", color="#6c757d", edgecolor="white")

        ax.set_title(method_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=10)
        ax.set_ylabel("Macro-F1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=9, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/6_full_comparison_macrof1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot 6")


# ── PLOT 7: Per-dataset detailed heatmap (embedding × shots, best LLM) ────────
def plot7_heatmap_detail():
    shots_list = [1, 2, 4, 8]
    embeddings = ["contriever", "minilm", "tfidf"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Macro-F1 Heatmap: Embedding × Shots\n(best LLM & classifier per cell)",
                 fontsize=13, fontweight="bold")

    for col, dataset in enumerate(DATASETS):
        for row, (method_key, method_label) in enumerate([("cicle", "CICLe"), ("fewshot", "Few-shot")]):
            ax = axes[row][col]
            mat = np.zeros((len(embeddings), len(shots_list)))
            for i, emb in enumerate(embeddings):
                for j, s in enumerate(shots_list):
                    recs = R(method=method_key, dataset=dataset, embedding=emb, shots=s)
                    b = best(recs)
                    mat[i, j] = b["macro_f1"] if b else np.nan

            sns.heatmap(mat, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                        vmin=0.0, vmax=1.0,
                        xticklabels=[f"{s}s" for s in shots_list],
                        yticklabels=[EMB_LABELS[e] for e in embeddings],
                        cbar=col == 2)
            if row == 0:
                ax.set_title(DATASET_LABELS[dataset], fontsize=11)
            ax.set_ylabel(method_label if col == 0 else "")
            ax.set_xlabel("Shots" if row == 1 else "")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/7_heatmap_emb_shots_macrof1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot 7")


# ── Run all plots ─────────────────────────────────────────────────────────────
plot1_overview()
plot2_shots()
plot3_embeddings()
plot4_llm_heatmap()
plot5_alpha()
plot6_full_comparison()
plot7_heatmap_detail()

print(f"\nAll plots saved to: {OUT_DIR}/")
