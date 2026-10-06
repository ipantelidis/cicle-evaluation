#!/usr/bin/env python3
"""
Builds every figure and table of the paper from the per-instance result files
(results/<variant>/seed-<n>/*.json), following paper/plan/figures_and_tables.md.

Figures are written as PDF and tables as LaTeX fragments (booktabs) into
paper/figures/out/. Values that depend on runs that have not finished are left
out and reported as "pending" on stdout, so the script can be re-run as results
land.

Usage:
    .venv/bin/python paper/figures/make_figures.py                 # everything, 1000 bootstraps
    .venv/bin/python paper/figures/make_figures.py --bootstrap 5000
    .venv/bin/python paper/figures/make_figures.py --only fig1,tab2
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import analyze  # noqa: E402  (loader, macro_f1)

RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures", "out")
SMALL = ["llama-3.2-3b", "ministral-3b", "qwen-2.5-3b", "mistral-7b-v0.3", "qwen-2.5-7b", "llama-3.1-8b"]
LARGE = ["mistral-nemo-2407", "qwen-2.5-32b"]
SEEDS = [42, 43, 44]
CORE = ["yahoo-answers", "sst", "semeval-18", "go-emotions", "ohsumed"]
NAME = {"yahoo-answers": "Yahoo Answers", "sst": "SST-5", "semeval-18": "SemEval-18",
        "go-emotions": "GoEmotions", "ohsumed": "Ohsumed"}
MODEL_NAME = {"llama-3.2-3b": "Llama-3.2-3B", "ministral-3b": "Ministral-3B", "qwen-2.5-3b": "Qwen2.5-3B",
              "mistral-7b-v0.3": "Mistral-7B", "qwen-2.5-7b": "Qwen2.5-7B", "llama-3.1-8b": "Llama-3.1-8B",
              "mistral-nemo-2407": "Mistral-Nemo-12B", "qwen-2.5-32b": "Qwen2.5-32B"}
METHOD_NAME = {"zeroshot": "Zero-shot", "fewshot": "Few-shot", "cicle": "CICLe (class-cond. CP)",
               "topk": "Top-$k$", "mass": "Prob. mass", "marginal": "Marginal CP", "oracle": "Oracle"}

# validated categorical palette (dataviz reference instance), fixed slot per method
COLOR = {"fewshot": "#52514e", "cicle": "#2a78d6", "topk": "#eb6834", "mass": "#1baf7a",
         "marginal": "#eda100", "oracle": "#4a3aa7", "zeroshot": "#9a9892"}
REF_COLOR = {"base": "#7a7975", "finetuned": "#0b0b0b"}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "pdf.fonttype": 42, "figure.dpi": 150,
})
TEXTWIDTH, COLWIDTH = 6.3, 3.03  # inches, ACL two-column

_CACHE = {}


def runs(variant_ds):
    """All runs of one dataset variant, keyed as in analyze.load."""
    if variant_ds not in _CACHE:
        _CACHE[variant_ds] = analyze.load(RESULTS, variant_ds)
    return _CACHE[variant_ds]


def key(method, variant=None, k=None, model=None, emb="minilm", clf="lr", alpha=0.05, seed=42):
    if method in ("zeroshot",):
        return (method, None, None, model, None, None, None, seed)
    if method == "fewshot":
        return (method, variant, k, model, emb, None, None, seed)
    if method in ("base",):
        return (method, None, None, "none", emb, clf, None, seed)
    if method == "finetuned":
        return (method, None, None, "none", "roberta-base", None, None, seed)
    return (method, variant, k, model, emb, clf, alpha, seed)


def f1_of(run, index=None):
    g, p = run["gold"], run["pred"]
    if index is not None:
        g, p = g[index], p[index]
    return 100 * analyze.macro_f1(g, p, run["n_classes"])


def mean_metric(variant_ds, method, variant=None, k=None, models=SMALL, metric="macro_f1", **kw):
    """Mean of a stored metric over models x seeds; None if nothing exists."""
    vals = []
    for m in (models if method not in ("base", "finetuned") else ["none"]):
        for s in SEEDS:
            r = runs(variant_ds).get(key(method, variant, k, m, seed=s, **kw))
            if r is not None:
                vals.append(100 * r["metrics"][metric] if metric != "mean_prompt_tokens"
                            else r["metrics"][metric])
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def paired(variant_ds, other, variant, ks, models=SMALL, alpha=0.05, n_boot=1000, rng=None,
           method="cicle", **kw):
    """`method` minus `other`, paired on (model, k, seed); bootstrap over test
    instances with one resample per seed applied to every cell. Returns
    (mean, lo, hi, p, n_pairs) or None."""
    rng = rng or np.random.default_rng(0)
    R = runs(variant_ds)
    pairs = defaultdict(list)
    for m in models:
        for k in ks:
            for s in SEEDS:
                a = R.get(key(method, variant, k, m, alpha=alpha, seed=s, **kw))
                b = R.get(key(other, variant, k, m, alpha=alpha, seed=s, **kw))
                if a is not None and b is not None:
                    pairs[s].append((a, b))
    n = sum(len(v) for v in pairs.values())
    if not n:
        return None

    def delta(index):
        return np.mean([f1_of(a, index[s]) - f1_of(b, index[s])
                        for s, items in pairs.items() for a, b in items])

    sizes = {s: len(items[0][0]["gold"]) for s, items in pairs.items()}
    obs = delta({s: np.arange(z) for s, z in sizes.items()})
    boots = np.array([delta({s: rng.integers(0, z, z) for s, z in sizes.items()}) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return obs, lo, hi, p, n


def mean_ci(variant_ds, method, variant, k, models=SMALL, n_boot=500, rng=None, **kw):
    """Mean macro-F1 over models x seeds with a bootstrap CI over test instances."""
    rng = rng or np.random.default_rng(0)
    R = runs(variant_ds)
    cells = defaultdict(list)
    for m in models:
        for s in SEEDS:
            r = R.get(key(method, variant, k, m, seed=s, **kw))
            if r is not None:
                cells[s].append(r)
    if not cells:
        return None
    sizes = {s: len(v[0]["gold"]) for s, v in cells.items()}
    stat = lambda idx: np.mean([f1_of(r, idx[s]) for s, rs in cells.items() for r in rs])
    obs = stat({s: np.arange(z) for s, z in sizes.items()})
    boots = [stat({s: rng.integers(0, z, z) for s, z in sizes.items()}) for _ in range(n_boot)]
    return obs, *np.percentile(boots, [2.5, 97.5])


def fmt_delta(res, bold_sig=True):
    if res is None:
        return "pending"
    m, lo, hi, p, n = res
    sig = lo > 0 or hi < 0
    body = f"{m:+.2f}"
    if bold_sig and sig:
        body = r"\textbf{" + body + "}"
    return body + (r" \scriptsize{[" + f"{lo:+.2f}, {hi:+.2f}" + "]}" if True else "") + ("" if sig else r" \scriptsize{n.s.}")


def num(v, d=1):
    return "--" if v is None else f"{v:.{d}f}"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=200)  # for proofreading
    plt.close(fig)
    print("  wrote", name + ".pdf")


def write_table(name, body):
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(body)
    print("  wrote", name + ".tex")


# ---------------------------------------------------------------------------
# Table 2: main results at k=4 with pooled deltas
# ---------------------------------------------------------------------------
def tab2(B):
    rows = []
    head = " & ".join(NAME[d] for d in CORE)
    def line(label, method, variant=None, k=None, tokens=True, **kw):
        cells = []
        for d in CORE:
            v, n = mean_metric(d, method, variant, k, **kw)
            t, _ = mean_metric(d, method, variant, k, metric="mean_prompt_tokens", **kw) if tokens and n else (None, 0)
            cell = num(v) + (r" \scriptsize{" + f"{t:,.0f}" + "}" if t else "")
            cells.append(cell)
        return label + " & " + " & ".join(cells) + r" \\"
    rows.append(line("MiniLM + LR (no LLM)", "base", tokens=False))
    rows.append(line("RoBERTa-base, fine-tuned", "finetuned", tokens=False))
    rows.append(r"\midrule")
    rows.append(line("Zero-shot", "zeroshot"))
    rows.append(line("Few-shot, Fixed", "fewshot", "fixed", 4))
    rows.append(line("CICLe, Fixed", "cicle", "fixed", 4))
    rows.append(line("Few-shot, Per-Class", "fewshot", "pc", 4))
    rows.append(line("CICLe, Per-Class", "cicle", "pc", 4))
    rows.append(r"\midrule")
    for variant, label in (("fixed", r"$\Delta$ Fixed (all $k$)"), ("pc", r"$\Delta$ Per-Class (all $k$)")):
        cells = []
        for d in CORE:
            ks = [1, 4] if d == "ohsumed" else [1, 2, 4, 8]
            res = paired(d, "fewshot", variant, ks, n_boot=B)
            cells.append(fmt_delta(res) if res else "--")
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    body = (r"\begin{tabular}{l" + "c" * len(CORE) + "}\n\\toprule\n & " + head + r" \\" + "\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("tab2_main", body)


# ---------------------------------------------------------------------------
# Figure 1: macro-F1 against prompt tokens, one panel per dataset
# ---------------------------------------------------------------------------
def fig1(B):
    fig, axes = plt.subplots(1, len(CORE), figsize=(TEXTWIDTH, 1.9), sharey=False)
    rng = np.random.default_rng(1)
    for ax, d in zip(axes, CORE):
        series = [("zeroshot", None, [None], "o", "none"),
                  ("fewshot", "fixed", [1, 2, 4, 8], "o", "none"), ("cicle", "fixed", [1, 2, 4, 8], "o", "none"),
                  ("fewshot", "pc", [1, 2, 4, 8], "s", "full"), ("cicle", "pc", [1, 2, 4, 8], "s", "full")]
        for method, variant, ks, marker, fill in series:
            xs, ys, los, his = [], [], [], []
            for k in ks:
                r = mean_ci(d, method, variant, k, n_boot=min(B, 300), rng=rng)
                t, _ = mean_metric(d, method, variant, k, metric="mean_prompt_tokens")
                if r is None or t is None:
                    continue
                xs.append(t); ys.append(r[0]); los.append(r[0] - r[1]); his.append(r[2] - r[0])
            if not xs:
                continue
            c = COLOR[method]
            ax.errorbar(xs, ys, yerr=[los, his], color=c, marker=marker, ls="-" if len(xs) > 1 else "none",
                        mfc=c if fill == "full" else "white", mec=c, capsize=1.5, elinewidth=0.7,
                        label=f"{METHOD_NAME[method].split(' (')[0]}" + (f", {variant.replace('pc', 'Per-Class').replace('fixed', 'Fixed')}" if variant else ""))
        for ref, ls in (("base", ":"), ("finetuned", "-.")):
            v, _ = mean_metric(d, ref)
            if v is not None:
                ax.axhline(v, color=REF_COLOR[ref], ls=ls, lw=0.9)
                ax.annotate("LR" if ref == "base" else "RoBERTa", xy=(1, v), xycoords=("axes fraction", "data"),
                            fontsize=6, color=REF_COLOR[ref], ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_xticks([300, 1000, 3000]); ax.set_xticklabels(["300", "1k", "3k"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(NAME[d])
    axes[0].set_ylabel("macro-F1 (%)")
    fig.supxlabel("mean prompt tokens per LLM call (log scale)", fontsize=8, y=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    save(fig, "fig1_f1_vs_tokens")


# ---------------------------------------------------------------------------
# Figure 2: how you narrow matters under a long-tailed labelled pool
# ---------------------------------------------------------------------------
def imb_tag(d, imb):
    return d if imb == 1 else f"{d}-imb{imb}"


def fig2(B, variant="pc", name="fig2_narrowing_imbalance"):
    datasets = ["yahoo-answers", "sst"]
    comparators = ["fewshot", "topk", "mass", "marginal"]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.9))
    rng = np.random.default_rng(2)
    for row, d in enumerate(datasets):
        ax = axes[row, 0]
        for comp in comparators:
            ys, los, his, xs = [], [], [], []
            for i, imb in enumerate((1, 10, 100)):
                res = paired(imb_tag(d, imb), comp, variant, [1, 4], n_boot=B, rng=rng)
                if res is None:
                    continue
                xs.append(i); ys.append(res[0]); los.append(res[0] - res[1]); his.append(res[2] - res[0])
            if xs:
                ax.errorbar(xs, ys, yerr=[los, his], color=COLOR[comp], marker="o", capsize=2,
                            label=METHOD_NAME[comp], mfc=COLOR[comp] if variant == "pc" else "white")
        # oracle ceiling: oracle minus CICLe (shown as CICLe minus oracle, i.e. negative headroom)
        ys, xs = [], []
        for i, imb in enumerate((1, 10, 100)):
            res = paired(imb_tag(d, imb), "oracle", variant, [1, 4], n_boot=max(50, B // 10), rng=rng)
            if res is not None:
                xs.append(i); ys.append(res[0])
        if xs:
            ax.plot(xs, ys, color=COLOR["oracle"], ls="--", marker="x", label="Oracle (ceiling)")
        ax.axhline(0, color="#9a9892", lw=0.8)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"1$\times$", r"10$\times$", r"100$\times$"])
        ax.set_ylabel(f"{NAME[d]}\nCICLe minus alternative (pp)")
        if row == 1:
            ax.set_xlabel("imbalance of the labelled pool")
        if row == 0:
            ax.set_title("(a) paired gain of CICLe")

        ax = axes[row, 1]
        for method in ("cicle", "topk", "mass", "marginal"):
            pts = []
            for imb, size in ((1, 18), (10, 32), (100, 50)):
                R = runs(imb_tag(d, imb))
                ms = [r["metrics"] for kk, r in R.items() if kk[0] == method and kk[3] in SMALL]
                if ms:
                    pts.append((np.mean([m["mean_set_size"] for m in ms]), 100 * np.mean([m["coverage"] for m in ms]), size, imb))
            for x, y, size, imb in pts:
                ax.scatter([x], [y], s=size, color=COLOR[method], edgecolor="white", linewidth=0.5, zorder=3,
                           label=METHOD_NAME[method] if imb == 1 else None)
                if method == "cicle":
                    ax.annotate(f"{imb}×", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6)
        ax.axhline(95, color="#9a9892", lw=0.8, ls=":")
        ax.set_ylabel("coverage of the true label (%)")
        if row == 1:
            ax.set_xlabel("mean candidate-set size")
        if row == 0:
            ax.set_title("(b) coverage vs. set size")

        ax = axes[row, 2]
        drawn = rank_coverage_panel(ax, d, 100, variant)
        if row == 0:
            ax.set_title(r"(c) per-class coverage at 100$\times$")
        if row == 1:
            ax.set_xlabel("class rank by frequency in the pool")
        if not drawn:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout()
    save(fig, name)


def rank_coverage_panel(ax, d, imb, variant):
    """Per-class coverage by frequency rank, from the per-instance records
    (gold_in_set), aggregated over seeds and models."""
    import json, glob
    try:
        from experiment import Data
    except Exception:
        return False
    cov = defaultdict(lambda: defaultdict(list))
    for s in SEEDS:
        try:
            data = Data(d, s, imbalance=imb)
        except Exception as e:
            print("  rank panel skipped:", str(e)[:80])
            return False
        order = [c for c, _ in Counter(data.y_train).most_common()]
        rank = {c: i for i, c in enumerate(order)}
        for method in ("cicle", "topk", "mass", "marginal"):
            # the candidate sets do not depend on the LLM, so one model is enough
            for path in glob.glob(os.path.join(RESULTS, imb_tag(d, imb), f"seed-{s}",
                                               f"*-{SMALL[-1]}-{method}-minilm-lr-4-shots-{variant}-0.05-alpha.json")):
                recs = json.load(open(path))["records"]
                by_class = defaultdict(list)
                for r in recs:
                    by_class[r["gold"]].append(r["gold_in_set"])
                for c, v in by_class.items():
                    cov[method][rank[c]].append(100 * np.mean(v))
    if not cov:
        return False
    for method, ranks in cov.items():
        xs = sorted(ranks)
        ax.plot(xs, [np.mean(ranks[x]) for x in xs], color=COLOR[method], marker="o", label=METHOD_NAME[method])
    ax.axhline(95, color="#9a9892", lw=0.8, ls=":")
    ax.set_ylabel("coverage (%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x + 1) for x in xs])
    return True


# ---------------------------------------------------------------------------
# Table 3: label renaming
# ---------------------------------------------------------------------------
def tab3(B):
    rows = []
    for d in ("yahoo-answers", "go-emotions"):
        for tag, label in ((d, "original"), (d + "-relabel", "nonsense words")):
            for k in (1, 4):
                cells = [num(mean_metric(tag, m, v, k)[0]) for m, v in
                         (("fewshot", "fixed"), ("cicle", "fixed"), ("fewshot", "pc"), ("cicle", "pc"))]
                rows.append(f"{NAME[d]} & {label} & {k} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows.append(r"\multicolumn{7}{l}{$\Delta$ CICLe $-$ few-shot with nonsense words (36 pairs)}\\")
    for d in ("yahoo-answers", "go-emotions"):
        cells = [fmt_delta(paired(d + "-relabel", "fewshot", v, [1, 4], n_boot=B)) for v in ("fixed", "pc")]
        rows.append(f"{NAME[d]} & & & \\multicolumn{{2}}{{c}}{{{cells[0]}}} & \\multicolumn{{2}}{{c}}{{{cells[1]}}} \\\\")
    body = ("\\begin{tabular}{llrcccc}\n\\toprule\nDataset & Labels & $k$ & FS Fixed & CICLe Fixed & FS PC & CICLe PC \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("tab3_relabel", body)


# ---------------------------------------------------------------------------
# Table 4: supervised baselines vs LLM pipelines (imbalance and pool size)
# ---------------------------------------------------------------------------
def tab4(B):
    cols = []
    for d in ("yahoo-answers", "sst"):
        for imb in (1, 10, 100):
            cols.append((imb_tag(d, imb), f"{NAME[d]} {imb}" + r"$\times$"))
        for n in (250, 500, 1000):
            cols.append((f"{d}-n{n}", f"{NAME[d]} $n$={n}"))
    for d in ("semeval-18", "go-emotions", "ohsumed"):
        cols.append((d, NAME[d]))
    cols = [(t, l) for t, l in cols if runs(t)]
    rows = []
    def row(label, method, variant=None, k=None):
        vals = [mean_metric(t, method, variant, k)[0] for t, _ in cols]
        return label + " & " + " & ".join(num(v) for v in vals) + r" \\"
    rows.append(row("MiniLM + LR", "base"))
    rows.append(row("RoBERTa-base, fine-tuned", "finetuned"))
    rows.append(row("Zero-shot", "zeroshot"))
    rows.append(row("Few-shot, Per-Class, $k$=4", "fewshot", "pc", 4))
    rows.append(row("CICLe, Per-Class, $k$=4", "cicle", "pc", 4))
    rows.append(row("Few-shot, Fixed, $k$=4", "fewshot", "fixed", 4))
    rows.append(row("CICLe, Fixed, $k$=4", "cicle", "fixed", 4))
    best = []
    for t, _ in cols:
        cand = [(mean_metric(t, m, v, k)[0], f"{'CICLe' if m == 'cicle' else 'FS'} {v[:1].upper()} $k$={k}")
                for m in ("fewshot", "cicle") for v in ("fixed", "pc") for k in (1, 2, 4, 8)]
        cand = [c for c in cand if c[0] is not None]
        best.append(max(cand) if cand else (None, ""))
    rows.append("Best LLM pipeline & " + " & ".join(f"{num(v)} \\scriptsize{{{w}}}" for v, w in best) + r" \\")
    body = ("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n & " + " & ".join(l for _, l in cols)
            + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("tab4_supervised", body)


# ---------------------------------------------------------------------------
# Figure 3: choosing alpha
# ---------------------------------------------------------------------------
def fig3(B):
    models = ["llama-3.1-8b", "llama-3.2-3b"]
    alphas = [0.01, 0.05, 0.10, 0.20]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(COLWIDTH, 3.4), sharex=True)
    rng = np.random.default_rng(3)
    ds_colors = {"yahoo-answers": "#2a78d6", "sst": "#eb6834", "semeval-18": "#1baf7a", "go-emotions": "#4a3aa7"}
    for d in ("yahoo-answers", "sst", "semeval-18", "go-emotions"):
        ys, los, his, skip, xs = [], [], [], [], []
        for a in alphas:
            res_f = paired(d, "fewshot", "fixed", [1, 4], models=models, alpha=a, n_boot=B, rng=rng)
            res_p = paired(d, "fewshot", "pc", [1, 4], models=models, alpha=a, n_boot=B, rng=rng)
            if res_f is None or res_p is None:
                continue
            # pool Fixed and Per-Class cells
            m = (res_f[0] * res_f[4] + res_p[0] * res_p[4]) / (res_f[4] + res_p[4])
            lo = (res_f[1] * res_f[4] + res_p[1] * res_p[4]) / (res_f[4] + res_p[4])
            hi = (res_f[2] * res_f[4] + res_p[2] * res_p[4]) / (res_f[4] + res_p[4])
            xs.append(a); ys.append(m); los.append(m - lo); his.append(hi - m)
            ms = [r["metrics"] for kk, r in runs(d).items() if kk[0] == "cicle" and kk[3] in models and kk[6] == a]
            skip.append(100 * np.mean([1 - x["llm_call_rate"] for x in ms]))
        if not xs:
            continue
        top.errorbar(xs, ys, yerr=[los, his], color=ds_colors[d], marker="o", capsize=2, label=NAME[d])
        bot.plot(xs, skip, color=ds_colors[d], marker="o")
    top.axhline(0, color="#9a9892", lw=0.8)
    top.set_ylabel("CICLe $-$ few-shot (pp)")
    bot.set_ylabel("answered without LLM (%)")
    bot.set_xscale("log"); bot.set_xticks(alphas); bot.set_xticklabels([str(a) for a in alphas])
    bot.set_xlabel(r"miscoverage level $\alpha$")
    top.legend(frameon=False, ncol=2, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig3_alpha")


# ---------------------------------------------------------------------------
# Appendix tables
# ---------------------------------------------------------------------------
def tabC1(B):
    variants = [t for t in ["yahoo-answers", "yahoo-answers-imb10", "yahoo-answers-imb100", "yahoo-answers-relabel",
                            "sst", "sst-imb10", "sst-imb100", "semeval-18", "go-emotions", "go-emotions-relabel", "ohsumed"]
                if runs(t)]
    comps = ["fewshot", "topk", "mass", "marginal", "oracle"]
    rows = []
    for t in variants:
        for v in ("fixed", "pc"):
            cells = []
            for c in comps:
                ks = [1, 2, 4, 8] if c == "fewshot" and "imb" not in t and "relabel" not in t and t != "ohsumed" else [1, 4]
                res = paired(t, c, v, ks, n_boot=B)
                cells.append(fmt_delta(res) if res else "--")
            if all(c == "--" for c in cells):
                continue
            rows.append(f"{t} & {v} & " + " & ".join(cells) + r" \\")
    body = ("\\begin{tabular}{llccccc}\n\\toprule\nVariant & Retrieval & vs few-shot & vs top-$k$ & vs prob. mass & vs marginal CP & vs oracle \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("tabC1_narrowing_all", body)


def tabD1(B):
    cols = [(d, v) for d in CORE for v in (("fixed", "pc") if d != "ohsumed" else ("fixed",))]
    rows = []
    for m in SMALL:
        cells = []
        for d, v in cols:
            ks = [1, 4] if d == "ohsumed" else [1, 2, 4, 8]
            cells.append(fmt_delta(paired(d, "fewshot", v, ks, models=[m], n_boot=B)))
        rows.append(MODEL_NAME[m] + " & " + " & ".join(cells) + r" \\")
    head = " & ".join(f"{NAME[d]} {'F' if v == 'fixed' else 'PC'}" for d, v in cols)
    body = ("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\nModel & " + head + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("tabD1_per_model", body)


def tabF1(B):
    rows = []
    for m in LARGE:
        for d in ("yahoo-answers", "sst", "semeval-18", "go-emotions"):
            zs = mean_metric(d, "zeroshot", models=[m])[0]
            cells = [num(zs)]
            for v in ("fixed", "pc"):
                for k in (1, 4):
                    cells.append(num(mean_metric(d, "fewshot", v, k, models=[m])[0]) + " / " + num(mean_metric(d, "cicle", v, k, models=[m])[0]))
            res = paired(d, "fewshot", "pc", [1, 4], models=[m], n_boot=B)
            cells.append(fmt_delta(res) if res else "--")
            n = mean_metric(d, "zeroshot", models=[m])[1]
            rows.append(f"{MODEL_NAME[m]} & {NAME[d]} & {n} & " + " & ".join(cells) + r" \\")
    body = ("\\begin{tabular}{llrcccccc}\n\\toprule\nModel & Dataset & seeds & Zero-shot & Fixed $k$=1 & Fixed $k$=4 & PC $k$=1 & PC $k$=4 & $\\Delta$ PC \\\\\n"
            "& & & & \\multicolumn{4}{c}{few-shot / CICLe} & \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("tabF1_large_models", body)


def tabG1():
    rows = []
    for t in [x for x in ["yahoo-answers", "sst", "semeval-18", "go-emotions", "ohsumed", "yahoo-answers-imb100", "sst-imb100"] if runs(x)]:
        for v in ("pc", "fixed"):
            R = runs(t)
            fixed = broken = both = neither = out = out_broken = 0
            for m in SMALL:
                for s in SEEDS:
                    a, b = R.get(key("cicle", v, 4, m, seed=s)), R.get(key("fewshot", v, 4, m, seed=s))
                    if a is None or b is None:
                        continue
                    ar, br = a["pred"] == a["gold"], b["pred"] == b["gold"]
                    fixed += np.sum(ar & ~br); broken += np.sum(~ar & br); both += np.sum(ar & br); neither += np.sum(~ar & ~br)
                    gis = a.get("gold_in_set")
                    if gis is not None:
                        out += np.sum(~gis); out_broken += np.sum(~gis & ~ar & br)
            tot = fixed + broken + both + neither
            if not tot:
                continue
            rows.append(f"{t} & {v} & {tot:,} & {100*fixed/tot:.1f} & {100*broken/tot:.1f} & {100*both/tot:.1f} & {100*neither/tot:.1f} & "
                        f"{100*out/tot:.1f} & {100*out_broken/max(out,1):.1f} \\\\")
    body = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\nVariant & Retrieval & pairs & fixed & broken & both right & both wrong & gold outside set & broken among those \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("tabG1_fixes_breaks", body)


def tabG2():
    rows = []
    for m in SMALL + LARGE:
        cells = []
        for d in CORE:
            vals = defaultdict(list)
            for kk, r in runs(d).items():
                if kk[3] == m and kk[0] in ("zeroshot", "fewshot", "cicle"):
                    vals[kk[0]].append(100 * r["metrics"]["invalid_rate"])
            cells.append(" / ".join(f"{np.mean(vals[x]):.1f}" if vals[x] else "--" for x in ("zeroshot", "fewshot", "cicle")))
        rows.append(MODEL_NAME[m] + " & " + " & ".join(cells) + r" \\")
    body = ("\\begin{tabular}{l" + "c" * len(CORE) + "}\n\\toprule\nModel & " + " & ".join(NAME[d] for d in CORE)
            + " \\\\\n & \\multicolumn{" + str(len(CORE)) + "}{c}{invalid outputs (\\%): zero-shot / few-shot / CICLe} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write_table("tabG2_invalid", body)


# ---------------------------------------------------------------------------
STEPS = {"tab2": tab2, "fig1": fig1, "fig2": fig2, "tab3": tab3, "tab4": tab4, "fig3": fig3,
         "tabC1": tabC1, "tabD1": tabD1, "tabF1": tabF1, "tabG1": lambda B: tabG1(), "tabG2": lambda B: tabG2(),
         "figC1": lambda B: fig2(B, variant="fixed", name="figC1_narrowing_imbalance_fixed")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--only", type=lambda s: s.split(","), default=list(STEPS))
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in args.only:
        print(name)
        STEPS[name](args.bootstrap)


if __name__ == "__main__":
    main()
