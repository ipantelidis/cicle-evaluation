#!/usr/bin/env python3
"""
Builds the figures and tables of the paper from the per-instance result files.

    .venv/bin/python paper/figures/make_figures.py --all            # 5000 bootstraps
    .venv/bin/python paper/figures/make_figures.py --all --quick    # 500, for development
    .venv/bin/python paper/figures/make_figures.py --only fig2,tab2
    .venv/bin/python paper/figures/make_figures.py --list

PDFs go to paper/figures/, LaTeX table bodies (tabular only) to
paper/figures/tables/, PNG previews to cache/figures/preview/. Every number a
figure or table produces is also appended to paper/figures/NUMBERS.md.
Missing runs (pending experiments) leave their slot empty and are listed on
stdout as "[pending] ...".

Specification: paper/plan/figures_and_tables.md and figure_scripts_todo.md.
Data layer and statistics: figdata.py (next to this file).
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figdata as fd  # noqa: E402
from figdata import SMALL, LARGE, SEEDS, pending  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")
PREVIEW = os.path.join(fd.CACHE, "preview")
NUMBERS_JSON = os.path.join(fd.CACHE, "numbers.json")
NUMBERS_MD = os.path.join(HERE, "NUMBERS.md")

CORE = ["yahoo-answers", "sst", "semeval-18", "go-emotions", "ohsumed"]
SHORT = ["yahoo-answers", "sst", "semeval-18", "go-emotions"]
ALL_TAGS = ["yahoo-answers", "yahoo-answers-imb10", "yahoo-answers-imb100", "yahoo-answers-relabel",
            "sst", "sst-imb10", "sst-imb100", "semeval-18", "go-emotions", "go-emotions-relabel",
            "ohsumed"]
NAME = {"yahoo-answers": "Yahoo Answers", "sst": "SST-5", "semeval-18": "SemEval-18",
        "go-emotions": "GoEmotions", "ohsumed": "Ohsumed"}
MODEL_NAME = {"llama-3.2-3b": "Llama-3.2-3B", "ministral-3b": "Ministral-3B",
              "qwen-2.5-3b": "Qwen2.5-3B", "mistral-7b-v0.3": "Mistral-7B",
              "qwen-2.5-7b": "Qwen2.5-7B", "llama-3.1-8b": "Llama-3.1-8B",
              "mistral-nemo-2407": "Mistral-Nemo-12B", "qwen-2.5-32b": "Qwen2.5-32B"}
METHOD_NAME = {"zeroshot": "Zero-shot", "fewshot": "Few-shot", "cicle": "CICLe",
               "topk": "Top-$m$", "mass": "Prob. mass", "marginal": "Marginal CP",
               "oracle": "Oracle", "base": "MiniLM + LR", "finetuned": "RoBERTa-base (fine-tuned)"}
METHOD_PLAIN = {"zeroshot": "zero-shot", "fewshot": "few-shot", "cicle": "CICLe", "topk": "top-m",
                "mass": "prob. mass", "marginal": "marginal CP", "oracle": "oracle"}
VARIANT_NAME = {"fixed": "Fixed", "pc": "Per-Class"}
ALPHAS = [0.01, 0.05, 0.10, 0.20]
LLAMAS = ["llama-3.1-8b", "llama-3.2-3b"]


def tag_name(tag):
    """Human name of a dataset variant directory."""
    m = re.match(r"(.+?)(?:-imb(\d+))?(-relabel)?(?:-n(\d+))?$", tag)
    base, imb, relabel, n = m.groups()
    s = NAME.get(base, base)
    if imb:
        s += f" {imb}$\\times$"
    if relabel:
        s += " (renamed)"
    if n:
        s += f" ($n$={n})"
    return s


# ---------------------------------------------------------------------------
# Style: one colour per method everywhere (dataviz reference palette; grey,
# blue, orange and aqua clear the colour-vision checks in every pairing;
# marginal and oracle add a dashed line and distinct marker as a second cue).
# ---------------------------------------------------------------------------
COLOR = {"fewshot": "#6b6a66", "cicle": "#2a78d6", "topk": "#eb6834", "mass": "#1baf7a",
         "marginal": "#4a3aa7", "oracle": "#0b0b0b", "zeroshot": "#0b0b0b",
         "base": "#52514e", "finetuned": "#0b0b0b"}
MARKER = {"fewshot": "o", "cicle": "o", "topk": "s", "mass": "D", "marginal": "^", "oracle": "x",
          "zeroshot": "*"}
LSTYLE = {"fewshot": "-", "cicle": "-", "topk": "-", "mass": "-", "marginal": "--", "oracle": "--"}
DS_COLOR = {"yahoo-answers": "#2a78d6", "sst": "#eb6834", "semeval-18": "#1baf7a",
            "go-emotions": "#4a3aa7"}
DS_MARKER = {"yahoo-answers": "o", "sst": "s", "semeval-18": "D", "go-emotions": "^"}
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"
TEXTWIDTH, COLWIDTH = 6.5, 3.3

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.0,
    "pdf.fonttype": 42, "figure.dpi": 150, "axes.unicode_minus": True,
})
ERR = dict(capsize=1.5, elinewidth=0.7, capthick=0.7)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_NUMBERS = defaultdict(list)  # step -> lines
_TITLE = {}                   # step -> section title


def note(section, line):
    """Record a number for NUMBERS.md under the step being built."""
    step = fd.CURRENT_STEP[0]
    _TITLE.setdefault(step, section)
    _NUMBERS[step].append(line)


def num(v, d=1):
    return "--" if v is None else f"{v:.{d}f}"


def tok(t):
    return "--" if t is None else f"{t:,.0f}"


def sig(res):
    return res is not None and (res["lo"] > 0 or res["hi"] < 0)


def ci_str(res):
    return f"[{res['lo']:+.2f}, {res['hi']:+.2f}]"


def delta_tex(res, ci=True, n=False):
    """Δ cell: bold when the 95% CI excludes zero, 'n.s.' otherwise."""
    if res is None:
        return "--"
    body = f"{res['mean']:+.2f}"
    body = r"\textbf{" + body + "}" if sig(res) else body + r"\,\textsuperscript{n.s.}"
    if ci:
        body += r" {\scriptsize " + ci_str(res) + "}"
    if n:
        body += f" ({res['n']})"
    return body


def delta_txt(res):
    if res is None:
        return "pending"
    return (f"{res['mean']:+.2f} pp {ci_str(res)} p={res['p']:.3f} n={res['n']}"
            + ("" if sig(res) else " n.s."))


def tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def save_fig(fig, name):
    os.makedirs(PREVIEW, exist_ok=True)
    fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(PREVIEW, name + ".png"), bbox_inches="tight", pad_inches=0.05, dpi=220)
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf")


def write_table(name, colspec, header_rows, body_rows, comment=""):
    """tabular only (booktabs rules); no table environment or caption."""
    os.makedirs(TABLES, exist_ok=True)
    lines = []
    if comment:
        lines.append("% " + comment.replace("\n", "\n% "))
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")
    lines.extend(header_rows)
    lines.append(r"\midrule")
    lines.extend(body_rows)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(os.path.join(TABLES, name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote paper/figures/tables/{name}.tex")


def ks_for(tag, variant=None):
    """Planned k grid: {1,2,4,8} on the four main short-text sets, {1,4} elsewhere;
    Ohsumed Per-Class is run at k=1 only (190-word abstracts)."""
    if tag == "ohsumed" and variant == "pc":
        return [1]
    return [1, 2, 4, 8] if tag in SHORT else [1, 4]


def imb_tag(ds, imb):
    return ds if imb == 1 else f"{ds}-imb{imb}"


def legend_handle(method, variant=None, label=None):
    filled = variant != "fixed"
    c = COLOR[method]
    return Line2D([], [], color=c, marker=MARKER[method], ls=LSTYLE.get(method, "none"),
                  mfc=c if filled else "white", mec=c, label=label or METHOD_NAME[method],
                  markersize=5 if method != "zeroshot" else 7)


# ---------------------------------------------------------------------------
# Table 2 -- main results at k = 4 (tables/main_results.tex)
# ---------------------------------------------------------------------------
def tab2(B):
    sec = "Table 2 (main results, k=4; Δ over all k)"
    head = [" & " + " & ".join(NAME[d] for d in CORE) + r" \\"]
    rows = []

    def f1_row(label, method, variant=None, k=None, tokens=True, models=SMALL):
        cells = []
        for d in CORE:
            kk, mark = k, ""
            if d == "ohsumed" and variant == "pc" and k not in ks_for(d, "pc"):
                kk = ks_for(d, "pc")[0]  # Ohsumed Per-Class exists at k=1 only
                mark = r"\textsuperscript{$k$=" + str(kk) + "}"
            v, n = fd.mean_metric(d, method, variant, kk, models=models)
            if v is None:
                cells.append("--")
                pending(sec, f"{d} {method} {variant or ''} k={kk}")
                continue
            t, _ = fd.mean_metric(d, method, variant, kk, models=models, metric="mean_prompt_tokens")
            cells.append(num(v) + mark + (r" {\scriptsize " + tok(t) + "}" if tokens else ""))
            note(sec, f"{d} {method} {variant or ''} k={kk or ''}: macro-F1 {v:.2f} "
                      f"(n={n}" + (f", tokens {t:,.0f})" if tokens else ")"))
        return label + " & " + " & ".join(cells) + r" \\"

    rows.append(f1_row("MiniLM + LR (no LLM)", "base", tokens=False))
    rows.append(f1_row("RoBERTa-base, fine-tuned", "finetuned", tokens=False))
    rows.append(r"\midrule")
    rows.append(f1_row("Zero-shot", "zeroshot"))
    # k = 0: the candidate set alone, no examples (pending runs)
    k0 = {d: fd.variant(d).select(method="cicle", k=0) for d in CORE}
    if any(k0.values()):
        var0 = sorted({kk[1] for d in CORE for kk in k0[d]})
        for var in var0:
            rows.append(f1_row(f"CICLe, no examples ($k$=0, {VARIANT_NAME.get(var, var)})",
                               "cicle", var, 0))
    else:
        pending(sec, "k=0 row (no cicle runs with k=0 yet)")
    rows.append(f1_row("Few-shot, Fixed", "fewshot", "fixed", 4))
    rows.append(f1_row("CICLe, Fixed", "cicle", "fixed", 4))
    rows.append(f1_row("Few-shot, Per-Class", "fewshot", "pc", 4))
    rows.append(f1_row("CICLe, Per-Class", "cicle", "pc", 4))
    rows.append(r"\midrule")
    for var in ("fixed", "pc"):
        dcells, ccells = [], []
        for d in CORE:
            res = fd.paired_delta(d, "cicle", "fewshot", var, ks_for(d, var), B=B)
            if res is None:
                dcells.append("--"); ccells.append("")
                pending(sec, f"Δ {d} {var}")
                continue
            dcells.append(delta_tex(res, ci=False))
            ccells.append(r"{\scriptsize " + ci_str(res) + f", $n$={res['n']}" + "}")
            note(sec, f"Δ CICLe − few-shot {d} {var} (all k): {delta_txt(res)}")
        rows.append(f"$\\Delta$ {VARIANT_NAME[var]} (CICLe $-$ few-shot, all $k$) & "
                    + " & ".join(dcells) + r" \\")
        rows.append(" & " + " & ".join(ccells) + r" \\")
    write_table("main_results", "l" + "c" * len(CORE), head, rows,
                comment="Table 2. Macro-F1 (pp) at k=4; LLM rows mean over 6 models x 3 seeds, "
                        "small number = mean prompt tokens per LLM call; Δ rows pool all k "
                        "(72 pairs; Ohsumed Fixed 36, Ohsumed Per-Class 18 at k=1 only), 95% paired "
                        "bootstrap CI over test instances. Ohsumed Per-Class cells show k=1 (marked).")


# ---------------------------------------------------------------------------
# Figure 1 -- macro-F1 against prompt tokens (f1_vs_tokens.pdf)
# ---------------------------------------------------------------------------
def fig1(B):
    sec = "Figure 1 (macro-F1 vs prompt tokens)"
    fig, axes = plt.subplots(1, len(CORE), figsize=(TEXTWIDTH, 1.95))
    for ax, d in zip(axes, CORE):
        ks = ks_for(d)
        series = [("zeroshot", None, [None]), ("fewshot", "fixed", ks), ("cicle", "fixed", ks),
                  ("fewshot", "pc", ks), ("cicle", "pc", ks)]
        for method, var, kk in series:
            xs, ys, lo, hi, kl = [], [], [], [], []
            for k in kk:
                r = fd.mean_ci(d, method, var, k, B=B)
                t, _ = fd.mean_metric(d, method, var, k, metric="mean_prompt_tokens")
                if r is None or t is None:
                    if method != "zeroshot" or k is not None:
                        pending(sec, f"{d} {method} {var} k={k}")
                    continue
                xs.append(t); ys.append(r["mean"]); lo.append(r["mean"] - r["lo"])
                hi.append(r["hi"] - r["mean"]); kl.append(k)
                note(sec, f"{d} {method} {var or ''} k={k or 0}: F1 {r['mean']:.2f} "
                          f"[{r['lo']:.2f}, {r['hi']:.2f}], tokens {t:,.0f} (n={r['n']})")
            if not xs:
                continue
            c = COLOR[method]
            filled = var != "fixed"
            ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=MARKER[method],
                        ls="-" if len(xs) > 1 else "none", mfc=c if filled else "white", mec=c,
                        markersize=7 if method == "zeroshot" else 4, zorder=3, **ERR)
            if method == "cicle" and var == "pc" and len(xs) > 1:
                ax.annotate(f"$k$={kl[0]}", (xs[0], ys[0]), textcoords="offset points",
                            xytext=(2, -9), fontsize=7, color="#52514e", ha="left", va="top")
                ax.annotate(f"$k$={kl[-1]}", (xs[-1], ys[-1]), textcoords="offset points",
                            xytext=(4, -2), fontsize=7, color="#52514e", ha="left", va="top")
        refs = []
        for ref, ls, lab in (("base", ":", "MiniLM+LR"), ("finetuned", "-.", "RoBERTa")):
            v, _ = fd.mean_metric(d, ref)
            if v is None:
                pending(sec, f"{d} {ref}")
                continue
            ax.axhline(v, color=COLOR[ref], ls=ls, lw=0.8, zorder=1)
            refs.append((v, lab))
            note(sec, f"{d} {ref}: {v:.2f}")
        ax.set_xscale("log")
        ax.set_xlim(140, 9000)
        ax.set_xticks([200, 1000, 5000])
        ax.set_xticklabels(["200", "1k", "5k"])
        lo_y, hi_y = ax.get_ylim()
        pad = 0.04 * (hi_y - lo_y)
        ax.set_ylim(min(lo_y, min(v for v, _ in refs) - 2 * pad), max(hi_y, max(v for v, _ in refs) + 2 * pad))
        lo_y, hi_y = ax.get_ylim()
        for v, lab in refs:  # label at the left edge, below the line when it is near the top
            above = v < hi_y - 0.12 * (hi_y - lo_y)
            ax.annotate(lab, xy=(0.02, v), xycoords=("axes fraction", "data"), fontsize=7,
                        color="#52514e", ha="left", va="bottom" if above else "top",
                        xytext=(0, 1 if above else -1), textcoords="offset points")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(NAME[d], fontsize=8, pad=3)
        ax.tick_params(axis="x", labelrotation=0)
    axes[0].set_ylabel("macro-F1 (pp)")
    fig.supxlabel("mean prompt tokens per LLM call (log scale)", fontsize=7.5, y=-0.01)
    handles = [legend_handle("zeroshot"), legend_handle("fewshot", "fixed", "Few-shot, Fixed"),
               legend_handle("cicle", "fixed", "CICLe, Fixed"),
               legend_handle("fewshot", "pc", "Few-shot, Per-Class"),
               legend_handle("cicle", "pc", "CICLe, Per-Class"),
               Line2D([], [], color=COLOR["base"], ls=":", lw=0.8, label="MiniLM + LR"),
               Line2D([], [], color=COLOR["finetuned"], ls="-.", lw=0.8, label="RoBERTa-base")]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.2),
               handletextpad=0.5)
    fig.subplots_adjust(wspace=0.35)
    save_fig(fig, "f1_vs_tokens")


# ---------------------------------------------------------------------------
# Figure 2 -- narrowing under a long-tailed labelled pool (narrowing_imbalance.pdf)
# Figure C1 -- the same for the Fixed variant
# ---------------------------------------------------------------------------
def rank_map(ds, seed, imb):
    counts = fd.class_counts(ds, seed, imb)
    if counts is None:
        return None, None
    return {lab: i for i, (lab, _) in enumerate(counts)}, counts


def fig2(B, variant="pc", name="narrowing_imbalance"):
    sec = f"Figure 2 ({VARIANT_NAME[variant]}; narrowing under imbalance)" if variant == "pc" \
        else "Figure C1 (Figure 2 for the Fixed variant)"
    datasets = ["yahoo-answers", "sst"]
    comparators = ["fewshot", "topk", "mass", "marginal"]
    drawn = set()  # methods that got at least one mark (for the legend)
    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, 4.0))
    for row, d in enumerate(datasets):
        # (a) paired delta vs imbalance ------------------------------------
        ax = axes[row, 0]
        for comp in comparators:
            xs, ys, lo, hi = [], [], [], []
            for i, imb in enumerate((1, 10, 100)):
                res = fd.paired_delta(imb_tag(d, imb), "cicle", comp, variant, [1, 4], B=B)
                if res is None:
                    pending(sec, f"(a) {imb_tag(d, imb)} CICLe − {comp} {variant}")
                    continue
                xs.append(i); ys.append(res["mean"]); lo.append(res["mean"] - res["lo"])
                hi.append(res["hi"] - res["mean"])
                note(sec, f"(a) {imb_tag(d, imb)} {variant} k∈{{1,4}} CICLe − {comp}: {delta_txt(res)}")
            if xs:
                c = COLOR[comp]
                drawn.add(comp)
                ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=MARKER[comp], ls=LSTYLE[comp],
                            mfc=c if variant == "pc" else "white", mec=c, zorder=3, **ERR)
        xs, ys = [], []
        for i, imb in enumerate((1, 10, 100)):
            res = fd.paired_delta(imb_tag(d, imb), "oracle", "fewshot", variant, [1, 4], B=B)
            if res is None:
                pending(sec, f"(a) {imb_tag(d, imb)} oracle − few-shot {variant} (ceiling)")
                continue
            xs.append(i); ys.append(res["mean"])
            note(sec, f"(a) {imb_tag(d, imb)} {variant} oracle − few-shot (ceiling): {delta_txt(res)}")
        if xs:
            drawn.add("oracle")
            ax.plot(xs, ys, color=COLOR["oracle"], ls="--", marker="x", lw=0.9, zorder=2)
        ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"1$\times$", r"10$\times$", r"100$\times$"])
        ax.set_xlim(-0.4, 2.4)
        ax.set_ylabel(f"{NAME[d]}\nCICLe $-$ alternative (pp)")
        if row == 0:
            ax.set_title("(a) paired gain of CICLe", fontsize=7.5)
        if row == 1:
            ax.set_xlabel("imbalance of labelled pool")

        # (b) coverage vs mean set size -------------------------------------
        ax = axes[row, 1]
        for method in ("cicle", "topk", "mass", "marginal"):
            pts = []
            for imb, size in ((1, 14), (10, 30), (100, 60)):
                tag = imb_tag(d, imb)
                vals = [fd.one_set_run(tag, method, s) for s in SEEDS]
                vals = [r for r in vals if r is not None]
                if not vals:
                    pending(sec, f"(b) {tag} {method} candidate sets")
                    continue
                cov = 100 * np.mean([r["gold_in_set"].mean() for r in vals])
                sz = np.mean([r["set_size"].mean() for r in vals])
                pts.append((sz, cov, size, imb))
                note(sec, f"(b) {tag} {method}: coverage {cov:.1f}%, mean set size {sz:.2f} "
                          f"({len(vals)} seeds)")
            if not pts:
                continue
            c = COLOR[method]
            drawn.add(method)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=0.7, ls=LSTYLE[method],
                    zorder=2)
            for x, y, size, imb in pts:
                ax.scatter([x], [y], s=size, color=c, marker=MARKER[method], edgecolor="white",
                           linewidth=0.6, zorder=3)
            if method == "cicle":
                for x, y, size, imb in pts:
                    if imb in (1, 100):
                        ax.annotate(f"{imb}$\\times$", (x, y), textcoords="offset points",
                                    xytext=(0, 6), fontsize=7, color="#52514e", ha="center")
        ax.axhline(95, color=MUTED, lw=0.7, ls=":", zorder=1)
        ax.set_ylabel("coverage of gold label (%)")
        if row == 0:
            ax.set_title("(b) coverage vs. set size", fontsize=7.5)
        if row == 1:
            ax.set_xlabel("mean candidate-set size")

        # (c) per-class coverage by frequency rank at 100x -------------------
        ax = axes[row, 2]
        tag = imb_tag(d, 100)
        v = fd.variant(tag)
        drew = False
        for method in ("cicle", "topk", "mass", "marginal"):
            by_rank = defaultdict(list)
            for s in SEEDS:
                rank, counts = rank_map(d, s, 100)
                run = fd.one_set_run(tag, method, s)
                if rank is None or run is None:
                    if run is None and s == SEEDS[0]:
                        pending(sec, f"(c) {tag} {method} candidate sets")
                    continue
                for lab, cov in fd.per_class_coverage(v, run).items():
                    by_rank[rank[lab]].append(cov)
            if not by_rank:
                continue
            xs = sorted(by_rank)
            ys = [np.mean(by_rank[x]) for x in xs]
            ax.plot([x + 1 for x in xs], ys, color=COLOR[method], marker=MARKER[method],
                    ls=LSTYLE[method], markersize=3, zorder=3)
            note(sec, f"(c) {tag} {method} per-class coverage by rank (mean over seeds): "
                      + ", ".join(f"{x + 1}:{y:.0f}" for x, y in zip(xs, ys)))
            drew = True
        ax.axhline(95, color=MUTED, lw=0.7, ls=":", zorder=1)
        if drew:
            ax.set_xticks(range(1, len(xs) + 1))
            ax.set_xticklabels([str(i) if (i == 1 or i % 2 == 0 or i == len(xs)) else ""
                                for i in range(1, len(xs) + 1)])
        else:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("per-class coverage (%)")
        if row == 0:
            ax.set_title(r"(c) coverage by class, 100$\times$", fontsize=7.5)
        if row == 1:
            ax.set_xlabel("class rank (frequent to rare)")

        # (d) per-class F1 by frequency rank at 100x (k=4) -------------------
        ax = axes[row, 3]
        drew = False
        for method in ("fewshot", "cicle", "topk", "mass", "marginal"):
            by_rank = defaultdict(list)
            for s in SEEDS:
                rank, counts = rank_map(d, s, 100)
                if rank is None:
                    continue
                for m in SMALL:
                    run = v.get(method, variant, 4, m, seed=s)
                    if run is None:
                        continue
                    for lab, f in fd.per_class_f1(v, run).items():
                        by_rank[rank[lab]].append(f)
            if not by_rank:
                if method != "marginal":
                    pending(sec, f"(d) {tag} {method} {variant} k=4")
                continue
            xs = sorted(by_rank)
            ys = [np.mean(by_rank[x]) for x in xs]
            c = COLOR[method]
            ax.plot([x + 1 for x in xs], ys, color=c, marker=MARKER[method], ls=LSTYLE[method],
                    markersize=3, mfc=c if variant == "pc" else "white", zorder=3)
            note(sec, f"(d) {tag} {method} {variant} k=4 per-class F1 by rank "
                      f"(mean over 6 models x 3 seeds): " + ", ".join(f"{x + 1}:{y:.1f}" for x, y in zip(xs, ys)))
            drew = True
        if drew:
            ax.set_xticks(range(1, len(xs) + 1))
            ax.set_xticklabels([str(i) if (i == 1 or i % 2 == 0 or i == len(xs)) else ""
                                for i in range(1, len(xs) + 1)])
        else:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("per-class F1 (pp)")
        if row == 0:
            ax.set_title(r"(d) F1 by class, 100$\times$, $k$=4", fontsize=7.5)
        if row == 1:
            ax.set_xlabel("class rank (frequent to rare)")

    handles = [legend_handle(m, variant, METHOD_NAME[m]) for m in
               ("fewshot", "cicle", "topk", "mass", "marginal") if m in drawn]
    if "oracle" in drawn:
        handles.append(Line2D([], [], color=COLOR["oracle"], ls="--", marker="x", lw=0.9,
                              label="Oracle $-$ few-shot (ceiling, panel a)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(wspace=0.55, hspace=0.32, left=0.09, right=0.99, top=0.95, bottom=0.14)
    save_fig(fig, name)


# ---------------------------------------------------------------------------
# Table 3 -- label renaming (tables/relabel.tex)
# ---------------------------------------------------------------------------
def tab3(B):
    sec = "Table 3 (label renaming)"
    head = [r"Dataset & Labels & $k$ & FS Fixed & CICLe Fixed & FS PC & CICLe PC \\"]
    rows = []
    for d in ("yahoo-answers", "go-emotions"):
        for tag, label in ((d, "original"), (d + "-relabel", "nonsense words")):
            for k in (1, 4):
                cells = []
                for m, var in (("fewshot", "fixed"), ("cicle", "fixed"), ("fewshot", "pc"),
                               ("cicle", "pc")):
                    val, n = fd.mean_metric(tag, m, var, k)
                    cells.append(num(val))
                    if val is None:
                        pending(sec, f"{tag} {m} {var} k={k}")
                    else:
                        note(sec, f"{tag} {m} {var} k={k}: {val:.2f} (n={n})")
                rows.append(f"{NAME[d]} & {label} & {k} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows.append(r"\multicolumn{7}{l}{$\Delta$ CICLe $-$ few-shot, $k \in \{1, 4\}$ (36 pairs)} \\")
    for d in ("yahoo-answers", "go-emotions"):
        for tag, label in ((d, "original"), (d + "-relabel", "nonsense words")):
            cells = []
            for var in ("fixed", "pc"):
                res = fd.paired_delta(tag, "cicle", "fewshot", var, [1, 4], B=B)
                cells.append(delta_tex(res))
                if res is None:
                    pending(sec, f"Δ {tag} {var}")
                else:
                    note(sec, f"Δ CICLe − few-shot {tag} {var} k∈{{1,4}}: {delta_txt(res)}")
            rows.append(f"{NAME[d]} & {label} & & \\multicolumn{{2}}{{c}}{{{cells[0]}}} & "
                        f"\\multicolumn{{2}}{{c}}{{{cells[1]}}} \\\\")
    write_table("relabel", "llrcccc", head, rows,
                comment="Table 3. Macro-F1 with the original class names and with every name "
                        "replaced by a nonsense word (same mapping for all seeds); mean over 6 "
                        "models x 3 seeds. Δ rows: paired bootstrap CI, 36 pairs.")


# ---------------------------------------------------------------------------
# Table 4 -- supervised baselines vs LLM pipelines (tables/baselines.tex)
# ---------------------------------------------------------------------------
def tab4(B):
    sec = "Table 4 (supervised vs pipelines)"
    cols = []
    for d in ("yahoo-answers", "sst"):
        for imb in (1, 10, 100):
            cols.append((imb_tag(d, imb), f"{NAME[d]} {imb}$\\times$"))
        for n in (250, 500, 1000):
            cols.append((f"{d}-n{n}", f"{NAME[d]} $n$={n}"))
    for d in ("semeval-18", "go-emotions", "ohsumed"):
        cols.append((d, NAME[d]))
    for t, _ in cols:
        if not fd.has_results(t):
            pending(sec, f"column {t} (no results directory)")
    cols = [(t, l) for t, l in cols if fd.has_results(t)]
    values = {}  # (row, tag) -> value

    def fill(row, method, var=None, k=None, only_1x=False):
        for t, _ in cols:
            if only_1x and ("imb" in t):
                values[(row, t)] = None
                continue
            val, n = fd.mean_metric(t, method, var, k)
            if val is None and method in ("fewshot", "cicle") and t == "ohsumed" and var == "pc":
                val, n = fd.mean_metric(t, method, "fixed", k)  # Ohsumed: Fixed only
                values[(row, t)] = (val, "F")
            else:
                values[(row, t)] = (val, None) if val is not None else None
            if val is None and not only_1x:
                pending(sec, f"{t} {method} {var or ''} k={k or ''}")
            if val is not None:
                note(sec, f"{t} {method} {var or ''} k={k or ''}: {val:.2f} (n={n})")

    labels = ["MiniLM + LR", "RoBERTa-base, fine-tuned", "Zero-shot", "Few-shot PC, $k$=4",
              "CICLe PC, $k$=4"]
    fill(0, "base"); fill(1, "finetuned"); fill(2, "zeroshot", only_1x=True)
    fill(3, "fewshot", "pc", 4); fill(4, "cicle", "pc", 4)
    best = {}
    for t, _ in cols:
        cand = []
        for m in ("fewshot", "cicle"):
            for var in ("fixed", "pc"):
                for k in (1, 2, 4, 8):
                    val, _ = fd.mean_metric(t, m, var, k)
                    if val is not None:
                        cand.append((val, f"{'CICLe' if m == 'cicle' else 'FS'} "
                                          f"{'F' if var == 'fixed' else 'PC'} $k$={k}"))
        best[t] = max(cand) if cand else None
        if best[t]:
            note(sec, f"{t} best LLM pipeline: {best[t][0]:.2f} ({best[t][1]})")
    rows = []
    for i, lab in enumerate(labels):
        cells = []
        for t, _ in cols:
            x = values.get((i, t))
            if x is None:
                cells.append("--")
                continue
            val, flag = x
            col_vals = [values[(j, t)][0] for j in range(len(labels)) if values.get((j, t))]
            col_vals.append(best[t][0] if best[t] else -1)
            s = num(val) + (r"\textsuperscript{F}" if flag else "")
            cells.append(r"\textbf{" + s + "}" if val >= max(col_vals) - 1e-9 else s)
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    cells = []
    for t, _ in cols:
        if not best[t]:
            cells.append("--")
            continue
        col_vals = [values[(j, t)][0] for j in range(len(labels)) if values.get((j, t))]
        s = num(best[t][0])
        s = r"\textbf{" + s + "}" if best[t][0] >= max(col_vals + [best[t][0]]) - 1e-9 else s
        cells.append(s + r" {\scriptsize " + best[t][1] + "}")
    rows.append("Best LLM pipeline & " + " & ".join(cells) + r" \\")
    head = [" & " + " & ".join(l for _, l in cols) + r" \\"]
    write_table("baselines", "l" + "c" * len(cols), head, rows,
                comment="Table 4. Macro-F1; supervised rows mean over 3 seeds, LLM rows over 6 "
                        "models x 3 seeds. Zero-shot does not depend on the pool (1x column only). "
                        "F = Fixed retrieval (Ohsumed has no Per-Class runs). Bold = best per column.")
    # per-seed RoBERTa / LR values for the appendix version
    for t, _ in cols:
        for method in ("base", "finetuned"):
            vals = fd.metric_values(t, method)
            if vals:
                note(sec, f"{t} {method} per seed: " + ", ".join(f"{s}: {v:.1f}" for (_, s), v in sorted(vals.items())))


# ---------------------------------------------------------------------------
# Figure 3 -- choosing alpha (alpha.pdf)
# ---------------------------------------------------------------------------
def fig3(B):
    sec = "Figure 3 (alpha; Llama-3.1-8B + Llama-3.2-3B, 24 cells)"
    fig, (top, mid, bot) = plt.subplots(3, 1, figsize=(COLWIDTH, 3.9), sharex=True,
                                        gridspec_kw={"height_ratios": [2.2, 1, 1]})
    for d in SHORT:
        xs, ys, lo, hi, cov, skip = [], [], [], [], [], []
        for a in ALPHAS:
            res = fd.paired_delta(d, "cicle", "fewshot", None, [1, 4], models=LLAMAS, B=B, alpha=a,
                                  variants=["fixed", "pc"])
            if res is None:
                pending(sec, f"{d} alpha={a}")
                continue
            runs = [fd.variant(d).get("cicle", var, k, m, seed=s, alpha=a)
                    for var in ("fixed", "pc") for k in (1, 4) for m in LLAMAS for s in SEEDS]
            runs = [r for r in runs if r is not None]
            c = 100 * np.mean([r["metrics"]["coverage"] for r in runs])
            sk = 100 * np.mean([1 - r["metrics"]["llm_call_rate"] for r in runs])
            xs.append(a); ys.append(res["mean"]); lo.append(res["mean"] - res["lo"])
            hi.append(res["hi"] - res["mean"]); cov.append(c); skip.append(sk)
            note(sec, f"{d} alpha={a}: Δ {delta_txt(res)}; coverage {c:.1f}%, "
                      f"answered without LLM {sk:.1f}%, mean set size "
                      f"{np.mean([r['metrics']['mean_set_size'] for r in runs]):.2f}")
        if not xs:
            continue
        c, mk = DS_COLOR[d], DS_MARKER[d]
        top.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=mk, label=NAME[d], zorder=3, **ERR)
        mid.plot(xs, cov, color=c, marker=mk, zorder=3)
        bot.plot(xs, skip, color=c, marker=mk, zorder=3)
    top.axhline(0, color=MUTED, lw=0.7, zorder=1)
    top.set_ylabel("CICLe $-$ few-shot (pp)")
    mid.plot(ALPHAS, [100 * (1 - a) for a in ALPHAS], color=MUTED, ls=":", lw=0.8, zorder=1)
    mid.text(0.03, 0.08, r"dotted: $1-\alpha$", transform=mid.transAxes, fontsize=7, color="#52514e")
    mid.set_ylabel("coverage (%)")
    bot.set_ylabel("no LLM call (%)")
    bot.set_xscale("log"); bot.set_xticks(ALPHAS); bot.set_xticklabels([str(a) for a in ALPHAS])
    bot.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    bot.set_xlabel(r"miscoverage level $\alpha$")
    top.legend(ncol=2, loc="upper left", handletextpad=0.4)
    fig.subplots_adjust(hspace=0.12)
    save_fig(fig, "alpha")


# ---------------------------------------------------------------------------
# Appendix B -- full grids per dataset variant (tables/grid_<variant>.tex)
# ---------------------------------------------------------------------------
def tabB(B):
    sec = "Appendix B (full grids)"
    order = ["base", "finetuned", "zeroshot", "fewshot", "cicle", "topk", "mass", "marginal", "oracle"]
    for tag in ALL_TAGS:
        if not fd.has_results(tag):
            pending(sec, tag)
            continue
        v = fd.variant(tag)
        groups = defaultdict(list)
        for key, run in v.select(models=SMALL + ["none"]).items():
            m, var, k = key[0], key[1], key[2]
            if m == "base" and (key[4], key[5]) != ("minilm", "lr"):
                continue
            groups[(m, var or "-", k or 0)].append(run["metrics"])
        rows = []
        last = None
        for g in sorted(groups, key=lambda g: (order.index(g[0]), g[1], g[2])):
            ms = groups[g]
            mean = lambda name: np.mean([x[name] for x in ms])  # noqa: E731
            if last and last != g[0] and g[0] in ("zeroshot", "fewshot", "topk"):
                rows.append(r"\midrule")
            last = g[0]
            rows.append(f"{METHOD_NAME[g[0]]} & {VARIANT_NAME.get(g[1], '--')} & "
                        f"{g[2] if g[0] not in ('base', 'finetuned', 'zeroshot') else '--'} & "
                        f"{len(ms)} & {100 * mean('macro_f1'):.1f} & {100 * mean('accuracy'):.1f} & "
                        f"{100 * mean('invalid_rate'):.1f} & {mean('mean_shots'):.1f} & "
                        f"{mean('mean_prompt_tokens'):,.0f} \\\\")
        head = [r"Method & Retrieval & $k$ & runs & macro-F1 & acc. & invalid (\%) & shots & tokens \\"]
        write_table(f"grid_{tag}", "llrrrrrrr", head, rows,
                    comment=f"Table B: {tag}, six small models, reference setting (MiniLM, LR, "
                            "alpha 0.05); means over runs. A row with fewer than 18 runs (3 for the "
                            "supervised rows) is an incomplete, still-running configuration.")


# ---------------------------------------------------------------------------
# Appendix C1 -- narrowing comparisons over all variants (tables/narrowing_all.tex)
# ---------------------------------------------------------------------------
def tabC1(B):
    sec = "Table C1 (CICLe − alternative, all variants)"
    comps = ["fewshot", "topk", "mass", "marginal", "oracle"]
    head = [r"Variant & Retrieval & vs.\ few-shot & vs.\ top-$m$ & vs.\ prob.\ mass & "
            r"vs.\ marginal CP & vs.\ oracle \\"]
    rows = []
    for tag in ALL_TAGS:
        if not fd.has_results(tag):
            pending(sec, tag)
            continue
        for var in ("fixed", "pc"):
            cells, any_cell = [], False
            for comp in comps:
                ks = ks_for(tag, var) if comp == "fewshot" else [k for k in (1, 4) if k in ks_for(tag, var)]
                res = fd.paired_delta(tag, "cicle", comp, var, ks, B=B)
                if res is None:
                    cells.append("--")
                    if comp in ("fewshot", "topk", "mass") and not (tag == "ohsumed" and var == "pc"):
                        pending(sec, f"{tag} {var} vs {comp}")
                    continue
                any_cell = True
                cells.append(delta_tex(res, n=True))
                note(sec, f"{tag} {var} CICLe − {comp}: {delta_txt(res)}")
            if any_cell:
                rows.append(f"{tag_name(tag)} & {VARIANT_NAME[var]} & " + " & ".join(cells) + r" \\")
    write_table("narrowing_all", "llccccc", head, rows,
                comment="Table C1. Paired Δ macro-F1 (pp), CICLe minus alternative, 95% bootstrap "
                        "CI over test instances, (n) = number of (model, seed, k) cells; vs few-shot "
                        "pools all k, the other columns k in {1,4}. Bold: CI excludes 0.")


# ---------------------------------------------------------------------------
# Appendix C2 -- candidate-set statistics (tables/set_stats.tex)
# ---------------------------------------------------------------------------
def tabC2(B):
    sec = "Table C2 (candidate-set statistics)"
    head = [r"Variant & Method & coverage (\%) & mean size & singleton (\%) & "
            r"min.\ per-class coverage (\%) \\"]
    rows = []
    for tag in ALL_TAGS:
        if not fd.has_results(tag):
            continue
        v = fd.variant(tag)
        first = True
        for method in fd.NARROWING:
            runs = [fd.one_set_run(tag, method, s) for s in SEEDS]
            runs = [r for r in runs if r is not None]
            if not runs:
                continue
            cov = 100 * np.mean([r["gold_in_set"].mean() for r in runs])
            size = np.mean([r["set_size"].mean() for r in runs])
            single = 100 * np.mean([(r["set_size"] == 1).mean() for r in runs])
            mins = [min(fd.per_class_coverage(v, r).values()) for r in runs]
            rows.append(f"{tag_name(tag) if first else ''} & {METHOD_NAME[method]} & {cov:.1f} & "
                        f"{size:.2f} & {single:.1f} & {np.mean(mins):.1f} \\\\")
            first = False
            note(sec, f"{tag} {method}: coverage {cov:.1f}%, size {size:.2f}, singleton {single:.1f}%, "
                      f"min per-class coverage {np.mean(mins):.1f}% (per seed "
                      + ", ".join(f"{m:.0f}" for m in mins) + f"; {len(runs)} seeds)")
        if not first:
            rows.append(r"\addlinespace")
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()
    write_table("set_stats", "llrrrr", head, rows,
                comment="Table C2. Candidate sets at the reference setting (MiniLM, LR, alpha 0.05), "
                        "one run per seed (sets do not depend on the LLM, k or variant), mean over "
                        "3 seeds. Min. per-class coverage = min over classes of the coverage of that "
                        "gold class, averaged over seeds.")


# ---------------------------------------------------------------------------
# Appendix D1 -- per-model CICLe minus few-shot (tables/per_model.tex)
# ---------------------------------------------------------------------------
def tabD1(B):
    sec = "Table D1 (per-model Δ, all k)"
    cols = [(d, var) for d in CORE for var in (("fixed", "pc") if d != "ohsumed" else ("fixed",))]
    head = ["Model & " + " & ".join(f"{NAME[d]} {'F' if var == 'fixed' else 'PC'}" for d, var in cols)
            + r" \\"]
    rows = []
    for m in SMALL:
        cells = []
        for d, var in cols:
            res = fd.paired_delta(d, "cicle", "fewshot", var, ks_for(d, var), models=[m], B=B)
            if res is None:
                cells.append("--"); pending(sec, f"{m} {d} {var}")
                continue
            cells.append(delta_tex(res))
            note(sec, f"{MODEL_NAME[m]} {d} {var}: {delta_txt(res)}")
        rows.append(MODEL_NAME[m] + " & " + " & ".join(cells) + r" \\")
    write_table("per_model", "l" + "c" * len(cols), head, rows,
                comment="Table D1. CICLe minus few-shot per model, all k pooled (12 pairs, Ohsumed 6), "
                        "95% paired bootstrap CI. F = Fixed, PC = Per-Class.")


# ---------------------------------------------------------------------------
# Appendix E1 -- embedding / classifier / alpha ablation (tables/ablation.tex)
# ---------------------------------------------------------------------------
def tabE1(B):
    sec = "Table E1 (ablation; Llama-3.1-8B + Llama-3.2-3B, 24 cells)"
    settings = [("contriever", "lr", 0.05), ("minilm", "lr", 0.05), ("minilm", "svm", 0.05),
                ("tfidf", "lr", 0.05), ("minilm", "lr", 0.01), ("minilm", "lr", 0.10),
                ("minilm", "lr", 0.20)]
    head = [r"Dataset & Embedding & Classifier & $\alpha$ & CICLe & few-shot & $\Delta$ & "
            r"coverage (\%) & set size & no LLM call (\%) \\"]
    rows = []
    for d in SHORT:
        first = True
        for emb, clf, a in settings:
            res = fd.paired_delta(d, "cicle", "fewshot", None, [1, 4], models=LLAMAS, B=B, alpha=a,
                                  emb=emb, clf=clf, variants=["fixed", "pc"])
            if res is None:
                pending(sec, f"{d} {emb} {clf} {a}")
                continue
            runs = [fd.variant(d).get("cicle", var, k, m, seed=s, emb=emb, clf=clf, alpha=a)
                    for var in ("fixed", "pc") for k in (1, 4) for m in LLAMAS for s in SEEDS]
            runs = [r for r in runs if r is not None]
            cov = 100 * np.mean([r["metrics"]["coverage"] for r in runs])
            size = np.mean([r["metrics"]["mean_set_size"] for r in runs])
            skip = 100 * np.mean([1 - r["metrics"]["llm_call_rate"] for r in runs])
            if a == 0.01 and not first:
                rows.append(r"\addlinespace")
            rows.append(f"{NAME[d] if first else ''} & {emb} & {clf.upper()} & {a:.2f} & "
                        f"{res['mean_a']:.1f} & {res['mean_b']:.1f} & {delta_tex(res)} & "
                        f"{cov:.1f} & {size:.2f} & {skip:.1f} \\\\")
            first = False
            note(sec, f"{d} {emb}/{clf}/alpha={a}: CICLe {res['mean_a']:.2f}, few-shot "
                      f"{res['mean_b']:.2f}, Δ {delta_txt(res)}; coverage {cov:.1f}%, size {size:.2f}, "
                      f"no LLM call {skip:.1f}%")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    write_table("ablation", "lllrrrcrrr", head, rows,
                comment="Table E1. Llama-3.1-8B and Llama-3.2-3B, both retrieval variants, k in {1,4}, "
                        "3 seeds (24 cells). Few-shot uses the same embedding for retrieval. "
                        "Δ = CICLe − few-shot, paired bootstrap CI.")


# ---------------------------------------------------------------------------
# Appendix F1 -- larger models (tables/large_models.tex)
# ---------------------------------------------------------------------------
def tabF1(B):
    sec = "Table F1 (12B / 32B models)"
    head = [r"Model & Dataset & seeds & Zero-shot & \multicolumn{2}{c}{Fixed} & "
            r"\multicolumn{2}{c}{Per-Class} & $\Delta$ Fixed & $\Delta$ PC \\",
            r" & & & & $k$=1 & $k$=4 & $k$=1 & $k$=4 & & \\"]
    rows = []
    for m in LARGE:
        for d in SHORT:
            v = fd.variant(d)
            seeds = sorted({key[7] for key in v.select(models=[m])})
            if not seeds:
                pending(sec, f"{m} {d}")
                continue
            zs, _ = fd.mean_metric(d, "zeroshot", models=[m], allow_partial=True)
            cells = [num(zs)]
            for var in ("fixed", "pc"):
                for k in (1, 4):
                    f, nf = fd.mean_metric(d, "fewshot", var, k, models=[m], allow_partial=True)
                    c, nc = fd.mean_metric(d, "cicle", var, k, models=[m], allow_partial=True)
                    cells.append(f"{num(f)} / {num(c)}")
                    if f is None or c is None:
                        pending(sec, f"{m} {d} {var} k={k} (few-shot n={nf}, CICLe n={nc})")
                    else:
                        note(sec, f"{MODEL_NAME[m]} {d} {var} k={k}: few-shot {f:.1f}, CICLe {c:.1f} "
                                  f"(seeds {nf}/{nc})")
            if zs is not None:
                note(sec, f"{MODEL_NAME[m]} {d} zero-shot: {zs:.1f} ({len(seeds)} seeds)")
            for var in ("fixed", "pc"):
                res = fd.paired_delta(d, "cicle", "fewshot", var, [1, 4], models=[m], B=B,
                                      allow_partial=True)
                cells.append(delta_tex(res, n=True))
                if res is not None:
                    note(sec, f"{MODEL_NAME[m]} {d} Δ {var}: {delta_txt(res)}")
            rows.append(f"{MODEL_NAME[m]} & {NAME[d]} & {','.join(str(s) for s in seeds)} & "
                        + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    # best small model + CICLe PC k=4 (seed 42) vs the 32B zero-shot (seed 42)
    extra = []
    for d in ("yahoo-answers", "sst", "semeval-18", "go-emotions"):
        v = fd.variant(d)
        cand = [(fd.f1(v, r), mm) for mm in SMALL[:3]
                if (r := v.get("cicle", "pc", 4, mm, seed=42)) is not None]
        big = v.get("zeroshot", None, None, "qwen-2.5-32b", seed=42)
        if cand and big is not None:
            f, mm = max(cand)
            extra.append(f"{NAME[d]} & best 3B + CICLe PC $k$=4, seed 42: {MODEL_NAME[mm]} {f:.1f} & "
                         f"Qwen2.5-32B zero-shot, seed 42: {fd.f1(v, big):.1f} \\\\")
            note(sec, f"{d} seed 42: best 3B + CICLe PC k=4 = {MODEL_NAME[mm]} {f:.1f}; "
                      f"Qwen2.5-32B zero-shot {fd.f1(v, big):.1f}")
    if extra:
        rows.append(r"\multicolumn{10}{l}{\emph{Smallest models with CICLe against the largest "
                    r"model without examples (seed 42)}} \\")
        rows.extend(r"\multicolumn{10}{l}{" + e.replace(r" \\", "").replace(" & ", "; ") + r"} \\"
                    for e in extra)
    write_table("large_models", "llrcccccc c", head, rows,
                comment="Table F1. Mistral-Nemo-12B and Qwen2.5-32B: macro-F1 (mean over the seeds "
                        "listed), cells 'few-shot / CICLe'; Δ = CICLe − few-shot over k in {1,4}, "
                        "paired bootstrap CI, (n) cells.")


# ---------------------------------------------------------------------------
# Appendix G1 -- fixes and breaks (tables/fixes_breaks.tex)
# ---------------------------------------------------------------------------
def tabG1(B):
    sec = "Table G1 (fixes vs breaks, k=4)"
    head = [r"Variant & Retr. & pairs & fixed & broken & both right & both wrong & "
            r"\multicolumn{3}{c}{gold outside set} & \multicolumn{3}{c}{few-shot answer outside set} \\",
            r" & & & \multicolumn{4}{c}{(\% of pairs)} & share & fixed & broken & share & fixed & broken \\"]
    rows = []
    tags = ["yahoo-answers", "sst", "semeval-18", "go-emotions", "ohsumed", "yahoo-answers-imb100",
            "sst-imb100", "yahoo-answers-relabel", "go-emotions-relabel"]
    for tag in tags:
        if not fd.has_results(tag):
            pending(sec, tag)
            continue
        v = fd.variant(tag)
        for var in ("pc", "fixed"):
            agg = Counter()
            for m in SMALL:
                for s in SEEDS:
                    a, b = v.get("cicle", var, 4, m, seed=s), v.get("fewshot", var, 4, m, seed=s)
                    if a is None or b is None:
                        continue
                    ar, br = a["pred"] == a["gold"], b["pred"] == b["gold"]
                    gis = a["gold_in_set"]
                    sets = v.set_matrix(a)
                    bp = b["pred"].astype(int)
                    fs_in = np.where(bp >= 0, sets[np.arange(len(bp)), np.clip(bp, 0, None)], False)
                    agg["n"] += len(ar)
                    agg["fix"] += (ar & ~br).sum(); agg["brk"] += (~ar & br).sum()
                    agg["both"] += (ar & br).sum(); agg["neither"] += (~ar & ~br).sum()
                    agg["out"] += (~gis).sum(); agg["out_fix"] += (~gis & ar & ~br).sum()
                    agg["out_brk"] += (~gis & ~ar & br).sum()
                    agg["fsout"] += (~fs_in).sum(); agg["fsout_fix"] += (~fs_in & ar & ~br).sum()
                    agg["fsout_brk"] += (~fs_in & ~ar & br).sum()
            if not agg["n"]:
                if not (tag == "ohsumed" and var == "pc"):
                    pending(sec, f"{tag} {var}")
                continue
            n = agg["n"]
            pct = lambda x, d=n: 100 * x / max(d, 1)  # noqa: E731
            rows.append(f"{tag_name(tag)} & {VARIANT_NAME[var]} & {n:,} & {pct(agg['fix']):.1f} & "
                        f"{pct(agg['brk']):.1f} & {pct(agg['both']):.1f} & {pct(agg['neither']):.1f} & "
                        f"{pct(agg['out']):.1f} & {pct(agg['out_fix'], agg['out']):.1f} & "
                        f"{pct(agg['out_brk'], agg['out']):.1f} & {pct(agg['fsout']):.1f} & "
                        f"{pct(agg['fsout_fix'], agg['fsout']):.1f} & "
                        f"{pct(agg['fsout_brk'], agg['fsout']):.1f} \\\\")
            note(sec, f"{tag} {var} k=4 ({n:,} pairs): fixed {pct(agg['fix']):.1f}%, broken "
                      f"{pct(agg['brk']):.1f}%, both right {pct(agg['both']):.1f}%, both wrong "
                      f"{pct(agg['neither']):.1f}%; gold outside set {pct(agg['out']):.1f}% of pairs "
                      f"(within: fixed {pct(agg['out_fix'], agg['out']):.1f}%, broken "
                      f"{pct(agg['out_brk'], agg['out']):.1f}%); few-shot answer outside set "
                      f"{pct(agg['fsout']):.1f}% (within: fixed {pct(agg['fsout_fix'], agg['fsout']):.1f}%, "
                      f"broken {pct(agg['fsout_brk'], agg['fsout']):.1f}%)")
    write_table("fixes_breaks", "llrrrrrrrrrrr", head, rows,
                comment="Table G1. CICLe k=4 vs few-shot k=4, same retrieval variant, 6 models x 3 "
                        "seeds x 1,000 instances. fixed = CICLe right & few-shot wrong; broken = the "
                        "reverse. 'gold outside set' / 'few-shot answer outside set': share of pairs, "
                        "then fixed and broken as % of that subset.")


# ---------------------------------------------------------------------------
# Appendix G2 -- invalid outputs (tables/invalid.tex, tables/invalid_raw.tex)
# ---------------------------------------------------------------------------
def tabG2(B):
    sec = "Table G2 (invalid outputs)"
    head = ["Model & " + " & ".join(NAME[d] for d in CORE) + r" \\",
            r" & \multicolumn{" + str(len(CORE)) + r"}{c}{mean (max) invalid rate in \%: "
            r"zero-shot / few-shot / CICLe} \\"]
    rows = []
    raw = defaultdict(Counter)
    for m in SMALL + LARGE:
        cells = []
        for d in CORE:
            v = fd.variant(d)
            vals = defaultdict(list)
            for key, r in v.select(models=[m]).items():
                if key[0] in ("zeroshot", "fewshot", "cicle"):
                    vals[key[0]].append(100 * r["metrics"]["invalid_rate"])
                    raw[m].update(r["invalid_raw"])
            parts = []
            for meth in ("zeroshot", "fewshot", "cicle"):
                if vals[meth]:
                    parts.append(f"{np.mean(vals[meth]):.1f} ({np.max(vals[meth]):.1f})")
                    note(sec, f"{MODEL_NAME[m]} {d} {meth}: mean {np.mean(vals[meth]):.1f}%, "
                              f"max {np.max(vals[meth]):.1f}% over {len(vals[meth])} runs")
                else:
                    parts.append("--")
            cells.append(" / ".join(parts))
        rows.append(MODEL_NAME[m] + " & " + " & ".join(cells) + r" \\")
    write_table("invalid", "l" + "c" * len(CORE), head, rows,
                comment="Table G2. Invalid (unparseable) outputs per model and dataset, reference "
                        "setting; mean and maximum over (variant, k, seed) runs of each method.")
    rows = []
    for m in SMALL + LARGE:
        if not raw[m]:
            continue
        total = sum(raw[m].values())
        top = ", ".join(f"\\texttt{{{tex_escape(' '.join(s.split()))}}} ({c:,})"
                        for s, c in raw[m].most_common(8))
        rows.append(f"{MODEL_NAME[m]} & {total:,} & {top} \\\\")
        note(sec, f"{MODEL_NAME[m]} most frequent invalid raw outputs ({total:,} invalid in total): "
                  + "; ".join(f"{s!r} x{c}" for s, c in raw[m].most_common(8)))
    write_table("invalid_raw", "lrp{0.75\\linewidth}",
                [r"Model & invalid outputs & most frequent raw outputs (count) \\"], rows,
                comment="Table G2b. Raw outputs that matched no label (first 60 characters), "
                        "pooled over the five main datasets, all methods, reference setting.")


# ---------------------------------------------------------------------------
# Appendix G3 -- qualitative examples (tables/examples.tex)
# ---------------------------------------------------------------------------
def tabG3(B):
    sec = "Table G3 (qualitative examples; Llama-3.1-8B, seed 42, PC k=4)"
    rng = np.random.default_rng(0)
    head = [r"Case & Text & Gold & Candidate set & Few-shot & CICLe \\"]
    rows = []
    for d in ("yahoo-answers", "go-emotions"):
        v = fd.variant(d)
        a, b = v.get("cicle", "pc", 4, "llama-3.1-8b", seed=42), v.get("fewshot", "pc", 4, "llama-3.1-8b", seed=42)
        if a is None or b is None:
            pending(sec, f"{d} runs"); continue
        texts = fd.test_texts(d, 42)
        if texts is None:
            pending(sec, f"{d} test texts (HF cache)"); continue
        with open(a["_path"]) as f:
            ra = {r["idx"]: r for r in json.load(f)["records"]}
        with open(b["_path"]) as f:
            rb = {r["idx"]: r for r in json.load(f)["records"]}
        labels = v.labels
        sets = v.set_matrix(a)
        ar, br = a["pred"] == a["gold"], b["pred"] == b["gold"]
        gis = a["gold_in_set"]
        bp = b["pred"].astype(int)
        fs_in = np.where(bp >= 0, sets[np.arange(len(bp)), np.clip(bp, 0, None)], False)
        cases = [("fixed: few-shot answered outside the set", np.where(ar & ~br & ~fs_in)[0]),
                 ("broken: gold outside the set", np.where(~gis & ~ar & br)[0]),
                 ("broken: switched inside the set", np.where(gis & ~ar & br)[0])]
        for label, idx in cases:
            if not len(idx):
                continue
            pick = rng.choice(idx, size=min(3, len(idx)), replace=False)
            for i in sorted(pick):
                text = " ".join(texts[i].replace("\\n", " ").split())  # literal "\n" in Yahoo texts
                text = text[:140] + ("..." if len(text) > 140 else "")
                cset = ", ".join(ra[i]["conformal_set"])
                rows.append(f"{tex_escape(label)} & {tex_escape(text)} & {tex_escape(labels[a['gold'][i]])} & "
                            f"{tex_escape(cset)} & {tex_escape(rb[i]['raw'])} & "
                            f"{tex_escape(ra[i]['raw'] if ra[i]['raw'] is not None else '(no LLM call)')} \\\\")
            note(sec, f"{d}: {len(idx)} instances in case '{label}'; sampled idx {sorted(pick.tolist())}")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    write_table("examples", "p{0.14\\linewidth}p{0.36\\linewidth}p{0.08\\linewidth}p{0.2\\linewidth}p{0.08\\linewidth}p{0.08\\linewidth}",
                head, rows,
                comment="Table G3. Yahoo Answers and GoEmotions, Llama-3.1-8B, seed 42, Per-Class k=4; "
                        "three instances per case sampled with a fixed seed.")


# ---------------------------------------------------------------------------
# Appendix H1 -- controlled-imbalance class counts and relabel words (tables/imbalance_counts.tex)
# ---------------------------------------------------------------------------
def tabH1(B):
    sec = "Table H1 (imbalance class counts; nonsense words)"
    head = [r"Dataset & Imbalance & Seed & class: training count (descending) \\"]
    rows = []
    for d in ("yahoo-answers", "sst"):
        for imb in (10, 100):
            for s in SEEDS:
                counts = fd.class_counts(d, s, imb)
                if counts is None:
                    continue
                rows.append(f"{NAME[d]} & {imb}$\\times$ & {s} & "
                            + ", ".join(f"{tex_escape(l)} {c}" for l, c in counts) + r" \\")
                note(sec, f"{d} {imb}x seed {s}: " + ", ".join(f"{l} {c}" for l, c in counts))
    write_table("imbalance_counts", "llrp{0.7\\linewidth}", head, rows,
                comment="Table H1. Class order and training-pool counts (after the 80/20 calibration "
                        "split) under controlled imbalance; the class order is a seed-dependent "
                        "permutation.")
    rows = []
    for d in ("yahoo-answers", "go-emotions"):
        lm = fd.label_map(d)
        if lm is None:
            continue
        rows.append(f"{NAME[d]} & " + ", ".join(f"{tex_escape(k)} $\\to$ {tex_escape(w)}"
                                              for k, w in lm.items()) + r" \\")
        note(sec, f"{d} relabel map: " + ", ".join(f"{k}->{w}" for k, w in lm.items()))
    write_table("relabel_words", "lp{0.8\\linewidth}", [r"Dataset & original name $\to$ nonsense word \\"],
                rows, comment="Table H1b. Nonsense-word mapping used for label renaming (same for all seeds).")


# ---------------------------------------------------------------------------
# NUMBERS.md
# ---------------------------------------------------------------------------
def write_numbers(B, built):
    """Merge this run's numbers into cache/figures/numbers.json (one entry per
    step, so a --only run refreshes only its own sections) and regenerate
    NUMBERS.md from the merged store."""
    store = json.load(open(NUMBERS_JSON)) if os.path.exists(NUMBERS_JSON) else {}
    for step in built:
        store[step] = {"title": _TITLE.get(step, STEPS[step][0]), "B": B,
                       "lines": _NUMBERS.get(step, []), "pending": fd.pending_messages(step)}
    with open(NUMBERS_JSON, "w") as f:
        json.dump(store, f, indent=1, ensure_ascii=False)
    order = list(STEPS)
    rank = lambda s: order.index(s) if s in order else 99  # noqa: E731
    out = ["# Numbers produced by make_figures.py", "",
           "Generated file; do not edit by hand. Every line is one value a figure or table shows "
           "(macro-F1 in percentage points; Δ = CICLe − alternative with the 95% paired bootstrap CI "
           "over test instances, two-sided bootstrap p, n = number of (model, seed, k[, variant]) "
           "cells). Sections built with fewer than 5000 bootstrap resamples are marked.", ""]
    pend = [(step, m) for step in sorted(store, key=rank) for m in store[step].get("pending", [])]
    if pend:
        out += ["## Pending (slots left empty)", "",
                "Specific slots first, then the incomplete grids behind them "
                "(present / expected runs or cells).", ""]
        generic = ("mean_metric:", "mean_ci:", "paired_delta:", "class_counts:", "label_map:",
                   "test_texts:")
        out += [f"- {m}" for _, m in pend if not m.startswith(generic)]
        out += [f"- {m}" for _, m in pend if m.startswith(generic)] + [""]
    for step in sorted(store, key=rank):
        b = store[step]["B"]
        out.append(f"## {store[step]['title']}" + (f" (B={b}, development build)" if b < 5000 else ""))
        out.append("")
        out += [f"- {l}" for l in store[step]["lines"]]
        out.append("")
    with open(NUMBERS_MD, "w") as f:
        f.write("\n".join(out))
    print("  wrote paper/figures/NUMBERS.md")


# ---------------------------------------------------------------------------
STEPS = {
    "tab2": ("Table 2: main results at k=4 -> tables/main_results.tex", tab2),
    "fig1": ("Figure 1: macro-F1 vs prompt tokens -> f1_vs_tokens.pdf", fig1),
    "fig2": ("Figure 2: narrowing under imbalance, Per-Class -> narrowing_imbalance.pdf", fig2),
    "tab3": ("Table 3: label renaming -> tables/relabel.tex", tab3),
    "tab4": ("Table 4: supervised vs pipelines -> tables/baselines.tex", tab4),
    "fig3": ("Figure 3: alpha -> alpha.pdf", fig3),
    "tabB": ("Tables B1-B11: full grids -> tables/grid_<variant>.tex", tabB),
    "tabC1": ("Table C1: narrowing comparisons, all variants -> tables/narrowing_all.tex", tabC1),
    "tabC2": ("Table C2: candidate-set statistics -> tables/set_stats.tex", tabC2),
    "figC1": ("Figure C1: Figure 2 for the Fixed variant -> narrowing_imbalance_fixed.pdf",
              lambda B: fig2(B, variant="fixed", name="narrowing_imbalance_fixed")),
    "tabD1": ("Table D1: per-model deltas -> tables/per_model.tex", tabD1),
    "tabE1": ("Table E1: embedding / classifier / alpha ablation -> tables/ablation.tex", tabE1),
    "tabF1": ("Table F1: larger models -> tables/large_models.tex", tabF1),
    "tabG1": ("Table G1: fixes vs breaks -> tables/fixes_breaks.tex", tabG1),
    "tabG2": ("Table G2: invalid outputs -> tables/invalid.tex, invalid_raw.tex", tabG2),
    "tabG3": ("Table G3: qualitative examples -> tables/examples.tex", tabG3),
    "tabH1": ("Table H1: imbalance class counts, relabel words -> tables/imbalance_counts.tex, relabel_words.tex", tabH1),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="build everything")
    p.add_argument("--only", type=lambda s: s.split(","), default=None,
                   help="comma-separated subset of: " + ", ".join(STEPS))
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--quick", action="store_true", help="500 bootstrap resamples (development)")
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    args = p.parse_args()
    if args.list:
        for k, (desc, _) in STEPS.items():
            print(f"{k:7s} {desc}")
        return
    steps = list(STEPS) if args.all or not args.only else args.only
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        sys.exit(f"unknown step(s): {unknown}; choose from {list(STEPS)}")
    B = 500 if args.quick else args.bootstrap
    print(f"bootstrap resamples: {B}")
    for s in steps:
        print(f"{s}: {STEPS[s][0]}")
        fd.CURRENT_STEP[0] = s
        STEPS[s][1](B)
    fd.CURRENT_STEP[0] = None
    write_numbers(B, steps)
    pend = fd.pending_messages()
    if pend:
        print(f"\n{len(pend)} pending slot(s) (listed at the top of NUMBERS.md)")


if __name__ == "__main__":
    main()
