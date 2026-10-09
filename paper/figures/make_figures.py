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
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "16")  # shared server: cap BLAS threads
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
            "sst", "sst-imb10", "sst-imb100", "sst-relabel", "semeval-18", "semeval-18-relabel",
            "go-emotions", "go-emotions-relabel", "ohsumed", "ohsumed-relabel"]
POOL_SIZES = [250, 500, 1000, 2000]
NAME = {"yahoo-answers": "Yahoo Answers", "sst": "SST-5", "semeval-18": "SemEval-18",
        "go-emotions": "GoEmotions", "ohsumed": "Ohsumed", "massive": "MASSIVE"}
MODEL_NAME = {"llama-3.2-3b": "Llama-3.2-3B", "ministral-3b": "Ministral-3B",
              "qwen-2.5-3b": "Qwen2.5-3B", "mistral-7b-v0.3": "Mistral-7B",
              "qwen-2.5-7b": "Qwen2.5-7B", "llama-3.1-8b": "Llama-3.1-8B",
              "mistral-nemo-2407": "Mistral-Nemo-12B", "qwen-2.5-32b": "Qwen2.5-32B"}
METHOD_NAME = {"zeroshot": "Zero-shot", "fewshot": "Few-shot", "cicle": "CICLe",
               "topk": "Top-$m$", "mass": "Prob. mass", "marginal": "Marginal CP",
               "oracle": "Oracle", "base": "MiniLM + LR", "finetuned": "RoBERTa-base",
               "massmatch": "Prob. mass (matched)", "margmatch": "Marginal CP (matched)"}
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
         "base": "#52514e", "finetuned": "#0b0b0b",
         "massmatch": "#1baf7a", "margmatch": "#4a3aa7"}  # matched rules share their rule's hue
MARKER = {"fewshot": "o", "cicle": "o", "topk": "s", "mass": "D", "marginal": "^", "oracle": "x",
          "zeroshot": "*", "massmatch": "D", "margmatch": "^"}
LSTYLE = {"fewshot": "-", "cicle": "-", "topk": "-", "mass": "-", "marginal": "--", "oracle": "--",
          "massmatch": "-", "margmatch": "--"}
DS_COLOR = {"yahoo-answers": "#2a78d6", "sst": "#eb6834", "semeval-18": "#1baf7a",
            "go-emotions": "#4a3aa7"}
DS_MARKER = {"yahoo-answers": "o", "sst": "s", "semeval-18": "D", "go-emotions": "^"}
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"
TEXTWIDTH, COLWIDTH = 6.5, 3.3

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 5.5,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.0,
    "pdf.fonttype": 42, "figure.dpi": 150, "axes.unicode_minus": True,
})
ERR = dict(capsize=2.5, elinewidth=1.0, capthick=1.0)
FS = [1.0]  # font/mark scale of the figure being drawn (1 / printed scale)


def pt(x):
    """A point size that prints at x pt after the manuscript scales the figure."""
    return x * FS[0]


def scaled_rc(printed_scale):
    """rc overrides so that a figure printed at `printed_scale` of its drawn size keeps
    8 pt ticks/legend, 9 pt axis labels, 1.6 pt lines and 5.5 pt markers."""
    f = 1.0 / printed_scale
    FS[0] = f
    return {"font.size": 8 * f, "axes.titlesize": 9 * f, "axes.labelsize": 9 * f,
            "legend.fontsize": 8 * f, "xtick.labelsize": 8 * f, "ytick.labelsize": 8 * f,
            "lines.linewidth": 1.6 * f, "lines.markersize": 5.5 * f, "axes.linewidth": 0.6 * f,
            "grid.linewidth": 0.5 * f, "xtick.major.width": 0.6 * f, "ytick.major.width": 0.6 * f,
            "xtick.major.size": 2.5 * f, "ytick.major.size": 2.5 * f}


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


def delta_tex(res, ci=True, n=False, stacked=False, inline=False):
    """Δ cell: bold when the 95% CI excludes zero, 'n.s.' otherwise.
    stacked: CI on a second line inside the cell (narrow appendix tables);
    inline: '+1.40 [+1.04, +1.76]' in one line, whole cell bold when significant."""
    if res is None:
        return "--"
    if inline:
        body = f"{res['mean']:+.2f} {ci_str(res)}"
        return r"\textbf{" + body + "}" if sig(res) else body
    body = f"{res['mean']:+.2f}"
    body = r"\textbf{" + body + "}" if sig(res) else body + r"\,\textsuperscript{n.s.}"
    if n:
        body += f" ({res['n']})"
    if ci and stacked:
        return (r"\begin{tabular}[c]{@{}c@{}}" + body + r"\\{\scriptsize " + ci_str(res) + "}"
                + r"\end{tabular}")
    if ci:
        body += r" {\scriptsize " + ci_str(res) + "}"
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
                  markersize=6 if method != "zeroshot" else 11)


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

    # -- main-text body: macro-F1 only, one number per cell -------------------
    k0_var = sorted({kk[1] for d in CORE for kk in fd.variant(d).select(method="cicle", k=0)})
    if not k0_var:
        pending(sec, "k=0 row (no cicle runs with k=0 yet)")
    main = [f1_row("Zero-shot", "zeroshot", tokens=False)]
    if k0_var:
        main.append(f1_row("Candidate set only ($k$=0)", "cicle", k0_var[0], 0, tokens=False))
    main += [r"\midrule",
             f1_row("Few-shot, Fixed", "fewshot", "fixed", 4, tokens=False),
             f1_row("CICLe, Fixed", "cicle", "fixed", 4, tokens=False),
             r"\midrule",
             f1_row("Few-shot, Per-Class", "fewshot", "pc", 4, tokens=False),
             f1_row("CICLe, Per-Class", "cicle", "pc", 4, tokens=False),
             r"\midrule",
             f1_row("MiniLM + LR", "base", tokens=False),
             f1_row("RoBERTa-base", "finetuned", tokens=False)]
    main = [r.replace(r"\textsuperscript{$k$=1}", r"$^{\dagger}$") for r in main]
    write_table("body_main_results", "l" + "c" * len(CORE), head, main,
                comment="Table 2 body (main text). Macro-F1 (pp) at k=4, mean over 6 models x 3 seeds "
                        "(supervised rows over 3 seeds). dagger: Ohsumed Per-Class is run at k=1 only. "
                        "Prompt tokens are in tables/main_results.tex (appendix).")
    # -- main-text delta body: one line per cell, bold when the CI excludes 0 --
    drows = []
    for var in ("fixed", "pc"):
        cells = []
        for d in CORE:
            res = fd.paired_delta(d, "cicle", "fewshot", var, ks_for(d, var), B=B)
            if res is None:
                cells.append("--")
                pending(sec, f"Δ {d} {var}")
                continue
            cells.append(delta_tex(res, inline=True))
            note(sec, f"Δ CICLe − few-shot {d} {var} (all k): {delta_txt(res)}")
        drows.append(f"{VARIANT_NAME[var]} & " + " & ".join(cells) + r" \\")
    saved = []
    for d in CORE:
        k = 4 if 4 in ks_for(d, "pc") else ks_for(d, "pc")[0]
        tf, _ = fd.mean_metric(d, "fewshot", "pc", k, metric="mean_prompt_tokens")
        tc, _ = fd.mean_metric(d, "cicle", "pc", k, metric="mean_prompt_tokens")
        if tf is None or tc is None:
            saved.append("--")
            continue
        saved.append(f"{100 * (1 - tc / tf):.0f}\\%" + (r"$^{\dagger}$" if k != 4 else ""))
        note(sec, f"{d} tokens saved by CICLe Per-Class at k={k}: {100 * (1 - tc / tf):.1f}% "
                  f"({tf:,.0f} -> {tc:,.0f} tokens per call)")
    drows.append(r"\midrule")
    drows.append("Tokens saved & " + " & ".join(saved) + r" \\")
    write_table("body_main_delta", "l" + "c" * len(CORE), head, drows,
                comment="Table 2b body (main text). Paired Δ CICLe − few-shot (pp) with the 95% "
                        "bootstrap CI over test instances, pooled over all k: 72 (model, seed, k) "
                        "pairs per cell; Ohsumed Fixed 36, Ohsumed Per-Class 18 (k=1 only, dagger). "
                        "Bold: interval excludes zero. Last row: 1 - CICLe PC tokens / few-shot PC "
                        "tokens per LLM call at k=4 (Ohsumed k=1).")
    # -- appendix version with prompt tokens --------------------------------
    rows.append(f1_row("MiniLM + LR (no LLM)", "base", tokens=False))
    rows.append(f1_row("RoBERTa-base, fine-tuned", "finetuned", tokens=False))
    rows.append(r"\midrule")
    rows.append(f1_row("Zero-shot", "zeroshot"))
    if k0_var:
        rows.append(f1_row("Candidate set only ($k$=0)", "cicle", k0_var[0], 0))
    rows.append(f1_row("Few-shot, Fixed", "fewshot", "fixed", 4))
    rows.append(f1_row("CICLe, Fixed", "cicle", "fixed", 4))
    rows.append(f1_row("Few-shot, Per-Class", "fewshot", "pc", 4))
    rows.append(f1_row("CICLe, Per-Class", "cicle", "pc", 4))
    rows.append(r"\midrule")
    for var in ("fixed", "pc"):
        dcells = []
        for d in CORE:
            res = fd.paired_delta(d, "cicle", "fewshot", var, ks_for(d, var), B=B)
            dcells.append(delta_tex(res, stacked=True) if res else "--")
        rows.append(f"$\\Delta$ {VARIANT_NAME[var]} & " + " & ".join(dcells) + r" \\")
    write_table("main_results", "l" + "c" * len(CORE), head, rows,
                comment="Table 2 (appendix version with tokens). Macro-F1 (pp) at k=4; LLM rows mean "
                        "over 6 models x 3 seeds, small number = mean prompt tokens per LLM call; Δ rows "
                        "= CICLe − few-shot pooled over all k (72 pairs; Ohsumed Fixed 36, Ohsumed "
                        "Per-Class 18 at k=1 only), 95% paired bootstrap CI over test instances. "
                        "Ohsumed Per-Class cells show k=1 (marked).")


# ---------------------------------------------------------------------------
# Figure 1 -- macro-F1 against prompt tokens (fig_main_tokens.pdf, 2 x 3, legend in the 6th cell)
# ---------------------------------------------------------------------------
def fig1(B):
    with plt.rc_context(scaled_rc(0.7)):  # printed at 0.7 text width
        _fig1(B)
    FS[0] = 1.0


def _fig1(B):
    sec = "Figure 1 (macro-F1 vs prompt tokens)"
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.9))
    axes = axes.ravel()
    for ax, d in zip(axes, CORE):
        ks = ks_for(d)
        series = [("zeroshot", None, [None]), ("fewshot", "fixed", ks), ("cicle", "fixed", ks),
                  ("fewshot", "pc", ks_for(d, "pc")), ("cicle", "pc", ks_for(d, "pc"))]
        for method, var, kk in series:
            xs, ys, kl = [], [], []
            for k in kk:
                r = fd.mean_ci(d, method, var, k, B=B)
                t, _ = fd.mean_metric(d, method, var, k, metric="mean_prompt_tokens")
                if r is None or t is None:
                    pending(sec, f"{d} {method} {var} k={k}")
                    continue
                xs.append(t); ys.append(r["mean"]); kl.append(k)
                note(sec, f"{d} {method} {var or ''} k={k or 0}: F1 {r['mean']:.2f} "
                          f"[{r['lo']:.2f}, {r['hi']:.2f}], tokens {t:,.0f} (n={r['n']})")
            if not xs:
                continue
            c = COLOR[method]
            filled = var != "fixed"
            ax.plot(xs, ys, color=c, marker=MARKER[method], ls="-" if len(xs) > 1 else "none",
                    mfc=c if filled else "white", mec=c, markersize=11 if method == "zeroshot" else 5.5,
                    zorder=3)
            if method == "cicle" and var == "pc" and len(xs) > 1:
                ax.annotate(f"$k$={kl[0]}", (xs[0], ys[0]), textcoords="offset points",
                            xytext=(3, -10 * FS[0]), fontsize=pt(8), color="#52514e", ha="left", va="top")
                ax.annotate(f"$k$={kl[-1]}", (xs[-1], ys[-1]), textcoords="offset points",
                            xytext=(5, -3), fontsize=pt(8), color="#52514e", ha="left", va="top")
        refs = []
        for ref, ls, lab in (("base", ":", "MiniLM + LR"), ("finetuned", "-.", "RoBERTa-base")):
            v, _ = fd.mean_metric(d, ref)
            if v is None:
                pending(sec, f"{d} {ref}")
                continue
            ax.axhline(v, color=COLOR[ref], ls=ls, lw=pt(1.1), zorder=1)
            refs.append((v, lab))
            note(sec, f"{d} {ref}: {v:.2f}")
        ax.set_xscale("log")
        ax.set_xlim(140, 12000)
        ax.set_xticks([200, 1000, 5000])
        ax.set_xticklabels(["200", "1k", "5k"])
        lo_y, hi_y = ax.get_ylim()
        pad = 0.05 * (hi_y - lo_y)
        if refs:
            ax.set_ylim(min(lo_y, min(v for v, _ in refs) - 2 * pad),
                        max(hi_y, max(v for v, _ in refs) + 2 * pad))
        lo_y, hi_y = ax.get_ylim()
        for v, lab in refs:  # label at the left edge, below the line when it is near the top
            above = v < hi_y - 0.12 * (hi_y - lo_y)
            ax.annotate(lab, xy=(0.02, v), xycoords=("axes fraction", "data"), fontsize=pt(8),
                        color="#52514e", ha="left", va="bottom" if above else "top",
                        xytext=(0, 2 if above else -2), textcoords="offset points")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(NAME[d], fontsize=pt(9), pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
    handles = [legend_handle("zeroshot"), legend_handle("fewshot", "fixed", "Few-shot, Fixed"),
               legend_handle("cicle", "fixed", "CICLe, Fixed"),
               legend_handle("fewshot", "pc", "Few-shot, Per-Class"),
               legend_handle("cicle", "pc", "CICLe, Per-Class"),
               Line2D([], [], color=COLOR["base"], ls=":", lw=pt(1.1), label="MiniLM + LR"),
               Line2D([], [], color=COLOR["finetuned"], ls="-.", lw=pt(1.1), label="RoBERTa-base")]
    axes[-1].axis("off")
    axes[-1].legend(handles=handles, loc="center", ncol=1, fontsize=pt(8.5), handlelength=2.2,
                    labelspacing=0.8, borderaxespad=0, handletextpad=0.6)
    fig.supxlabel("mean prompt tokens per LLM call (log scale)", fontsize=pt(9), y=0.012)
    fig.supylabel("macro-F1 (pp)", fontsize=pt(9), x=0.01)
    fig.subplots_adjust(wspace=0.28, hspace=0.36, left=0.095, right=0.99, top=0.95, bottom=0.12)
    save_fig(fig, "fig_main_tokens")


# ---------------------------------------------------------------------------
# Figure 2 -- narrowing under a long-tailed labelled pool
#   fig_main_narrowing.pdf      (a) + (c), Per-Class
#   fig_app_narrowing_extra.pdf (b) + (d), Per-Class
#   fig_app_narrowing_fixed.pdf (a)-(d), Fixed
# ---------------------------------------------------------------------------
def rank_map(ds, seed, imb):
    counts = fd.class_counts(ds, seed, imb)
    if counts is None:
        return None, None
    return {lab: i for i, (lab, _) in enumerate(counts)}, counts


def _rank_ticks(ax, n):
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([str(i) if (i == 1 or i % 2 == 0 or i == n) else "" for i in range(1, n + 1)])


MATCHED = ["topk", "massmatch", "margmatch"]          # rules at CICLe's mean set size
UNMATCHED = ["mass", "marginal"]
EXTRA_SEEDS = [45, 46]


def present_seeds(tag):
    """Seeds with a results directory for a variant (42-44, plus 45/46 when they land)."""
    root = os.path.join(fd.RESULTS, tag)
    return sorted(int(x[5:]) for x in os.listdir(root) if x.startswith("seed-")) if os.path.isdir(root) else []


def extra_seeds_complete(tag, methods, variant, k=4):
    """Extra seeds (45, 46) for which every small model has every method at (variant, k)."""
    v = fd.variant(tag)
    return [s for s in EXTRA_SEEDS if s in present_seeds(tag)
            and all(v.get(m, variant, k, mdl, seed=s) is not None for m in methods for mdl in SMALL)]


def fig2_cells(tag, comp, variant):
    """Cells for the imbalance comparison: k in {1,4} on seeds 42-44, plus k=4 on
    seeds 45/46 when those runs are complete for both methods."""
    cells = fd.grid_cells([variant], SMALL, [1, 4], SEEDS)
    extra = extra_seeds_complete(tag, ["cicle", comp], variant)
    return cells + fd.grid_cells([variant], SMALL, [4], extra), extra


def set_seeds(tag):
    """Seeds whose candidate sets enter coverage statistics (those with results)."""
    return [s for s in present_seeds(tag) if s in SEEDS + EXTRA_SEEDS] or SEEDS


def cpu_set_stats(tag, method, seeds=None):
    """Per-seed set statistics of a rule on the base dataset of `tag` (CPU sets)."""
    ds, imb = fd.base_tag(tag)
    out = []
    for s in seeds or set_seeds(tag):
        S = fd.cpu_sets(ds, imb, s)
        if S is not None:
            out.append((s, S, fd.set_stats(S, method)))
    return out


def panel_a(ax, d, variant, B, sec, drawn, comps=("fewshot",) + tuple(MATCHED)):
    """Paired Δ CICLe − alternative against pool imbalance; oracle − few-shot dashed."""
    for comp in comps:
        xs, ys, lo, hi = [], [], [], []
        for i, imb in enumerate((1, 10, 100)):
            tag = imb_tag(d, imb)
            cells, extra = fig2_cells(tag, comp, variant)
            res = fd.paired_delta(tag, "cicle", comp, variant, None, cells=cells, B=B)
            if res is None:
                pending(sec, f"(a) {tag} CICLe − {comp} {variant}")
                continue
            xs.append(i); ys.append(res["mean"]); lo.append(res["mean"] - res["lo"])
            hi.append(res["hi"] - res["mean"])
            note(sec, f"(a) {tag} {variant} k∈{{1,4}}{' + k=4 seeds ' + str(extra) if extra else ''} "
                      f"CICLe − {comp}: {delta_txt(res)}")
        if xs:
            c = COLOR[comp]
            drawn.add(comp)
            ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=MARKER[comp], ls=LSTYLE[comp],
                        mfc=c if variant == "pc" else "white", mec=c, zorder=3, **ERR)
    xs, ys = [], []
    for i, imb in enumerate((1, 10, 100)):
        tag = imb_tag(d, imb)
        cells, extra = fig2_cells(tag, "oracle", variant)
        cells = [c for c in cells if c[3] in SEEDS] if not extra_seeds_complete(tag, ["fewshot", "oracle"], variant) else cells
        res = fd.paired_delta(tag, "oracle", "fewshot", variant, None, cells=cells, B=B)
        if res is None:
            pending(sec, f"(a) {tag} oracle − few-shot {variant} (ceiling)")
            continue
        xs.append(i); ys.append(res["mean"])
        note(sec, f"(a) {tag} {variant} oracle − few-shot (ceiling): {delta_txt(res)}")
    if xs:
        drawn.add("oracle")
        ax.plot(xs, ys, color=COLOR["oracle"], ls="--", marker="x", zorder=2)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"1$\times$", r"10$\times$", r"100$\times$"])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylabel("CICLe $-$ alternative (pp)")
    ax.set_xlabel("pool imbalance" if COMPACT[0] else "imbalance of the labelled pool")


def panel_b(ax, d, variant, B, sec, drawn,
            methods=("cicle", "topk", "massmatch", "margmatch", "mass", "marginal")):
    """Coverage against mean candidate-set size (CPU sets), marker size grows with
    imbalance; unmatched rules hollow."""
    for method in methods:
        pts = []
        for imb, size in ((1, 22), (10, 48), (100, 95)):
            tag = imb_tag(d, imb)
            st = cpu_set_stats(tag, method)
            if not st:
                pending(sec, f"(b) {tag} {method} candidate sets")
                continue
            cov = np.mean([x["cov"] for _, _, x in st])
            sz = np.mean([x["size"] for _, _, x in st])
            pts.append((sz, cov, size, imb))
            note(sec, f"(b) {tag} {method}: coverage {cov:.1f}%, mean set size {sz:.2f} "
                      f"({len(st)} seeds)")
        if not pts:
            continue
        c = COLOR[method]
        drawn.add(method)
        hollow = method in UNMATCHED
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=0.9,
                ls=":" if hollow else LSTYLE[method], zorder=2)
        for x, y, size, imb in pts:
            ax.scatter([x], [y], s=size, marker=MARKER[method], zorder=3,
                       facecolor="white" if hollow else c, edgecolor=c if hollow else "white",
                       linewidth=1.2 if hollow else 0.7)
        if method == "cicle":
            for x, y, size, imb in pts:
                if imb in (1, 100):
                    ax.annotate(f"{imb}$\\times$", (x, y), textcoords="offset points",
                                xytext=(0, -13 * FS[0]), fontsize=pt(8), color="#52514e", ha="center")
    ax.axhline(95, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_ylabel("coverage (%)" if COMPACT[0] else "coverage of the gold label (%)")
    ax.set_xlabel("mean set size" if COMPACT[0] else "mean candidate-set size")


def panel_c(ax, d, variant, B, sec, drawn, methods=("cicle",) + tuple(MATCHED)):
    """Per-class coverage at 100x by class-frequency rank in the pool (CPU sets,
    all seeds with results; aggregated by rank because the class order is a
    seed-dependent permutation)."""
    tag = imb_tag(d, 100)
    n = 0
    for method in methods:
        by_rank = defaultdict(list)
        st = cpu_set_stats(tag, method)
        for s, S, x in st:
            order = [lab for lab, _ in sorted(S["train_counts"].items(), key=lambda kv: -kv[1])]
            for lab, cov in x["per_class"].items():
                by_rank[order.index(lab)].append(cov)
        if not by_rank:
            pending(sec, f"(c) {tag} {method} candidate sets")
            continue
        xs = sorted(by_rank)
        ys = [np.mean(by_rank[r]) for r in xs]
        drawn.add(method)
        ax.plot([x + 1 for x in xs], ys, color=COLOR[method], marker=MARKER[method],
                ls=LSTYLE[method], mfc="white" if method in UNMATCHED and "massmatch" in methods else COLOR[method],
                zorder=3)
        note(sec, f"(c) {tag} {method} per-class coverage by rank (mean over {len(st)} seeds): "
                  + ", ".join(f"{x + 1}:{y:.0f}" for x, y in zip(xs, ys)))
        n = len(xs)
    ax.axhline(95, color=MUTED, lw=0.8, ls=":", zorder=1)
    if n:
        _rank_ticks(ax, n)
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel(r"class coverage, 100$\times$ (%)" if COMPACT[0] else r"coverage at 100$\times$ (%)")
    ax.set_xlabel("class rank" if COMPACT[0] else "class rank (frequent to rare)")


def panel_d(ax, d, variant, B, sec, drawn,
            methods=("fewshot", "cicle") + tuple(MATCHED) + ("oracle",)):
    """Per-class F1 at 100x, k = 4, by class-frequency rank (6 models x seeds 42-44)."""
    tag = imb_tag(d, 100)
    v = fd.variant(tag)
    n = 0
    for method in methods:
        by_rank = defaultdict(list)
        n_runs = 0
        for s in SEEDS:
            rank, counts = rank_map(d, s, 100)
            if rank is None:
                continue
            for m in SMALL:
                run = v.get(method, variant, 4, m, seed=s)
                if run is None:
                    continue
                n_runs += 1
                for lab, f in fd.per_class_f1(v, run).items():
                    by_rank[rank[lab]].append(f)
        if n_runs < len(SMALL) * len(SEEDS):
            pending(sec, f"(d) {tag} {method} {variant} k=4 ({n_runs}/{len(SMALL) * len(SEEDS)} runs)")
            continue
        xs = sorted(by_rank)
        ys = [np.mean(by_rank[x]) for x in xs]
        c = COLOR[method]
        drawn.add(method)
        ax.plot([x + 1 for x in xs], ys, color=c, marker=MARKER[method], ls=LSTYLE[method],
                mfc=c if variant == "pc" else "white", zorder=3)
        note(sec, f"(d) {tag} {method} {variant} k=4 per-class F1 by rank "
                  f"(mean over 6 models x 3 seeds): " + ", ".join(f"{x + 1}:{y:.1f}" for x, y in zip(xs, ys)))
        n = len(xs)
    if n:
        _rank_ticks(ax, n)
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel(r"class F1, 100$\times$, $k$=4 (pp)" if COMPACT[0] else r"per-class F1 at 100$\times$, $k$=4 (pp)")
    ax.set_xlabel("class rank" if COMPACT[0] else "class rank in the pool (frequent to rare)")


PANELS = {"a": panel_a, "b": panel_b, "c": panel_c, "d": panel_d}
COMPACT = [False]  # shorter axis labels for the four-column appendix figure


def narrowing_figure(B, variant, panels, name, sec, height, kwargs=None):
    """Rows Yahoo / SST-5, one column per panel letter, shared legend below.
    kwargs: {panel letter: dict of keyword arguments for that panel}."""
    datasets = ["yahoo-answers", "sst"]
    drawn = set()
    COMPACT[0] = len(panels) > 2
    kwargs = kwargs or {}
    fig, axes = plt.subplots(2, len(panels), figsize=(TEXTWIDTH, height), squeeze=False)
    for row, d in enumerate(datasets):
        for col, p in enumerate(panels):
            ax = axes[row, col]
            PANELS[p](ax, d, variant, B, sec, drawn, **kwargs.get(p, {}))
            if col == 0:
                ax.set_ylabel(f"{NAME[d]}\n" + ax.get_ylabel())
            if row == 0:
                ax.set_xlabel("")
    order = ["fewshot", "cicle", "topk", "massmatch", "margmatch", "mass", "marginal"]
    both = any(m in drawn for m in MATCHED[1:]) and any(m in drawn for m in UNMATCHED)
    handles = []
    for m in order:
        if m not in drawn:
            continue
        h = legend_handle(m, variant, METHOD_NAME[m] if not (both and m in UNMATCHED)
                          else METHOD_NAME[m] + " (unmatched)")
        if both and m in UNMATCHED:
            h.set_markerfacecolor("white"); h.set_linestyle(":")
        handles.append(h)
    if "oracle" in drawn:
        handles.append(Line2D([], [], color=COLOR["oracle"], ls="--", marker="x",
                              label="Oracle $-$ few-shot (ceiling, left)" if "a" in panels
                              else "Oracle"))
    ncol = 3 if len(handles) > 4 else len(handles)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.01),
               fontsize=pt(8.5), handlelength=2.2, columnspacing=1.4)
    legend_rows = -(-len(handles) // ncol)
    fig.subplots_adjust(wspace=0.36 if len(panels) <= 2 else 0.6, hspace=0.25,
                        left=0.12 if len(panels) <= 2 else 0.09, right=0.98, top=0.97,
                        bottom=0.10 + 0.045 * legend_rows * FS[0])
    COMPACT[0] = False
    save_fig(fig, name)


def fig2(B):
    with plt.rc_context(scaled_rc(0.8)):  # printed at 0.8 text width
        narrowing_figure(B, "pc", "ac", "fig_main_narrowing",
                         "Figure 2 (Per-Class; size-matched narrowing under imbalance, panels a and c)", 5.8)
    FS[0] = 1.0


def figAN(B):
    narrowing_figure(B, "pc", "bd", "fig_app_narrowing_extra",
                     "Figure 2 appendix extra (Per-Class; panels b and d)", 5.6)


def figAU(B):
    narrowing_figure(B, "pc", "ac", "fig_app_narrowing_unmatched",
                     "Figure 2 appendix (Per-Class; unmatched mass and marginal CP)", 5.6,
                     kwargs={"a": {"comps": ("fewshot", "topk", "mass", "marginal")},
                             "c": {"methods": ("cicle", "topk", "mass", "marginal")}})


def figC1(B):
    narrowing_figure(B, "fixed", "abcd", "fig_app_narrowing_fixed",
                     "Figure C1 (Fixed variant; panels a-d)", 5.2)


# ---------------------------------------------------------------------------
# Table 3 -- label renaming (tables/relabel.tex)
# ---------------------------------------------------------------------------
def tab3(B):
    sec = "Table 3 (label renaming, five datasets)"
    head = [r"Dataset & Labels & $k$ & FS F & CICLe F & FS PC & CICLe PC & "
            r"$\Delta$ F & $\Delta$ PC \\"]
    rows = []
    for d in CORE:
        variants = ("fixed", "pc") if d != "ohsumed" else ("fixed",)
        for tag, label in ((d, "original"), (d + "-relabel", "nonsense")):
            if not fd.has_results(tag):
                pending(sec, f"{tag} (no results directory)")
                continue
            # k = 0: candidate set only (CICLe Fixed column; nothing to compare against)
            v0, n0 = fd.mean_metric(tag, "cicle", "fixed", 0)
            if v0 is not None:
                rows.append(f"{NAME[d]} & {label} & 0 & -- & {num(v0)} & -- & -- & & \\\\")
                note(sec, f"{tag} cicle k=0 (candidate set only): {v0:.2f} (n={n0})")
            for k in (1, 4):
                cells = []
                for m, var in (("fewshot", "fixed"), ("cicle", "fixed"), ("fewshot", "pc"),
                               ("cicle", "pc")):
                    if var not in variants:
                        cells.append("--")
                        continue
                    val, n = fd.mean_metric(tag, m, var, k)
                    cells.append(num(val))
                    if val is None:
                        pending(sec, f"{tag} {m} {var} k={k}")
                    else:
                        note(sec, f"{tag} {m} {var} k={k}: {val:.2f} (n={n})")
                # matched-k paired delta in the row of k = 1 (over k in {1,4}, 36 pairs)
                dcells = []
                for var in ("fixed", "pc"):
                    if var not in variants or k != 1:
                        dcells.append("")
                        continue
                    res = fd.paired_delta(tag, "cicle", "fewshot", var, [1, 4], B=B)
                    dcells.append(r"\multirow{2}{*}{" + delta_tex(res, stacked=True) + "}")
                    if res is None:
                        pending(sec, f"Δ {tag} {var}")
                    else:
                        note(sec, f"Δ CICLe − few-shot {tag} {var} k∈{{1,4}}: {delta_txt(res)}")
                rows.append(f"{NAME[d]} & {label} & {k} & " + " & ".join(cells) + " & "
                            + " & ".join(dcells) + r" \\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    write_table("relabel", "llrcccccc", head, rows,
                comment="Table 3. Macro-F1 with the original class names and with every name "
                        "replaced by a nonsense word (same mapping for all seeds); mean over 6 "
                        "models x 3 seeds. k = 0: the candidate set alone, no examples. Δ columns: "
                        "CICLe − few-shot paired over k in {1,4} (36 pairs; needs \\usepackage{multirow}), "
                        "same k grid for original and renamed labels. Ohsumed: Fixed only.")


# ---------------------------------------------------------------------------
# Table 4 -- supervised baselines vs LLM pipelines (tables/baselines.tex)
# ---------------------------------------------------------------------------
def tab4(B):
    sec = "Table 4 (supervised vs pipelines)"
    cols = []
    for d in ("yahoo-answers", "sst"):
        for imb in (1, 10, 100):
            cols.append((imb_tag(d, imb), f"{NAME[d]} {imb}$\\times$"))
    for d in ("semeval-18", "go-emotions", "ohsumed"):
        cols.append((d, NAME[d]))
    for t, _ in cols:
        if not fd.has_results(t):
            pending(sec, f"column {t} (no results directory)")
    cols = [(t, l) for t, l in cols if fd.has_results(t)]
    values = {}  # (row, tag) -> value

    def fill(row, method, var=None, k=None, only_1x=False, **kw):
        for t, _ in cols:
            if only_1x and ("imb" in t):
                values[(row, t)] = None
                continue
            val, n = fd.mean_metric(t, method, var, k, **kw)
            if val is None and method in ("fewshot", "cicle") and t == "ohsumed" and var == "pc":
                val, n = fd.mean_metric(t, method, "fixed", k)  # Ohsumed: Fixed only
                values[(row, t)] = (val, "F")
            else:
                values[(row, t)] = (val, None) if val is not None else None
            if val is None and not only_1x:
                pending(sec, f"{t} {method} {var or ''} k={k or ''}")
            if val is not None:
                note(sec, f"{t} {method} {var or ''} {kw.get('emb', '')} k={k or ''}: {val:.2f} (n={n})")

    labels = ["MiniLM + LR", "RoBERTa-base, fine-tuned", "RoBERTa-large, fine-tuned", "Zero-shot",
              "Few-shot PC, $k$=4", "CICLe PC, $k$=4"]
    fill(0, "base"); fill(1, "finetuned"); fill(2, "finetuned", emb="roberta-large")
    fill(3, "zeroshot", only_1x=True); fill(4, "fewshot", "pc", 4); fill(5, "cicle", "pc", 4)
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
    # transposed layout: one row per dataset variant, one column per method
    short = ["MiniLM + LR", "RoBERTa-base", "RoBERTa-large", "Zero-shot", "Few-shot PC $k$=4",
             "CICLe PC $k$=4"]

    def row_cells(t, idx, with_best):
        vals = [values.get((i, t)) for i in idx]
        present = [v[0] for v in vals if v] + ([best[t][0]] if with_best and best[t] else [])
        top = max(present) if present else None
        cells = []
        for v in vals:
            if v is None:
                cells.append("--")
                continue
            s = num(v[0]) + (r"$^{\mathrm{F}}$" if v[1] else "")
            cells.append(r"\textbf{" + s + "}" if v[0] >= top - 1e-9 else s)
        if with_best:
            if best[t]:
                s = num(best[t][0])
                s = r"\textbf{" + s + "}" if best[t][0] >= top - 1e-9 else s
                cells.append(s + r" {\scriptsize " + best[t][1].replace("CICLe ", "C-").replace("FS ", "FS-").replace(" $k$=", "-") + "}")
            else:
                cells.append("--")
        return cells

    main_idx = [0, 1, 2, 4, 5]
    heads = ["MiniLM+LR", "RoB-base", "RoB-large", "Zero-shot", "FS PC", "CICLe PC"]
    head = [" & " + " & ".join(heads[i] for i in main_idx) + r" \\"]
    rows = []
    for t, label in cols:
        label = (label.replace("Yahoo Answers", "Yahoo").replace("SemEval-18", "SemEval")
                 .replace("GoEmotions", "GoEmo"))
        rows.append(label + " & " + " & ".join(row_cells(t, main_idx, False)) + r" \\")
        if t in ("yahoo-answers-imb100", "sst-imb100"):
            rows.append(r"\midrule")
    write_table("body_main_supervised", "l" + "c" * len(main_idx), head, rows,
                comment="Table 4 body (main text). Macro-F1 (pp): supervised columns mean over 3 "
                        "seeds, LLM columns over 6 models x 3 seeds, Per-Class k=4 (Ohsumed: Fixed, "
                        "marked F). The test sample is the same at every imbalance level. Bold = best "
                        "per row. Pool sizes: tables/pool_size.tex.")
    app_idx = [0, 1, 2, 3, 4, 5]
    head = [" & " + " & ".join(short[i] for i in app_idx) + r" & best LLM pipeline \\"]
    rows = []
    for t, label in cols:
        rows.append(label + " & " + " & ".join(row_cells(t, app_idx, True)) + r" \\")
        if t in ("yahoo-answers-imb100", "sst-imb100"):
            rows.append(r"\midrule")
    write_table("baselines", "l" + "c" * (len(app_idx) + 1), head, rows,
                comment="Table 4 (appendix version). As the main body plus zero-shot (1x only; it does "
                        "not depend on the pool) and the best LLM pipeline (argmax over method, "
                        "variant, k of the 18-run mean; name in small type). Bold = best per row.")
    # per-seed RoBERTa / LR values for the appendix version
    for t, _ in cols:
        for method, kw in (("base", {}), ("finetuned", {}), ("finetuned", {"emb": "roberta-large"})):
            vals = fd.metric_values(t, method, **kw)
            if vals:
                note(sec, f"{t} {method} {kw.get('emb', '')} per seed: "
                          + ", ".join(f"{s}: {v:.1f}" for (_, s), v in sorted(vals.items())))


# ---------------------------------------------------------------------------
# Figure 3 -- choosing alpha (alpha.pdf)
# ---------------------------------------------------------------------------
def fig3(B):
    sec = "Figure 3 (alpha; Llama-3.1-8B + Llama-3.2-3B, 24 cells)"
    fig, (top, mid, bot) = plt.subplots(3, 1, figsize=(COLWIDTH, 5.0), sharex=True,
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
    mid.text(0.03, 0.08, r"dotted: $1-\alpha$", transform=mid.transAxes, fontsize=8, color="#52514e")
    mid.set_ylabel("coverage (%)")
    bot.set_ylabel("no LLM call (%)")
    bot.set_xscale("log"); bot.set_xticks(ALPHAS); bot.set_xticklabels([str(a) for a in ALPHAS])
    bot.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    bot.set_xlabel(r"miscoverage level $\alpha$")
    top.legend(ncol=2, loc="upper left", handletextpad=0.4, fontsize=8)
    fig.subplots_adjust(hspace=0.15, left=0.17, right=0.98, top=0.98, bottom=0.09)
    save_fig(fig, "fig_app_alpha")


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
        head = [r"Method & Retr. & $k$ & runs & F1 & acc. & inv.\ (\%) & shots & tokens \\"]
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
    rows = []
    ns = set()
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
                    if not (tag.startswith("ohsumed") and var == "pc"):
                        pending(sec, f"{tag} {var} vs {comp}")
                    continue
                any_cell = True
                ns.add((comp == "fewshot", res["n"]))
                cells.append(delta_tex(res, stacked=True))
                note(sec, f"{tag} {var} CICLe − {comp}: {delta_txt(res)}")
            if any_cell:
                rows.append((f"{tag_name(tag)} & {VARIANT_NAME[var]} & ", cells))
    comment = ("Paired Δ macro-F1 (pp), CICLe minus alternative, 95% bootstrap CI over test "
               "instances; vs few-shot pools all k (72 cells on the four main sets, 36 elsewhere, "
               "18 for Ohsumed Per-Class), the other comparators k in {1,4} (36 cells, Ohsumed "
               "Per-Class 18). Bold: CI excludes 0.")
    write_table("narrowing_all_a", "llccc",
                [r"Variant & Retrieval & vs.\ few-shot & vs.\ top-$m$ & vs.\ prob.\ mass \\"],
                [lab + " & ".join(c[:3]) + r" \\" for lab, c in rows], comment="Table C1a. " + comment)
    write_table("narrowing_all_b", "llcc",
                [r"Variant & Retrieval & vs.\ marginal CP & vs.\ oracle \\"],
                [lab + " & ".join(c[3:]) + r" \\" for lab, c in rows if any(x != "--" for x in c[3:])],
                comment="Table C1b. " + comment)


# ---------------------------------------------------------------------------
# Appendix C2 -- candidate-set statistics (tables/set_stats.tex)
# ---------------------------------------------------------------------------
def tabC2(B):
    sec = "Table C2 (candidate-set statistics)"
    head = [r"Variant & Method & coverage (\%) & mean size & singleton (\%) & "
            r"min.\ class cov.\ (\%) \\"]
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
    for var in ("fixed", "pc"):
        head = ["Model & " + " & ".join(NAME[d] for d in CORE) + r" \\"]
        rows = []
        for m in SMALL:
            cells = []
            for d in CORE:
                if var == "pc" and d == "ohsumed" and False:
                    cells.append("--"); continue
                res = fd.paired_delta(d, "cicle", "fewshot", var, ks_for(d, var), models=[m], B=B)
                if res is None:
                    cells.append("--"); pending(sec, f"{m} {d} {var}")
                    continue
                cells.append(delta_tex(res, stacked=True))
                note(sec, f"{MODEL_NAME[m]} {d} {var}: {delta_txt(res)}")
            rows.append(MODEL_NAME[m] + " & " + " & ".join(cells) + r" \\")
        write_table(f"per_model_{var}", "l" + "c" * len(CORE), head, rows,
                    comment=f"Table D1 ({VARIANT_NAME[var]}). CICLe minus few-shot per model, all k "
                            "pooled (12 pairs; Ohsumed Fixed 6, Ohsumed Per-Class 3 at k=1), 95% paired "
                            "bootstrap CI over test instances. Bold: CI excludes 0.")


# ---------------------------------------------------------------------------
# Appendix E1 -- embedding / classifier / alpha ablation (tables/ablation.tex)
# ---------------------------------------------------------------------------
def tabE1(B):
    sec = "Table E1 (ablation; Llama-3.1-8B + Llama-3.2-3B, 24 cells)"
    settings = [("contriever", "lr", 0.05), ("minilm", "lr", 0.05), ("minilm", "svm", 0.05),
                ("tfidf", "lr", 0.05), ("minilm", "lr", 0.01), ("minilm", "lr", 0.10),
                ("minilm", "lr", 0.20)]
    head = [r"Dataset & Setting & CICLe & few-shot & $\Delta$ & "
            r"cov.\ (\%) & set size & no LLM (\%) \\"]
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
            rows.append(f"{NAME[d] if first else ''} & {emb}/{clf.upper()}, $\\alpha$={a:.2f} & "
                        f"{res['mean_a']:.1f} & {res['mean_b']:.1f} & {delta_tex(res, stacked=True)} & "
                        f"{cov:.1f} & {size:.2f} & {skip:.1f} \\\\")
            first = False
            note(sec, f"{d} {emb}/{clf}/alpha={a}: CICLe {res['mean_a']:.2f}, few-shot "
                      f"{res['mean_b']:.2f}, Δ {delta_txt(res)}; coverage {cov:.1f}%, size {size:.2f}, "
                      f"no LLM call {skip:.1f}%")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    write_table("ablation", "llrrcrrr", head, rows,
                comment="Table E1. Llama-3.1-8B and Llama-3.2-3B, both retrieval variants, k in {1,4}, "
                        "3 seeds (24 cells). Few-shot uses the same embedding for retrieval. "
                        "Δ = CICLe − few-shot, paired bootstrap CI.")


# ---------------------------------------------------------------------------
# Appendix F1 -- larger models (tables/large_models.tex)
# ---------------------------------------------------------------------------
def tabF1(B):
    sec = "Table F1 (12B / 32B models)"
    head = [r"Model & Dataset & seeds & Zero-shot & \multicolumn{2}{c}{Fixed} & "
            r"\multicolumn{2}{c}{Per-Class} \\",
            r" & & & & $k$=1 & $k$=4 & $k$=1 & $k$=4 \\"]
    rows, drows = [], []
    for m in LARGE:
        for d in CORE:
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
            dcells = []
            for var in ("fixed", "pc"):
                res = fd.paired_delta(d, "cicle", "fewshot", var, [1, 4], models=[m], B=B,
                                      allow_partial=True)
                dcells.append(delta_tex(res, n=True, stacked=True))
                if res is not None:
                    note(sec, f"{MODEL_NAME[m]} {d} Δ {var}: {delta_txt(res)}")
            rows.append(f"{MODEL_NAME[m]} & {NAME[d]} & {len(seeds)} & "
                        + " & ".join(cells) + r" \\")
            drows.append(f"{MODEL_NAME[m]} & {NAME[d]} & " + " & ".join(dcells) + r" \\")
        rows.append(r"\midrule"); drows.append(r"\midrule")
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
        rows.append(r"\multicolumn{8}{l}{\emph{Smallest models with CICLe against the largest "
                    r"model without examples (seed 42)}} \\")
        rows.extend(r"\multicolumn{8}{l}{" + e.replace(r" \\", "").replace(" & ", "; ") + r"} \\"
                    for e in extra)
    write_table("large_models", "llrccccc", head, rows,
                comment="Table F1a. Mistral-Nemo-12B and Qwen2.5-32B: macro-F1, mean over the number "
                        "of seeds given (seed ids in NUMBERS.md), cells 'few-shot / CICLe'.")
    if drows and drows[-1] == r"\midrule":
        drows.pop()
    write_table("large_models_delta", "llcc",
                [r"Model & Dataset & $\Delta$ Fixed & $\Delta$ Per-Class \\"], drows,
                comment="Table F1b. Δ = CICLe − few-shot over k in {1,4} for the 12B / 32B models, "
                        "paired bootstrap CI over test instances, (n) = cells (2 per seed).")


# ---------------------------------------------------------------------------
# Appendix G1 -- fixes and breaks (tables/fixes_breaks.tex)
# ---------------------------------------------------------------------------
def tabG1(B):
    sec = "Table G1 (fixes vs breaks, k=4)"
    head = [r"Variant & Retr. & pairs & fixed & broken & both right & both wrong \\"]
    head2 = [r"Variant & Retr. & \multicolumn{3}{c}{gold outside the set} & "
             r"\multicolumn{3}{c}{few-shot answer outside the set} \\",
             r" & & share & fixed & broken & share & fixed & broken \\"]
    rows, rows2 = [], []
    tags = ["yahoo-answers", "sst", "semeval-18", "go-emotions", "ohsumed", "yahoo-answers-imb100",
            "sst-imb100", "yahoo-answers-relabel", "sst-relabel", "semeval-18-relabel",
            "go-emotions-relabel", "ohsumed-relabel"]
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
                if not (tag.startswith("ohsumed") and var == "pc"):
                    pending(sec, f"{tag} {var}")
                continue
            n = agg["n"]
            pct = lambda x, d=n: 100 * x / max(d, 1)  # noqa: E731
            rows.append(f"{tag_name(tag)} & {VARIANT_NAME[var]} & {n:,} & {pct(agg['fix']):.1f} & "
                        f"{pct(agg['brk']):.1f} & {pct(agg['both']):.1f} & {pct(agg['neither']):.1f} \\\\")
            rows2.append(f"{tag_name(tag)} & {VARIANT_NAME[var]} & "
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
    write_table("fixes_breaks", "llrrrrr", head, rows,
                comment="Table G1a. CICLe k=4 vs few-shot k=4, same retrieval variant, 6 models x 3 "
                        "seeds x 1,000 instances; % of pairs. fixed = CICLe right & few-shot wrong; "
                        "broken = the reverse.")
    write_table("fixes_breaks_sets", "llrrrrrr", head2, rows2,
                comment="Table G1b. Same pairs split by the candidate set: share of pairs whose gold "
                        "label (left) or few-shot answer (right) lies outside CICLe's set, then fixed "
                        "and broken as % of that subset.")


# ---------------------------------------------------------------------------
# Appendix G2 -- invalid outputs (tables/invalid.tex, tables/invalid_raw.tex)
# ---------------------------------------------------------------------------
def tabG2(B):
    sec = "Table G2 (invalid outputs)"
    head = ["Model & Method & " + " & ".join(NAME[d] for d in CORE) + r" \\"]
    rows = []
    raw = defaultdict(Counter)
    for m in SMALL + LARGE:
        per_method = {}
        for d in CORE:
            v = fd.variant(d)
            vals = defaultdict(list)
            for key, r in v.select(models=[m]).items():
                if key[0] in ("zeroshot", "fewshot", "cicle"):
                    vals[key[0]].append(100 * r["metrics"]["invalid_rate"])
                    raw[m].update(r["invalid_raw"])
            for meth in ("zeroshot", "fewshot", "cicle"):
                if vals[meth]:
                    per_method.setdefault(meth, []).append(f"{np.mean(vals[meth]):.1f} ({np.max(vals[meth]):.1f})")
                    note(sec, f"{MODEL_NAME[m]} {d} {meth}: mean {np.mean(vals[meth]):.1f}%, "
                              f"max {np.max(vals[meth]):.1f}% over {len(vals[meth])} runs")
                else:
                    per_method.setdefault(meth, []).append("--")
        for i, meth in enumerate(("zeroshot", "fewshot", "cicle")):
            rows.append(f"{MODEL_NAME[m] if i == 0 else ''} & {METHOD_NAME[meth]} & "
                        + " & ".join(per_method.get(meth, ["--"] * len(CORE))) + r" \\")
        rows.append(r"\addlinespace")
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()
    write_table("invalid", "ll" + "c" * len(CORE), head, rows,
                comment="Table G2. Invalid (unparseable) outputs in %, per model, method and dataset, "
                        "reference setting: mean (maximum) over the (variant, k, seed) runs.")
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
# Pool size -- table (tables/pool_size.tex) and figure (pool_size.pdf)
# ---------------------------------------------------------------------------
def pool_tag(d, n):
    return d if n == 2000 else f"{d}-n{n}"


POOL_SERIES = [("base", {}, "MiniLM + LR"), ("finetuned", {}, "RoBERTa-base"),
               ("finetuned", {"emb": "roberta-large"}, "RoBERTa-large"),
               ("fewshot", {"variant_name": "pc", "k": 4}, "Few-shot PC, $k$=4"),
               ("cicle", {"variant_name": "pc", "k": 4}, "CICLe PC, $k$=4")]


def pool_values(sec):
    """(dataset, n, series label) -> (value, n_runs); best pipeline under 'best'."""
    out = {}
    for d in ("yahoo-answers", "sst"):
        for n in POOL_SIZES:
            tag = pool_tag(d, n)
            if not fd.has_results(tag):
                pending(sec, f"{tag} (no results directory)")
                continue
            for method, kw, label in POOL_SERIES:
                val, nr = fd.mean_metric(tag, method, **kw)
                if val is None:
                    pending(sec, f"{tag} {label}")
                else:
                    out[(d, n, label)] = (val, nr)
                    per_seed = ""
                    if method in ("base", "finetuned"):
                        vs = fd.metric_values(tag, method, **kw)
                        per_seed = "; per seed " + ", ".join(f"{s}: {x:.1f}" for (_, s), x in sorted(vs.items()))
                    note(sec, f"{tag} {label}: {val:.2f} (n={nr}){per_seed}")
            cand = []
            for m in ("fewshot", "cicle"):
                for var in ("fixed", "pc"):
                    for k in (1, 2, 4, 8):
                        val, _ = fd.mean_metric(tag, m, var, k)
                        if val is not None:
                            cand.append((val, f"{'CICLe' if m == 'cicle' else 'FS'} "
                                              f"{'F' if var == 'fixed' else 'PC'} $k$={k}"))
            if cand:
                out[(d, n, "best")] = max(cand)
                note(sec, f"{tag} best LLM pipeline: {max(cand)[0]:.2f} ({max(cand)[1]})")
    return out


def tabP(B):
    sec = "Table P (pool size)"
    vals = pool_values(sec)
    labels = [s[2] for s in POOL_SERIES]
    head = [r"Dataset & pool & " + " & ".join(l.replace("Few-shot PC, $k$=4", "FS PC").replace("CICLe PC, $k$=4", "CICLe PC") for l in labels) + r" & best pipeline \\"]
    rows = []
    for d in ("yahoo-answers", "sst"):
        for n in POOL_SIZES:
            cells = [vals.get((d, n, lab)) for lab in labels]
            best = vals.get((d, n, "best"))
            present = [c[0] for c in cells if c] + ([best[0]] if best else [])
            if not present:
                continue
            top = max(present)
            tex = [(r"\textbf{" + num(c[0]) + "}" if c[0] >= top - 1e-9 else num(c[0])) if c else "--"
                   for c in cells]
            if best:
                b = num(best[0])
                tex.append((r"\textbf{" + b + "}" if best[0] >= top - 1e-9 else b)
                           + r" {\scriptsize " + best[1] + "}")
            else:
                tex.append("--")
            rows.append(f"{NAME[d] if n == POOL_SIZES[0] else ''} & {n:,} & " + " & ".join(tex) + r" \\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    write_table("pool_size", "lr" + "c" * (len(labels) + 1), head, rows,
                comment="Table P. Macro-F1 against the size of the labelled pool (train + calibration; "
                        "the base classifier, retrieval pool and fine-tuned models all see 80% of it). "
                        "Supervised rows mean over 3 seeds, LLM rows over 6 models x 3 seeds; the test "
                        "sample is the same at every pool size. Best pipeline: argmax over (method, "
                        "variant, k) of the 18-run mean (k in {1,4} below 2,000). Bold = best per row.")


def figP(B):
    with plt.rc_context(scaled_rc(0.9)):  # margin for the tight bounding box at column width
        _figP(B)
    FS[0] = 1.0


def _figP(B):
    sec = "Figure P (macro-F1 vs pool size)"
    vals = pool_values(sec)
    style = {"MiniLM + LR": dict(color=COLOR["base"], ls=":", marker="v"),
             "RoBERTa-base": dict(color=COLOR["finetuned"], ls="-.", marker="^"),
             "RoBERTa-large": dict(color=COLOR["finetuned"], ls="--", marker="D", mfc="white"),
             "Few-shot PC, $k$=4": dict(color=COLOR["fewshot"], ls="-", marker="o"),
             "CICLe PC, $k$=4": dict(color=COLOR["cicle"], ls="-", marker="o")}
    fig, axes = plt.subplots(2, 1, figsize=(COLWIDTH, 5.6), sharex=True)
    for ax, d in zip(axes, ("yahoo-answers", "sst")):
        for lab, st in style.items():
            pts = [(n, vals[(d, n, lab)][0]) for n in POOL_SIZES if (d, n, lab) in vals]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], label=lab, **st)
        zs, _ = fd.mean_metric(d, "zeroshot")
        if zs is not None:
            ax.axhline(zs, color=COLOR["zeroshot"], lw=0.8, ls=(0, (1, 2)))
            ax.annotate("zero-shot", xy=(0.02, zs), xycoords=("axes fraction", "data"), fontsize=pt(8),
                        color="#52514e", va="bottom", xytext=(0, 2), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xticks(POOL_SIZES)
        ax.set_xticklabels([f"{n:,}" for n in POOL_SIZES])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(NAME[d], fontsize=pt(9), pad=4)
        ax.set_ylabel("macro-F1 (pp)")
    axes[1].set_xlabel("labelled pool size (train + calibration)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=pt(8), handlelength=2.2,
               bbox_to_anchor=(0.54, -0.005), columnspacing=1.2)
    fig.subplots_adjust(hspace=0.22, left=0.17, right=0.98, top=0.96, bottom=0.21)
    save_fig(fig, "fig_main_poolsize")


# ---------------------------------------------------------------------------
# Legacy protocol -- tables/legacy.tex
# ---------------------------------------------------------------------------
def tabL(B):
    sec = "Table L (legacy vs corrected protocol, seed 42, Fixed, k in {1,4})"
    head = [r"Model & \multicolumn{2}{c}{invalid outputs (\%): ZS / FS / CICLe} & "
            r"\multicolumn{2}{c}{macro-F1: ZS / FS / CICLe} \\",
            r" & legacy & corrected & legacy & corrected \\"]
    rows, drows = [], []
    for d in CORE:
        leg, cor = "legacy:" + d, d
        if not fd.has_results(leg):
            pending(sec, f"{d}: no results_legacy directory")
            continue
        for m in SMALL:
            cells = []
            for tag in (leg, cor):
                parts_inv, parts_f1 = [], []
                for method, k_list in (("zeroshot", [None]), ("fewshot", [1, 4]), ("cicle", [1, 4])):
                    invs, f1s = [], []
                    for k in k_list:
                        inv, n = fd.mean_metric(tag, method, "fixed" if k else None, k, models=[m],
                                                seeds=[42], metric="invalid_rate")
                        f, _ = fd.mean_metric(tag, method, "fixed" if k else None, k, models=[m],
                                              seeds=[42])
                        if inv is not None:
                            invs.append(inv); f1s.append(f)
                    if len(invs) < len(k_list):
                        pending(sec, f"{tag} {m} {method}")
                        parts_inv.append("--"); parts_f1.append("--")
                    else:
                        parts_inv.append(f"{np.mean(invs):.1f}"); parts_f1.append(f"{np.mean(f1s):.1f}")
                        note(sec, f"{tag} {MODEL_NAME[m]} {method} (mean over k): invalid "
                                  f"{np.mean(invs):.1f}%, macro-F1 {np.mean(f1s):.2f}")
                cells.append((" / ".join(parts_inv), " / ".join(parts_f1)))
            deltas = []
            for tag in (leg, cor):
                res = fd.paired_delta(tag, "cicle", "fewshot", "fixed", [1, 4], models=[m], seeds=[42], B=B)
                deltas.append(delta_tex(res, stacked=True))
                if res is None:
                    pending(sec, f"Δ {tag} {m}")
                else:
                    note(sec, f"Δ CICLe − few-shot {tag} {MODEL_NAME[m]} (2 cells): {delta_txt(res)}")
            if m == SMALL[0]:
                rows.append(r"\multicolumn{5}{l}{\emph{" + NAME[d] + r"}} \\")
            rows.append(f"{MODEL_NAME[m]} & {cells[0][0]} & "
                        f"{cells[1][0]} & {cells[0][1]} & {cells[1][1]} \\\\")
            drows.append(f"{NAME[d] if m == SMALL[0] else ''} & {MODEL_NAME[m]} & {deltas[0]} & "
                         f"{deltas[1]} \\\\")
        rows.append(r"\midrule"); drows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop(); drows.pop()
    write_table("legacy_delta", "llcc",
                [r"Dataset & Model & $\Delta$ legacy & $\Delta$ corrected \\"], drows,
                comment="Table Lb. Per-model Δ CICLe − few-shot (Fixed, k in {1,4}, seed 42) under the "
                        "legacy and the corrected protocol, paired over the 1,000 instances and 2 "
                        "cells, 95% bootstrap CI.")
    write_table("legacy", "lcccc", head, rows,
                comment="Table L. Original (DS2026) protocol vs the corrected one on the same seed-42 "
                        "test sample: zero-shot, few-shot Fixed and CICLe Fixed at k in {1,4} (FS / CICLe "
                        "cells are means over k). Legacy = no label list in the few-shot/CICLe prompts, "
                        "different wording, 5-token generation cap, exact matching. Δ = CICLe − few-shot "
                        "paired over the 1,000 instances and k in {1,4} (2 cells), 95% bootstrap CI.")


# ---------------------------------------------------------------------------
# Prompt robustness and random retrieval -- tables/robustness.tex
# ---------------------------------------------------------------------------
def tabR(B):
    sec = "Table R (alternative prompt / random retrieval; seed 42, 6 models, k in {1,4})"
    head = [r"Dataset & Condition & FS Fixed & CICLe Fixed & $\Delta$ Fixed & FS PC & CICLe PC & $\Delta$ PC \\"]
    rows = []
    conds = [("", "default"), ("-altprompt", "alt. prompt"), ("-random", "random retr.")]
    for d in CORE:
        variants = ("fixed", "pc") if d != "ohsumed" else ("fixed",)
        for suffix, label in conds:
            tag = d + suffix
            if not fd.has_results(tag):
                pending(sec, f"{tag} (no results directory)")
                rows.append(f"{NAME[d] if suffix == '' else ''} & {label} & "
                            + " & ".join(["[pending]"] * 6) + r" \\")
                continue
            cells = []
            for var in ("fixed", "pc"):
                if var not in variants:
                    cells += ["--", "--", "--"]
                    continue
                res = fd.paired_delta(tag, "cicle", "fewshot", var, [1, 4], seeds=[42], B=B)
                if res is None:
                    pending(sec, f"{tag} {var}")
                    cells += ["[pending]"] * 3
                    continue
                cells += [num(res["mean_b"]), num(res["mean_a"]), delta_tex(res, stacked=True)]
                note(sec, f"{tag} {var}: few-shot {res['mean_b']:.2f}, CICLe {res['mean_a']:.2f}, "
                          f"Δ {delta_txt(res)}")
            rows.append(f"{NAME[d] if suffix == '' else ''} & {label} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    write_table("robustness", "llcccccc", head, rows,
                comment="Table R. CICLe vs few-shot under the default prompt, an alternative prompt "
                        "wording and random (instead of similarity-based) example retrieval; seed 42, "
                        "six small models, k in {1,4}: means over the 12 runs and the paired Δ over "
                        "the 1,000 instances x 12 cells, 95% bootstrap CI. Ohsumed: Fixed only.")


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
    "tab2": ("Table 2: main results -> tables/body_main_results.tex, body_main_delta.tex, main_results.tex", tab2),
    "fig1": ("Figure 1: macro-F1 vs prompt tokens -> fig_main_tokens.pdf", fig1),
    "fig2": ("Figure 2: narrowing under imbalance, Per-Class, panels a+c -> fig_main_narrowing.pdf", fig2),
    "tab3": ("Table 3: label renaming -> tables/relabel.tex", tab3),
    "tab4": ("Table 4: supervised vs pipelines -> tables/body_main_supervised.tex, baselines.tex", tab4),
    "fig3": ("Figure 3: alpha -> fig_app_alpha.pdf", fig3),
    "tabB": ("Tables B: full grids -> tables/grid_<variant>.tex", tabB),
    "tabC1": ("Table C1: narrowing comparisons, all variants -> tables/narrowing_all_a.tex, narrowing_all_b.tex", tabC1),
    "tabC2": ("Table C2: candidate-set statistics -> tables/set_stats.tex", tabC2),
    "figAN": ("Figure 2 extra: panels b+d, Per-Class -> fig_app_narrowing_extra.pdf", figAN),
    "figC1": ("Figure C1: panels a-d, Fixed -> fig_app_narrowing_fixed.pdf", figC1),
    "tabD1": ("Table D1: per-model deltas -> tables/per_model_fixed.tex, per_model_pc.tex", tabD1),
    "tabE1": ("Table E1: embedding / classifier / alpha ablation -> tables/ablation.tex", tabE1),
    "tabF1": ("Table F1: larger models -> tables/large_models.tex, large_models_delta.tex", tabF1),
    "tabG1": ("Table G1: fixes vs breaks -> tables/fixes_breaks.tex, fixes_breaks_sets.tex", tabG1),
    "tabG2": ("Table G2: invalid outputs -> tables/invalid.tex, invalid_raw.tex", tabG2),
    "tabG3": ("Table G3: qualitative examples -> tables/examples.tex", tabG3),
    "tabH1": ("Table H1: imbalance class counts, relabel words -> tables/imbalance_counts.tex, relabel_words.tex", tabH1),
    "tabP": ("Table P: pool size -> tables/pool_size.tex", tabP),
    "figP": ("Figure P: macro-F1 vs pool size -> fig_main_poolsize.pdf", figP),
    "tabL": ("Table L: legacy vs corrected protocol -> tables/legacy.tex, legacy_delta.tex", tabL),
    "tabR": ("Table R: alternative prompt / random retrieval -> tables/robustness.tex", tabR),
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
