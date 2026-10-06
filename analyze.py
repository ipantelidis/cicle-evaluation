#!/usr/bin/env python3
"""
Summarises the per-instance results written by experiment.py and baselines.py.

For every dataset it prints
  * macro-F1 per method, variant and k, averaged over models and seeds,
    with the invalid-output rate and prompt length;
  * coverage, set size and LLM-skip rate of every narrowing method;
  * paired macro-F1 differences between CICLe and each alternative
    (few-shot, top-k, probability mass) with a bootstrap 95% confidence
    interval and p-value, overall and per model.

The paired test resamples test instances; the same resample is applied to
both methods and to every model / k sharing that seed's test set, so the
interval reflects test-sample noise rather than the spread over
hyperparameters.

Usage:
    python analyze.py
    python analyze.py --datasets sst,yahoo-answers-imb100 --bootstrap 5000 --per-model
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METHOD_ORDER = ["base", "finetuned", "zeroshot", "fewshot", "cicle", "topk", "mass"]
NARROWING = ["cicle", "topk", "mass"]


def macro_f1(gold, pred, n_classes):
    """Macro-F1 over the classes present in `gold`. Inputs are integer
    codes; invalid predictions are coded as n_classes."""
    cm = np.bincount(gold * (n_classes + 1) + pred,
                     minlength=n_classes * (n_classes + 1)).reshape(n_classes, n_classes + 1)
    tp, support, predicted = np.diag(cm[:, :n_classes]), cm.sum(1), cm[:, :n_classes].sum(0)
    f1 = np.where(support + predicted > 0, 2 * tp / np.maximum(support + predicted, 1), 0.0)
    return f1[support > 0].mean()


def load(results_dir, dataset):
    """(method, variant, k, model, emb, clf, alpha, seed) -> run."""
    runs = {}
    for path in glob.glob(os.path.join(results_dir, dataset, "seed-*", "*.json")):
        with open(path) as f:
            d = json.load(f)
        c, records = d["config"], d["records"]
        if c.get("legacy_prompt") or c["n_test"] != len(records):
            continue
        labels = sorted({r["gold"] for r in records} | {r["pred"] for r in records if r["pred"]})
        code = {l: i for i, l in enumerate(labels)}
        key = (c["method"], c.get("variant"), c.get("k"), c["model"], c.get("emb"),
               c.get("clf"), c.get("alpha"), c["seed"])
        runs[key] = {
            "gold": np.array([code[r["gold"]] for r in records]),
            "pred": np.array([code[r["pred"]] if r["pred"] is not None else len(labels)
                              for r in records]),
            "n_classes": len(labels), "metrics": d["metrics"],
        }
    return runs


def summary_table(runs):
    groups = defaultdict(list)
    for (method, variant, k, *_), run in runs.items():
        groups[(method, variant or "-", k or 0)].append(run["metrics"])
    print(f"  {'method':9s}{'variant':8s}{'k':>2s} {'runs':>5s} {'macro-F1':>9s} {'acc':>6s} "
          f"{'invalid':>8s} {'shots':>6s} {'tokens':>7s}")
    for key in sorted(groups, key=lambda g: (METHOD_ORDER.index(g[0]), g[1], g[2])):
        ms = groups[key]
        mean = lambda name: np.mean([m[name] for m in ms])
        print(f"  {key[0]:9s}{key[1]:8s}{key[2]:2d} {len(ms):5d} {100 * mean('macro_f1'):9.2f} "
              f"{100 * mean('accuracy'):6.2f} {100 * mean('invalid_rate'):7.1f}% "
              f"{mean('mean_shots'):6.1f} {mean('mean_prompt_tokens'):7.0f}")


def narrowing_table(runs):
    for method in NARROWING:
        seen = {}  # the sets do not depend on the LLM, variant or k
        for (m, _, _, _, emb, clf, alpha, seed), run in runs.items():
            if m == method:
                seen[(emb, clf, alpha, seed)] = run["metrics"]
        if not seen:
            continue
        mean = lambda name: np.mean([m[name] for m in seen.values()])
        print(f"  candidate sets [{method:5s}]: coverage {100 * mean('coverage'):.1f}%, "
              f"mean size {mean('mean_set_size'):.2f}, "
              f"LLM skipped {100 * (1 - mean('llm_call_rate')):.1f}%")


def ablation_table(runs):
    """CICLe under every (embedding, classifier, alpha) setting, next to
    few-shot with the same embedding, restricted to the models, k and
    variants that every setting was run with."""
    settings = sorted({k[4:7] for k in runs if k[0] == "cicle"})
    if len(settings) < 2:
        return
    cells = [{(k[1], k[2], k[3], k[7]) for k in runs if k[0] == "cicle" and k[4:7] == s}
             for s in settings]
    shared = set.intersection(*cells)
    print(f"  ablation over {len(shared)} shared (variant, k, model, seed) cells:")
    print(f"    {'emb':11s}{'clf':5s}{'alpha':>6s} {'CICLe':>7s} {'few-shot':>9s} {'diff':>6s} "
          f"{'coverage':>9s} {'set size':>9s} {'LLM skipped':>12s}")
    for emb, clf, alpha in settings:
        cicle = [runs[("cicle", v, k, m, emb, clf, alpha, s)]["metrics"] for v, k, m, s in shared]
        few = [runs[key]["metrics"] for v, k, m, s in shared
               if (key := ("fewshot", v, k, m, emb, None, None, s)) in runs]
        mean = lambda ms, name: np.mean([x[name] for x in ms])
        f_c = 100 * mean(cicle, "macro_f1")
        f_f = 100 * mean(few, "macro_f1") if len(few) == len(cicle) else float("nan")
        print(f"    {emb:11s}{clf:5s}{alpha:6.2f} {f_c:7.2f} {f_f:9.2f} {f_c - f_f:+6.2f} "
              f"{100 * mean(cicle, 'coverage'):8.1f}% {mean(cicle, 'mean_set_size'):9.2f} "
              f"{100 * (1 - mean(cicle, 'llm_call_rate')):11.1f}%")


def paired_test(runs, other, variant, n_boot, rng, model=None):
    """CICLe minus `other` macro-F1, paired on everything but the method."""
    pairs = defaultdict(list)  # seed -> [(cicle run, other run)]
    for key, run in runs.items():
        method, var, k, mdl, emb, clf, alpha, seed = key
        if method != "cicle" or var != variant or (model and mdl != model):
            continue
        partner = ((other, var, k, mdl, emb, None, None, seed) if other == "fewshot"
                   else (other, var, k, mdl, emb, clf, alpha, seed))
        if partner in runs:
            pairs[seed].append((run, runs[partner]))
    n_pairs = sum(len(v) for v in pairs.values())
    if not n_pairs:
        return None

    def mean_delta(index):
        return np.mean([
            macro_f1(a["gold"][index[s]], a["pred"][index[s]], a["n_classes"])
            - macro_f1(b["gold"][index[s]], b["pred"][index[s]], b["n_classes"])
            for s, items in pairs.items() for a, b in items])

    sizes = {s: len(items[0][0]["gold"]) for s, items in pairs.items()}
    observed = mean_delta({s: np.arange(n) for s, n in sizes.items()})
    boots = np.array([mean_delta({s: rng.integers(0, n, n) for s, n in sizes.items()})
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p-value for "the difference is zero"
    p = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return (f"{100 * observed:+.2f} pp [95% CI {100 * lo:+.2f}, {100 * hi:+.2f}], "
            f"p = {p:.3f}, {n_pairs} pairs")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=os.path.join(BASE_DIR, "results"))
    p.add_argument("--datasets", type=lambda s: s.split(","), default=None,
                   help="default: every directory under the results directory")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--per-model", action="store_true", help="also test each model separately")
    # the reference configuration; other settings are ablations and are
    # summarised separately instead of being pooled into the main comparison
    p.add_argument("--emb", default="minilm")
    p.add_argument("--clf", default="lr")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()
    rng = np.random.default_rng(0)

    datasets = args.datasets or sorted(
        d for d in os.listdir(args.results_dir)
        if glob.glob(os.path.join(args.results_dir, d, "seed-*")))
    for dataset in datasets:
        every_run = load(args.results_dir, dataset)
        runs = {k: v for k, v in every_run.items()
                if k[4] in (None, args.emb) and k[5] in (None, args.clf)
                and k[6] in (None, args.alpha)}
        if not runs:
            continue
        models = sorted({k[3] for k in runs} - {"none"})
        seeds = sorted({k[7] for k in runs})
        print(f"\n===== {dataset}: {len(runs)} runs, {len(models)} models, seeds {seeds}")
        summary_table(runs)
        narrowing_table(runs)
        for other in ("fewshot", "topk", "mass"):
            for variant in ("fixed", "pc"):
                res = paired_test(runs, other, variant, args.bootstrap, rng)
                if res:
                    print(f"  CICLe - {other:7s} ({variant:5s}): {res}")
                if res and args.per_model:
                    for model in models:
                        res_m = paired_test(runs, other, variant, args.bootstrap // 4, rng, model)
                        if res_m:
                            print(f"      {model:16s} {res_m}")
        ablation_table(every_run)


if __name__ == "__main__":
    main()
