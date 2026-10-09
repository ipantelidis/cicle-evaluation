"""
Data layer for the paper's figures and tables.

Loads every per-instance result file (results/<variant>/seed-<s>/*.json) once,
reduces it to compact arrays and caches the reduction under cache/figures/, one
pickle per dataset variant. The cache is incremental: files that are new or
whose (mtime, size) changed are re-read, everything else comes from the pickle,
so the same command works before and after pending experiments land.

Statistics live here as well: the fast macro-F1 of analyze.py and a vectorised
version of its paired bootstrap over test instances (one resample of the test
indices per seed, applied to every cell that shares that seed). Every bootstrap
in this module draws its resamples from a fixed matrix per (seed, n, B), which
lets the per-run bootstrap statistics be computed once and reused by every test
that touches the run; the test itself is then a mean over cells of the cached
per-run vectors.
"""
import glob
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import analyze  # noqa: E402  (macro_f1 on integer codes)

RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache", "figures")

SMALL = ["llama-3.2-3b", "ministral-3b", "qwen-2.5-3b", "mistral-7b-v0.3", "qwen-2.5-7b",
         "llama-3.1-8b"]
LARGE = ["mistral-nemo-2407", "qwen-2.5-32b"]
SEEDS = [42, 43, 44]
NARROWING = ["cicle", "topk", "mass", "marginal", "oracle"]
INVALID_MAX_RAW = 2000  # raw outputs kept per run for the invalid-output table

_PENDING = []  # (step, message) collected while building
CURRENT_STEP = [None]  # set by make_figures.main before each step


def pending(where, what):
    msg = f"{where}: {what}"
    if (CURRENT_STEP[0], msg) not in _PENDING:
        _PENDING.append((CURRENT_STEP[0], msg))
        print("  [pending] " + msg)


def pending_messages(step=None):
    return [m for s, m in _PENDING if step is None or s == step]


# ---------------------------------------------------------------------------
# Loading and caching
# ---------------------------------------------------------------------------
def run_key(method, variant=None, k=None, model=None, emb="minilm", clf="lr", alpha=0.05,
            seed=42):
    """The key under which a run is stored; mirrors analyze.load's key and the
    file-name conventions of experiment.py / baselines.py."""
    if method == "zeroshot":
        return (method, None, None, model, None, None, None, seed)
    if method == "fewshot":
        return (method, variant, k, model, emb, None, None, seed)
    if method == "base":
        return (method, None, None, "none", emb, clf, None, seed)
    if method == "finetuned":  # emb names the fine-tuned encoder (roberta-base / roberta-large)
        enc = emb if str(emb).startswith("roberta") else "roberta-base"
        return (method, None, None, "none", enc, None, None, seed)
    return (method, variant, k, model, emb, clf, alpha, seed)


def _reduce(d, vocab):
    """One result file -> compact run dict. `vocab` (label -> code) is extended
    in place with labels not seen before, so codes stay stable across files."""
    c, records = d["config"], d["records"]
    records = sorted(records, key=lambda r: r["idx"])
    assert [r["idx"] for r in records] == list(range(len(records))), "records not 0..n-1"
    for r in records:
        for lab in [r["gold"]] + ([r["pred"]] if r["pred"] is not None else []) \
                + list(r.get("conformal_set") or []):
            if lab not in vocab:
                vocab[lab] = len(vocab)
    gold = np.array([vocab[r["gold"]] for r in records], dtype=np.int16)
    pred = np.array([vocab[r["pred"]] if r["pred"] is not None else -1 for r in records],
                    dtype=np.int16)
    run = {
        "config": c, "metrics": d["metrics"], "n": len(records),
        "gold": gold, "pred": pred,
        "prompt_tokens": np.array([r["prompt_tokens"] for r in records], dtype=np.int32),
        "llm_called": np.array([bool(r["llm_called"]) for r in records]),
        "gold_in_set": None, "set_size": None, "sets": None,
        "invalid_raw": Counter(),
    }
    if "conformal_set" in records[0]:
        run["gold_in_set"] = np.array([bool(r["gold_in_set"]) for r in records])
        run["set_size"] = np.array([len(r["conformal_set"]) for r in records], dtype=np.int16)
        # candidate sets as a packed boolean matrix (n x |vocab|); the vocab may
        # grow later, so the width is stored with it
        width = len(vocab)
        mat = np.zeros((len(records), width), dtype=bool)
        for i, r in enumerate(records):
            for lab in r["conformal_set"]:
                mat[i, vocab[lab]] = True
        run["sets"] = (np.packbits(mat, axis=1), width)
    if c["method"] != "base" and c["method"] != "finetuned":
        run["invalid_raw"] = Counter(
            (r["raw"] or "<empty>").strip()[:60] for r in records
            if r["pred"] is None and r.get("llm_called", True))
    return run


LEGACY_PREFIX = "legacy:"  # variant("legacy:<dataset>") reads results_legacy/<dataset>/


def _root_and_dir(tag):
    """(results root, directory name, cache stem) of a variant tag."""
    if tag.startswith(LEGACY_PREFIX):
        d = tag[len(LEGACY_PREFIX):]
        return os.path.join(ROOT, "results_legacy"), d, "legacy_" + d
    return RESULTS, tag, tag


class Variant:
    """All runs of one dataset variant (a directory under results/, or under
    results_legacy/ for a "legacy:<dataset>" tag, where the original-prompt
    files that analyze.load skips are kept)."""

    def __init__(self, tag):
        self.tag = tag
        self.root, self.dirname, self.stem = _root_and_dir(tag)
        self.legacy = tag.startswith(LEGACY_PREFIX)
        self.runs = {}        # run_key -> run dict
        self.vocab = {}       # label -> code
        self.manifest = {}    # file path -> (mtime, size)
        self.load()

    @property
    def labels(self):
        inv = {v: k for k, v in self.vocab.items()}
        return [inv[i] for i in range(len(inv))]

    def load(self):
        os.makedirs(CACHE, exist_ok=True)
        cache_path = os.path.join(CACHE, f"{self.stem}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
            self.runs, self.vocab, self.manifest = state["runs"], state["vocab"], state["manifest"]
        files = {p: (os.path.getmtime(p), os.path.getsize(p))
                 for p in glob.glob(os.path.join(self.root, self.dirname, "seed-*", "*.json"))}
        stale = [p for p in self.manifest if p not in files]
        fresh = [p for p, sig in files.items() if self.manifest.get(p) != sig]
        if not stale and not fresh:
            return
        if stale:
            keep = {k for k, r in self.runs.items() if r.get("_path") not in stale}
            self.runs = {k: r for k, r in self.runs.items() if k in keep}
            for p in stale:
                self.manifest.pop(p, None)
        for i, p in enumerate(sorted(fresh)):
            with open(p) as f:
                d = json.load(f)
            c = d["config"]
            if bool(c.get("legacy_prompt")) != self.legacy or c["n_test"] != len(d["records"]):
                continue  # same filters as analyze.load (inverted for the legacy directory)
            run = _reduce(d, self.vocab)
            run["_path"] = p
            key = (c["method"], c.get("variant"), c.get("k"), c["model"], c.get("emb"),
                   c.get("clf"), c.get("alpha"), c["seed"])
            self.runs[key] = run
            self.manifest[p] = files[p]
            if (i + 1) % 100 == 0:
                print(f"    {self.tag}: {i + 1}/{len(fresh)} files")
        tmp = cache_path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"runs": self.runs, "vocab": self.vocab, "manifest": self.manifest}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
        print(f"  cached {self.tag}: {len(self.runs)} runs ({len(fresh)} files read)")

    def get(self, method, variant=None, k=None, model=None, seed=42, **kw):
        return self.runs.get(run_key(method, variant, k, model, seed=seed, **kw))

    def select(self, method=None, variant=None, k=None, models=None, seed=None, emb="minilm",
               clf="lr", alpha=0.05, any_setting=False):
        """Runs matching the filters (None = any). `any_setting` ignores emb,
        clf and alpha (for the ablation tables)."""
        out = {}
        for key, run in self.runs.items():
            m, v, kk, mdl, e, c, a, s = key
            if method is not None and m != method:
                continue
            if variant is not None and v != variant:
                continue
            if k is not None and kk != k:
                continue
            if models is not None and mdl not in models:
                continue
            if seed is not None and s != seed:
                continue
            if not any_setting and m not in ("base", "finetuned"):
                if e not in (None, emb) or c not in (None, clf) or a not in (None, alpha):
                    continue
            out[key] = run
        return out

    def set_matrix(self, run):
        """Candidate sets of a narrowing run as a boolean (n x |vocab|) matrix."""
        packed, width = run["sets"]
        mat = np.unpackbits(packed, axis=1, count=width).astype(bool)
        if mat.shape[1] < len(self.vocab):
            mat = np.concatenate([mat, np.zeros((mat.shape[0], len(self.vocab) - mat.shape[1]),
                                                dtype=bool)], axis=1)
        return mat


_VARIANTS = {}


def variant(tag):
    if tag not in _VARIANTS:
        _VARIANTS[tag] = Variant(tag)
    return _VARIANTS[tag]


def has_results(tag):
    root, d, _ = _root_and_dir(tag)
    return bool(glob.glob(os.path.join(root, d, "seed-*", "*.json")))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def n_classes(v):
    return len(v.vocab)


def f1(v, run, index=None):
    """Macro-F1 (pp) of one run over the classes present in gold[index]."""
    C = n_classes(v)
    gold, pred = run["gold"].astype(np.int64), run["pred"].astype(np.int64)
    pred = np.where(pred < 0, C, pred)
    if index is not None:
        gold, pred = gold[index], pred[index]
    return 100 * analyze.macro_f1(gold, pred, C)


def per_class_f1(v, run):
    """Per-class F1 (pp) as a dict label -> F1 for the classes present in gold."""
    C = n_classes(v)
    gold, pred = run["gold"].astype(np.int64), run["pred"].astype(np.int64)
    pred = np.where(pred < 0, C, pred)
    cm = np.bincount(gold * (C + 1) + pred, minlength=C * (C + 1)).reshape(C, C + 1)
    tp, support, predicted = np.diag(cm[:, :C]), cm.sum(1), cm[:, :C].sum(0)
    f = np.where(support + predicted > 0, 2 * tp / np.maximum(support + predicted, 1), 0.0)
    labels = v.labels
    return {labels[c]: 100 * f[c] for c in range(C) if support[c] > 0}


_RESAMPLES = {}


def resamples(seed, n, B):
    """The fixed (B x n) matrix of bootstrap indices for a test seed."""
    key = (seed, n, B)
    if key not in _RESAMPLES:
        rng = np.random.default_rng(10_000 * B + seed)
        _RESAMPLES[key] = rng.integers(0, n, size=(B, n), dtype=np.int32)
    return _RESAMPLES[key]


def boot_f1(v, run, B):
    """Bootstrap vector (B,) of the run's macro-F1 (pp) under the fixed
    resamples of its seed; cached on the run dict."""
    cache = run.setdefault("_boot", {})
    if B in cache:
        return cache[B]
    C = n_classes(v)
    seed, n = run["config"]["seed"], run["n"]
    idx = resamples(seed, n, B)
    gold = run["gold"].astype(np.int64)[idx]
    pred = run["pred"].astype(np.int64)
    pred = np.where(pred < 0, C, pred)[idx]
    codes = gold * (C + 1) + pred + (np.arange(B, dtype=np.int64) * (C * (C + 1)))[:, None]
    cm = np.bincount(codes.ravel(), minlength=B * C * (C + 1)).reshape(B, C, C + 1)
    tp = cm[:, np.arange(C), np.arange(C)]
    support, predicted = cm.sum(2), cm[:, :, :C].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(support + predicted > 0, 2 * tp / np.maximum(support + predicted, 1), 0.0)
    present = support > 0
    out = 100 * (f * present).sum(1) / present.sum(1)
    cache[B] = out
    return out


def paired_delta(tag, method_a, method_b, variant_name, ks, models=SMALL, seeds=SEEDS, B=5000,
                 alpha=0.05, emb="minilm", clf="lr", variants=None, allow_partial=False):
    """`method_a` minus `method_b` macro-F1 (pp), paired on (model, seed, k[, variant]).

    Returns dict(mean, lo, hi, p, n, cells) or None when no pair exists.
    `variants` (list) pools several retrieval variants into one test; otherwise
    `variant_name` is used. few-shot partners never carry clf / alpha. Unless
    `allow_partial`, a grid with missing cells is treated as pending (None), so
    a half-finished experiment never enters a paper number."""
    v = variant(tag)
    variants = variants or [variant_name]
    a_vecs, b_vecs, a_obs, b_obs, cells = [], [], [], [], []
    for var in variants:
        for m in models:
            for k in ks:
                for s in seeds:
                    a = v.get(method_a, var, k, m, seed=s, emb=emb, clf=clf, alpha=alpha)
                    b = v.get(method_b, var, k, m, seed=s, emb=emb, clf=clf, alpha=alpha)
                    if a is None or b is None:
                        continue
                    a_vecs.append(boot_f1(v, a, B)); b_vecs.append(boot_f1(v, b, B))
                    a_obs.append(f1(v, a)); b_obs.append(f1(v, b))
                    cells.append((var, m, k, s))
    if not cells:
        return None
    expected = len(variants) * len(models) * len(ks) * len(seeds)
    if len(cells) < expected and not allow_partial:
        pending("paired_delta", f"{tag} {method_a} - {method_b} {variants} k={ks}: "
                                f"{len(cells)}/{expected} cells present, slot left empty")
        return None
    boots = np.mean(a_vecs, axis=0) - np.mean(b_vecs, axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    return {"mean": float(np.mean(a_obs) - np.mean(b_obs)), "lo": float(lo), "hi": float(hi),
            "p": float(p), "n": len(cells), "cells": cells,
            "mean_a": float(np.mean(a_obs)), "mean_b": float(np.mean(b_obs))}


def mean_ci(tag, method, variant_name=None, k=None, models=SMALL, seeds=SEEDS, B=5000,
            allow_partial=False, **kw):
    """Mean macro-F1 over models x seeds and its bootstrap CI over test
    instances (same resamples as the paired test, unpaired statistic)."""
    v = variant(tag)
    vecs, obs = [], []
    models = models if method not in ("base", "finetuned") else ["none"]
    for m in models:
        for s in seeds:
            r = v.get(method, variant_name, k, m, seed=s, **kw)
            if r is not None:
                vecs.append(boot_f1(v, r, B)); obs.append(f1(v, r))
    if not obs:
        return None
    if len(obs) < len(models) * len(seeds) and not allow_partial:
        pending("mean_ci", f"{tag} {method} {variant_name} k={k}: {len(obs)}/{len(models) * len(seeds)} "
                           "runs present, slot left empty")
        return None
    boots = np.mean(vecs, axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(np.mean(obs)), "lo": float(lo), "hi": float(hi), "n": len(obs)}


def mean_metric(tag, method, variant_name=None, k=None, models=SMALL, seeds=SEEDS,
                metric="macro_f1", allow_partial=False, **kw):
    """Mean of a stored metric over models x seeds -> (value, n_runs). Rates are
    returned in percent, tokens / sizes / shots as stored. (None, n) if no run
    or, unless `allow_partial`, if the models x seeds grid is incomplete."""
    v = variant(tag)
    vals = []
    models = models if method not in ("base", "finetuned") else ["none"]
    for m in models:
        for s in seeds:
            r = v.get(method, variant_name, k, m, seed=s, **kw)
            if r is not None and metric in r["metrics"]:
                x = r["metrics"][metric]
                vals.append(x if metric in ("mean_prompt_tokens", "mean_set_size", "mean_shots")
                            else 100 * x)
    if vals and len(vals) < len(models) * len(seeds) and not allow_partial:
        pending("mean_metric", f"{tag} {method} {variant_name} k={k} {metric}: {len(vals)}/"
                               f"{len(models) * len(seeds)} runs present, slot left empty")
        return (None, len(vals))
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def metric_values(tag, method, variant_name=None, k=None, models=SMALL, seeds=SEEDS,
                  metric="macro_f1", **kw):
    """Per-run values (dict (model, seed) -> value) of a stored metric."""
    v = variant(tag)
    out = {}
    for m in (models if method not in ("base", "finetuned") else ["none"]):
        for s in seeds:
            r = v.get(method, variant_name, k, m, seed=s, **kw)
            if r is not None and metric in r["metrics"]:
                x = r["metrics"][metric]
                out[(m, s)] = (x if metric in ("mean_prompt_tokens", "mean_set_size", "mean_shots")
                               else 100 * x)
    return out


def one_set_run(tag, method, seed, alpha=0.05, emb="minilm", clf="lr", models=None):
    """One run of a narrowing method for a seed; the candidate sets do not
    depend on the LLM, k or variant, so any run will do (the first found)."""
    v = variant(tag)
    for key, run in sorted(v.runs.items(), key=lambda kv: str(kv[0])):
        m, var, k, mdl, e, c, a, s = key
        if m == method and s == seed and e == emb and c == clf and a == alpha \
                and (models is None or mdl in models) and run["sets"] is not None:
            return run
    return None


def per_class_coverage(v, run):
    """label -> coverage (pp) of the candidate set for that gold class."""
    labels = v.labels
    gis, gold = run["gold_in_set"], run["gold"]
    return {labels[c]: 100 * gis[gold == c].mean() for c in np.unique(gold)}


# ---------------------------------------------------------------------------
# Dataset metadata from experiment.Data (CPU, offline Hugging Face cache)
# ---------------------------------------------------------------------------
_META_PATH = os.path.join(CACHE, "data_meta.json")
_META = None


def _data(dataset, seed, **kw):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from experiment import Data
    return Data(dataset, seed, **kw)


def data_meta():
    global _META
    if _META is None:
        _META = json.load(open(_META_PATH)) if os.path.exists(_META_PATH) else {}
    return _META


def _save_meta():
    os.makedirs(CACHE, exist_ok=True)
    with open(_META_PATH, "w") as f:
        json.dump(_META, f, indent=1)


def class_counts(dataset, seed, imbalance):
    """Training-pool class counts (after the 80/20 calibration split) in
    descending order: list of [label, count]. Cached in cache/figures."""
    meta = data_meta()
    key = f"counts/{dataset}/{seed}/{imbalance}"
    if key not in meta:
        try:
            d = _data(dataset, seed, imbalance=imbalance)
        except Exception as e:  # HF cache missing etc.
            pending("class_counts", f"{dataset} seed {seed} imb {imbalance}: {str(e)[:80]}")
            return None
        meta[key] = [[str(l), int(c)] for l, c in Counter(d.y_train).most_common()]
        _save_meta()
    return meta[key]


def label_map(dataset):
    meta = data_meta()
    key = f"labelmap/{dataset}"
    if key not in meta:
        try:
            d = _data(dataset, 42, relabel=True)
        except Exception as e:
            pending("label_map", f"{dataset}: {str(e)[:80]}")
            return None
        meta[key] = {str(k): str(v) for k, v in d.label_map.items()}
        _save_meta()
    return meta[key]


def test_texts(dataset, seed):
    """The 1,000 test texts of a (dataset, seed) in record order."""
    meta = data_meta()
    key = f"texts/{dataset}/{seed}"
    if key not in meta:
        try:
            d = _data(dataset, seed)
        except Exception as e:
            pending("test_texts", f"{dataset} seed {seed}: {str(e)[:80]}")
            return None
        meta[key] = [str(x) for x in d.X_test]
        _save_meta()
    return meta[key]
