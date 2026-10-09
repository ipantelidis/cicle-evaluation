#!/usr/bin/env python3
"""
Single-script experiment runner for the CICLe evaluation suite.

Replaces the one-notebook-per-configuration setup: one invocation loads an
LLM once and runs every requested (method, variant, embedding, classifier,
k, alpha, seed) configuration for one dataset, skipping any configuration
whose result file already exists.

Differences from the original notebooks (all disabled by --legacy-prompt,
which reproduces the notebooks' behaviour for validation):

  * every prompt lists the labels the LLM may answer with -- all classes
    for zero-shot / few-shot, the conformal prediction set for CICLe;
  * few-shot and CICLe prompts use the same wording, so the candidate
    label list and the retrieved examples are the only difference;
  * the generation budget is derived from the longest label instead of a
    fixed 5 tokens, and the raw output is normalised before matching;
  * outputs that still match no label are counted as wrong and reported
    as an invalid-output rate;
  * per-instance records (gold, raw output, prediction, conformal set,
    retrieved examples, prompt length) are saved for every test instance;
  * macro-F1 is computed over the classes that occur in the test sample.

Examples:
    # CICLe and few-shot, both variants, all k, for one model on one GPU
    python experiment.py --dataset yahoo-answers --model llama-3.1-8b \\
        --methods fewshot,cicle --variants fixed,pc --shots 1,2,4,8 --gpu 0

    # zero-shot, three data seeds
    python experiment.py --dataset sst --model qwen-2.5-7b \\
        --methods zeroshot --seeds 42,43,44

    # reproduce a notebook result with the original prompt
    python experiment.py --dataset yahoo-answers --model ministral-3b \\
        --methods fewshot --variants fixed --shots 1 --legacy-prompt
"""
import argparse
import itertools
import json
import os
import re
import sys
import unicodedata

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")

MODELS = {
    "llama-3.2-3b":      "meta-llama/Llama-3.2-3B-Instruct",
    "ministral-3b":      "mistralai/Ministral-3-3B-Instruct-2512",
    "qwen-2.5-3b":       "Qwen/Qwen2.5-3B-Instruct",
    "mistral-7b-v0.3":   "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-2.5-7b":       "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.1-8b":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral-nemo-2407": "mistralai/Mistral-Nemo-Instruct-2407",
    "qwen-2.5-32b":      "Qwen/Qwen2.5-32B-Instruct",
    "llama-3.1-70b":     "meta-llama/Meta-Llama-3.1-70B-Instruct",
}

EMBEDDINGS = ["tfidf", "minilm", "contriever"]
# methods that narrow the label set with the base classifier before prompting:
#   cicle  class-conditional conformal prediction set at miscoverage alpha
#   topk   the m most probable classes, m = the mean conformal set size (same
#          average budget as cicle, but not adaptive per instance)
#   mass   most probable classes until their uncalibrated probabilities sum to
#          1 - alpha (adaptive, but without conformal calibration)
#   marginal  conformal set with standard (marginal) instead of class-conditional
#          calibration, isolating the effect of calibrating per class
#   oracle the true label plus random other classes, at the mean conformal set
#          size: the ceiling for what narrowing at that budget could achieve
NARROWING = ["cicle", "topk", "mass", "marginal", "oracle"]
CLASSIFIERS = ["lr", "svm"]


# ---------------------------------------------------------------------------
# Datasets. Each loader returns (train_df, test_df) with string columns
# "text" and "label" plus the integer "label_id" the notebooks stratified
# on (so seed 42 reproduces their exact subsample); subsampling is shared and controlled by the seed.
# ---------------------------------------------------------------------------
def _load_yahoo():
    from datasets import load_dataset
    name = "community-datasets/yahoo_answers_topics"
    train, test = load_dataset(name, split="train"), load_dataset(name, split="test")
    # shorten the label names for clarity:
    rename = {
        "Society & Culture": "Culture", "Science & Mathematics": "Science",
        "Education & Reference": "Education", "Computers & Internet": "Tech",
        "Business & Finance": "Business", "Entertainment & Music": "Entertainment",
        "Family & Relationships": "Family", "Politics & Government": "Politics",
    }
    names = [rename.get(n, n) for n in train.features["topic"].names]
    out = []
    for split in (train, test):
        df = split.to_pandas()
        df = df.rename(columns={"question_content": "text"})
        df["label"] = df["topic"].map(lambda i: names[i])
        out.append(df.rename(columns={"topic": "label_id"})[["text", "label", "label_id"]])
    return out


def _load_sst():
    from datasets import load_dataset
    out = []
    for split in ("train", "test"):
        df = load_dataset("SetFit/sst5", split=split).to_pandas()
        out.append(df.rename(columns={"label": "label_id", "label_text": "label"}))
    return out


def _load_semeval():
    from datasets import load_dataset
    # SemEval-2018 Task 2 English emoji labels (index -> emoji):
    emojis = ["❤", "😍", "😂", "💕", "🔥", "😊", "😎", "✨", "💙", "😘",
              "📷", "🇺🇸", "☀", "💜", "😉", "💯", "😁", "🎄", "📸", "😜"]
    out = []
    for split in ("train", "test"):
        df = load_dataset("Karim-Gamal/SemEval-2018-Task-2-english-emojis", split=split).to_pandas()
        df = df.rename(columns={"sentence": "text"})
        df["label_id"] = df["label"]
        df["label"] = df["label_id"].map(lambda i: emojis[i])
        out.append(df[["text", "label", "label_id"]])
    return out


def _load_go_emotions():
    from datasets import load_dataset
    out = []
    for split in ("train", "test"):
        data = load_dataset("go_emotions", "simplified", split=split)
        names = data.features["labels"].feature.names
        df = data.to_pandas()
        # keep only single-label examples:
        df = df[df["labels"].map(len) == 1]
        df["label_id"] = df["labels"].map(lambda l: l[0])
        df["label"] = df["label_id"].map(lambda i: names[i])
        out.append(df[["text", "label", "label_id"]])
    return out


def _load_fol():
    import pandas as pd
    out = []
    for split in ("train", "test"):
        df = pd.read_csv(os.path.join(BASE_DIR, "fol-reasoning", f"fol_reasoning_{split}.csv"))
        df["label_id"] = df["label"]
        df["label"] = df["label_id"].astype(str)
        out.append(df[["text", "label", "label_id"]])
    return out


def _load_ohsumed():
    from datasets import load_dataset
    out = []
    for split in ("train", "test"):
        data = load_dataset("joao-luz/ohsumed-single", split=split)
        names = data.features["label"].names
        df = data.to_pandas().rename(columns={"label": "label_id"})
        df["label"] = df["label_id"].map(lambda i: names[i])
        out.append(df[["text", "label", "label_id"]])
    return out


def _load_massive():
    import pandas as pd
    from datasets import load_dataset
    out = []
    name = "SetFit/amazon_massive_intent_en-US"
    # the official test split has a single example of one intent, too few to
    # stratify, so the validation and test splits together form the test pool
    for splits in (["train"], ["validation", "test"]):
        df = pd.concat([load_dataset(name, split=s).to_pandas() for s in splits])
        df = df.rename(columns={"label": "label_id"})
        df["label"] = df["label_text"]
        out.append(df[["text", "label", "label_id"]])
    # one intent (cooking_query) has 4 training examples, too few to appear in
    # both the training and calibration splits of a 2,000-example subsample;
    # it is dropped, leaving 59 intents
    counts = out[0]["label"].value_counts()
    keep = set(counts[counts >= 10].index)
    return [df[df["label"].isin(keep)] for df in out]


DATASETS = {
    "massive": {
        "loader": _load_massive,
        "task": "We classify voice-assistant commands by their intent.",
    },
    "ohsumed": {
        "loader": _load_ohsumed,
        "task": "We classify medical abstracts into disease categories based on their text.",
    },
    "yahoo-answers": {
        "loader": _load_yahoo,
        "task": "We classify user questions into topic categories based on their text.",
    },
    "sst": {
        "loader": _load_sst,
        "task": "We classify movie review snippets into sentiment categories based on their text.",
    },
    "semeval-18": {
        "loader": _load_semeval,
        "task": "We classify tweets by their most likely emoji.",
    },
    "go-emotions": {
        "loader": _load_go_emotions,
        "task": "We classify Reddit comments by their emotion.",
        # the few-shot and CICLe notebooks used the Yahoo Answers task
        # sentence for this dataset; kept only to reproduce their numbers
        "legacy_task": "We classify user questions into topic categories based on their text.",
    },
    "fol-reasoning": {
        "loader": _load_fol,
        "task": "We classify short passages containing logical facts and rules, ending in a "
                "yes/no question, into the number of reasoning steps required to answer that "
                "question based on their text.",
    },
}


def dataset_tag(dataset, imbalance=1.0, relabel=False, n_train=2000, prompt="default",
                retrieval="similar"):
    """Name of the results directory for a dataset / protocol variant."""
    tag = dataset if imbalance <= 1 else f"{dataset}-imb{imbalance:g}"
    tag += "-relabel" if relabel else ""
    tag += "" if n_train == 2000 else f"-n{n_train}"
    tag += "" if prompt == "default" else f"-{prompt}prompt"
    return tag + ("" if retrieval == "similar" else f"-{retrieval}")


def nonsense_words(n, seed=0):
    """n distinct pronounceable nonsense words, none a prefix of another."""
    import random
    rng = random.Random(seed)
    words = []
    while len(words) < n:
        w = "".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou") for _ in range(3))
        if not any(w.startswith(o[:4]) or o.startswith(w[:4]) for o in words):
            words.append(w)
    return words


class Data:
    """Stratified 1,600 / 400 / 1,000 train / calibration / test subsample.

    imbalance > 1 replaces the stratified training sample with a long-tailed
    one: class sizes decay geometrically so that the largest class is
    `imbalance` times the smallest, with a seed-dependent class order. The
    test sample is left untouched, so results stay paired with imbalance=1."""

    def __init__(self, dataset, seed, n_train=2000, n_test=1000, imbalance=1.0, relabel=False):
        self.n_train = n_train
        import pandas as pd
        from sklearn.model_selection import train_test_split
        self.dataset, self.seed, self.imbalance = dataset, seed, imbalance
        self.tag = dataset_tag(dataset, imbalance, relabel, n_train)
        train_df, test_df = DATASETS[dataset]["loader"]()
        train_df = train_df[train_df["text"].str.strip().astype(bool)]
        test_df = test_df[test_df["text"].str.strip().astype(bool)]

        if imbalance > 1:
            rng = np.random.default_rng(seed)
            available = train_df["label_id"].value_counts()
            weights = imbalance ** (-np.arange(len(available)) / (len(available) - 1))
            # at least 5 per class, so one example is left for calibration
            counts = np.maximum(5, np.round(n_train * weights / weights.sum())).astype(int)
            # random class order; redrawn only if a class is too small for its share
            while True:
                ids = rng.permutation(np.sort(available.index))
                if all(available[c] >= n for c, n in zip(ids, counts)):
                    break
            train_df = pd.concat([train_df[train_df["label_id"] == c].sample(n, random_state=seed)
                                  for c, n in zip(ids, counts)])
        else:
            train_df, _ = train_test_split(
                train_df, stratify=train_df["label_id"], train_size=n_train, random_state=seed)
        test_df, _ = train_test_split(
            test_df, stratify=test_df["label_id"], train_size=n_test, random_state=seed)
        train_df, dev_df = train_test_split(
            train_df, stratify=train_df["label_id"], test_size=0.2, shuffle=True, random_state=seed)

        self.X_train, self.y_train = train_df["text"].to_numpy(), train_df["label"].to_numpy()
        self.X_dev, self.y_dev = dev_df["text"].to_numpy(), dev_df["label"].to_numpy()
        self.X_test, self.y_test = test_df["text"].to_numpy(), test_df["label"].to_numpy()
        if relabel:
            # semantically unrelated labels: every class name becomes a nonsense
            # word (the same mapping for every seed), so the LLM can only learn
            # the label meanings from the examples in the prompt
            names = nonsense_words(len(np.unique(self.y_train)))
            self.label_map = dict(zip(np.unique(self.y_train), names))
            rename = np.vectorize(self.label_map.get, otypes=[object])
            self.y_train, self.y_dev = rename(self.y_train), rename(self.y_dev)
            self.y_test = rename(self.y_test)
        self.classes = np.unique(self.y_train)
        self.label2id = {l: i for i, l in enumerate(self.classes)}
        self._emb, self._sets, self._pred_sets = {}, {}, {}

    # -- embeddings ---------------------------------------------------------
    def embeddings(self, emb):
        """(train, dev, test) embeddings; dense ones are cached on disk."""
        if emb in self._emb:
            return self._emb[emb]
        if emb == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer().fit(self.X_train)
            out = tuple(tfidf.transform(X).toarray() for X in (self.X_train, self.X_dev, self.X_test))
        else:
            # the cache is keyed on the texts themselves, so a change to the
            # subsample can never be paired with stale embeddings
            import hashlib
            digest = hashlib.sha1("\x00".join(
                [*self.X_train, *self.X_dev, *self.X_test]).encode()).hexdigest()[:12]
            path = os.path.join(CACHE_DIR, f"{self.dataset}-seed{self.seed}-{emb}-{digest}.npz")
            if os.path.exists(path):
                z = np.load(path)
                out = (z["train"], z["dev"], z["test"])
            else:
                phi = _dense_encoder(emb)
                out = tuple(phi(X.tolist()) for X in (self.X_train, self.X_dev, self.X_test))
                os.makedirs(CACHE_DIR, exist_ok=True)
                # write-then-rename: parallel jobs may embed the same data
                tmp = f"{path}.{os.getpid()}.tmp.npz"
                np.savez(tmp, train=out[0], dev=out[1], test=out[2])
                os.replace(tmp, path)
        self._emb[emb] = out
        return out

    # -- conformal base classifier ------------------------------------------
    def base_classifier(self, emb, clf, class_cond=True):
        if (emb, clf, class_cond) in self._sets:
            return self._sets[(emb, clf, class_cond)]
        from crepes import WrapClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        E_train, E_dev, _ = self.embeddings(emb)
        learner = LogisticRegression() if clf == "lr" else SVC(probability=True, random_state=0)
        wrapped = WrapClassifier(learner)
        wrapped.fit(E_train, [self.label2id[y] for y in self.y_train])
        wrapped.calibrate(E_dev, [self.label2id[y] for y in self.y_dev], class_cond=class_cond)
        self._sets[(emb, clf, class_cond)] = wrapped
        return wrapped

    def prediction_sets(self, emb, clf, alpha, class_cond=True):
        """Per-test-instance conformal prediction sets (arrays of label strings)
        and the base classifier's own point predictions."""
        key = (emb, clf, alpha, class_cond)
        if key not in self._pred_sets:
            wrapped = self.base_classifier(emb, clf, class_cond)
            E_test = self.embeddings(emb)[2]
            # crepes smooths p-values with random tie-breaking; seeding it makes
            # every configuration sharing (emb, clf, alpha) see the same sets
            sets = wrapped.predict_set(E_test, confidence=1 - alpha, seed=self.seed).astype(bool)
            point = self.classes[wrapped.predict(E_test)]
            self._pred_sets[key] = ([self.classes[s] for s in sets], point)
        return self._pred_sets[key]

    def candidate_sets(self, method, emb, clf, alpha):
        """Candidate label sets for any NARROWING method, plus point predictions."""
        conformal, point = self.prediction_sets(emb, clf, alpha)
        if method == "cicle":
            return conformal, point
        if method == "marginal":
            return self.prediction_sets(emb, clf, alpha, class_cond=False)
        if method == "oracle":
            rng = np.random.default_rng(self.seed)
            m = max(1, int(round(np.mean([len(s) for s in conformal]))))
            sets = []
            for y in self.y_test:
                others = self.classes[self.classes != y]
                chosen = rng.choice(others, size=min(m - 1, len(others)), replace=False)
                sets.append(np.sort(np.append(chosen, y)))
            return sets, point
        proba = self.base_classifier(emb, clf).predict_proba(self.embeddings(emb)[2])
        ranked = np.argsort(-proba, axis=1)
        if method == "topk":
            m = max(1, int(round(np.mean([len(s) for s in conformal]))))
            sizes = np.full(len(proba), m)
        else:  # mass
            cumulative = np.cumsum(np.take_along_axis(proba, ranked, axis=1), axis=1)
            sizes = np.minimum((cumulative < 1 - alpha).sum(axis=1) + 1, proba.shape[1])
        # keep the label order used for conformal sets (alphabetical)
        return [self.classes[np.sort(r[:n])] for r, n in zip(ranked, sizes)], point

    # -- few-shot example retrieval -----------------------------------------
    def retrieve(self, emb, test_idx, allowed, k, variant, random=False):
        """Indices into the training set of the retrieved examples, most
        similar first. `allowed` restricts retrieval to those classes.
        fixed: the k most similar examples overall; pc: k per class.
        random=True draws the examples at random instead (seeded per instance)."""
        E_train, _, E_test = self.embeddings(emb)
        if random:
            rng = np.random.default_rng([self.seed, test_idx])
            if variant == "fixed":
                cand = np.flatnonzero(np.isin(self.y_train, list(allowed)))
                return rng.choice(cand, size=min(k, len(cand)), replace=False)
            chosen = [rng.choice(c, size=min(k, len(c)), replace=False)
                      for y in allowed for c in [np.flatnonzero(self.y_train == y)]]
            chosen = np.concatenate(chosen).astype(int)
            return rng.permutation(chosen)
        sim = _cosine(E_test[test_idx], E_train)
        if variant == "fixed":
            cand = np.flatnonzero(np.isin(self.y_train, list(allowed)))
            chosen = cand[np.argsort(sim[cand])[::-1][:k]]
        else:
            chosen = []
            for y in allowed:
                cand = np.flatnonzero(self.y_train == y)
                chosen.extend(cand[np.argsort(sim[cand])[::-1][:k]])
            chosen = np.array(chosen, dtype=int)
        return chosen[np.argsort(sim[chosen])[::-1]]


def _cosine(v, M):
    norms = np.linalg.norm(M, axis=1) * np.linalg.norm(v)
    return (M @ v) / np.where(norms == 0, 1.0, norms)


def _dense_encoder(emb):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if emb == "minilm":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        return lambda texts: model.encode(texts, convert_to_numpy=True, batch_size=32)
    if emb == "contriever":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        model = AutoModel.from_pretrained("facebook/contriever").to(device).eval()

        def phi(texts, batch_size=32):
            chunks = []
            for i in range(0, len(texts), batch_size):
                inputs = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                   return_tensors="pt", max_length=512).to(device)
                with torch.no_grad():
                    hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                chunks.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
            return np.vstack(chunks)
        return phi
    raise ValueError(f"unknown embedding: {emb}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def _quote(s):
    return s.replace('"', "'")


def build_alt_prompt(task, text, examples, candidates):
    """A second template for the prompt-robustness check: different wording
    and layout, labels in the (shuffled) order given, and examples in the
    order given (the caller passes the most similar example last)."""
    sep = "; " if any("," in str(c) for c in candidates) else ", "
    lines = [f"{task} Choose exactly one of these labels: {sep.join(candidates)}."]
    if examples:
        lines.append("Labelled examples:")
        for x, y in examples:
            lines.append(f"Text: {_quote(x)}\nLabel: {y}")
    lines.append(f"Now label this text. Answer with the label only.\nText: {_quote(text)}\nLabel:")
    return "\n\n".join(lines)


def build_prompt(task, text, examples, candidates):
    """examples: list of (text, label), most similar first.
    candidates: labels the LLM may answer with."""
    # label names that contain commas are separated by semicolons
    sep = "; " if any("," in str(c) for c in candidates) else ", "
    context = f"{task} The possible classes are: {sep.join(candidates)}."
    if examples:
        context += " Here are some labelled examples:\n"
        for x, y in examples:
            context += f'\n"{_quote(x)}" => {y}'
    return (f"{context}\n\nPlease predict the correct class for the following sample. "
            f'Only provide the class label.\n\n"{_quote(text)}" => ')


def build_legacy_prompt(task, text, examples, candidates, method):
    """The prompts used in the original notebooks: only zero-shot lists
    the classes, and CICLe / few-shot differ in their wording."""
    task = task.rstrip(".")
    if method == "zeroshot":
        return (f"{task}. The possible classes are: {', '.join(candidates)}.\n\n"
                "Please predict the correct class for the following sample. "
                f'Output only the class label, nothing else.\n\n"{_quote(text)}" => ')
    intro = ("Here are some labelled examples sorted from most probable to least probable:"
             if method == "cicle" else "Here are some labelled examples:")
    context = f"{task}. {intro}\n"
    for x, y in examples:
        context += f'\n"{_quote(x)}" => {y}'
    return (f"{context}\n\nPlease predict the correct class for the following sample. "
            f'Only provide the class label.\n\n"{_quote(text)}" => ')


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
_STRIP = " \t\r\n\"'`*_.:,;!()[]{}<>=#-"


def _normalise(s):
    s = unicodedata.normalize("NFKC", s).replace("️", "")  # emoji variation selector
    return s.strip(_STRIP).casefold()


def parse_output(raw, classes):
    """Map a raw LLM output to one of `classes`, or None if it matches none.
    Only surface form is normalised (case, quotes, markdown, punctuation,
    trailing text after the label); no semantic matching is attempted."""
    lines = [l for l in raw.splitlines() if l.strip(_STRIP)]
    if not lines:
        return None
    out = _normalise(lines[0])
    norm = {_normalise(c): c for c in classes}
    if out in norm:
        return norm[out]
    # label followed by extra text, e.g. "Sports (football)" or "7 steps":
    hits = [n for n in norm
            if out.startswith(n) and not (out[len(n)].isalnum() and n[-1].isalnum())]
    if hits:
        return norm[max(hits, key=len)]
    return None


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
class LLM:
    def __init__(self, name, token_budget=12000):
        import torch
        import transformers
        transformers.logging.set_verbosity_error()
        # the text-generation pipeline resolves the right model class for
        # every model we use (Ministral-3 is not an AutoModelForCausalLM)
        pipe = transformers.pipeline("text-generation", model=MODELS[name],
                                     model_kwargs={"dtype": torch.bfloat16}, device_map="auto")
        self.torch = torch
        self.model, self.tokenizer = pipe.model.eval(), pipe.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.token_budget = token_budget  # max padded tokens per batch

    def label_budget(self, classes):
        """Enough new tokens to spell out the longest label."""
        longest = max(len(self.tokenizer(str(c), add_special_tokens=False)["input_ids"])
                      for c in classes)
        return longest + 4

    def encode(self, prompts):
        return [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True,
            return_dict=True)["input_ids"] for p in prompts]

    def generate(self, prompts, max_new_tokens, batched=True):
        """Greedy completions for a list of user prompts.
        Returns (outputs, prompt token counts)."""
        from tqdm import tqdm
        torch, tok = self.torch, self.tokenizer
        ids = self.encode(prompts)
        lengths = [len(x) for x in ids]
        order = sorted(range(len(ids)), key=lambda i: -lengths[i])
        outputs = [None] * len(ids)
        bar = tqdm(total=len(ids), desc="  prompting", leave=False)
        start = 0
        while start < len(order):
            size = max(1, self.token_budget // lengths[order[start]]) if batched else 1
            while True:
                batch = order[start:start + size]
                width = lengths[batch[0]]
                input_ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, i in enumerate(batch):
                    input_ids[row, width - lengths[i]:] = torch.tensor(ids[i])
                    mask[row, width - lengths[i]:] = 1
                try:
                    with torch.no_grad():
                        gen = self.model.generate(
                            input_ids=input_ids.to(self.model.device),
                            attention_mask=mask.to(self.model.device),
                            max_new_tokens=max_new_tokens, do_sample=False,
                            temperature=None, top_p=None, top_k=None,
                            pad_token_id=tok.pad_token_id)
                    break
                except torch.OutOfMemoryError:
                    # halve the batch and retry; a single prompt that does not fit is fatal
                    torch.cuda.empty_cache()
                    if size == 1:
                        raise
                    size = max(1, size // 2)
            for row, i in enumerate(batch):
                outputs[i] = tok.decode(gen[row, width:], skip_special_tokens=True).strip()
            start += len(batch)
            bar.update(len(batch))
        bar.close()
        return outputs, lengths


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------
def config_name(cfg):
    parts = [cfg["dataset"], cfg["model"], cfg["method"]]
    if cfg["method"] != "zeroshot":
        parts.append(cfg["emb"])
    if cfg["method"] in NARROWING:
        parts.append(cfg["clf"])
    if cfg["method"] != "zeroshot":
        parts += [f"{cfg['k']}-shots", cfg["variant"]]
    if cfg["method"] in NARROWING:
        parts.append(f"{cfg['alpha']:.2f}-alpha")
    return "-".join(parts)


def run_config(cfg, data, llm, legacy=False, limit=None, batched=True):
    method = cfg["method"]
    n = len(data.X_test) if limit is None else min(limit, len(data.X_test))
    all_classes = list(data.classes)
    task = DATASETS[data.dataset]["task"]
    # "conformal_set" in the records holds the candidate set of whichever
    # narrowing method is used (the key name predates topk / mass)
    if legacy and method != "zeroshot":
        task = DATASETS[data.dataset].get("legacy_task", task)

    narrowing = method in NARROWING
    if narrowing:
        sets, point = data.candidate_sets(method, cfg["emb"], cfg["clf"], cfg["alpha"])

    records, prompts, prompt_rows = [], [], []
    for i in range(n):
        rec = {"idx": i, "gold": data.y_test[i], "llm_called": True}
        candidates = all_classes
        if narrowing:
            ps = list(sets[i])
            rec.update(conformal_set=ps, gold_in_set=bool(data.y_test[i] in ps))
            if len(ps) <= 1:
                # singleton: return it directly; empty: fall back to the base classifier
                rec.update(llm_called=False, raw=None, shots=[], prompt_tokens=0,
                           pred=ps[0] if ps else point[i])
                records.append(rec)
                continue
            candidates = ps
        shots = []
        if method != "zeroshot":
            shots = data.retrieve(cfg["emb"], i, candidates, cfg["k"], cfg["variant"],
                                  random=cfg.get("retrieval") == "random")
        examples = [(data.X_train[j], data.y_train[j]) for j in shots]
        rec["shots"] = [int(j) for j in shots]
        if legacy:
            prompts.append(build_legacy_prompt(task, data.X_test[i], examples, candidates, method))
        elif cfg.get("prompt") == "alt":
            rng = np.random.default_rng([data.seed, i])
            shuffled = [candidates[j] for j in rng.permutation(len(candidates))]
            prompts.append(build_alt_prompt(task, data.X_test[i], examples[::-1], shuffled))
        else:
            prompts.append(build_prompt(task, data.X_test[i], examples, candidates))
        prompt_rows.append(len(records))
        records.append(rec)

    max_new = 5 if legacy else llm.label_budget(all_classes)
    outputs, lengths = llm.generate(prompts, max_new, batched=batched)
    for row, raw, length in zip(prompt_rows, outputs, lengths):
        rec = records[row]
        if legacy:
            pred = raw if raw in all_classes else None  # exact string match
        else:
            pred = parse_output(raw, all_classes)
        rec.update(raw=raw, pred=pred, prompt_tokens=length)
        if narrowing:
            rec["pred_in_set"] = pred in rec["conformal_set"]

    return {"config": {**cfg, "seed": data.seed, "imbalance": data.imbalance, "n_train": data.n_train,
                       "label_map": getattr(data, "label_map", None),
                       "legacy_prompt": legacy, "n_test": n},
            "example_prompt": prompts[0] if prompts else None,
            "metrics": compute_metrics(records, all_classes),
            "records": records}


def compute_metrics(records, classes):
    from sklearn.metrics import accuracy_score, f1_score
    gold = [r["gold"] for r in records]
    pred = [r["pred"] if r["pred"] is not None else "<invalid>" for r in records]
    called = [r for r in records if r["llm_called"]]
    present = sorted(set(gold))
    m = {
        "accuracy": accuracy_score(gold, pred),
        # macro-F1 over the classes that occur in the test sample
        "macro_f1": f1_score(gold, pred, labels=present, average="macro", zero_division=0),
        # as in the original notebooks: over every training class
        "macro_f1_all_classes": f1_score(gold, pred, labels=list(classes), average="macro",
                                         zero_division=0),
        "invalid_rate": float(np.mean([r["pred"] is None for r in records])),
        "llm_call_rate": len(called) / len(records),
        "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in called])) if called else 0.0,
        "mean_shots": float(np.mean([len(r["shots"]) for r in called])) if called else 0.0,
    }
    if "conformal_set" in records[0]:
        sizes = [len(r["conformal_set"]) for r in records]
        m.update(coverage=float(np.mean([r["gold_in_set"] for r in records])),
                 mean_set_size=float(np.mean(sizes)),
                 singleton_rate=float(np.mean([s == 1 for s in sizes])),
                 empty_rate=float(np.mean([s == 0 for s in sizes])),
                 pred_outside_set_rate=float(np.mean(
                     [not r.get("pred_in_set", True) for r in records])))
    return m


# ---------------------------------------------------------------------------
def expand_configs(args):
    configs = []
    for method in args.methods:
        if method == "zeroshot":
            configs.append({"method": method})
            continue
        for emb, k, variant in itertools.product(args.embeddings, args.shots, args.variants):
            base = {"method": method, "emb": emb, "k": k, "variant": variant}
            if method == "fewshot":
                configs.append(base)
            else:
                for clf, alpha in itertools.product(args.classifiers, args.alphas):
                    configs.append({**base, "clf": clf, "alpha": alpha})
    tag = dataset_tag(args.dataset, args.imbalance, args.relabel, args.n_train,
                      args.prompt, args.retrieval)
    return [{"dataset": tag, "model": args.model, "prompt": args.prompt,
             "retrieval": args.retrieval, **c} for c in configs]


def parse_args():
    csv = lambda cast: (lambda s: [cast(x) for x in s.split(",")])
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--methods", type=csv(str), default=["fewshot", "cicle"],
                   help="comma-separated subset of zeroshot,fewshot,cicle,topk,mass,marginal,oracle")
    p.add_argument("--variants", type=csv(str), default=["fixed", "pc"])
    p.add_argument("--embeddings", type=csv(str), default=["minilm"])
    p.add_argument("--classifiers", type=csv(str), default=["lr"])
    p.add_argument("--shots", type=csv(int), default=[1, 2, 4, 8])
    p.add_argument("--alphas", type=csv(float), default=[0.05])
    p.add_argument("--seeds", type=csv(int), default=[42])
    p.add_argument("--imbalance", type=float, default=1.0,
                   help="resample the training pool to this largest/smallest class ratio; "
                        "results go to <dataset>-imb<ratio>/")
    p.add_argument("--n-train", type=int, default=2000,
                   help="size of the labelled pool (train + calibration); results go to "
                        "<dataset>-n<size>/ when not 2000")
    p.add_argument("--prompt", choices=["default", "alt"], default="default",
                   help="alt: second template, labels in random order, most similar example "
                        "last; results go to <dataset>-altprompt/")
    p.add_argument("--retrieval", choices=["similar", "random"], default="similar",
                   help="random: examples drawn at random from the allowed classes; "
                        "results go to <dataset>-random/")
    p.add_argument("--relabel", action="store_true",
                   help="replace every label name with a nonsense word; "
                        "results go to <dataset>-relabel/")
    p.add_argument("--gpu", type=str, default=None, help="sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--out-dir", type=str, default=os.path.join(BASE_DIR, "results"))
    p.add_argument("--legacy-prompt", action="store_true",
                   help="reproduce the original notebooks: no label list in few-shot/CICLe "
                        "prompts, 5 new tokens, exact string matching")
    p.add_argument("--limit", type=int, default=None, help="only the first N test instances")
    p.add_argument("--no-batching", action="store_true", help="one prompt per forward pass")
    p.add_argument("--force", action="store_true", help="re-run existing results")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    todo = []
    for seed in args.seeds:
        for cfg in expand_configs(args):
            out_dir = os.path.join(args.out_dir, cfg["dataset"], f"seed-{seed}")
            path = os.path.join(out_dir, config_name(cfg) + ".json")
            if args.force or not os.path.exists(path):
                todo.append((seed, cfg, path))
    print(f"{len(todo)} configuration(s) to run")
    if args.dry_run or not todo:
        for _, _, path in todo:
            print("  [dry-run]", os.path.basename(path))
        return

    # compute every embedding before the LLM takes the GPU:
    datasets = {}
    for seed in sorted({s for s, _, _ in todo}):
        datasets[seed] = Data(args.dataset, seed, n_train=args.n_train,
                              imbalance=args.imbalance, relabel=args.relabel)
        for emb in sorted({c["emb"] for s, c, _ in todo if s == seed and "emb" in c}):
            datasets[seed].embeddings(emb)

    llm = LLM(args.model)
    failed = 0
    for seed, cfg, path in todo:
        print(f"[seed {seed}] {os.path.basename(path)}")
        try:
            result = run_config(cfg, datasets[seed], llm, legacy=args.legacy_prompt,
                                limit=args.limit, batched=not args.no_batching)
        except Exception as e:  # e.g. out of memory on one long-prompt configuration
            print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
            failed += 1
            llm.torch.cuda.empty_cache()
            continue
        m = result["metrics"]
        print(f"  macro-F1 {100 * m['macro_f1']:.2f}  acc {100 * m['accuracy']:.2f}  "
              f"invalid {100 * m['invalid_rate']:.1f}%"
              + (f"  coverage {100 * m['coverage']:.1f}%  set size {m['mean_set_size']:.2f}"
                 if "coverage" in m else ""))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=1, ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
