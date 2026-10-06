#!/usr/bin/env python3
"""
Supervised baselines that use no LLM: the base classifier CICLe narrows
with, evaluated on its own. Results are written next to the LLM results,
in the same per-instance format, as
results/<dataset>/seed-<n>/<dataset>-base-<embedding>-<classifier>.json.

With --finetune it instead fine-tunes an encoder (e.g. roberta-base) on the
same 1,600 training examples, picking the epoch on the 400 calibration
examples, and writes <dataset>-finetuned-<encoder>.json.

Usage:
    python baselines.py
    python baselines.py --finetune FacebookAI/roberta-base --gpu 0
    python baselines.py --datasets yahoo-answers --imbalance 100 --embeddings minilm,tfidf
"""
import argparse
import itertools
import json
import os

import numpy as np

from experiment import BASE_DIR, DATASETS, Data, compute_metrics


def finetune(data, encoder, epochs=10, lr=2e-5, batch_size=16, max_length=256):
    """Fine-tunes `encoder` on the training split and returns test predictions
    from the epoch with the best macro-F1 on the calibration split."""
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(data.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    model = AutoModelForSequenceClassification.from_pretrained(
        encoder, num_labels=len(data.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(data.X_train) / batch_size))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / (0.1 * steps), 1.0) * max(0.0, 1 - step / steps))

    def batches(texts):
        for i in range(0, len(texts), batch_size):
            yield tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)

    def predict(texts):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(**b).logits.argmax(-1).cpu().numpy()
                                   for b in batches(texts)])

    y_train = np.array([data.label2id[y] for y in data.y_train])
    y_dev = np.array([data.label2id[y] for y in data.y_dev])
    rng = np.random.default_rng(data.seed)
    best_f1, best_pred = -1.0, None
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(y_train))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = tokenizer(list(data.X_train[idx]), padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt").to(device)
            loss = model(**batch, labels=torch.tensor(y_train[idx], device=device)).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        dev_f1 = f1_score(y_dev, predict(data.X_dev), average="macro", zero_division=0)
        if dev_f1 > best_f1:
            best_f1, best_pred = dev_f1, predict(data.X_test)
    return data.classes[best_pred]


def main():
    csv = lambda cast: (lambda s: [cast(x) for x in s.split(",")])
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=csv(str),
                   default=["yahoo-answers", "sst", "semeval-18", "go-emotions"])
    p.add_argument("--embeddings", type=csv(str), default=["minilm", "tfidf"])
    p.add_argument("--classifiers", type=csv(str), default=["lr", "svm"])
    p.add_argument("--seeds", type=csv(int), default=[42, 43, 44])
    p.add_argument("--imbalance", type=float, default=1.0)
    p.add_argument("--finetune", default=None, help="encoder to fine-tune, e.g. FacebookAI/roberta-base")
    p.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--out-dir", default=os.path.join(BASE_DIR, "results"))
    args = p.parse_args()
    assert all(d in DATASETS for d in args.datasets)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    for dataset, seed in itertools.product(args.datasets, args.seeds):
        data = Data(dataset, seed, imbalance=args.imbalance)
        if args.finetune:
            settings = [(args.finetune.split("/")[-1], None)]
        else:
            settings = list(itertools.product(args.embeddings, args.classifiers))
        for emb, clf in settings:
            name = (f"{data.tag}-finetuned-{emb}" if args.finetune
                    else f"{data.tag}-base-{emb}-{clf}")
            path = os.path.join(args.out_dir, data.tag, f"seed-{seed}", name + ".json")
            if os.path.exists(path):
                continue
            if args.finetune:
                pred = finetune(data, args.finetune)
            else:
                # the classifier is fitted on the 1,600 training examples only,
                # exactly as it is inside CICLe
                wrapped = data.base_classifier(emb, clf)
                pred = data.classes[wrapped.predict(data.embeddings(emb)[2])]
            records = [{"idx": i, "gold": g, "pred": p, "llm_called": False, "raw": None,
                        "shots": [], "prompt_tokens": 0}
                       for i, (g, p) in enumerate(zip(data.y_test, pred))]
            metrics = compute_metrics(records, list(data.classes))
            cfg = {"dataset": data.tag, "model": "none",
                   "method": "finetuned" if args.finetune else "base", "emb": emb,
                   "clf": clf, "seed": seed, "imbalance": data.imbalance,
                   "legacy_prompt": False, "n_test": len(records)}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"config": cfg, "metrics": metrics, "records": records}, f, indent=1,
                          ensure_ascii=False,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o))
            print(f"[seed {seed}] {name}: macro-F1 {100 * metrics['macro_f1']:.2f}  "
                  f"acc {100 * metrics['accuracy']:.2f}")


if __name__ == "__main__":
    main()
