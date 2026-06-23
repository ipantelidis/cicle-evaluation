"""
Ablation generator for the FOL-Reasoning synthetic benchmark.

Reuses every piece of generate_dataset.py's text-rendering logic
unchanged, and exposes three independent knobs as separate CLI flags,
so each axis can be varied while holding the other two at the baseline
("medium") level used by the main paper's FOL-Reasoning benchmark:

  --max-depth N        number of classes (= N), varies class cardinality
  --skew S             depth-sampling skew, S=1.0 is uniform (balanced);
                        S<1.0 makes shallow depths exponentially more
                        common, varying class imbalance
  --distractor-range LO HI
                        range for both n_distractor_entities and
                        n_distractor_rules, varies passage length

Calibration (max_depth=12, n_train_pool=3000): skew=1.0 -> ~1.0x
imbalance (current paper benchmark), skew=0.83 -> ~7.8x (comparable to
SemEval-18's 8.7x), skew=0.65 -> ~114x (comparable to GoEmotions' 184.7x,
without pushing the rarest class below ~9 examples in the train pool).

Example variant grid (one axis at a time, baseline = max_depth=12,
skew=1.0, distractor_range=(2,4)):

  classes:    max-depth in {4, 8, 12, 16}
  imbalance:  skew      in {1.0, 0.83, 0.65}     (at max-depth=12)
  length:     distractor-range in {(0,1), (2,4), (6,10)}  (at max-depth=12, skew=1.0)
"""
import argparse
import csv
import random

import numpy as np
import pandas as pd

from generate_dataset import (
    build_vocab,
    generate_example,
)


def sample_depth(rng, max_depth, skew):
    if skew >= 1.0:
        return rng.randint(1, max_depth)
    weights = [skew ** d for d in range(max_depth)]
    total = sum(weights)
    weights = [w / total for w in weights]
    return rng.choices(range(1, max_depth + 1), weights=weights, k=1)[0]


def generate_dataset_variant(n_samples, max_depth, skew, distractor_range,
                              seed=0, n_entities=40, n_predicates_per_depth=12):
    rng = random.Random(seed)
    n_predicates = max(30, n_predicates_per_depth * max_depth)
    entity_pool, predicate_pool = build_vocab(rng, n_entities, n_predicates)

    rows = []
    for _ in range(n_samples):
        depth = sample_depth(rng, max_depth, skew)
        text, label = generate_example(
            rng, depth, predicate_pool, entity_pool,
            n_distractor_entities=distractor_range,
            n_distractor_rules=distractor_range,
        )
        rows.append({"text": text, "label": label})

    rng.shuffle(rows)
    return pd.DataFrame(rows)


def report_imbalance(df):
    counts = df["label"].value_counts()
    return counts.max() / counts.min(), counts.min(), counts.max()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-name", type=str, required=True,
                         help="e.g. classes_4, imbalance_high, length_long")
    parser.add_argument("--n-train-pool", type=int, default=3000)
    parser.add_argument("--n-test-pool", type=int, default=1500)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--skew", type=float, default=1.0)
    parser.add_argument("--distractor-range", type=int, nargs=2, default=[2, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                         default="/home/v25/ippa6201/cicle-evaluation/fol-reasoning/ablation")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    drange = tuple(args.distractor_range)
    train_df = generate_dataset_variant(args.n_train_pool, args.max_depth, args.skew,
                                         drange, seed=args.seed)
    test_df = generate_dataset_variant(args.n_test_pool, args.max_depth, args.skew,
                                        drange, seed=args.seed + 1)

    ratio, lo, hi = report_imbalance(train_df)
    print(f"variant={args.variant_name}: max_depth={args.max_depth}, skew={args.skew}, "
          f"distractor_range={drange}")
    print(f"  imbalance ratio={ratio:.1f}x (min={lo}, max={hi})")
    print(f"  mean passage length (chars): {train_df['text'].str.len().mean():.0f}")

    train_path = f"{args.out_dir}/fol_reasoning_{args.variant_name}_train.csv"
    test_path = f"{args.out_dir}/fol_reasoning_{args.variant_name}_test.csv"
    train_df.to_csv(train_path, index=False, quoting=csv.QUOTE_ALL)
    test_df.to_csv(test_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"  -> {train_path}")
    print(f"  -> {test_path}")
