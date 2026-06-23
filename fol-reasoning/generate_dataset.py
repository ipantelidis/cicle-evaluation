"""
Synthetic first-order-logic reasoning dataset generator.

Each example is a short passage containing:
  - several ground facts ("X is <predicate>.")
  - several universally-quantified rules ("If something is <P>, then it is <Q>.")
  - a query statement that is entailed by chaining `d` of those rules
    starting from one of the given facts.

The label is `d`, the minimal number of rule applications needed to prove
the query - i.e. the number of reasoning steps.

All entity names and predicates are procedurally-generated nonsense words,
so the resulting text (and the specific chains of reasoning) should not
overlap with anything an LLM has seen during pretraining.
"""

import argparse
import csv
import random

import pandas as pd


# ---------------------------------------------------------------------------
# Nonsense vocabulary generation
# ---------------------------------------------------------------------------

CONSONANTS = list("bcdfgklmnprstvz")
VOWELS = list("aeiou")


def make_word(rng, min_syllables=2, max_syllables=3):
    n = rng.randint(min_syllables, max_syllables)
    word = ""
    for _ in range(n):
        word += rng.choice(CONSONANTS) + rng.choice(VOWELS)
    if rng.random() < 0.5:
        word += rng.choice(CONSONANTS)
    return word


def build_vocab(rng, n_entities, n_predicates):
    entities, predicates = set(), set()
    while len(entities) < n_entities:
        entities.add(make_word(rng, 2, 3).capitalize())
    while len(predicates) < n_predicates:
        predicates.add(make_word(rng, 2, 4))
    return sorted(entities), sorted(predicates)


# ---------------------------------------------------------------------------
# Sentence templates
# ---------------------------------------------------------------------------

FACT_TEMPLATES = [
    "{entity} is {pred}.",
    "{entity} is known to be {pred}.",
    "It is given that {entity} is {pred}.",
]

RULE_TEMPLATES = [
    "If something is {p}, then it is {q}.",
    "All things that are {p} are also {q}.",
    "Anything that is {p} is also {q}.",
    "Every {p} thing is {q}.",
]

QUERY_TEMPLATES = [
    "Question: is {entity} {pred}?",
    "Question: based on the above, is {entity} {pred}?",
]


def render_fact(rng, entity, pred):
    return rng.choice(FACT_TEMPLATES).format(entity=entity, pred=pred)


def render_rule(rng, p, q):
    return rng.choice(RULE_TEMPLATES).format(p=p, q=q)


def render_query(rng, entity, pred):
    return rng.choice(QUERY_TEMPLATES).format(entity=entity, pred=pred)


# ---------------------------------------------------------------------------
# Single example generation
# ---------------------------------------------------------------------------

def generate_example(rng, depth, predicate_pool, entity_pool,
                      n_distractor_entities=(2, 4), n_distractor_rules=(2, 4)):
    """Generate one (text, label) pair with reasoning depth == `depth`."""

    # --- main proof chain ---------------------------------------------------
    # pick depth+1 distinct predicates: P0 -> P1 -> ... -> Pd
    chain_predicates = rng.sample(predicate_pool, depth + 1)
    main_entity = rng.choice(entity_pool)

    sentences = []

    # base fact: main_entity is P0
    sentences.append(render_fact(rng, main_entity, chain_predicates[0]))

    # chain rules: P_{i-1} -> P_i for i = 1..depth
    for i in range(1, depth + 1):
        sentences.append(render_rule(rng, chain_predicates[i - 1], chain_predicates[i]))

    used_predicates = set(chain_predicates)

    # --- distractor entities with their own (irrelevant) facts -------------
    other_entities = [e for e in entity_pool if e != main_entity]
    n_dent = rng.randint(*n_distractor_entities)
    distractor_entities = rng.sample(other_entities, min(n_dent, len(other_entities)))

    remaining_predicates = [p for p in predicate_pool if p not in used_predicates]

    for ent in distractor_entities:
        if not remaining_predicates:
            break
        pred = rng.choice(remaining_predicates)
        sentences.append(render_fact(rng, ent, pred))
        used_predicates.add(pred)
        remaining_predicates = [p for p in predicate_pool if p not in used_predicates]

    # --- distractor rules (not part of the proof chain) ---------------------
    n_drules = rng.randint(*n_distractor_rules)
    for _ in range(n_drules):
        remaining_predicates = [p for p in predicate_pool if p not in used_predicates]
        if len(remaining_predicates) < 2:
            break
        p, q = rng.sample(remaining_predicates, 2)
        sentences.append(render_rule(rng, p, q))
        used_predicates.update({p, q})

    # --- shuffle context and append query ------------------------------------
    rng.shuffle(sentences)
    query = render_query(rng, main_entity, chain_predicates[-1])

    text = " ".join(sentences) + " " + query
    return text, depth


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def generate_dataset(n_samples, max_depth, seed=0,
                      n_entities=40, n_predicates_per_depth=12):
    rng = random.Random(seed)

    # need at least (max_depth + 1) predicates for the longest chain, plus
    # room for distractors; scale pool size with max_depth.
    n_predicates = max(30, n_predicates_per_depth * max_depth)
    entity_pool, predicate_pool = build_vocab(rng, n_entities, n_predicates)

    # depth is sampled uniformly at random per example, so class counts are
    # naturally unbalanced rather than forced to be exactly equal.
    rows = []
    for _ in range(n_samples):
        depth = rng.randint(1, max_depth)
        text, label = generate_example(rng, depth, predicate_pool, entity_pool)
        rows.append({"text": text, "label": label})

    rng.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # other datasets sample 2000 stratified train examples (-> 1600 train / 400 dev)
    # and 1000 stratified test examples from a "train"/"test" pool. We generate
    # generously-sized pools here so the notebook's stratified sampling has room.
    parser.add_argument("--n-train-pool", type=int, default=3000)
    parser.add_argument("--n-test-pool", type=int, default=1500)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="/home/v25/ippa6201/cicle-evaluation/fol-reasoning")
    args = parser.parse_args()

    train_df = generate_dataset(args.n_train_pool, args.max_depth, seed=args.seed)
    test_df = generate_dataset(args.n_test_pool, args.max_depth, seed=args.seed + 1)

    print("train pool:")
    print(train_df["label"].value_counts().sort_index())
    print("test pool:")
    print(test_df["label"].value_counts().sort_index())
    print(train_df.iloc[0]["text"])
    print("label:", train_df.iloc[0]["label"])

    train_path = f"{args.out_dir}/fol_reasoning_train.csv"
    test_path = f"{args.out_dir}/fol_reasoning_test.csv"
    train_df.to_csv(train_path, index=False, quoting=csv.QUOTE_ALL)
    test_df.to_csv(test_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"train: {len(train_df)} rows -> {train_path}")
    print(f"test: {len(test_df)} rows -> {test_path}")
