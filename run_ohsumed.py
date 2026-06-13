import subprocess
import os

BASE = "/home/v25/ippa6201/cicle-evaluation/ohsumed"

# pin everything to a single GPU
GPU_ID = "6"

# models used by cicle/fewshot, ordered smallest -> largest
CICLE_MODELS = [
    "llama-3.2-3b", "qwen-2.5-3b", "ministral-3b",
    "mistral-7b-v0.3", "qwen-2.5-7b", "llama-3.1-8b",
]
# large zero-shot-only anchor models, ordered smallest -> largest (run last)
LARGE_ZEROSHOT_MODELS = ["mistral-nemo-2407", "qwen-2.5-32b", "llama-3.1-70b"]

EMBEDDINGS = ["contriever", "minilm", "tfidf"]
CLASSIFIERS = ["lr", "svm"]
SHOTS = [1, 2, 4, 8]
ALPHAS = [0.01, 0.05, 0.10, 0.20]

notebooks = []

# tfidf baseline (CPU-only, run first)
for clf in CLASSIFIERS:
    notebooks.append(f"{BASE}/ohsumed-tfidf-{clf}-2.0k-samples.ipynb")

# fewshot (small -> large models)
for model in CICLE_MODELS:
    for emb in EMBEDDINGS:
        for shots in SHOTS:
            for variant in ["fixed", "pc"]:
                notebooks.append(
                    f"{BASE}/ohsumed-{model}-fewshot-{emb}-2.0k-samples-{shots}-shots-{variant}.ipynb"
                )

# cicle (small -> large models)
for model in CICLE_MODELS:
    for emb in EMBEDDINGS:
        for clf in CLASSIFIERS:
            for shots in SHOTS:
                alphas = ALPHAS if shots == 2 else [0.05]
                for alpha in alphas:
                    for variant in ["fixed", "pc"]:
                        notebooks.append(
                            f"{BASE}/ohsumed-{model}-cicle-{emb}-{clf}-2.0k-samples-"
                            f"{shots}-shots-{variant}-{alpha:.2f}-α.ipynb"
                        )

# zeroshot for the 6 cicle/fewshot models (small -> large)
for model in CICLE_MODELS:
    notebooks.append(f"{BASE}/ohsumed-{model}-zeroshot-2.0k-samples.ipynb")

# zeroshot for the large anchor models (smallest -> largest, run very last)
for model in LARGE_ZEROSHOT_MODELS:
    notebooks.append(f"{BASE}/ohsumed-{model}-zeroshot-2.0k-samples.ipynb")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = GPU_ID

for nb in notebooks:
    print(f"Running {nb}...")
    subprocess.run([
        "/home/v25/ippa6201/cicle-evaluation/.venv/bin/jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=7200",
        nb
    ], check=True, env=env)
