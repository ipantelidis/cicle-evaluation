#!/usr/bin/env python3
"""
Runs experiment.py for every (dataset, model) pair across a pool of GPUs.

Each job is one experiment.py process (one LLM load) covering every
requested configuration and seed for that pair; one job runs per GPU at a
time. Finished configurations are skipped, so re-running the same command
resumes after an interruption. Any argument this script does not know is
passed straight through to experiment.py.

Examples:
    python run_grid.py --gpus 0,1,2,3,4,5,6,7 --seeds 42,43,44 \\
        --methods zeroshot,fewshot,cicle

    python run_grid.py --datasets sst --models llama-3.1-8b,qwen-2.5-7b --gpus 0,1

    # a 32B model spread over four GPUs, one job at a time
    python run_grid.py --models qwen-2.5-32b --gpus 0+1+2+3
"""
import argparse
import os
import queue
import subprocess
import sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

CORE_DATASETS = ["yahoo-answers", "sst", "semeval-18", "go-emotions"]
CORE_MODELS = ["llama-3.2-3b", "ministral-3b", "qwen-2.5-3b",
               "mistral-7b-v0.3", "qwen-2.5-7b", "llama-3.1-8b"]


def worker(gpu, jobs, extra):
    while True:
        try:
            dataset, model = jobs.get_nowait()
        except queue.Empty:
            return
        log = os.path.join(LOG_DIR, f"{dataset}-{model}.log")
        # "0+1+2" gives one job several GPUs (for models that do not fit on one)
        cmd = [sys.executable, os.path.join(BASE_DIR, "experiment.py"),
               "--dataset", dataset, "--model", model, "--gpu", gpu.replace("+", ","), *extra]
        print(f"[gpu {gpu}] start  {dataset} / {model}", flush=True)
        with open(log, "a") as f:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        status = "done  " if code == 0 else f"FAILED (exit {code}, see {log})"
        print(f"[gpu {gpu}] {status} {dataset} / {model}", flush=True)


def main():
    csv = lambda s: s.split(",")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=csv, default=CORE_DATASETS)
    p.add_argument("--models", type=csv, default=CORE_MODELS)
    p.add_argument("--gpus", type=csv, default=["0"])
    args, extra = p.parse_known_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    jobs = queue.Queue()
    # larger models first, so the longest jobs are not left for the end
    for model in reversed(args.models):
        for dataset in args.datasets:
            jobs.put((dataset, model))

    threads = [threading.Thread(target=worker, args=(gpu, jobs, extra)) for gpu in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("all jobs finished", flush=True)


if __name__ == "__main__":
    main()
