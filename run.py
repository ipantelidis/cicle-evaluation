#!/usr/bin/env python3
"""
Unified experiment runner for the CICLe evaluation suite.

Discovers and executes the Jupyter notebooks for one or more datasets,
skipping any notebook whose result file already exists. Replaces the
old per-dataset run_<dataset>.py scripts with a single, configurable
entry point: a "dataset" is any top-level directory containing both
.ipynb files and a results/predictions/ folder, so newly added datasets
are picked up automatically without editing this script.

Examples:
    # List available datasets and how many notebooks each has
    python run.py --list

    # Run everything in one dataset
    python run.py --datasets yahoo-answers

    # Run several datasets in one invocation
    python run.py --datasets yahoo-answers,sst,semeval-18,go-emotions

    # Only the CICLe notebooks for a dataset, on a specific GPU
    python run.py --datasets go-emotions --filter cicle --gpu 0

    # See what would run without actually executing anything
    python run.py --datasets sst --dry-run

    # Re-run notebooks even if a result file already exists
    python run.py --datasets sst --filter zeroshot --force
"""
import argparse
import glob
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JUPYTER = os.path.join(BASE_DIR, ".venv", "bin", "jupyter")


def discover_datasets():
    datasets = []
    for entry in sorted(os.listdir(BASE_DIR)):
        path = os.path.join(BASE_DIR, entry)
        if not os.path.isdir(path):
            continue
        has_notebooks = bool(glob.glob(os.path.join(path, "*.ipynb")))
        has_results_dir = os.path.isdir(os.path.join(path, "results", "predictions"))
        if has_notebooks and has_results_dir:
            datasets.append(entry)
    return datasets


def discover_notebooks(dataset, name_filter=None):
    notebooks = sorted(glob.glob(os.path.join(BASE_DIR, dataset, "*.ipynb")))
    if name_filter:
        notebooks = [nb for nb in notebooks if name_filter in os.path.basename(nb)]
    return notebooks


def result_path(dataset, notebook):
    stem = os.path.basename(notebook)[: -len(".ipynb")]
    return os.path.join(BASE_DIR, dataset, "results", "predictions", f"{stem}.json")


def run_notebook(notebook, timeout):
    cmd = [
        JUPYTER, "nbconvert", "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}", notebook,
    ]
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CICLe experiment notebooks for one or more datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset names (e.g. 'yahoo-answers,sst'). "
             "Defaults to every discovered dataset.",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only run notebooks whose filename contains this substring "
             "(e.g. 'cicle', 'zeroshot', 'contriever').",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run notebooks even if a result file already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run, without executing anything.",
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="Restrict execution to this CUDA device, e.g. '0' or '0,1' "
             "(sets CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--timeout", type=int, default=7200,
        help="Per-notebook execution timeout in seconds (default: 7200).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List discovered datasets and notebook counts, then exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    available = discover_datasets()

    if args.list:
        print("Available datasets:")
        for ds in available:
            print(f"  {ds:20s} ({len(discover_notebooks(ds))} notebooks)")
        return

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
        unknown = [d for d in datasets if d not in available]
        if unknown:
            print(f"Unknown dataset(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)
    else:
        datasets = available

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    grand_ran = grand_skipped = grand_failed = grand_total = 0

    for ds in datasets:
        notebooks = discover_notebooks(ds, name_filter=args.filter)
        print(f"\n=== {ds}: {len(notebooks)} notebook(s) match ===")
        ran = skipped = failed = 0

        for nb in notebooks:
            stem = os.path.basename(nb)[: -len(".ipynb")]
            result = result_path(ds, nb)
            if os.path.exists(result) and not args.force:
                skipped += 1
                continue
            if args.dry_run:
                print(f"  [dry-run] would run: {stem}")
                continue
            print(f"  Running {stem} ...")
            try:
                run_notebook(nb, args.timeout)
                ran += 1
            except subprocess.CalledProcessError as e:
                print(f"  FAILED {stem}: {e}")
                failed += 1

        print(f"  -> {ran} ran, {skipped} skipped, {failed} failed, {len(notebooks)} total")
        grand_ran += ran
        grand_skipped += skipped
        grand_failed += failed
        grand_total += len(notebooks)

    if len(datasets) > 1:
        print(
            f"\n=== Overall: {grand_ran} ran, {grand_skipped} skipped, "
            f"{grand_failed} failed, {grand_total} total ==="
        )


if __name__ == "__main__":
    main()
