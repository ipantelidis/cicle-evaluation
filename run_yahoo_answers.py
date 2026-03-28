import subprocess
import os
import sys

BASE = "/home/v25/ippa6201/cicle-evaluation/yahoo-answers"

notebooks = sorted([
    os.path.join(BASE, f)
    for f in os.listdir("yahoo-answers")
    if f.endswith(".ipynb")
])

print(f"Found {len(notebooks)} notebooks to run.\n")

failed = []
for i, nb in enumerate(notebooks, 1):
    print(f"[{i}/{len(notebooks)}] Running {os.path.basename(nb)} ...")
    result = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=3600",
            nb,
        ]
    )
    if result.returncode != 0:
        print(f"  FAILED: {os.path.basename(nb)}")
        failed.append(nb)
    else:
        print(f"  OK")

print(f"\nDone. {len(notebooks) - len(failed)}/{len(notebooks)} succeeded.")
if failed:
    print("Failed notebooks:")
    for nb in failed:
        print(f"  {nb}")
    sys.exit(1)
