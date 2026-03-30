import os
import nbformat

def fix_notebook(path):
    try:
        nb = nbformat.read(path, as_version=4)
        changed = False

        for cell in nb.cells:
            if cell.cell_type == "code":
                # Fix missing execution_count
                if "execution_count" not in cell:
                    cell["execution_count"] = None
                    changed = True

                # Fix missing outputs
                if "outputs" not in cell:
                    cell["outputs"] = []
                    changed = True

        if changed:
            nbformat.write(nb, path)
            print(f"✔ Fixed: {path}")
        else:
            print(f"– OK: {path}")

    except Exception as e:
        print(f"✘ Error in {path}: {e}")


def fix_all_notebooks(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                fix_notebook(os.path.join(root, file))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing notebooks")
    args = parser.parse_args()

    fix_all_notebooks(args.directory)