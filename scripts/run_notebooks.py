"""Re-run the project notebooks in place, in order.

Usage (from the repository root):
    python scripts/run_notebooks.py            # all notebooks/NN_*.ipynb
    python scripts/run_notebooks.py 02         # only notebooks whose name starts with 02

Execution timestamps are not stored in the notebooks, so rerunning unchanged
code on unchanged data produces byte-identical files (all random steps are
seeded through src.preprocessing.RANDOM_STATE).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

notebooks = sorted((ROOT / 'notebooks').glob('[0-9][0-9]_*.ipynb'))
prefixes = sys.argv[1:]
if prefixes:
    notebooks = [nb for nb in notebooks if nb.name.startswith(tuple(prefixes))]

for nb in notebooks:
    print(f'Running {nb.name} ...', flush=True)
    subprocess.run([
        sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
        '--ExecutePreprocessor.timeout=1800',
        '--ExecutePreprocessor.record_timing=False',
        '--ClearMetadataPreprocessor.enabled=True',
        '--ClearMetadataPreprocessor.clear_notebook_metadata=False',
        str(nb),
    ], check=True)
print('Done.')
