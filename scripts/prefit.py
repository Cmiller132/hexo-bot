"""Thin launcher for the shrimp behavior-cloning prefit (imports shrimp via
PYTHONPATH; shrimp is never pip-installed — see README).

Usually invoked through scripts/prefit_launch.sh, but can be run directly:
    python scripts/prefit.py --data data/prefit --out runs/shrimp_bc_1 --epochs 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "shrimp" / "python"))

from shrimp.prefit import main

if __name__ == "__main__":
    main()
