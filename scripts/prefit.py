"""Thin launcher for the hexfield behavior-cloning prefit (imports hexfield via
PYTHONPATH; hexfield is never pip-installed — see README).

Usually invoked through scripts/prefit_launch.sh, but can be run directly:
    python scripts/prefit.py --data data/prefit --out runs/hexfield_bc_1 --epochs 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "hexfield" / "python"))

from hexfield.prefit import main

if __name__ == "__main__":
    main()
