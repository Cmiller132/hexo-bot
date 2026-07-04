"""Fetch the behavior-cloning bootstrap corpus (human Hexo games) from Hugging Face.

Downloads a game corpus used to warm-start training via scripts/prefit_launch.sh.
Requires `huggingface_hub` (pip install huggingface_hub).

Usage:
    python scripts/fetch_corpus.py --out data/hexo-bootstrap-corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO / "data" / "hexo-bootstrap-corpus"),
        help="local directory to download the corpus into",
    )
    parser.add_argument(
        "--repo-id",
        default="timmyburn/hexo-bootstrap-corpus",
        help="Hugging Face dataset repo id",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.out,
    )
    print("downloaded to", path)


if __name__ == "__main__":
    main()
