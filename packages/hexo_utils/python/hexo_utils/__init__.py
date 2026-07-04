"""Shared Python utility package for Hexo training and replay.

Search, model encoding, and sample generation live inside the model packages.
This package keeps stable cross-model utilities such as records, the D6
symmetry contract, and the `.hxr` codec facade.

Subsystem status (see README.md for the full map):

- `records.py` (ACTIVE): Python facade over the Rust `.hxr` codec in
  `rust/src/records.rs` + `rust/src/pybridge.rs`. Production callers reach it
  through `packages/hexo_runner/python/hexo_runner/records/record.py`, which
  re-exports these classes; `scripts/_wf_r4_health.py` and
  `analysis/exploration_diversity.py` import it directly.
- `encoding/` (ACTIVE): the D6 symmetry contract (`D6_SIZE`, `D6Symmetry`),
  consumed live by `packages/hexo_train/python/hexo_train/symmetry.py` for
  training-time symmetry augmentation.

The Rust crate (`rust/src`) also exports `hash_state` (state_hash.rs), the
MCTS evaluator-cache key used by the hexo_models and hexgnn crates; it has no
Python surface.
"""

__version__ = "0.1.0"
