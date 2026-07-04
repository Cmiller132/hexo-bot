# Model Card — hexfield (main_7)

The shipped weights are a snapshot of the `hexfield_main_7` training run: an
AlphaZero-style, Gumbel self-play RL agent for Hexo (a Connect6-style game on an
unbounded hex grid). The network is the PyTorch/Triton "hexfield" model.

- **Snapshot epoch:** 18
- **Snapshot date:** 2026-07-03
- **Parameters:** 8,128,812 (~8.1 M)

> The run was **live and early** when this snapshot was taken (epoch 18 of a
> planned 200). Treat the strength numbers below as an early-training signal on
> small samples, not a converged result.

## Files

| File | Contents | Size | Use it for |
| --- | --- | --- | --- |
| `hexfield_main7_infer.pt` | Weights only (`{"meta", "model"}`) | ~31 MB | Playing / studying the bot: dashboard Debug + Match tools, inference, analysis. |
| `hexfield_main7_full.pt` | Full training checkpoint (`{"meta", "model", "optimizer"}`) | ~93 MB | Resuming or continuing training from epoch 18 (carries optimizer state + step counters). |

Both files are stored via Git LFS (see `.gitattributes`: `models/*.pt`). After a
clone you need `git lfs pull` to fetch the real weights.

The inference file is the training checkpoint with the optimizer state stripped
and a small arch-metadata block embedded; it was produced by
`scripts/export_weights.py` and re-verified to load strict into `HexfieldNet`.

## Architecture — env vars are load-bearing

The network architecture is read from environment variables **at import time**,
not from the checkpoint. A checkpoint only loads into a net built with the same
values, so you must export these before instantiating the model or launching any
tool that loads the weights:

```
export HEXFIELD_CHANNELS=192
export HEXFIELD_ATTENTION_HEADS=3
export HEXFIELD_TRUNK=CCACCACCACCACCA
export HEXFIELD_SUPPORT_RADIUS=4
```

- `CHANNELS=192`, `ATTENTION_HEADS=3` (head_dim = 192/3 = 64), and
  `TRUNK=CCACCACCACCACCA` (a stack of conv/conv/attn blocks) fully determine the
  8.1 M-parameter weight shapes. Getting any of these wrong makes `load_state_dict`
  fail.
- `SUPPORT_RADIUS=4` does **not** change the weight shapes (input feature count
  is fixed at 15), but it **is** what the model was trained with: it controls the
  featurization / legal-move support the net expects. Leaving it unset defaults to
  radius 8, which mismatches training and degrades play. Always set it to 4 for
  these weights.

## Training provenance

1. **Warm start (behavioral cloning).** A prefit checkpoint was trained by
   behavioral cloning on the public corpus
   [timmyburn/hexo-bootstrap-corpus](https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus)
   (see `scripts/prefit_launch.sh`; wired via `checkpoint.initialize_from` in
   `configs/hexfield_main_7.toml`). The RL run initialized from prefit epoch 3.
   The warm start is optional — without it, training starts from a random net.
2. **Self-play RL.** 18 epochs of Gumbel-AlphaZero self-play + supervised updates
   + gated eval, on a single **RTX 4070 Ti** (12 GB). The repo deliberately ships
   only the Gumbel search path; classic PUCT+Dirichlet exploration knobs were
   stripped.

Full recipe: `configs/hexfield_main_7.toml`.

## Strength (early, small-sample)

Evaluated vs [SealBot](https://github.com/Ramora0/SealBot) and prior-run anchors
during the run's multi-stage eval (512 search visits/side, unpaired SealBot
games). These are the SealBot edges recorded in the run's eval pool:

| Candidate | Opponent | Wins (cand–opp) | Win rate |
| --- | --- | --- | --- |
| main_7 epoch 5 | SealBot | 22 – 10 | 68.8% |
| main_7 epoch 10 | SealBot | 19 – 13 | 59.4% |

For orientation, at epoch 10 the candidate also went 21–11 vs its own epoch-5
snapshot (improving over itself), but lost to the stronger prior-run anchors
(10–16.7 eff. vs `main5_ep105`; 6–26 vs `main6_ep73`).

**Caveats.** SealBot edges are unpaired "zero-point" measurements over ~32 games
each and are down-weighted in the run's own difference inference — treat them as
directional, not precise Elo. No fresh SealBot rematch was run at epoch 18; the
epoch-18 gated eval that exists is a moves-left-head audit (passed: conv Spearman
0.58, near-end MAE 4.4). The snapshot is early-training on a single consumer GPU.

## How to load

Set the arch env vars (above), then:

```python
import os, torch
from hexfield.model import HexfieldNet   # PYTHONPATH: packages/hexfield/python

model = HexfieldNet(
    channels=int(os.environ["HEXFIELD_CHANNELS"]),            # 192
    attention_heads=int(os.environ["HEXFIELD_ATTENTION_HEADS"]),  # 3
    trunk_layout=os.environ["HEXFIELD_TRUNK"],                # CCACCACCACCACCA
)
payload = torch.load("models/hexfield_main7_infer.pt", map_location="cpu", weights_only=False)
model.load_state_dict(payload["model"], strict=True)
model.eval()
```

**Or just open the dashboard** (`scripts/dashboard.sh`) and let the frontend
loader (`hexo_frontend.debug_infer.load_checkpoint`) read the embedded arch
metadata and instantiate the net for you — no arch env vars needed to play.
The dashboard discovers weights as **runs**, not loose files, so stage the
inference weights into a minimal run layout first
(`runs/shipped/checkpoints/…` + a one-line `manifest.json`). The README quick
start walks through this in [§1c](../README.md#1c-stage-the-shipped-weights-as-a-run-the-dashboard-can-see);
once staged, the `shipped` run appears in both the Debug workbench and the
Match arena.

## Resuming training

`hexfield_main7_full.pt` is a complete checkpoint (model + optimizer + step
counters). Point `checkpoint.initialize_from` in a config at it to seed a new run
from epoch 18's weights and optimizer state. Use the same arch env vars.
