"""hexfield_eq serve-evaluator raylen threading (Phase L3 gate, spec D-S18/D-S34).

The serve evaluator must thread the wire payload's ``raylen`` buffer
((total_nodes, RAYLEN_SLOTS) u8, CSR-flat like ``nbr``) into
``forward_policy_value`` exactly when the net's trunk layout contains ``L``:

  1. L-NET SERVE PARITY — an L-layout net (blockers on) now serves, and the
     serve reply (binned-value decode + legal-prefix prior softmax +
     moves-left) matches the train-path ``forward_policy_value`` over
     ``batching.collate_rows`` on the same positions to the shipped 3e-3 serve
     tolerance (the payload's f16 feats are mirrored into the reference so the
     comparison isolates the serve path, not the wire rounding).
  2. C/A PATH UNCHANGED — a net without L blocks serves through the pre-L
     argument list: every forward call carries exactly the four positional
     tensors + ``request_moves_left`` (no raylen tensor is ever built), and the
     reply is deterministic call-over-call. (Bit-identity vs the pre-change
     tree was verified once with a sha256 baseline harness; this test pins the
     structural invariant that guarantees it.)
  3. GUARDS — an L net refuses a payload with no ``raylen`` key (a pre-L0
     featurizer .so) and refuses an all-zero first raylen buffer when blockers
     are on (the serve twin of ``trainer._check_raylen_once``, spec D-S31); a
     blockers-OFF (geometric) L net accepts all-zero raylen (it never reads
     the values, spec D-S16).
  4. CUDA (gated, skips on CPU) — the same L-net parity through the CUDA
     evaluator (eager, fp16 autocast), plus the Rust-pack path when the .so
     provides ``build_serve_groups``.

Payloads are assembled in-test from ``_rust.featurize_states`` rows following
the payload.rs wire contract (size-DESCENDING rows, u16 nbr sentinel, f16
feats), the ``scripts/_hexfield_serve_ref.py`` harness pattern. Runs in the
hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python (plus the shared
packages). CPU except test 4.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield_eq import constants as C
from hexfield_eq.batching import collate_rows
from hexfield_eq.engine_facts import facts_from_engine
from hexfield_eq.features import build_position, build_ray_lengths
from hexfield_eq.geometry import unpack_action_id
from hexfield_eq.inference import HexfieldEvaluator
from hexfield_eq.losses import decode_binned_value, decode_moves_left
from hexfield_eq.model import HexfieldNet

try:
    from hexfield_eq import _rust
except ImportError:  # pragma: no cover
    _rust = None

needs_rust = pytest.mark.skipif(
    _rust is None, reason="hexfield_eq._rust not built (see the Phase-1 build gate)"
)

RL = C.RAYLEN_SLOTS
NBR_SENTINEL = 0xFFFF
SERVE_ATOL = 3e-3  # the shipped serve parity tolerance (deployment checklist §5)
L_LAYOUT = "CLA"


# --- state / payload assembly (payload.rs wire contract) --------------------------


def _random_state(seed: int, plies: int):
    state = api.new_game()
    rng = random.Random(seed)
    for _ in range(plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            break
    return state


def _battery_states(n_seeds: int = 2, plies=(1, 5, 11)):
    states = []
    for seed in range(n_seeds):
        for p in plies:
            st = _random_state(7000 + seed * 100 + p, p)
            if api.terminal(st) is None:
                states.append(st)
    assert states
    return states


def _build_payload(states, *, request_ml=True, raylen_override=None, drop_raylen=False):
    """Assemble the serve wire payload from featurize_states rows, mirroring
    payload.rs: rows size-DESCENDING (stable by request index), f16 feats, u16
    nbr with the 0xFFFF sentinel, CSR raylen. Returns (payload, ordered_states)
    with ordered_states in payload row order."""
    rows = _rust.featurize_states(states)
    order = sorted(range(len(states)), key=lambda i: (-int(rows[i]["num_nodes"]), i))
    feats, qr, nbr, rl, offsets, legal = [], [], [], [], [0], []
    for i in order:
        r = rows[i]
        n = int(r["num_nodes"])
        feats.append(np.frombuffer(r["feats"], dtype=np.float32).astype(np.float16))
        qr.append(np.frombuffer(r["coords"], dtype=np.int16))
        nb = np.frombuffer(r["nbr"], dtype=np.int32)
        nbr.append(np.where(nb < 0, NBR_SENTINEL, nb).astype(np.uint16))
        rl.append(np.frombuffer(r["raylen"], dtype=np.uint8))
        offsets.append(offsets[-1] + n)
        legal.append(int(r["legal_count"]))
    raylen_bytes = np.concatenate(rl).tobytes()
    if raylen_override is not None:
        raylen_bytes = raylen_override(np.concatenate(rl)).tobytes()
    payload = {
        "abi": 1,
        "shape": (len(order), offsets[-1]),
        "node_feats": np.concatenate(feats).tobytes(),
        "node_qr": np.concatenate(qr).tobytes(),
        "nbr": np.concatenate(nbr).tobytes(),
        "raylen": raylen_bytes,
        "node_row_offsets": offsets,
        "legal_counts": np.asarray(legal, dtype=np.int32).tobytes(),
        "request_moves_left": request_ml,
    }
    if drop_raylen:
        payload.pop("raylen")
    return payload, [states[i] for i in order]


def _train_reference(model, ordered_states, *, request_ml=True):
    """Train-path forward over collate_rows on the SAME positions, with the
    payload's f16 feats rounding mirrored in (isolates the serve path). Returns
    (values, priors_flat, moves_left) as numpy, rows in ordered_states order."""
    rows, raylen_rows, legal_counts = [], [], []
    for st in ordered_states:
        facts = facts_from_engine(api.to_python_state(st))
        sup, feats = build_position(facts)
        feats16 = feats.astype(np.float16).astype(np.float32)  # wire rounding
        rows.append((sup, feats16))
        raylen_rows.append(build_ray_lengths(facts, sup))
        legal_counts.append(sup.legal_count)
    batch = collate_rows(rows, raylen=raylen_rows)
    kwargs = {}
    if "L" in model._trunk_layout:
        kwargs["raylen"] = batch["raylen"]
    with torch.no_grad():
        out = model.forward_policy_value(
            batch["feats"], batch["nbr"], batch["mask"], batch["coords"],
            request_moves_left=request_ml, **kwargs,
        )
    values = decode_binned_value(out["value"].float()).numpy()
    priors = []
    for k, lc in enumerate(legal_counts):
        logits = out["policy"][k, :lc].float()
        priors.append(torch.softmax(logits, dim=0).numpy())
    ml = (
        decode_moves_left(out["moves_left"].float()).numpy() if request_ml else None
    )
    return values, np.concatenate(priors), ml


def _reply_arrays(reply, *, request_ml=True):
    values = np.frombuffer(reply["values_bytes"], dtype=np.float32)
    priors = np.frombuffer(reply["priors_bytes"], dtype=np.float32)
    ml = (
        np.frombuffer(reply["moves_left_bytes"], dtype=np.float32)
        if request_ml
        else None
    )
    return values, priors, ml


def _randomize(model: HexfieldNet, seed: int) -> None:
    """Blanket-randomize params (the ray_block-test idiom): the default init is
    near-identity (LayerScale ~0), which would make the L mask numerically
    invisible and every parity trivially uniform."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)


def _build_l_net(seed: int = 11, **kwargs) -> HexfieldNet:
    torch.manual_seed(seed)
    model = HexfieldNet(trunk_layout=L_LAYOUT, **kwargs).eval()
    _randomize(model, seed + 1)
    return model


# --- 1. L-net serve parity (CPU) ---------------------------------------------------


@needs_rust
def test_l_net_serves_and_matches_train_forward() -> None:
    states = _battery_states()
    payload, ordered = _build_payload(states, request_ml=True)
    model = _build_l_net()
    assert model._ray_blockers  # env default: blockers on — raylen is live
    ev = HexfieldEvaluator(model, device="cpu")
    assert ev._needs_raylen
    reply = ev.evaluate_payload(payload)
    sv, sp, sml = _reply_arrays(reply)
    tv, tp, tml = _train_reference(model, ordered)
    assert sv.shape == tv.shape and sp.shape == tp.shape
    np.testing.assert_allclose(sv, tv, atol=SERVE_ATOL, rtol=0)
    np.testing.assert_allclose(sp, tp, atol=SERVE_ATOL, rtol=0)
    np.testing.assert_allclose(sml, tml, atol=1.0, rtol=1e-3)  # decoded plies

    # Size-1 group (the pad-batch corner) serves too.
    payload1, ordered1 = _build_payload(states[:1], request_ml=False)
    sv1, sp1, _ = _reply_arrays(
        ev.evaluate_payload(payload1), request_ml=False
    )
    tv1, tp1, _ = _train_reference(model, ordered1, request_ml=False)
    np.testing.assert_allclose(sv1, tv1, atol=SERVE_ATOL, rtol=0)
    np.testing.assert_allclose(sp1, tp1, atol=SERVE_ATOL, rtol=0)


@needs_rust
def test_deferred_host_legal_gather_is_byte_exact(monkeypatch) -> None:
    """The XPU sync-removal path changes scheduling, never reply values."""
    states = _battery_states(n_seeds=1)
    payload, _ = _build_payload(states, request_ml=True)
    payload["request_logits"] = True
    model = _build_l_net()

    monkeypatch.setenv("HEXFIELD_DEFER_DECODE", "0")
    monkeypatch.setenv("HEXFIELD_HOST_LEGAL_GATHER", "0")
    monkeypatch.setenv("HEXFIELD_DECODE_CACHE", "0")
    baseline = HexfieldEvaluator(model, device="cpu").evaluate_payload(dict(payload))

    monkeypatch.setenv("HEXFIELD_DEFER_DECODE", "1")
    monkeypatch.setenv("HEXFIELD_HOST_LEGAL_GATHER", "1")
    monkeypatch.setenv("HEXFIELD_DECODE_CACHE", "1")
    optimized = HexfieldEvaluator(model, device="cpu").evaluate_payload(dict(payload))

    assert optimized.keys() == baseline.keys()
    for key in baseline:
        assert optimized[key] == baseline[key], key


@needs_rust
def test_l_net_raylen_actually_changes_outputs() -> None:
    """The threaded raylen is LIVE: zeroing it (fresh evaluator, latch aside —
    blockers OFF net so the guard does not fire) changes the L-block mask, so
    the geometric net's outputs differ between true raylen and a blockers-on
    net. Guards against a silently-dropped tensor (e.g. a kwarg typo) that
    would still pass parity if both sides ignored raylen."""
    states = _battery_states(n_seeds=1, plies=(9,))
    payload, ordered = _build_payload(states)
    on = _build_l_net(seed=11)  # blockers on (env default)
    off = _build_l_net(seed=11, ray_blockers=False)  # same weights, mask variant
    rv_on, rp_on, _ = _reply_arrays(
        HexfieldEvaluator(on, device="cpu").evaluate_payload(payload)
    )
    rv_off, rp_off, _ = _reply_arrays(
        HexfieldEvaluator(off, device="cpu").evaluate_payload(payload)
    )
    # A mid-game board has blocked rays, so the two mask semantics disagree.
    assert not np.allclose(rp_on, rp_off, atol=1e-6)


# --- 2. C/A path unchanged ----------------------------------------------------------


@needs_rust
def test_ca_net_serve_path_builds_no_raylen() -> None:
    states = _battery_states(n_seeds=1)
    payload, _ = _build_payload(states)
    torch.manual_seed(12)
    model = HexfieldNet(trunk_layout="CCA").eval()

    calls: list[tuple[int, frozenset]] = []
    orig = model.forward_policy_value

    def spy(*args, **kwargs):
        calls.append((len(args), frozenset(kwargs)))
        return orig(*args, **kwargs)

    model.forward_policy_value = spy  # instance attr shadows the bound method
    ev = HexfieldEvaluator(model, device="cpu")
    assert not ev._needs_raylen
    reply_a = ev.evaluate_payload(payload)
    # Every forward carried exactly the pre-L argument list: 4 positional
    # tensors, request_moves_left only — no raylen tensor exists on this path
    # (the byte-identity guarantee; sha256-verified against the pre-change
    # tree once, at landing).
    assert calls and all(
        n_args == 4 and kw == frozenset({"request_moves_left"}) for n_args, kw in calls
    )
    # Deterministic call-over-call.
    reply_b = ev.evaluate_payload(payload)
    assert reply_a["values_bytes"] == reply_b["values_bytes"]
    assert reply_a["priors_bytes"] == reply_b["priors_bytes"]
    # A C/A net also serves a payload with NO raylen key (older wire).
    payload_nr, _ = _build_payload(states, drop_raylen=True)
    reply_c = ev.evaluate_payload(payload_nr)
    assert reply_c["values_bytes"] == reply_a["values_bytes"]


# --- 3. guards -----------------------------------------------------------------------


@needs_rust
def test_l_net_missing_raylen_key_fails_loudly() -> None:
    states = _battery_states(n_seeds=1, plies=(3,))
    payload, _ = _build_payload(states, drop_raylen=True)
    ev = HexfieldEvaluator(_build_l_net(), device="cpu")
    with pytest.raises(ValueError, match="raylen"):
        ev.submit_payload(payload)


@needs_rust
def test_l_net_all_zero_raylen_guard_fires() -> None:
    states = _battery_states(n_seeds=1, plies=(3,))
    payload, _ = _build_payload(
        states, raylen_override=lambda rl: np.zeros_like(rl)
    )
    ev = HexfieldEvaluator(_build_l_net(), device="cpu")
    with pytest.raises(ValueError, match="all-zero"):
        ev.submit_payload(payload)
    # ... but a geometric (blockers-off) L net accepts it: raylen is threaded
    # yet never read (spec D-S16), and the latch only guards live-mask nets.
    geo = HexfieldEvaluator(_build_l_net(ray_blockers=False), device="cpu")
    assert geo._needs_raylen and not geo._ray_blockers
    reply = geo.evaluate_payload(payload)
    assert np.isfinite(np.frombuffer(reply["values_bytes"], dtype=np.float32)).all()


@needs_rust
def test_l_net_raylen_byte_count_mismatch_fails() -> None:
    states = _battery_states(n_seeds=1, plies=(3,))
    payload, _ = _build_payload(
        states, raylen_override=lambda rl: rl[: rl.shape[0] - RL]
    )
    ev = HexfieldEvaluator(_build_l_net(), device="cpu")
    with pytest.raises(ValueError, match="byte count"):
        ev.submit_payload(payload)


# --- 4. CUDA-gated L serve (skips on CPU) ---------------------------------------------


@needs_rust
@pytest.mark.skipif(not torch.cuda.is_available(), reason="serve fast path is CUDA-only")
def test_l_net_serve_cuda_parity(monkeypatch) -> None:
    """The CUDA evaluator (eager forward, fp16 autocast + f16 wire feats)
    threads raylen and matches the fp32 CPU train-path reference to the serve
    tolerance; the Rust-pack path (grp['raylen'] consumption) agrees with the
    CSR path bitwise on the same device."""
    import copy

    monkeypatch.setenv("HEXFIELD_NO_COMPILE", "1")  # keep the gate brief
    states = _battery_states(n_seeds=1)
    payload, ordered = _build_payload(states)
    model = _build_l_net()
    # Reference on a separate COPY: the no-grad CPU forward populates the
    # serve dense-weight cache with CPU tensors, and the evaluator's in-place
    # .to(cuda) does not regenerate it (plan §0 gotcha 2).
    tv, tp, _ = _train_reference(copy.deepcopy(model), ordered)

    ev = HexfieldEvaluator(model, device="cuda")
    assert ev._needs_raylen
    sv, sp, _ = _reply_arrays(ev.evaluate_payload(dict(payload)))
    np.testing.assert_allclose(sv, tv, atol=SERVE_ATOL, rtol=0)
    np.testing.assert_allclose(sp, tp, atol=SERVE_ATOL, rtol=0)

    if hasattr(_rust, "build_serve_groups"):
        monkeypatch.setenv("HEXFIELD_RUST_PACK", "1")
        ev_rp = HexfieldEvaluator(model, device="cuda")
        assert ev_rp._rust_pack
        rv, rp, _ = _reply_arrays(ev_rp.evaluate_payload(dict(payload)))
        np.testing.assert_allclose(rv, sv, atol=1e-6, rtol=0)
        np.testing.assert_allclose(rp, sp, atol=1e-6, rtol=0)
