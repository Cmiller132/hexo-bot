"""Public synthesized-shard generator for the katago-buffer tests.

The retired reference oracle and the private development-run
live-run tree are unavailable in the public environment, so every test that
previously read real shards now reads shards SYNTHESIZED here: a random game is
played with the public engine, finalized into ``ShrimpSampleData`` rows, and
written as a ``shrimp_compact_v1`` shard (``.npz`` + ``.json`` sidecar) via the
same ``write_compact_shard`` the production writer uses.

Games are played long enough (``max_plies >= 16``) that the finalized rows carry
standing-hot cells (non-empty ``own_hot`` / ``opp_hot``), which the p3 concat
test requires to exercise the qr-CSR rebase path.

Layout mirrors the live tree: ``<samples>/epoch_00000E/game_<key>.npz`` with
``game_key = epoch * 1_000_000 + index`` (the manifest derives generation from
that key), so the manifest scanner and window math see realistic (generation,
game_key) ordering.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Make ``shrimp_testkit`` importable regardless of how pytest is invoked (the
# harness command only puts packages/shrimp/python on PYTHONPATH). testkit in
# turn adds the shrimp package path.
_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from shrimp_testkit import api  # noqa: E402  (path set up above)

from shrimp.features import window_scan  # noqa: E402
from shrimp.geometry import pack_action_id, unpack_action_id  # noqa: E402
from shrimp.samples import ShrimpSampleData, finalize_game_samples  # noqa: E402
from shrimp.engine_facts import facts_from_engine, player_int  # noqa: E402
from shrimp.shards import write_compact_shard  # noqa: E402
from hexo_engine.types import AxialCoord, PlacementAction  # noqa: E402


def _hex_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Axial hex distance between two (q, r) cells."""
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2


def _sample_from_state(state, rng: random.Random, turn_index: int) -> ShrimpSampleData:
    """One pre-decision sample carrying the engine facts + a soft policy.

    The policy is restricted to legal actions within a small hex radius of an
    already-placed stone (or all legal actions at ply 0). This keeps every policy
    target inside the model support even at a REDUCED support radius (the p7
    radius-4 gate expands these rows at radius 4 and requires that nothing is
    naturally off-legal there — a uniform over-the-whole-board policy would land
    outside radius 4 of any stone). Mirrors how a real MCTS policy stays local.
    """
    facts = facts_from_engine(api.to_python_state(state))
    legal = sorted(api.legal_action_ids(state))
    stones = [(q, r) for (q, r, _o, _i) in facts.records]
    if stones:
        near = [
            aid
            for aid in legal
            if any(_hex_dist(unpack_action_id(aid), s) <= 3 for s in stones)
        ]
        pool = near if near else legal
    else:
        pool = legal
    chosen = rng.sample(pool, k=min(3, len(pool)))
    weights = [rng.random() + 0.1 for _ in chosen]
    total = sum(weights)
    policy = tuple((aid, w / total) for aid, w in zip(chosen, weights))
    return ShrimpSampleData(
        game_id="synth",
        turn_index=turn_index,
        current_player=facts.current_player,
        phase=facts.phase,
        records=facts.records,
        first_stone=facts.first_stone,
        own_hot=facts.own_hot,
        opp_hot=facts.opp_hot,
        own_win=facts.own_win,
        opp_win=facts.opp_win,
        policy=policy,
        metadata={"pcr_full": True},
    )


def make_game(seed: int, max_plies: int = 24):
    """Play one random game; return ``(pending, winner)``.

    ``pending`` is the list of ``(player, sample, root_value)`` decisions that
    ``finalize_game_samples`` consumes. ``max_plies >= 16`` is enough for the
    game to accumulate standing-hot windows.
    """
    rng = random.Random(seed)
    state = api.new_game()
    pending: list[tuple[int, ShrimpSampleData, float]] = []
    winner = None
    for ply in range(max_plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        sample = _sample_from_state(state, rng, ply)
        pending.append((sample.current_player, sample, rng.uniform(-0.8, 0.8)))
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            winner = player_int(api.terminal(state).winner)
            break
    return pending, winner


def _hot_records(base_len: int) -> tuple[tuple[int, int, int, int], ...]:
    """Records with four collinear player-0 stones on the q-axis (a length-6
    window of count 4 -> a hot window once >= HOT_MIN_PLACEMENTS placements),
    interleaved with player-1 stones far away so the window stays single-colour.
    ``base_len`` pads with extra scattered player-1 stones so placements_made
    clears the HOT_MIN_PLACEMENTS floor. Returns ``(q, r, owner, placement)``.
    """
    recs: list[tuple[int, int, int, int]] = [
        (0, 0, 0, 1), (5, 5, 1, 2), (1, 0, 0, 3), (6, 5, 1, 4),
        (2, 0, 0, 5), (7, 5, 1, 6), (3, 0, 0, 7),
    ]
    # A few more scattered player-1 stones to lift the placement count.
    for k in range(base_len):
        recs.append((8 + k, 5, 1, 8 + k))
    return tuple(recs)


def make_hot_game(seed: int, plies: int = 6):
    """A short scripted "game" whose rows carry standing-hot cells.

    Real random games almost never place four collinear same-owner stones, so
    the p3 concat test's qr-CSR rebase path (which needs a non-empty
    own_hot/opp_hot segment) would go unexercised. This builds a handful of
    pre-decision samples over a constructed hot position and returns a
    ``(pending, winner)`` pair shaped exactly like :func:`make_game`.
    """
    rng = random.Random(seed)
    records = _hot_records(base_len=3)
    pending: list[tuple[int, ShrimpSampleData, float]] = []
    for ply in range(plies):
        cp = ply % 2
        own_hot, opp_hot, own_win, opp_win = window_scan(records, cp, len(records))
        # A small valid policy over arbitrary cells (parity only decodes it).
        policy = tuple(
            (pack_action_id(10 + j, -(10 + j)), w)
            for j, w in enumerate((0.5, 0.3, 0.2))
        )
        sample = ShrimpSampleData(
            game_id="synth-hot",
            turn_index=ply,
            current_player=cp,
            phase="FirstStone",
            records=records,
            first_stone=None,
            own_hot=own_hot,
            opp_hot=opp_hot,
            own_win=own_win,
            opp_win=opp_win,
            policy=policy,
            metadata={"pcr_full": True},
        )
        pending.append((cp, sample, rng.uniform(-0.8, 0.8)))
    return pending, 0  # winner = player 0 (arbitrary completed outcome)


def write_game_shard(
    path: Path, *, seed: int, epoch: int, max_plies: int = 24, hot: bool = False
) -> int:
    """Synthesize one game and write it as a compact shard at ``path``.

    Returns the number of rows written. The sidecar carries the ``epoch`` /
    ``game_key`` fields the manifest cross-checks.
    """
    if hot:
        pending, winner = make_hot_game(seed)
    else:
        pending, winner = make_game(seed, max_plies=max_plies)
    finalized = finalize_game_samples(pending, winner)
    game_key = int(path.stem.split("_", 1)[1])
    sidecar = {"epoch": int(epoch), "game_key": game_key}
    return write_compact_shard(path, finalized, sidecar=sidecar)


def generate_samples_tree(
    samples_dir: Path,
    *,
    epochs: int = 3,
    games_per_epoch: int = 4,
    max_plies: int = 24,
    base_seed: int = 1000,
    hot_first: bool = False,
) -> int:
    """Populate ``samples_dir`` with ``epochs`` x ``games_per_epoch`` synthesized
    shards laid out as ``epoch_00000E/game_<key>.npz`` (+ sidecars).

    Idempotent-friendly: creates parent dirs. Returns the total shard count.
    Epoch numbering starts at 1 (matching the live ``epoch_000001`` convention).

    ``hot_first``: when True the first game of every epoch is a scripted hot game
    so each epoch subdir carries standing-hot cells (needed by the p3 concat
    qr-CSR rebase path). Hot games use a hand-built position whose policy cells
    are NOT on the engine's legal set, so they must NOT be used where
    ``expand_sample`` runs (p5/p7); those consumers pass ``hot_first=False`` and
    get fully expandable random self-play rows.
    """
    samples_dir = Path(samples_dir)
    written = 0
    seed = base_seed
    for e in range(1, epochs + 1):
        epoch_dir = samples_dir / f"epoch_{e:06d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        for i in range(games_per_epoch):
            game_key = e * 1_000_000 + i
            path = epoch_dir / f"game_{game_key}.npz"
            # Vary the seed so shards differ; retry a few seeds if a game turns
            # out empty (no legal moves at ply 0 never happens, but guard rows>0).
            # Optionally make the first game of every epoch a scripted hot game
            # so every epoch subdir carries standing-hot cells (the p3 concat
            # qr-CSR rebase path needs a non-empty own_hot/opp_hot segment); the
            # rest are random self-play games.
            hot = hot_first and i == 0
            rows = 0
            attempt = 0
            while rows == 0 and attempt < 8:
                rows = write_game_shard(
                    path, seed=seed + attempt * 101, epoch=e, max_plies=max_plies, hot=hot
                )
                attempt += 1
            seed += 7
            written += 1
    return written
