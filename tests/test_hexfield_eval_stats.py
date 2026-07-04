"""Unit tests for the hexfield eval statistics layer (pure numpy/CPU).

Covers the functions in ``packages/hexfield/python/hexfield/eval_stats.py``:

  - Wilson CI coverage (Monte-Carlo) and edge cases.
  - LOS values and monotonicity.
  - Pentanomial / paired SE: pair-level SE vs the naive per-game binomial, and
    the effective-count deflation under within-pair correlation.
  - Bradley-Terry: convergence (max|grad| < 1e-6), Elo recovery on synthetic
    data, and anchor pinned at 0.
  - var_diff: difference variance under shared free-anchor covariance.
  - Over-dispersion down-weight.
  - SPRT boundary monotonicity and verdicts.
  - Elo<->win-rate round-trips and single-epoch resolution numbers.

hexfield is not installed in a shared venv; its source is added to ``sys.path``
directly. eval_stats has no engine/torch/CUDA dependency (pure numpy + optional
scipy) and runs on a CPU-only interpreter.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "hexfield" / "python"))

from hexfield.eval_stats import (  # noqa: E402
    ELO_PER_LOGIT,
    BTEdge,
    bonferroni_alpha,
    bradley_terry,
    effective_counts,
    elo_ci_from_winrate,
    elo_from_winrate,
    elo_winrate_se,
    expected_se_elo,
    games_for_se,
    los,
    overdispersion_weight,
    paired_winrate,
    pentanomial_summary,
    primary_verdict,
    sprt,
    sprt_bounds,
    sprt_llr,
    var_diff,
    wilson_ci,
    winrate_from_elo,
)


# --------------------------------------------------------------------------- #
# Wilson CI.
# --------------------------------------------------------------------------- #
def test_wilson_empty_and_extremes() -> None:
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25  # all losses: lower bound at 0
    lo, hi = wilson_ci(20, 20)
    assert hi == 1.0 and 0.75 < lo < 1.0  # all wins: upper bound at 1
    lo, hi = wilson_ci(10, 20)
    assert lo < 0.5 < hi
    assert math.isclose((lo + hi) / 2.0, 0.5, abs_tol=1e-9)  # symmetric at p=.5


def test_wilson_contains_point_and_orders() -> None:
    for wins, n in [(1, 10), (5, 10), (9, 10), (50, 200), (3, 7)]:
        lo, hi = wilson_ci(wins, n)
        assert 0.0 <= lo <= wins / n <= hi <= 1.0
        assert hi - lo > 0.0


def test_wilson_coverage_montecarlo() -> None:
    """A nominal-95% Wilson interval covers the true p >= 90% of the time.

    Asserts empirical coverage lands in [0.90, 0.995] across several p, n.
    """

    rng = random.Random(20240613)
    for p in (0.2, 0.5, 0.75):
        for n in (40, 128):
            covered = 0
            trials = 4000
            for _ in range(trials):
                wins = sum(1 for _ in range(n) if rng.random() < p)
                lo, hi = wilson_ci(wins, n)
                if lo <= p <= hi:
                    covered += 1
            frac = covered / trials
            assert 0.90 <= frac <= 0.995, f"p={p} n={n} coverage={frac:.3f}"


# --------------------------------------------------------------------------- #
# LOS.
# --------------------------------------------------------------------------- #
def test_los_basics_and_monotone() -> None:
    assert los(0, 0) == 0.5
    assert math.isclose(los(10, 10), 0.5, abs_tol=1e-12)
    assert los(20, 5) > 0.95
    assert los(5, 20) < 0.05
    # More net wins at fixed total -> higher LOS.
    prev = -1.0
    for w in range(0, 41):
        cur = los(w, 40 - w)
        assert cur >= prev - 1e-12
        prev = cur


# --------------------------------------------------------------------------- #
# Elo <-> win rate.
# --------------------------------------------------------------------------- #
def test_elo_winrate_roundtrip() -> None:
    for p in (0.05, 0.3, 0.5, 0.7, 0.95):
        assert math.isclose(winrate_from_elo(elo_from_winrate(p)), p, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(elo_from_winrate(0.5), 0.0, abs_tol=1e-12)
    # 0.75 win rate maps to ~190.84 Elo.
    assert math.isclose(elo_from_winrate(0.75), 190.84, abs_tol=0.05)
    assert elo_from_winrate(1.0) == float("inf")
    assert elo_from_winrate(0.0) == float("-inf")


def test_elo_delta_method_matches_numeric_derivative() -> None:
    p, se_p = 0.62, 0.03
    analytic = elo_winrate_se(p, se_p)
    h = 1e-6
    slope = (elo_from_winrate(p + h) - elo_from_winrate(p - h)) / (2 * h)
    assert math.isclose(analytic, abs(slope) * se_p, rel_tol=1e-4)
    assert elo_winrate_se(1.0, 0.01) == float("inf")  # diverges at rail


def test_elo_ci_brackets_point() -> None:
    lo, hi = elo_ci_from_winrate(0.6, 0.04)
    assert lo < elo_from_winrate(0.6) < hi


# --------------------------------------------------------------------------- #
# Pentanomial / paired SE.
# --------------------------------------------------------------------------- #
def test_pentanomial_winrate_value() -> None:
    # 10 pairs: 4 WW (2 pts), 3 even (1 pt), 3 LL (0 pts) -> mean pts = 1.1/2.
    res = pentanomial_summary([3, 0, 3, 0, 4])
    assert res.n_pairs == 10
    expected_wr = (4 * 1.0 + 3 * 0.5 + 3 * 0.0) / 10
    assert math.isclose(res.win_rate, expected_wr, abs_tol=1e-12)
    assert res.se > 0.0


def test_pentanomial_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        pentanomial_summary([1, 2, 3])
    with pytest.raises(ValueError):
        pentanomial_summary([1, -1, 0, 0, 0])


def test_pairing_correlation_changes_se_vs_naive_binomial() -> None:
    """Pair-level SE depends on within-pair correlation.

    Two pentanomials with the same win rate (0.5) and same game count (2*N),
    differing only in within-pair correlation:

      - HIGH correlation: every pair is WW or LL -> large pair-level variance.
      - LOW correlation: every pair splits 1-1 (all 'even') -> small pair-level
        variance.

    The naive per-game binomial SE (sqrt(.25/2N)) is identical for both. The
    pair-level SE differs: high-corr SE >> low-corr SE.
    """

    n = 50
    high = pentanomial_summary([n // 2, 0, 0, 0, n // 2])   # WW/LL only
    low = pentanomial_summary([0, 0, n, 0, 0])              # all 1-1 splits
    assert math.isclose(high.win_rate, 0.5, abs_tol=1e-12)
    assert math.isclose(low.win_rate, 0.5, abs_tol=1e-12)

    naive_se_p = math.sqrt(0.25 / (2 * n))  # per-game binomial SE
    # Low correlation: pairs nearly constant => SE much smaller than naive.
    assert low.se < naive_se_p * 0.25
    # High correlation: pair is one effective coin => SE larger than naive.
    assert high.se > naive_se_p * 1.2
    # The two pairing regimes differ from each other.
    assert high.se > low.se * 5


def test_paired_winrate_matches_pentanomial() -> None:
    scores = [1.0] * 4 + [0.5] * 3 + [0.0] * 3
    a = paired_winrate(scores)
    b = pentanomial_summary([3, 0, 3, 0, 4])
    assert a.n_pairs == b.n_pairs == 10
    assert math.isclose(a.win_rate, b.win_rate, abs_tol=1e-12)
    assert math.isclose(a.se, b.se, rel_tol=1e-9)


def test_paired_winrate_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        paired_winrate([0.5, 1.5])


def test_effective_counts_deflates_with_positive_correlation() -> None:
    """Effective game count drops below 2*N_pairs when pairs are correlated."""

    n = 40
    # Perfectly correlated pairs (WW/LL only) at p=0.5: effective games == N_pairs.
    high = pentanomial_summary([n // 2, 0, 0, 0, n // 2])
    w, l, n_eff = effective_counts(high)
    assert math.isclose(w, l, abs_tol=1e-9)             # symmetric at p=0.5
    assert math.isclose(n_eff, n, rel_tol=0.05)         # ~N_pairs, not 2N
    assert n_eff < 2 * n - 1e-6                          # strictly deflated

    # Low correlation (all splits): n_eff capped at the physical 2N games.
    low = pentanomial_summary([0, 0, n, 0, 0])
    _, _, n_eff_low = effective_counts(low)
    assert n_eff_low <= 2 * n + 1e-6


def test_effective_counts_degenerate_sweep() -> None:
    res = pentanomial_summary([0, 0, 0, 0, 10])  # all WW (p=1)
    w, l, n_eff = effective_counts(res)
    assert math.isclose(w, n_eff, abs_tol=1e-9)
    assert l == 0.0
    assert n_eff == 20.0  # falls back to physical game count
    assert effective_counts(pentanomial_summary([0, 0, 0, 0, 0])) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Over-dispersion down-weight.
# --------------------------------------------------------------------------- #
def test_overdispersion_weight() -> None:
    # Over-dispersed: deff=2 -> weight 0.5.
    assert math.isclose(overdispersion_weight(0.5, 0.25), 0.5, rel_tol=1e-9)
    # Not over-dispersed (deff<=1): clamp to 1.0.
    assert overdispersion_weight(0.1, 0.25) == 1.0
    assert overdispersion_weight(0.25, 0.25) == 1.0
    # Degenerate inputs -> neutral weight.
    assert overdispersion_weight(0.0, 0.25) == 1.0
    assert overdispersion_weight(0.5, 0.0) == 1.0


# --------------------------------------------------------------------------- #
# Bradley-Terry: convergence + Elo recovery + anchor pin.
# --------------------------------------------------------------------------- #
def _simulate_edges(true_elo: dict[str, float], n_games: int, seed: int) -> list[BTEdge]:
    """Synthetic head-to-head games from known true Elos (round-robin)."""

    rng = random.Random(seed)
    labels = list(true_elo)
    edges = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            p = winrate_from_elo(true_elo[a] - true_elo[b])
            wa = sum(1 for _ in range(n_games) if rng.random() < p)
            edges.append(BTEdge(a, b, float(wa), float(n_games - wa)))
    return edges


def test_bradley_terry_converges_and_recovers_elos() -> None:
    true_elo = {"sealbot": 0.0, "bc": 80.0, "ep5": 160.0, "cand": 240.0}
    edges = _simulate_edges(true_elo, n_games=4000, seed=7)
    res = bradley_terry(edges, anchor="sealbot")

    # Fit converged.
    assert res.converged
    assert res.max_grad < 1e-6

    # Anchor pinned exactly at 0 Elo.
    assert res.rating("sealbot") == 0.0
    assert res.cov[res.players["sealbot"], :].sum() == 0.0
    assert res.cov[:, res.players["sealbot"]].sum() == 0.0

    # Recovers the true Elos within 25 Elo.
    for label, elo in true_elo.items():
        assert abs(res.rating(label) - elo) < 25.0, f"{label}: {res.rating(label):.1f} vs {elo}"
    # Ordering preserved.
    ratings = [res.rating(l) for l in ("sealbot", "bc", "ep5", "cand")]
    assert ratings == sorted(ratings)


def test_bradley_terry_converges_on_ill_conditioned_ladder() -> None:
    """Convergence on a chain topology and comparison to fixed-step GD.

    The ladder is a chain: sealbot - e5 - e10 - e20 - e40 - e80 - e160, with
    only nearest-neighbour edges, each a lopsided high-count edge. The chain
    Hessian is ill-conditioned. bradley_terry converges (max|grad| < 1e-6) in
    fewer than 50 iterations. A fixed-step GD (step 0.002, 500 iters) on the
    same data does not reach a stationary point (max|grad| > 1e-3).
    """

    chain = ["sealbot", "e5", "e10", "e20", "e40", "e80", "e160"]
    edges = [BTEdge(a, b, 420.0, 180.0) for a, b in zip(chain[1:], chain[:-1])]
    res = bradley_terry(edges, anchor="sealbot")
    assert res.max_grad < 1e-6
    assert res.iterations < 50
    ratings = [res.rating(p) for p in chain]
    assert ratings == sorted(ratings)  # monotone up the ladder
    assert res.rating("sealbot") == 0.0

    # Fixed-step GD on the same data does not reach a stationary point.
    labels = sorted({lbl for e in edges for lbl in (e.a, e.b)})
    idx = {p: i for i, p in enumerate(labels)}
    n = len(labels)
    r = np.zeros(n)
    raw = [(e.a, e.b, e.wins_a, e.wins_b) for e in edges]
    for _ in range(500):
        g = np.zeros(n)
        for a, b, wa, wb in raw:
            ia, ib = idx[a], idx[b]
            pa = 1.0 / (1.0 + math.exp(-(r[ia] - r[ib])))
            g[ia] -= wa - (wa + wb) * pa
            g[ib] += wa - (wa + wb) * pa
        g[idx["sealbot"]] = 0.0
        r -= 0.002 * g
        r -= r[idx["sealbot"]]
    g_final = np.zeros(n)
    for a, b, wa, wb in raw:
        ia, ib = idx[a], idx[b]
        pa = 1.0 / (1.0 + math.exp(-(r[ia] - r[ib])))
        s = wa - (wa + wb) * pa
        g_final[ia] -= s
        g_final[ib] += s
    g_final[idx["sealbot"]] = 0.0
    gd_max_grad = float(np.max(np.abs(g_final)))
    assert gd_max_grad > 1e-3, (
        f"expected fixed-step GD to stall on the ladder, got {gd_max_grad:.2e}"
    )
    assert res.max_grad < gd_max_grad


def test_bradley_terry_tuple_edges_and_iterations() -> None:
    edges = [("cand", "sealbot", 70, 30), ("cand", "ep5", 55, 45), ("ep5", "sealbot", 65, 35)]
    res = bradley_terry(edges, anchor="sealbot")
    assert res.max_grad < 1e-6
    assert res.iterations >= 1
    assert res.rating("cand") > res.rating("ep5") > 0.0  # both beat sealbot


def test_bradley_terry_handles_undefeated_via_prior() -> None:
    # cand never loses -> unregularized MLE is +inf; the ridge prior keeps it finite.
    edges = [BTEdge("cand", "sealbot", 50.0, 0.0), BTEdge("ep5", "sealbot", 30.0, 20.0)]
    res = bradley_terry(edges, anchor="sealbot", prior_sd_elo=800.0)
    assert res.max_grad < 1e-6
    assert math.isfinite(res.rating("cand"))
    assert res.rating("cand") > res.rating("ep5")


def test_bradley_terry_anchor_must_exist() -> None:
    with pytest.raises(ValueError):
        bradley_terry([("a", "b", 5, 5)], anchor="sealbot")


def test_bradley_terry_newton_matches_scipy() -> None:
    """The pure-numpy Newton path and the scipy path agree (both converge)."""

    true_elo = {"sealbot": 0.0, "bc": 100.0, "cand": 220.0}
    edges = _simulate_edges(true_elo, n_games=2000, seed=3)
    res_scipy = bradley_terry(edges, anchor="sealbot", use_scipy=True)
    res_newton = bradley_terry(edges, anchor="sealbot", use_scipy=False)
    assert res_scipy.max_grad < 1e-6 and res_newton.max_grad < 1e-6
    for label in true_elo:
        assert abs(res_scipy.rating(label) - res_newton.rating(label)) < 1e-3
    # Covariances agree too (same stationary point, same Hessian).
    assert np.allclose(res_scipy.cov, res_newton.cov, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# var_diff & shared-anchor covariance.
# --------------------------------------------------------------------------- #
def test_var_diff_formula() -> None:
    cov = np.array([[4.0, 1.5], [1.5, 9.0]])
    assert math.isclose(var_diff(cov, 0, 1), 4.0 + 9.0 - 2 * 1.5)


def test_var_diff_shrinks_with_shared_anchor_covariance() -> None:
    """Two players measured against a shared free anchor get positive Cov_ij, so
    the difference variance is smaller than the sum of marginal variances.

    'cand' and 'champ' both play a shared free anchor 'bc' in addition to the
    pinned 'sealbot'. The shared free anchor couples their estimates:
    Cov(cand, champ) > 0 -> Var(cand - champ) < Var(cand) + Var(champ).

    Contrast: if the two players shared only the pinned sealbot and no other
    common opponent, the Hessian is block-diagonal and Cov_ij is exactly 0 (a
    pinned anchor carries no covariance). See the test below.
    """

    edges = [
        BTEdge("cand", "sealbot", 70.0, 30.0),
        BTEdge("champ", "sealbot", 64.0, 36.0),
        BTEdge("cand", "bc", 60.0, 40.0),
        BTEdge("champ", "bc", 55.0, 45.0),
        BTEdge("bc", "sealbot", 62.0, 38.0),
    ]
    res = bradley_terry(edges, anchor="sealbot")
    i, j = res.players["cand"], res.players["champ"]
    cov_ij = res.cov[i, j]
    assert cov_ij > 0.0, "shared free anchor should induce positive covariance"

    vd = res.var_diff("cand", "champ")
    sum_marginals = res.cov[i, i] + res.cov[j, j]
    assert vd < sum_marginals  # -2*Cov_ij term reduces the difference variance
    # The difference SE is smaller than the quadrature-of-marginals SE.
    se_diff = res.se_diff("cand", "champ")
    se_quad = math.sqrt(res.se("cand") ** 2 + res.se("champ") ** 2)
    assert se_diff < se_quad


def test_var_diff_zero_covariance_with_only_pinned_anchor() -> None:
    """Sharing only the pinned anchor gives Cov_ij == 0 (block-diagonal Hessian).

    A pinned zero-point induces no covariance between two players who otherwise
    never meet, so their difference variance equals the sum of marginals.
    """

    edges = [
        BTEdge("cand", "sealbot", 70.0, 30.0),
        BTEdge("champ", "sealbot", 64.0, 36.0),
    ]
    res = bradley_terry(edges, anchor="sealbot")
    i, j = res.players["cand"], res.players["champ"]
    assert math.isclose(res.cov[i, j], 0.0, abs_tol=1e-9)
    assert math.isclose(
        res.var_diff("cand", "champ"), res.cov[i, i] + res.cov[j, j], rel_tol=1e-9
    )


def test_diff_ci_brackets_point_estimate() -> None:
    edges = [BTEdge("cand", "sealbot", 72.0, 28.0), BTEdge("champ", "sealbot", 60.0, 40.0)]
    res = bradley_terry(edges, anchor="sealbot")
    lo, hi = res.diff_ci("cand", "champ")
    d = res.diff("cand", "champ")
    assert lo < d < hi
    assert d > 0.0  # cand beat sealbot harder than champ did


# --------------------------------------------------------------------------- #
# SPRT.
# --------------------------------------------------------------------------- #
def test_sprt_bounds_symmetry_and_monotonicity() -> None:
    lo, hi = sprt_bounds(0.05, 0.05)
    assert math.isclose(lo, -hi, rel_tol=1e-12)
    assert math.isclose(hi, math.log(0.95 / 0.05), rel_tol=1e-12)
    # Tighter alpha => wider boundaries (need more evidence).
    _, hi_tight = sprt_bounds(0.01, 0.01)
    _, hi_loose = sprt_bounds(0.10, 0.10)
    assert hi_tight > hi_loose
    with pytest.raises(ValueError):
        sprt_bounds(0.0, 0.05)


def test_sprt_llr_monotone_in_wins() -> None:
    # H0: -10 Elo, H1: 0 Elo.
    prev = -1e9
    for w in range(0, 51):
        cur = sprt_llr(w, 50 - w, elo0=-10.0, elo1=0.0)
        assert cur > prev  # each extra win raises the LLR toward H1
        prev = cur


def test_sprt_verdicts() -> None:
    # Dominant win record -> accept H1.
    r = sprt(wins=120, losses=20, elo0=-15.0, elo1=0.0)
    assert r.verdict == "accept_h1"
    assert r.llr >= r.upper
    # Dominant loss record -> accept H0.
    r = sprt(wins=20, losses=120, elo0=-15.0, elo1=0.0)
    assert r.verdict == "accept_h0"
    assert r.llr <= r.lower
    # Near-indifference, small sample -> continue.
    r = sprt(wins=16, losses=14, elo0=-15.0, elo1=0.0)
    assert r.verdict == "continue"
    assert r.lower < r.llr < r.upper


# --------------------------------------------------------------------------- #
# Resolution helpers and Bonferroni.
# --------------------------------------------------------------------------- #
def test_single_epoch_resolution_is_honest() -> None:
    """Single-rate and paired-difference Elo SE at 128 games.

    - expected_se_elo(128) is the SE of one win-rate's Elo from 128 independent
      decided games: ~30.7 Elo.
    - The paired difference SE r_L - r_B enlarges this via a sqrt(2) factor and
      an effective-N reduction (N_eff = 128/deff for design effect deff),
      landing in the ~40-55 Elo band across a deff sweep.
    """

    se_128 = expected_se_elo(128, p=0.5)
    assert 28.0 < se_128 < 33.0, se_128  # single-rate, independent games

    # Paired difference SE = single-rate SE(N_eff) * sqrt(2), with N_eff = 128/deff
    # for design effect deff. Assert the 40-55 Elo band is bracketed by the sweep
    # and every case exceeds the single-rate SE.
    diff_se = {
        deff: expected_se_elo(128 / deff, p=0.5) * math.sqrt(2.0)
        for deff in (1.0, 1.3, 1.6, 2.0)
    }
    assert diff_se[1.0] < 45.0 < 50.0 < diff_se[2.0]      # band 40-55 is bracketed
    assert min(diff_se.values()) > se_128                  # difference always > single rate
    assert all(v > 40.0 for v in diff_se.values())

    # Reaching ~15 Elo SE on a single rate needs more than 4x128 games.
    need = games_for_se(15.0, p=0.5)
    assert need > 128 * 4
    # games_for_se inverts expected_se_elo.
    assert math.isclose(expected_se_elo(need, 0.5), 15.0, rel_tol=0.02)


def test_expected_se_edges() -> None:
    assert expected_se_elo(0) == float("inf")
    assert games_for_se(0.0) > 10**8


def test_bonferroni_alpha() -> None:
    assert math.isclose(bonferroni_alpha(0.05, 5), 0.01)
    assert bonferroni_alpha(0.05, 1) == 0.05
    with pytest.raises(ValueError):
        bonferroni_alpha(0.05, 0)


# --------------------------------------------------------------------------- #
# Verdict label.
# --------------------------------------------------------------------------- #
def test_primary_verdict_labels() -> None:
    assert primary_verdict((5.0, 40.0)) == "PROMOTE"        # CI entirely > 0
    assert primary_verdict((-50.0, -10.0)) == "REGRESS"     # CI entirely < 0
    assert primary_verdict((-20.0, 30.0)) == "INCONCLUSIVE"  # straddles 0
    # Custom regression threshold.
    assert primary_verdict((-40.0, -25.0), regress_elo=-20.0) == "REGRESS"
    assert primary_verdict((-15.0, -5.0), regress_elo=-20.0) == "INCONCLUSIVE"


def test_elo_per_logit_constant() -> None:
    assert math.isclose(ELO_PER_LOGIT, 400.0 / math.log(10.0), rel_tol=1e-15)
