//! Unit tests for the rule machine: every [`MoveError`] variant, the precedence table,
//! atomicity on rejection, the turn machine, windows, the legal set, and hashing.

use super::*;
use crate::action::ActionId;
use crate::window::WINDOWS_PER_PLACEMENT;
use crate::{COORD_LIMIT, DISK_CELLS};

fn act(q: i16, r: i16) -> Action {
    Action::new(HexCoord::new(q, r))
}

/// Replay a move list through the normal rule machine.
fn play(moves: &[(i16, i16)]) -> Position {
    let mut p = Position::new();
    for &(q, r) in moves {
        p.advance(act(q, r))
            .unwrap_or_else(|e| panic!("({q}, {r}) rejected: {e}"));
    }
    p
}

/// A move list as the actions `replay` takes.
fn actions(moves: &[(i16, i16)]) -> Vec<Action> {
    moves.iter().map(|&(q, r)| act(q, r)).collect()
}

/// P0's filler ladder: `(0, 1), (0, 3), (0, 5), ...`.
const P0_LADDER: [(i16, i16); 8] = [
    (0, 1),
    (0, 3),
    (0, 5),
    (0, 7),
    (0, 9),
    (0, 11),
    (0, 13),
    (0, 15),
];

/// P1 wins with the **second** stone of a turn: six in a row ending at `(6, 0)`.
const SECOND_STONE_WIN: [(i16, i16); 11] = [
    (0, 0),
    (1, 0),
    (2, 0),
    P0_LADDER[0],
    P0_LADDER[1],
    (3, 0),
    (4, 0),
    P0_LADDER[2],
    P0_LADDER[3],
    (5, 0),
    (6, 0),
];

/// P1 wins with the **first** stone of a turn, filling a gap to make seven in a row.
const FIRST_STONE_WIN: [(i16, i16); 14] = [
    (0, 0),
    (1, 0),
    (2, 0),
    P0_LADDER[0],
    P0_LADDER[1],
    (3, 0),
    (4, 0),
    P0_LADDER[2],
    P0_LADDER[3],
    (6, 0),
    (7, 0),
    P0_LADDER[4],
    P0_LADDER[5],
    (5, 0),
];

pub(super) fn second_stone_win() -> Position {
    play(&SECOND_STONE_WIN)
}

pub(super) fn first_stone_win() -> Position {
    play(&FIRST_STONE_WIN)
}

#[test]
fn error_illegal_opening() {
    let mut p = Position::new();
    assert_eq!(p.advance(act(1, 0)), Err(MoveError::IllegalOpening));
    assert_eq!(p.advance(act(-3, 4)), Err(MoveError::IllegalOpening));
    assert!(p.advance(act(0, 0)).is_ok());
}

#[test]
fn error_occupied() {
    let mut p = play(&[(0, 0)]);
    assert_eq!(
        p.advance(act(0, 0)),
        Err(MoveError::Occupied(HexCoord::ORIGIN))
    );
}

#[test]
fn error_too_far_from_stones() {
    let mut p = play(&[(0, 0)]);
    assert_eq!(
        p.advance(act(9, 0)),
        Err(MoveError::TooFarFromStones(HexCoord::new(9, 0)))
    );
    assert!(p.advance(act(8, 0)).is_ok());
}

/// An off-domain cell far from every stone breaks the distance rule before it
/// breaks the representation, and is reported as the rule violation. The
/// `CoordOutOfBounds` case — off-domain but within the radius — needs a stone
/// at a face and is pinned in `tests/boundary.rs`.
#[test]
fn error_coord_out_of_bounds_is_reserved_for_rule_legal_placements() {
    let mut p = play(&[(0, 0)]);
    let far = HexCoord::new(COORD_LIMIT + 1, 0);
    let err = p.advance(Action::new(far)).expect_err("off-domain");
    assert_eq!(err, MoveError::TooFarFromStones(far));
    assert!(err.is_rule_violation());
}

/// The rule "the second stone of a turn may not reuse the first" has no code of its own:
/// occupancy already forbids the cell. This pins the variant that reports it.
#[test]
fn the_second_stone_of_a_turn_reuses_the_first_only_as_occupied() {
    let mut p = play(&[(0, 0), (1, 0)]);
    assert_eq!(p.phase(), TurnPhase::SecondStone);
    assert_eq!(p.get(HexCoord::new(1, 0)), Some(Player::P1));
    assert_eq!(
        p.advance(act(1, 0)),
        Err(MoveError::Occupied(HexCoord::new(1, 0)))
    );
}

#[test]
fn error_terminal_state() {
    let mut p = second_stone_win();
    assert!(p.is_terminal());
    assert_eq!(p.advance(act(-1, 0)), Err(MoveError::TerminalState));
}

#[test]
fn error_board_extent_exceeded() {
    let mut p = Position::new();
    p.advance(act(0, 0)).expect("opening");
    let mut r = 0i16;
    for _ in 0..300 {
        r += 8;
        p.advance(act(0, r)).expect("r walk");
    }
    let mut q = 0i16;
    let err = loop {
        q += 8;
        assert!(q < 8000, "the arena never refused to grow");
        match p.advance(act(q, 0)) {
            Ok(_) => {}
            Err(e) => break e,
        }
    };
    match err {
        MoveError::BoardExtentExceeded { cells } => {
            assert!(
                cells > crate::MAX_GRID_CELLS,
                "{cells} is within the ceiling"
            );
            assert!(!err.is_rule_violation());
        }
        other => panic!("wrong error: {other:?}"),
    }
    assert!(p.advance(act(q - 8, 1)).is_ok());
}

/// A legal straight walk along `q` must remain representable through this test
/// length.
#[test]
fn a_straight_q_walk_is_never_refused_by_the_arena() {
    let mut p = play(&[(0, 0)]);
    let mut q = 0i16;
    for ply in 1..=600 {
        q += 8;
        let a = act(q, 0);
        assert!(p.is_legal(a), "ply {ply} at ({q}, 0) is not legal");
        p.advance(a)
            .unwrap_or_else(|e| panic!("ply {ply} at ({q}, 0) refused: {e}"));
    }
    assert_eq!(p.grid.row_words(), 2, "a q-only walk widened r");
    assert!(p.grid.rows() as u64 * p.grid.row_words() as u64 * 64 <= crate::MAX_GRID_CELLS);
}

/// `(q, r) -> (r, q)` is a symmetry of the rules: it maps axis `Q` to `R`, fixes `QR`
/// up to sign, preserves `hex_distance`, and fixes the origin.
#[test]
fn walks_along_q_and_r_reach_the_same_ply() {
    fn walk(along_q: bool) -> (usize, Option<MoveError>) {
        let mut p = play(&[(0, 0)]);
        for k in 1..=400i16 {
            let a = if along_q {
                act(k * 8, 0)
            } else {
                act(0, k * 8)
            };
            if let Err(e) = p.advance(a) {
                return (k as usize - 1, Some(e));
            }
        }
        (400, None)
    }
    assert_eq!(walk(true), walk(false));
    assert_eq!(walk(true), (400, None), "an axis walk was refused");
}

/// Hazard H9, the sharp form: a `Search` that grows the arena and then unwinds must not
/// consume the position's extent budget.
#[test]
fn a_rewound_search_does_not_consume_the_extent_budget() {
    let mut searched = play(&[(0, 0)]);
    let flat = play(&[(0, 0)]);
    {
        let mut s = crate::Search::new(&mut searched);
        let mut q = 0i16;
        for _ in 0..60 {
            q -= 8;
            s.apply(act(q, 0)).expect("legal");
        }
        s.unwind();
        assert!(s.at_floor());
    }
    assert_eq!(searched, flat);
    assert_eq!(searched.zobrist(), flat.zobrist());
    assert_eq!(
        searched.legal_actions().collect::<Vec<_>>(),
        flat.legal_actions().collect::<Vec<_>>()
    );
    searched.audit().expect("audit after the excursion");

    let mut a = searched;
    let mut b = flat;
    let (mut q, mut r) = (0i16, 0i16);
    let mut refused = None;
    for step in 0..1400 {
        if step % 2 == 0 {
            q += 8;
        } else {
            r += 8;
        }
        let m = act(q, r);
        let (ra, rb) = (a.advance(m), b.advance(m));
        assert_eq!(
            ra.is_err(),
            rb.is_err(),
            "step {step} at ({q}, {r}): searched {ra:?} vs flat {rb:?}"
        );
        assert_eq!(a.zobrist(), b.zobrist(), "step {step}");
        if let Err(e) = ra {
            assert_eq!(Err(e), rb, "different refusals at ({q}, {r})");
            assert!(matches!(e, MoveError::BoardExtentExceeded { .. }), "{e}");
            refused = Some(step);
            break;
        }
    }
    assert!(refused.is_some(), "the diagonal never reached the ceiling");
    assert_eq!(a, b);
}

/// The sharper form of the same hazard: an excursion that widens `r` must not shorten a
/// later walk along `q`.
#[test]
fn an_excursion_along_r_does_not_shorten_a_later_q_walk() {
    let mut searched = play(&[(0, 0)]);
    let flat = play(&[(0, 0)]);
    {
        let mut s = crate::Search::new(&mut searched);
        let mut r = 0i16;
        for _ in 0..64 {
            r += 8;
            s.apply(act(0, r)).expect("legal");
        }
        s.unwind();
    }
    assert_eq!(searched, flat);

    let (mut a, mut b) = (searched, flat);
    let mut q = 0i16;
    for ply in 1..=1200 {
        q += 8;
        let m = act(q, 0);
        let ra = a.advance(m);
        let rb = b.advance(m);
        assert!(
            ra.is_ok(),
            "ply {ply} at ({q}, 0) refused in the searched position: {ra:?}"
        );
        assert_eq!(ra, rb, "ply {ply} at ({q}, 0)");
    }
    // Arena geometry may differ, but all public observables must agree.
    assert_eq!(a, b);
}

#[test]
fn precedence_terminal_beats_everything_reachable() {
    let mut p = second_stone_win();
    assert_eq!(p.advance(act(0, 0)), Err(MoveError::TerminalState));
    assert_eq!(p.advance(act(300, 300)), Err(MoveError::TerminalState));
    assert_eq!(
        p.advance(Action::new(HexCoord::new(crate::COORD_LIMIT + 1, 0))),
        Err(MoveError::TerminalState)
    );
}

#[test]
fn precedence_illegal_opening_beats_coord_out_of_bounds() {
    let mut p = Position::new();
    assert_eq!(
        p.advance(Action::new(HexCoord::new(crate::COORD_LIMIT + 1, 0))),
        Err(MoveError::IllegalOpening)
    );
}

#[test]
fn precedence_illegal_opening_beats_too_far_from_stones() {
    let mut p = Position::new();
    assert_eq!(p.advance(act(500, 0)), Err(MoveError::IllegalOpening));
}

#[test]
fn precedence_the_rules_speak_before_the_domain() {
    let mut p = play(&[(0, 0)]);
    let far = HexCoord::new(COORD_LIMIT + 1, 0);
    assert!(p.get(far).is_none());
    assert_eq!(
        p.advance(Action::new(far)),
        Err(MoveError::TooFarFromStones(far))
    );
}

#[test]
fn precedence_too_far_from_stones_beats_board_extent_exceeded() {
    // (8000, 8000) violates both halves at once: it is 16,000 steps from the only
    // stone, and its least padded box — 8017 rows by 127 words — is 65,162,176
    // cells, over MAX_GRID_CELLS. The rule must speak before the representation
    // limit, which a probe growth could actually represent would not establish.
    let mut p = play(&[(0, 0)]);
    let c = HexCoord::new(8000, 8000);
    assert!(c.is_valid());
    assert_eq!(
        p.advance(Action::new(c)),
        Err(MoveError::TooFarFromStones(c))
    );
}

#[test]
fn every_rejection_is_atomic() {
    let mut positions = [
        Position::new(),
        play(&[(0, 0)]),
        play(&[(0, 0), (1, 0)]),
        second_stone_win(),
        first_stone_win(),
    ];
    let probes = [
        (0i16, 0i16),
        (1, 0),
        (9, 0),
        (9000, 0),
        (8000, 8000),
        (crate::COORD_LIMIT + 1, 0),
        (-crate::COORD_LIMIT - 1, 0),
        (300, -300),
    ];
    // Atomicity includes arena geometry, which public equality and hashes omit.
    let shape = |p: &Position| {
        (
            p.grid.rows(),
            p.grid.row_words(),
            p.grid.origin_q(),
            p.grid.origin_r(),
        )
    };
    for p in &mut positions {
        for &(q, r) in &probes {
            let before = p.clone();
            let z = p.zobrist();
            let geometry = shape(p);
            if let Err(e) = p.advance(act(q, r)) {
                assert_eq!(*p, before, "({q}, {r}) rejected with {e} but mutated");
                assert_eq!(p.zobrist(), z);
                assert_eq!(
                    shape(p),
                    geometry,
                    "({q}, {r}) rejected with {e} but the arena moved"
                );
                p.audit().expect("audit after a rejection");
            } else {
                *p = before;
            }
        }
    }
}

#[test]
fn opening_is_forced_and_hands_over_to_p1() {
    let mut p = Position::new();
    assert_eq!(p.current_player(), Player::P0);
    assert_eq!(p.phase(), TurnPhase::Opening);
    assert_eq!(p.legal_count(), 1);
    assert_eq!(
        p.legal_actions().collect::<Vec<_>>(),
        vec![Action::new(HexCoord::ORIGIN)]
    );
    let applied = p.advance(act(0, 0)).expect("opening");
    assert_eq!(applied.mover, Player::P0);
    assert_eq!(applied.phase_before, TurnPhase::Opening);
    assert_eq!(applied.phase_after, TurnPhase::FirstStone);
    assert_eq!(applied.outcome, None);
    assert_eq!(applied.wins, [None, None, None]);
    assert_eq!(p.current_player(), Player::P1);
    assert_eq!(p.legal_count(), DISK_CELLS - 1);
}

#[test]
fn ply_pattern_is_p0_then_pairs() {
    let expected = [
        Player::P0,
        Player::P1,
        Player::P1,
        Player::P0,
        Player::P0,
        Player::P1,
        Player::P1,
        Player::P0,
        Player::P0,
    ];
    let mut p = Position::new();
    let mut r = 0i16;
    for (ply, want) in expected.into_iter().enumerate() {
        assert_eq!(p.current_player(), want, "ply {ply}");
        let a = if ply == 0 {
            act(0, 0)
        } else {
            r += 2;
            act(0, r)
        };
        p.advance(a).expect("legal");
    }
}

#[test]
fn first_stone_win_freezes_the_phase_and_the_mover() {
    let p = first_stone_win();
    assert_eq!(p.outcome(), Some(Outcome { winner: Player::P1 }));
    assert_eq!(p.phase(), TurnPhase::FirstStone);
    assert_eq!(p.current_player(), Player::P1);
    assert_eq!(p.legal_count(), 0);
    assert_eq!(p.legal_actions().count(), 0);
    assert!(
        p.frontier_cells() > 0,
        "frontier is geometric, not rule-level"
    );
    p.audit().expect("audit");
}

#[test]
fn second_stone_win_freezes_at_second_stone_with_first_occupied() {
    let p = second_stone_win();
    assert_eq!(p.outcome(), Some(Outcome { winner: Player::P1 }));
    assert_eq!(p.phase(), TurnPhase::SecondStone);
    // The first stone remains occupied after the terminal phase clears its
    // phase-local coordinate.
    assert_eq!(p.get(HexCoord::new(5, 0)), Some(Player::P1));
    assert_eq!(p.current_player(), Player::P1);
    assert_eq!(p.legal_count(), 0);
    p.audit().expect("audit");
}

#[test]
fn windows_through_puts_the_query_cell_at_its_offset() {
    let p = play(&[(0, 0), (1, 0), (2, 0)]);
    for c in [HexCoord::ORIGIN, HexCoord::new(1, 0), HexCoord::new(-4, 7)] {
        let ws = p.windows_through(c);
        assert_eq!(ws.len(), WINDOWS_PER_PLACEMENT);
        for axis in Axis::ALL {
            for k in 0..WINDOW_LEN {
                let slot = &ws[axis.index() * WINDOW_LEN + k];
                assert_eq!(slot.window.axis, axis);
                assert_eq!(slot.window.cell(k), c, "axis {axis:?} offset {k}");
            }
        }
    }
}

#[test]
fn windows_through_agrees_with_the_per_cell_window_query() {
    let p = first_stone_win();
    for q in -6i16..=10 {
        for r in -6i16..=10 {
            let c = HexCoord::new(q, r);
            for slot in p.windows_through(c) {
                assert_eq!(
                    p.window(slot.window),
                    slot.mask,
                    "strip gather and per-cell read disagree at {c:?} {:?}",
                    slot.window
                );
            }
        }
    }

    let face = HexCoord::new(-COORD_LIMIT, 0);
    let mut skipped = 0;
    for slot in p.windows_through(face) {
        if !slot.window.start.is_valid() {
            skipped += 1;
            continue;
        }
        assert_eq!(
            p.window(slot.window),
            slot.mask,
            "on-domain face slot did not round-trip: {:?}",
            slot.window
        );
    }
    assert!(
        skipped > 0,
        "the face query did not exercise off-domain starts"
    );
}

#[test]
fn windows_are_total_far_outside_the_arena() {
    let p = play(&[(0, 0)]);
    for c in [
        HexCoord::new(5000, 0),
        HexCoord::new(-5000, 0),
        HexCoord::new(0, 5000),
        HexCoord::new(0, -5000),
    ] {
        for slot in p.windows_through(c) {
            assert_eq!(slot.mask, WindowMask::EMPTY);
            assert_eq!(p.window(slot.window), WindowMask::EMPTY);
        }
    }
}

#[test]
fn window_masks_report_ownership() {
    let p = play(&[(0, 0), (1, 0), (2, 0)]);
    let w = Window {
        start: HexCoord::ORIGIN,
        axis: Axis::Q,
    };
    let m = p.window(w);
    assert_eq!(m.mask(Player::P0), 0b000001);
    assert_eq!(m.mask(Player::P1), 0b000110);
    assert_eq!(m.occupied(), 0b000111);
    assert_eq!(m.empty(), 0b111000);
    assert!(!m.is_full_for(Player::P0));
}

#[test]
fn legal_actions_are_strictly_ascending_action_ids() {
    let p = play(&[(0, 0), (1, 0), (2, 0), (0, 5), (-3, 1)]);
    let ids: Vec<ActionId> = p.legal_actions().map(Action::id).collect();
    assert!(!ids.is_empty());
    for w in ids.windows(2) {
        assert!(w[0] < w[1], "not ascending: {:?} then {:?}", w[0], w[1]);
    }
    let coords: Vec<HexCoord> = p.legal_actions().map(Action::coord).collect();
    let mut sorted = coords.clone();
    sorted.sort_unstable();
    assert_eq!(coords, sorted);
}

#[test]
fn legal_actions_is_exact_size_and_fused() {
    let p = play(&[(0, 0), (1, 0)]);
    let mut it = p.legal_actions();
    assert_eq!(it.len(), p.legal_count());
    let mut n = 0;
    for _a in it.by_ref() {
        n += 1;
    }
    assert_eq!(n, p.legal_count());
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
}

#[test]
fn is_legal_agrees_with_legal_actions() {
    for p in [
        Position::new(),
        play(&[(0, 0)]),
        play(&[(0, 0), (1, 0)]),
        second_stone_win(),
    ] {
        let listed: std::collections::HashSet<HexCoord> =
            p.legal_actions().map(Action::coord).collect();
        for q in -12i16..=12 {
            for r in -12i16..=12 {
                let c = HexCoord::new(q, r);
                assert_eq!(
                    p.is_legal(Action::new(c)),
                    listed.contains(&c),
                    "disagreement at {c:?}"
                );
            }
        }
    }
}

#[test]
fn stones_iterate_in_canonical_order_regardless_of_play_order() {
    let a = play(&[(0, 0), (1, 0), (2, 0), (0, 5), (0, 7)]);
    let b = play(&[(0, 0), (2, 0), (1, 0), (0, 5), (0, 7)]);
    let sa: Vec<_> = a.stones().collect();
    let sb: Vec<_> = b.stones().collect();
    assert_eq!(sa, sb);
    assert_eq!(sa.len(), a.stone_count() as usize);
    for w in sa.windows(2) {
        assert!(w[0].0 < w[1].0);
    }
    assert_eq!(a.stones().len(), 5);
    assert_eq!(a.stone_count_for(Player::P0), 3);
    assert_eq!(a.stone_count_for(Player::P1), 2);
}

#[test]
fn turn_slot_covers_kind_mover_and_terminal() {
    let mut seen = std::collections::HashSet::new();
    for p in [
        Position::new(),
        play(&[(0, 0)]),
        play(&[(0, 0), (1, 0)]),
        play(&[(0, 0), (1, 0), (2, 0)]),
        play(&[(0, 0), (1, 0), (2, 0), (0, 1)]),
        first_stone_win(),
        second_stone_win(),
    ] {
        let slot = p.turn_slot();
        assert!(slot < crate::zobrist::TURN_SLOTS);
        assert_eq!(
            slot,
            p.phase().kind_index() * 4 + p.current_player().index() * 2 + p.is_terminal() as usize
        );
        assert_eq!(p.zobrist(), p.hash_cells() ^ crate::zobrist::TURN_KEY[slot]);
        seen.insert(slot);
    }
    assert!(
        seen.len() >= 5,
        "the fixtures should span several turn slots"
    );
}

#[test]
fn partial_eq_ignores_arena_growth() {
    let mut grown = play(&[(0, 0)]);
    let flat = play(&[(0, 0)]);
    {
        let mut s = crate::Search::new(&mut grown);
        let mut q = 0i16;
        for _ in 0..30 {
            q += 8;
            s.apply(act(q, 0)).expect("legal");
        }
    }
    assert_eq!(grown, flat);
    assert_eq!(grown.zobrist(), flat.zobrist());
    grown.audit().expect("audit");
}

#[test]
fn turn_closed_form_matches_the_documented_pattern() {
    assert_eq!(turn_closed_form(0, false), Some((0, Player::P0)));
    assert_eq!(turn_closed_form(0, true), None);
    assert_eq!(turn_closed_form(1, true), None);
    let want = [
        (1usize, Player::P1),
        (2, Player::P1),
        (1, Player::P0),
        (2, Player::P0),
        (1, Player::P1),
        (2, Player::P1),
        (1, Player::P0),
        (2, Player::P0),
    ];
    for (i, w) in want.into_iter().enumerate() {
        assert_eq!(
            turn_closed_form(i as u32 + 1, false),
            Some(w),
            "n = {}",
            i + 1
        );
    }
    assert_eq!(turn_closed_form(6, true), turn_closed_form(5, false));
}

#[test]
fn audit_passes_on_a_fresh_and_a_played_position() {
    Position::new().audit().expect("empty");
    play(&[(0, 0)]).audit().expect("opening");
    first_stone_win().audit().expect("first-stone win");
    second_stone_win().audit().expect("second-stone win");
}

#[test]
fn stone_count_tracks_every_placement() {
    let mut p = Position::new();
    assert_eq!(p.stone_count(), 0);
    for (i, &(q, r)) in [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2)]
        .iter()
        .enumerate()
    {
        p.advance(act(q, r)).expect("legal");
        assert_eq!(p.stone_count() as usize, i + 1);
    }
}

#[test]
fn replaying_a_move_list_reproduces_the_position() {
    for moves in [
        &[(0, 0)][..],
        &[(0, 0), (1, 0), (2, 0), (0, 1)][..],
        &FIRST_STONE_WIN[..],
        &SECOND_STONE_WIN[..],
    ] {
        let p = play(moves);
        let rebuilt = Position::replay(&actions(moves)).expect("the move list must replay");
        assert_eq!(rebuilt, p);
        assert_eq!(rebuilt.zobrist(), p.zobrist());
        assert_eq!(rebuilt.phase(), p.phase());
        assert_eq!(rebuilt.current_player(), p.current_player());
        assert_eq!(rebuilt.outcome(), p.outcome());
        rebuilt.audit().expect("a replayed position must audit");
    }
}

#[test]
fn replaying_every_prefix_reproduces_that_ply() {
    let moves = actions(&SECOND_STONE_WIN);
    let mut incremental = Position::new();
    for k in 0..moves.len() {
        let from_prefix = Position::replay(&moves[..k]).expect("prefix replays");
        assert_eq!(from_prefix, incremental, "prefix of length {k}");
        assert_eq!(from_prefix.stone_count() as usize, k);
        incremental.advance(moves[k]).expect("legal");
    }
}

#[test]
fn replay_of_an_empty_slice_is_the_empty_position() {
    let p = Position::replay(&[]).expect("empty replays");
    assert_eq!(p, Position::new());
    assert_eq!(p.stone_count(), 0);
}

#[test]
fn replay_reports_the_ply_that_failed() {
    let err = Position::replay(&[act(0, 0), act(1, 0), act(0, 0)]).expect_err("must fail");
    assert_eq!(err.ply, 2);
    assert_eq!(err.action, act(0, 0));
    assert_eq!(err.cause, MoveError::Occupied(HexCoord::ORIGIN));
    assert!(format!("{err}").contains("ply 2"));

    let err = Position::replay(&[act(0, 0), act(1, 0), act(1, 0)]).expect_err("must fail");
    assert_eq!(err.ply, 2);
    assert_eq!(err.cause, MoveError::Occupied(HexCoord::new(1, 0)));
}

#[test]
fn replay_rejects_an_illegal_opening_at_ply_zero() {
    let err = Position::replay(&[act(1, 0)]).expect_err("must fail");
    assert_eq!(err.ply, 0);
    assert_eq!(err.cause, MoveError::IllegalOpening);
}

#[test]
fn replay_past_a_win_is_a_terminal_error_not_a_silent_stop() {
    let mut too_long = actions(&FIRST_STONE_WIN);
    too_long.push(act(1, 1));
    let err = Position::replay(&too_long).expect_err("must fail");
    assert_eq!(err.ply, FIRST_STONE_WIN.len());
    assert_eq!(err.cause, MoveError::TerminalState);
}

#[test]
fn replay_from_continues_an_existing_position() {
    const MOVES: [(i16, i16); 5] = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)];
    let full = play(&MOVES);
    let moves = actions(&MOVES);
    let mut partial = Position::replay(&moves[..2]).expect("prefix");
    partial.replay_from(&moves[2..]).expect("suffix");
    assert_eq!(partial, full);
    assert_eq!(partial.stone_count() as usize, moves.len());
}

#[test]
fn replay_from_reports_a_ply_relative_to_its_own_slice() {
    let mut p = play(&[(0, 0), (1, 0)]);
    let err = p
        .replay_from(&[act(2, 0), act(2, 0)])
        .expect_err("must fail");
    assert_eq!(err.ply, 1);
    assert_eq!(err.cause, MoveError::Occupied(HexCoord::new(2, 0)));
    assert_eq!(p.stone_count(), 3);
}

#[test]
fn undo_names_the_placement_it_reversed_and_restores_the_stone_count() {
    let mut p = play(&[(0, 0), (1, 0)]);
    {
        let mut s = crate::search::Search::new(&mut p);
        s.apply(act(2, 0)).expect("legal");
        s.apply(act(3, 0)).expect("legal");
        assert_eq!(s.position().stone_count(), 4);
        assert_eq!(s.undo(), Some(act(3, 0)));
        assert_eq!(s.position().stone_count(), 3);
    }
    assert_eq!(p.stone_count(), 2);
}

#[test]
fn equality_ignores_move_order() {
    let a = play(&[(0, 0), (1, 0), (2, 0)]);
    let b = play(&[(0, 0), (2, 0), (1, 0)]);
    assert_eq!(a, b, "the two move orders reach the same position");
    assert_eq!(a.zobrist(), b.zobrist(), "and the same hash");
}

#[test]
fn legal_rank_and_nth_legal_agree_with_the_iterator() {
    for p in [
        play(&[(0, 0)]),
        play(&[(0, 0), (1, 0)]),
        play(&[(0, 0), (1, 0), (2, 0), (0, 1), (5, -3)]),
    ] {
        let listed: Vec<Action> = p.legal_actions().collect();
        assert_eq!(listed.len(), p.legal_count());
        for (i, &a) in listed.iter().enumerate() {
            assert_eq!(p.legal_rank(a), Some(i), "rank of {a:?}");
            assert_eq!(p.nth_legal(i), Some(a), "nth_legal({i})");
        }
        assert_eq!(p.nth_legal(listed.len()), None, "past the end");
    }
}

#[test]
fn legal_rank_is_none_for_illegal_actions() {
    let p = play(&[(0, 0), (1, 0), (2, 0)]);
    assert_eq!(p.legal_rank(act(0, 0)), None, "occupied");
    assert_eq!(p.legal_rank(act(1, 0)), None, "occupied");
    assert_eq!(p.legal_rank(act(400, 400)), None, "far away");
    assert_eq!(p.legal_rank(act(i16::MAX, i16::MAX)), None, "out of bounds");
}

#[test]
fn legal_rank_agrees_with_is_legal_over_a_neighbourhood() {
    let p = play(&[(0, 0), (1, 0), (2, 0), (0, 1)]);
    for q in -12i16..=12 {
        for r in -12i16..=12 {
            let a = act(q, r);
            assert_eq!(
                p.legal_rank(a).is_some(),
                p.is_legal(a),
                "disagreement at ({q}, {r})"
            );
        }
    }
}

#[test]
fn the_opening_ranks_only_the_origin() {
    let p = Position::new();
    assert_eq!(p.legal_rank(Action::new(HexCoord::ORIGIN)), Some(0));
    assert_eq!(p.nth_legal(0), Some(Action::new(HexCoord::ORIGIN)));
    assert_eq!(p.nth_legal(1), None);
    assert_eq!(p.legal_rank(act(1, 0)), None);
    assert_eq!(p.legal_rank(act(0, 1)), None);
}

#[test]
fn the_second_stone_of_a_turn_cannot_rank_the_first() {
    let p = play(&[(0, 0), (1, 0)]);
    assert_eq!(p.phase(), TurnPhase::SecondStone);
    assert_eq!(p.legal_rank(act(1, 0)), None, "the first stone is occupied");
    for (i, a) in p.legal_actions().enumerate() {
        assert_eq!(p.legal_rank(a), Some(i));
    }
}

#[test]
fn a_terminal_position_ranks_nothing() {
    for p in [first_stone_win(), second_stone_win()] {
        assert_eq!(p.legal_count(), 0);
        assert_eq!(p.nth_legal(0), None);
        for (c, _) in p.stones() {
            assert_eq!(p.legal_rank(Action::new(c)), None);
        }
        assert_eq!(p.legal_rank(act(3, 3)), None);
    }
}
