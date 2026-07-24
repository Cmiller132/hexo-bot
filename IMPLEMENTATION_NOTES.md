# Live Search Telemetry — Implementation Notes

## Feasibility verdict

**PASS: genuine, concurrent, parity-safe Hexfield Gumbel SH streaming is
feasible and implemented.**

The implementation records one real search initialized once at its full visit
budget. It never repeats or incrementally restarts `search()`. Compact and wide
CPU fixtures prove byte-identical authoritative results with telemetry off and
on, as well as byte-identical evaluator requests in the same order.

The shipped egress is option **(a)** from the brief:

`Rust callback → worker _PROGRESS frame → existing multiprocessing result queue
→ web-loop decoder → bounded replay hub → owner-only SSE → EventSource`

This path gives genuine mid-search updates. The authoritative action remains
the normal search return and both bot stones are still committed atomically by
`GameSession.apply_bot_actions`.

## What was built

### Native read-only recorder

- `HexfieldMctsSession.search(..., telemetry_callback=None)` adds one final,
  optional argument. The default is off.
- `start` is emitted after the real root evaluation and Gumbel-Top-m
  initialization. It contains the full bare-model policy reconstructed from
  the evaluator's stored raw logits, the genuine initial candidate set, target
  visits, and root value. It does not run another forward.
- `round` is captured only when the existing search reaches a drained SH
  barrier. Candidate order and heat weights come from the exact SH scores
  already used to rank that halving; per-candidate visit counts are included
  separately.
- `complete` copies the already-built authoritative result fields before root
  advance. It does not recompute the final policy, value, or action.
- Recorder snapshots are detached copies/immutable byte buffers. They do not
  expose the search tree, draw RNG, alter eval batches, or call Python while
  Rust has released the GIL.
- Callback exceptions are swallowed. The search result path remains
  authoritative.

### Serve-base port specifics

- This port is based on `1d1287a` (`perf/hexfield-main5-serve`), the live
  main_5 ep35 serve branch, not the older reference feature base.
- The reference `search.rs` and `tree.rs` were not copied. The recorder was
  re-hooked onto this engine's own `run_searches_to_targets` /
  `select_leaf_batch` flow and `maybe_advance_gumbel_round` implementation.
- A round snapshot is permitted only when both this engine's ordinary
  in-flight leaves and its newer parked-TSS leaves are drained. Telemetry-off
  continues to call the original halving method; telemetry-on calls a sibling
  method that returns detached copies of the already-computed SH ranking.
- The worker/server/SSE/frontend patch applied independently of the engine and
  was reconciled onto the serve branch without replacing unrelated showcase
  code.
- `packages/hexfield_eq/python/hexfield_eq/model.py` is unchanged. The
  `_XPU_LEAN_BIAS`, `_XPU_RAY_COEFF_LUT`, and `_XPU_ATTN_HEAD_SPLIT` serve
  paths and all `HEXFIELD_XPU_*` gates remain exactly as they were at the base
  commit.

### Worker, server, and stream

- Telemetry-off `bot_turn` retains the previous branch and does not pass a new
  search kwarg, allocate event payloads, or run another model evaluation.
- With telemetry enabled, the callback retains scalar fields and immutable byte
  buffers only. The worker forwards them through the existing ordered
  multiprocessing result queue. Wide per-cell JSON expansion happens in the
  web event loop, after IPC, rather than on the search thread.
- Per-job progress callbacks are removed on success, timeout, recycle, error,
  and shutdown. Late frames and callback exceptions cannot resolve or fail the
  move future.
- `LiveSearchHub` uses a 64-event replay ring, 64-event subscriber queues, and
  at most four viewers per game. Slow or disconnected viewers never
  back-pressure search.
- `GET /api/game/{id}/search-stream` is owner-cookie-only SSE with replay,
  monotonic IDs, heartbeats, `Cache-Control: no-cache`, and
  `X-Accel-Buffering: no`.
- Every public event has `seq`, `run_id`, `attempt`, `base_ply`, and `kind`.
  Public kinds are `turn_start`, `bare_policy`, `candidate_set`,
  `search_round`, `search_complete`, `stone`, `turn_complete`,
  `turn_failed`, and `turn_cancelled`.
- A new user retry gets a new run ID and cleared replay. A transparent worker
  retry emits a new `turn_start` barrier inside the same run.
- Streamed `stone` events are presentation-only. The server still applies the
  real one- or two-stone result in one authoritative operation after the worker
  returns.

### Viewer

- One checked-by-default **watch live search** toggle controls F1, F2, and F3.
- Create, move, and retry mutations explicitly tell the server whether to
  record. Retry explicitly sends both true and false so an unchecked viewer
  cannot accidentally retain a prior opt-in.
- The play board has keyed heat cells with eased opacity transitions and a
  genuine SH-leader ring. Existing analysis heat behavior is unchanged.
- Preview stones live in a separate SVG layer and never enter authoritative
  moves, occupied cells, legal cells, or touch staging.
- The animation queues real events in order: stone 1 search → stone 1 preview →
  stone 2 search → stone 2 preview. A retry `turn_start` is an immediate queue
  barrier, so frames cannot cross attempts.
- Normal game polling remains active and authoritative. A missing EventSource,
  malformed frame, disconnect, cancellation, or timeout clears optional
  visuals and falls back to the final polled result.
- Final authoritative snapshots are briefly deferred while queued frames drain,
  then replace previews and heat in the same board update. Reduced-motion users
  skip the dwell.

Hexfield checkpoints receive genuine concurrent native SH events. Other model
families do not expose this native recorder; they retain a clearly bounded
post-search start/final fallback derived from the one completed search, with no
fabricated rounds and no extra forward.

## Enable and disable

- Native recorder: omit `telemetry_callback` (default/off), or pass a callable.
- HTTP create/move: `?watch_search=1` enables; omission/default is off.
- HTTP retry: `?watch_search=1` enables and `?watch_search=0` disables.
- Browser: the checked viewer toggle sends the opt-in. Unchecking it closes the
  optional stream and leaves polling/gameplay unchanged.
- Match API, training, self-play, and non-viewer server callers do not opt in.

There is no global environment flag and no telemetry state is written into
snapshots, the database, or HXR records.

## Parity proof and verification

All commands were run from this worktree using `./.venv`, with
`CARGO_HOME=./.cargo`, `CUDA_VISIBLE_DEVICES=""`,
`ROCR_VISIBLE_DEVICES=""`, and `SHOWCASE_DEVICE=cpu`.

Build the worktree extension:

```powershell
$env:CARGO_HOME = Join-Path (Get-Location) ".cargo"
$env:CUDA_VISIBLE_DEVICES = ""
$env:ROCR_VISIBLE_DEVICES = ""
$env:SHOWCASE_DEVICE = "cpu"
.\.venv\Scripts\maturin.exe develop --release --locked `
  -m packages\hexfield_eq\Cargo.toml
```

Hard parity gate:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_hexfield_eq_live_telemetry.py -q
```

Result: **4 passed**. The fixtures have 216 and 891 legal cells. For each
fixture, telemetry off/on has:

- identical result key sets;
- byte-identical `action_id` (u32), `root_value` (f32), and `visits` (u32);
- byte-identical `visit_policy_action_ids_bytes`;
- byte-identical `visit_policy_weights_bytes`; and
- byte-identical evaluator payloads/order (`node_row_offsets`, legal counts,
  features, coordinates, neighbors, ray lengths, shapes, and request flags).

The same gate also verifies genuine start/round/complete payloads and proves a
raising callback cannot fail the search.

Additional CPU results:

- Rust `hexfield_eq` suite with the Python feature: **77 passed**.
- Existing Hexfield parity/golden files: **15 passed**.
- Live-search backend/authorized-SSE tests: **8 passed**.
- Existing showcase files overlapping the changed API/pool/family/profile/unit
  paths: **57 passed, 1 skipped**.
- JavaScript syntax checks for `app.js`, `api.js`, and `board.js`: passed.
- Python compile checks and `git diff --check`: passed.

No timing microbenchmark was repeated for this port. The stronger semantic
check did run: the automated trace proves the number, contents, and ordering of
every evaluator request are unchanged, and the telemetry-off worker path
retains its pre-feature branch.

## Not verified

- No browser was launched because this machine is running protected GPU
  training and a browser could not be guaranteed never to initialize hardware
  acceleration. Rendered visual QA (actual paint/easing on target browsers)
  remains for an operator on a safe machine. Static DOM/SVG review, syntax
  checks, event-contract tests, and reduced-motion logic were completed.
- The production `hexfield_eq_main5_ep35` checkpoint is not present in this
  worktree (`deploy/models` contains only `.gitkeep`). Native telemetry used a
  deterministic CPU evaluator and server egress used fakes/tiny CPU fixtures.
  A production-checkpoint visual smoke test and production-load latency
  measurement remain for deployment staging.
- The aggregate full-showcase invocation did not complete inside a five-minute
  run window, so no full-suite verdict is claimed. The focused live-search
  suite and the 57 overlapping existing showcase tests listed above passed.
- A port-specific on/off timing benchmark was not run; correctness and request
  schedule identity were verified instead.
- Non-Hexfield families have only the honest post-search fallback described
  above; genuine Gumbel SH rounds are specific to `hexfield_eq`.
- The first `maturin develop` invocation used maturin's default system temporary
  directory for transient wheel staging before `TEMP`/`TMP` were redirected to
  `./.cargo/tmp`. Subsequent build/cache writes were worktree-local. No other
  worktree, training process, or accelerator was accessed.
- Nothing was deployed, no image was built or pushed, and no Git remote was
  contacted.

## Deployment handoff

No deployment, image build, push, or remote contact was performed. The
operator should build and stage the committed serve branch with the real
main_5 ep35 checkpoint, then visually confirm the two-stone animation and
polling fallback on the target host.
