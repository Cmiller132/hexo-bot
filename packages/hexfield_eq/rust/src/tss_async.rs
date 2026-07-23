//! Background deep-solve pool (Stage 4 async rung, PLAN_TSS_DEEPENING.md §10).
//!
//! Moves the deep leaf solves off the search's critical path: gated leaves
//! ENQUEUE a solve request and proceed to the normal GPU eval; pool workers
//! run the identical verified path (`tree::tss_solve_verified` — solver →
//! independent certificate verifier → sealed `HardValue` mint) on their own
//! threads; the driver drains completed results back into the owning search's
//! per-move memo, where the descent-stop in `select_pending_leaf` consumes
//! them on every later visit through the proven position.
//!
//! Soundness is inherited wholesale: nothing here can mint a hard value —
//! only route one that `tss_core::hard_value_from_verified` already accepted.
//! What the pool DOES change is timing: which visit first sees a proof is
//! wall-clock dependent, so flag-on self-play is NOT bit-reproducible under a
//! fixed seed (the flag-off golden digest remains the bit-identity anchor).
//!
//! Staleness: every request carries the pool-global GENERATION its search
//! held at enqueue time (re-assigned on every move/rebind). A response whose
//! generation no longer matches the slot's live search is dropped — except
//! its fatal `deep_verify_failed` count, which is never dropped.
//!
//! Memory: the request queue is bounded. A full legacy queue evicts its oldest
//! request; a full park queue rejects fresh work so no already-parked leaf is
//! orphaned. Both outcomes are counted and selection never waits for capacity.
//! Each worker's solver TT is byte-capped per solve exactly like the inline
//! path, and responses carry only scalars + the small `RootBinding`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use hexo_engine::HexoState as RustHexoState;
use hexo_utils::StateHash;

use crate::tree::{tss_solve_verified, SolverHorizon, TssCounters};
use crate::tss_core::{HardValue, ProofStatus, SolveGoal, ZoneSearchCaps};
use crate::tss_solver::TssSolver;
use crate::tss_verify::RootBinding;

/// Bounded request-queue depth. The legacy queue is a LIFO with oldest-eviction
/// (ep32 first-contact finding): workers always serve the NEWEST request, so
/// result freshness is bounded by workers × solve-time (milliseconds) instead
/// of the whole backlog (ep32 ran 47s of FIFO latency => 65% of responses
/// arrived after their move died). A full queue evicts the OLDEST entry —
/// the least likely to still matter — and the eviction is counted as
/// `async_dropped`; fresh legacy work is never rejected. Park mode is FIFO
/// and rejects a fresh request at capacity rather than evicting accepted
/// work. Memory cost is one state clone per entry (~KBs each).
pub const TSS_ASYNC_QUEUE_CAP: usize = 16384;

/// Queue discipline is frozen when the pool is constructed. Keeping this as
/// an enum makes the flag-off legacy behavior explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QueueDiscipline {
    LegacyLifoEvict,
    ParkFifoNoEvict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PushOutcome {
    evicted: u32,
    depth: usize,
}

/// The request deque shared by producers and workers.
struct RequestQueue {
    queue: Mutex<QueueState>,
    ready: Condvar,
    discipline: QueueDiscipline,
    capacity: usize,
}

struct QueueState {
    entries: VecDeque<SolveRequest>,
    disconnected: bool,
    /// Requests popped by a worker whose response has not been sent yet.
    /// Tracked under the queue lock so `is_idle` can never miss a request
    /// that is between pop and completion (quiesce correctness).
    in_flight: u32,
}

impl RequestQueue {
    fn new(park: bool) -> Self {
        Self::with_capacity(park, TSS_ASYNC_QUEUE_CAP)
    }

    /// Capacity injection is private and lets the queue semantics be tested
    /// without cloning sixteen thousand engine states.
    fn with_capacity(park: bool, capacity: usize) -> Self {
        Self {
            queue: Mutex::new(QueueState {
                entries: VecDeque::with_capacity(1024),
                disconnected: false,
                in_flight: 0,
            }),
            ready: Condvar::new(),
            discipline: if park {
                QueueDiscipline::ParkFifoNoEvict
            } else {
                QueueDiscipline::LegacyLifoEvict
            },
            capacity,
        }
    }

    /// Push a fresh request (newest end). Returns the eviction count and depth,
    /// or None if the pool is shut down / a park-mode queue is at capacity.
    fn push(&self, request: SolveRequest) -> Option<PushOutcome> {
        let mut state = match self.queue.lock() {
            Ok(state) => state,
            Err(_) => return None,
        };
        if state.disconnected {
            return None;
        }
        let mut evicted = 0u32;
        if state.entries.len() >= self.capacity {
            match self.discipline {
                QueueDiscipline::LegacyLifoEvict => {
                    state.entries.pop_front(); // oldest
                    evicted = 1;
                }
                QueueDiscipline::ParkFifoNoEvict => {
                    // A parked leaf exists only after an accepted enqueue, so
                    // reject fresh work rather than invalidating an existing
                    // parked leaf's request. The caller routes this leaf to
                    // ordinary eval. This preserves the hard memory bound.
                    return None;
                }
            }
        }
        state.entries.push_back(request);
        let depth = state.entries.len();
        drop(state);
        self.ready.notify_one();
        Some(PushOutcome { evicted, depth })
    }

    /// Pop according to the frozen discipline, waiting up to 50ms. `None` =>
    /// timed out or disconnected (caller rechecks its shutdown flag either
    /// way). A popped request is marked in-flight under the same lock; the
    /// worker MUST pair it with `finish_one` once the solve is resolved (sent,
    /// dropped, or panicked).
    fn pop_next(&self) -> Option<SolveRequest> {
        let mut state = self.queue.lock().ok()?;
        if state.entries.is_empty() && !state.disconnected {
            let (next, _timeout) = self
                .ready
                .wait_timeout(state, std::time::Duration::from_millis(50))
                .ok()?;
            state = next;
        }
        let popped = match self.discipline {
            QueueDiscipline::LegacyLifoEvict => state.entries.pop_back(),
            QueueDiscipline::ParkFifoNoEvict => state.entries.pop_front(),
        };
        if popped.is_some() {
            state.in_flight = state.in_flight.saturating_add(1);
        }
        popped
    }

    /// Mark one popped request resolved (response sent, dropped, or panicked).
    fn finish_one(&self) {
        if let Ok(mut state) = self.queue.lock() {
            state.in_flight = state.in_flight.saturating_sub(1);
        }
    }

    /// Discard every pending (not yet popped) request, returning the count.
    fn clear_pending(&self) -> u32 {
        match self.queue.lock() {
            Ok(mut state) => {
                let cleared = state.entries.len() as u32;
                state.entries.clear();
                cleared
            }
            Err(_) => 0,
        }
    }

    /// True when no request is queued and none is mid-solve.
    fn is_idle(&self) -> bool {
        match self.queue.lock() {
            Ok(state) => state.entries.is_empty() && state.in_flight == 0,
            Err(_) => true,
        }
    }

    fn disconnect(&self) {
        if let Ok(mut state) = self.queue.lock() {
            state.disconnected = true;
            state.entries.clear();
        }
        self.ready.notify_all();
    }
}

/// Out-of-band alarm channel, written by WORKERS at solve time so the fatal
/// signal exists the moment it happens — it can never be lost to a dropped,
/// stale, or never-drained response (Codex review 4). The drain passes fold
/// pending failures into a live search's counters (=> epoch telemetry); an
/// untaken residue is screamed about on pool drop.
#[derive(Default)]
pub struct PoolAlarms {
    pub verify_failed: AtomicU32,
    pub worker_panics: AtomicU32,
}

/// A gated leaf's solve request. `state` is a clone taken at enqueue time;
/// `binding` re-asserts full-position identity on the way back (the 64-bit
/// hash is never trusted alone for a value-bearing result, §2.5).
pub struct SolveRequest {
    pub slot: u32,
    pub generation: u64,
    pub hash: StateHash,
    pub binding: RootBinding,
    pub state: RustHexoState,
    pub node_cap: u64,
    pub goal: SolveGoal,
    pub zone: ZoneSearchCaps,
    pub horizon: SolverHorizon,
    pub dual_pass: bool,
    pub loss_reserve_nodes: u32,
    pub group2: bool,
    pub j2near: bool,
}

/// A completed, already-verified solve. `hard` is `Some` only when the
/// independent verifier accepted the certificate inside `tss_solve_verified`
/// on the worker thread; `counters` carries that solve's telemetry deltas
/// (deep_calls/win/loss/unknown/nodes/verify_failed) for the owning move.
pub struct SolveResponse {
    pub slot: u32,
    pub generation: u64,
    pub hash: StateHash,
    pub binding: RootBinding,
    pub status: ProofStatus,
    pub hard: Option<HardValue>,
    pub counters: TssCounters,
}

/// State shared by the owning pool and every enqueue handle. Dynamic workers
/// are registered behind a mutex because enqueue happens through `&self` on a
/// handle. Worker creation never happens while the request queue is locked.
struct PoolShared {
    requests: Arc<RequestQueue>,
    response_tx: Sender<SolveResponse>,
    alarms: Arc<PoolAlarms>,
    shutdown: Arc<AtomicBool>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    base_workers: usize,
    max_workers: usize,
    park: bool,
    /// Successful dynamic scale-up spawns since the last telemetry take.
    workers_spawned: AtomicU32,
}

impl PoolShared {
    fn spawn_worker(&self, index: usize) -> std::io::Result<JoinHandle<()>> {
        let rx = Arc::clone(&self.requests);
        let tx = self.response_tx.clone();
        let alarms = Arc::clone(&self.alarms);
        let stop = Arc::clone(&self.shutdown);
        std::thread::Builder::new()
            .name(format!("tss-solve-{index}"))
            .spawn(move || worker_loop(rx, tx, alarms, stop))
    }

    /// Spawn at most one worker for this push. The queue-depth snapshot was
    /// captured under the queue lock, but this registry lock (and OS thread
    /// creation) happens only after that lock has been released.
    fn maybe_scale(&self, queue_depth: usize) {
        if self.shutdown.load(Ordering::Relaxed) {
            return;
        }
        let mut workers = self
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let current = workers.len();
        if self.shutdown.load(Ordering::Relaxed)
            || current >= self.max_workers
            || queue_depth <= current.saturating_mul(2)
        {
            return;
        }
        if let Ok(worker) = self.spawn_worker(current) {
            workers.push(worker);
            self.workers_spawned.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn worker_count(&self) -> usize {
        self.workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

/// Per-search enqueue handle: a reference to the pool's queue/scaler plus the
/// slot/generation identity stamped on every request. Rewired by the driver
/// at every search creation, reuse-rebind, and move advance.
#[derive(Clone)]
pub struct TssAsyncHandle {
    shared: Arc<PoolShared>,
    pub slot: u32,
    pub generation: u64,
}

impl std::fmt::Debug for TssAsyncHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TssAsyncHandle")
            .field("slot", &self.slot)
            .field("generation", &self.generation)
            .finish()
    }
}

impl TssAsyncHandle {
    /// Enqueue at the fresh end. `Some(evicted)` => accepted
    /// (with `evicted` OLD entries discarded to make room — the caller counts
    /// them as `async_dropped`); `None` => pool shut down or park queue full
    /// (the caller counts the request itself as dropped and the leaf takes the
    /// plain net eval).
    /// Legacy mode never rejects fresh work; park mode rejects at capacity.
    pub fn try_enqueue(&self, request: SolveRequest) -> Option<u32> {
        let outcome = self.shared.requests.push(request)?;
        // Preserve the pre-park fixed-size async path exactly. Dynamic scale
        // is a park-mode capacity mechanism and is never consulted flag-off.
        if self.shared.park {
            self.shared.maybe_scale(outcome.depth);
        }
        Some(outcome.evicted)
    }
}

/// The worker pool. Owned by the MCTS session so worker solvers (each with
/// its own persistent positive-proof-fragment cache) stay warm across
/// `run_continuous` calls. Dropping the pool closes the request channel and
/// the workers exit on their next `recv`.
pub struct TssAsyncPool {
    shared: Arc<PoolShared>,
    results: Receiver<SolveResponse>,
    generation: AtomicU64,
}

impl std::fmt::Debug for TssAsyncPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TssAsyncPool")
            .field("base_workers", &self.shared.base_workers)
            .field("max_workers", &self.shared.max_workers)
            .field("workers", &self.worker_count())
            .field("park", &self.shared.park)
            .finish()
    }
}

impl TssAsyncPool {
    /// Resolve a constructor request to its effective worker ceiling. Park-off
    /// is deliberately fixed at the base for flag-off behavioral identity.
    pub fn resolved_max_worker_count(threads: u32, threads_max: u32, park: bool) -> usize {
        let base_workers = threads.clamp(1, 32) as usize;
        if !park {
            return base_workers;
        }
        if threads_max == 0 {
            let available = std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(base_workers);
            available
                .saturating_sub(6)
                .max(base_workers)
                .min(24usize.max(base_workers))
        } else {
            (threads_max as usize).clamp(base_workers, 64)
        }
    }

    /// Construct a pool with a fixed base, a dynamic ceiling, and a frozen
    /// queue discipline. Park-off always fixes the ceiling at base. In park
    /// mode, `threads_max == 0` selects the auto ceiling:
    /// clamp(available_parallelism - 6, base, 24). A base above 24 naturally
    /// makes the auto ceiling equal to the base.
    pub fn new(threads: u32, threads_max: u32, park: bool) -> Self {
        let base_workers = threads.clamp(1, 32) as usize;
        let max_workers = Self::resolved_max_worker_count(threads, threads_max, park);
        let requests = Arc::new(RequestQueue::new(park));
        let (response_tx, results) = std::sync::mpsc::channel::<SolveResponse>();
        let alarms = Arc::new(PoolAlarms::default());
        let shutdown = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(PoolShared {
            requests,
            response_tx,
            alarms,
            shutdown,
            workers: Mutex::new(Vec::with_capacity(max_workers)),
            base_workers,
            max_workers,
            park,
            workers_spawned: AtomicU32::new(0),
        });
        {
            let mut workers = shared
                .workers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for index in 0..base_workers {
                workers.push(
                    shared
                        .spawn_worker(index)
                        .expect("spawn tss async solve worker"),
                );
            }
        }
        Self {
            shared,
            results,
            generation: AtomicU64::new(1),
        }
    }

    /// Mint a fresh generation (monotone, pool-global — unique per
    /// (search, move) so cross-move and cross-call responses can never
    /// masquerade as live).
    pub fn next_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::Relaxed)
    }

    /// A handle stamped for `slot` at a fresh generation.
    pub fn handle_for(&self, slot: u32) -> TssAsyncHandle {
        TssAsyncHandle {
            shared: Arc::clone(&self.shared),
            slot,
            generation: self.next_generation(),
        }
    }

    /// Drain every completed response without blocking.
    pub fn try_drain(&self) -> Vec<SolveResponse> {
        let mut drained = Vec::new();
        while let Ok(response) = self.results.try_recv() {
            drained.push(response);
        }
        drained
    }

    /// Take (swap to 0) the accumulated fatal verify-failure count. Drain
    /// passes call this with a live search in hand so the count reaches the
    /// epoch telemetry no matter which response carried the failure.
    pub fn take_verify_failures(&self) -> u32 {
        self.shared.alarms.verify_failed.swap(0, Ordering::Relaxed)
    }

    /// Take (swap to 0) the accumulated worker-panic count (ops signal; each
    /// panic lost one request and recycled that worker's solver).
    pub fn take_worker_panics(&self) -> u32 {
        self.shared.alarms.worker_panics.swap(0, Ordering::Relaxed)
    }

    /// Take (swap to zero) successful DYNAMIC worker spawns. Base workers are
    /// configuration, not scale-up telemetry, and are therefore excluded.
    pub fn take_workers_spawned(&self) -> u32 {
        self.shared.workers_spawned.swap(0, Ordering::Relaxed)
    }

    /// End-of-run quiesce (Codex review, late-alarm loss): discard every
    /// pending request (all are stale once the scheduler loop ends — their
    /// generations died with the finished slots) and wait, bounded, for
    /// in-flight solves to resolve, so the alarm bank and result channel are
    /// FINAL before the caller takes its tail drain into the epoch telemetry.
    /// Returns the number of discarded pending requests. On timeout (a solve
    /// still mid-flight) the residue still reaches the next drain pass or the
    /// Drop-time stderr backstop.
    pub fn quiesce_for_telemetry(&self, max_wait: std::time::Duration) -> u32 {
        let cleared = self.shared.requests.clear_pending();
        let deadline = std::time::Instant::now() + max_wait;
        while !self.shared.requests.is_idle() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        cleared
    }

    /// Current live worker-thread count, including dynamic scale-up workers.
    pub fn worker_count(&self) -> usize {
        self.shared.worker_count()
    }

    /// Construction-time base worker count (stable across dynamic scale-up).
    pub fn base_worker_count(&self) -> usize {
        self.shared.base_workers
    }

    /// Resolved dynamic ceiling (`threads_max=0` has already become auto).
    pub fn max_worker_count(&self) -> usize {
        self.shared.max_workers
    }

    /// Whether this pool uses the park-safe FIFO/no-eviction queue.
    pub fn park_mode(&self) -> bool {
        self.shared.park
    }
}

impl Drop for TssAsyncPool {
    fn drop(&mut self) {
        // Quiesce BEFORE the final alarm read (a worker mid-solve could bank
        // a failure after an early read): raise the shutdown flag (workers
        // exit after at most their CURRENT solve), disconnect + clear the
        // queue (waking every parked worker), then join. Only then is the
        // alarm bank final. Handle clones parked on persisted searches can
        // no longer stall this: the disconnect wakes waiters and every
        // pop/timeout path rechecks the flag.
        self.shared.shutdown.store(true, Ordering::Relaxed);
        self.shared.requests.disconnect();
        let workers = {
            let mut workers = self
                .shared
                .workers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::take(&mut *workers)
        };
        for worker in workers {
            let _ = worker.join();
        }
        let verify_failed = self.shared.alarms.verify_failed.load(Ordering::Relaxed);
        let panics = self.shared.alarms.worker_panics.load(Ordering::Relaxed);
        if verify_failed > 0 {
            eprintln!(
                "hexfield tss_async: {verify_failed} UNREPORTED certificate verify \
                 FAILURE(s) at pool shutdown — investigate immediately"
            );
        }
        if panics > 0 {
            eprintln!("hexfield tss_async: {panics} unreported worker panic(s) at pool shutdown");
        }
    }
}

fn worker_loop(
    rx: Arc<RequestQueue>,
    tx: Sender<SolveResponse>,
    alarms: Arc<PoolAlarms>,
    shutdown: Arc<AtomicBool>,
) {
    // One persistent solver per worker: its shared positive-proof-fragment TT
    // warms across solves (O16); byte caps are enforced per solve inside
    // `tss_solve_verified` exactly as on the inline path. Configured to the
    // campaign leaf-decided profile (§3), identical to the inline/root paths.
    let mut solver = TssSolver::default();
    solver.configure_leaf_profile();
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return; // pool dropping
        }
        // Discipline-selected pop with a bounded wait, so the shutdown flag
        // is rechecked at least every 50ms regardless of queue traffic.
        let Some(request) = rx.pop_next() else {
            continue; // timeout or disconnect: loop top rechecks shutdown
        };
        if shutdown.load(Ordering::Relaxed) {
            rx.finish_one();
            return; // checked again post-pop: skip the doomed solve
        }
        // Panic shield (Codex review 7): a panicking solve loses its request
        // (the Pending entry falls out at the owner's next move) but the
        // worker survives with a FRESH solver (the old one's state is
        // suspect), and the panic is counted instead of silently shrinking
        // the pool.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut counters = TssCounters::default();
            solver.set_dual_pass(request.dual_pass);
            solver.set_loss_reserve_nodes(request.loss_reserve_nodes);
            solver.set_group2(request.group2);
            solver.set_leaf_j2near(request.j2near);
            let solved = tss_solve_verified(
                &request.state,
                request.node_cap,
                request.goal,
                request.zone,
                request.horizon,
                &mut solver,
                &mut counters,
            );
            (solved.status, solved.hard, counters)
        }));
        let (status, hard, mut counters) = match outcome {
            Ok(result) => result,
            Err(_) => {
                alarms.worker_panics.fetch_add(1, Ordering::Relaxed);
                solver = TssSolver::default();
                solver.configure_leaf_profile();
                rx.finish_one();
                continue;
            }
        };
        // The alarm atomic is the SOLE carrier of the fatal signal (single
        // channel — no drain-vs-response double count): strip it from the
        // response counters after banking it.
        if counters.deep_verify_failed > 0 {
            alarms
                .verify_failed
                .fetch_add(counters.deep_verify_failed, Ordering::Relaxed);
            counters.deep_verify_failed = 0;
        }
        let response = SolveResponse {
            slot: request.slot,
            generation: request.generation,
            hash: request.hash,
            binding: request.binding,
            status,
            hard,
            counters,
        };
        let sent = tx.send(response);
        rx.finish_one();
        if sent.is_err() {
            return; // pool dropped
        }
    }
}
