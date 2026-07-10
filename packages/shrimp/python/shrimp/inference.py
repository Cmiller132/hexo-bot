"""ShrimpEvaluator — serve-side half of the wire ABI.

Consumes the Rust payload (CSR over support nodes, rows sorted by support size
descending), packs rows into quantized static shapes under the inference pair
ceiling, runs `forward_policy_value`, and returns the reply: `values_bytes`
(f32 x B, clamped [-1, 1]), `priors_bytes` (f32 x sum L_g, positional over each
row's legal prefix, fp32 softmax), `moves_left_bytes` (f32 x B) when requested,
and `priors_logits_bytes` (f32 x sum L_g, raw pre-softmax policy logits, same
positional layout as `priors_bytes`) when `request_logits` is set.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch._dynamo  # noqa: F401  (mark_dynamic / config used in the serve path)

from .constants import NUM_FEATURES, NUM_TOKENS
from .losses import decode_binned_value, decode_moves_left
from .model import ShrimpNet

NBR_SENTINEL = 0xFFFF
# Upper bound on B * S_pad^2 per group. The 3.8e7 default keeps the fp16
# (B, 4, S, S) bias transient roughly under ~305 MB on the MATERIALIZED path.
# On the flex serve path no (B, 4, S, S) tensor exists (the largest S^2 object
# is the flex-pair uint8 index at B*S^2 BYTES), so the ceiling can be raised
# via SHRIMP_PAIR_CEILING to pack fewer, fatter groups per flush — each group
# costs host-side dispatch (dynamo guards + kernel enqueue), which is the
# serve bottleneck once the kernels themselves are fast. Distinct from the
# training pair budget.
PAIR_CEILING = float(os.environ.get("SHRIMP_PAIR_CEILING", 0) or 3.8e7)
# Padded cell-count quantum (rows pad up to a multiple of this).
QUANT_NODES = 64
# Padding-waste bound for grouping: a row is not padded up to a group anchor
# more than WASTE_FRACTION larger than its own size (or QUANT_NODES, whichever
# is larger). Bounds the squared attention padding cost (sum B*S_pad^2).
# SHRIMP_WASTE_FRACTION overrides: once the serve wall is per-GROUP host
# dispatch (guard eval + kernel enqueue) rather than GPU compute, padding more
# rows into fewer groups trades idle-GPU FLOPs for dispatch — a win while
# gpu_idle_fraction is high.
WASTE_FRACTION = float(os.environ.get("SHRIMP_WASTE_FRACTION", 0) or 0.18)


def _ceil_quant(n: int) -> int:
    return max(QUANT_NODES, -(-int(n) // QUANT_NODES) * QUANT_NODES)


def serve_autocast(device: torch.device | str) -> bool:
    """Whether the serve forward autocasts to fp16 on `device`.

    cuda: on unless SHRIMP_SERVE_AUTOCAST=0 (long-proven path).
    xpu: OPT-IN via SHRIMP_SERVE_AUTOCAST=1. fp16 halves bytes moved and the
    activation VRAM that wedges small cards, but the first fp16 kernel
    compile on an Arc A310 wedged the i915/xe KERNEL driver in production —
    and an LXC-shared kernel turns that into a host-wide outage, not a
    recycleable worker fault. Enable deliberately, watch dmesg, keep a finger
    on the power button. Exposed so the showcase device self-check verifies
    the exact numeric path the evaluator serves (device.py builds its parity
    forward with the same flag).
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        return os.environ.get("SHRIMP_SERVE_AUTOCAST", "1") != "0"
    if dev.type == "xpu":
        return os.environ.get("SHRIMP_SERVE_AUTOCAST", "0") == "1"
    return False


class _PinnedRing:
    """Ring of pinned staging buffer-sets for the copy-stream H2D path
    (SHRIMP_COPY_STREAM). A pageable H2D copy serializes the submitting host
    thread with the GPU stream (the driver stages it), which made
    submit_payload's duration track the whole flush's device time. Staging each
    group's four wire buffers through persistent PINNED memory and issuing the
    copies on a dedicated copy stream makes submit a true enqueue: the compute
    stream waits per group on a copy-completion event, and slot reuse is gated
    on that same event so a buffer is never overwritten while its copy is in
    flight. Buffers grow on demand and are never shrunk."""

    SLOTS = 4
    KEYS = ("feats", "nbr", "mask", "coords")

    def __init__(self) -> None:
        self.slots = [
            {"feats": None, "nbr": None, "mask": None, "coords": None, "event": None}
            for _ in range(self.SLOTS)
        ]
        self.i = 0

    def acquire(self) -> dict:
        slot = self.slots[self.i]
        self.i = (self.i + 1) % self.SLOTS
        if slot["event"] is not None:
            slot["event"].synchronize()  # usually already complete
        return slot

    @staticmethod
    def stage(slot: dict, key: str, raw: bytes) -> torch.Tensor:
        """Copy `raw` into the slot's pinned buffer for `key`; return the
        filled uint8 pinned view (length == len(raw))."""
        n = len(raw)
        buf = slot[key]
        if buf is None or buf.numel() < n:
            cap = max(n, 1 << 20)
            buf = torch.empty(cap, dtype=torch.uint8, pin_memory=True)
            slot[key] = buf
        buf[:n].numpy()[:] = np.frombuffer(raw, dtype=np.uint8)
        return buf[:n]


class PerfTrace:
    """cuda.Event-based GPU-busy instrument (bench-only, SHRIMP_PERF_TRACE=1).

    Measures the GPU-busy fraction of forward compute using cuda.Events rather
    than a sampler.

    Mechanism: a pair of `cuda.Event`s brackets the forward enqueues of each
    flush (recorded in submit, on the same stream the forwards run on). When the
    flush is drained (`result()` forces the D2H sync, so both events are
    complete), `start.elapsed_time(end)` gives the wall-clock GPU time that
    flush's forwards occupied the device. Summing those over a measured wall
    window yields busy_fraction = sum(device_ms) / wall_ms. Idle fraction =
    1 - busy.

    Adds only `cuda.Event.record()` calls to the forward path (no extra sync, no
    D2H); the events are read in `result()` which already syncs. Inert unless the
    flag is set (the evaluator holds `_perf = None`).
    """

    def __init__(self) -> None:
        self.device_ms = 0.0      # sum of per-flush forward device time
        self.flushes = 0          # flushes whose events were measured
        self.states = 0           # total states (rows) evaluated
        self.per_flush_ms: list[float] = []
        self.per_flush_states: list[int] = []
        self._t0: float | None = None   # wall-clock anchor (first submit)
        self._t_last: float = 0.0        # wall-clock of the last drained flush

    def make_events(self):
        """Allocate a (start, end) cuda.Event pair for one flush, or None if
        CUDA is unavailable (tracing is a no-op on the CPU path)."""
        if not torch.cuda.is_available():
            return None
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def on_submit(self) -> None:
        import time as _time
        if self._t0 is None:
            self._t0 = _time.perf_counter()

    def on_result(self, events, n_states: int) -> None:
        """Called from result() AFTER the D2H sync (events guaranteed complete).
        Accumulate this flush's device-busy ms and the wall anchor."""
        import time as _time
        self._t_last = _time.perf_counter()
        if events is None:
            return
        start_ev, end_ev = events
        ms = start_ev.elapsed_time(end_ev)
        self.device_ms += ms
        self.flushes += 1
        self.states += n_states
        self.per_flush_ms.append(ms)
        self.per_flush_states.append(n_states)

    def report(self) -> dict:
        import statistics
        wall_ms = (
            (self._t_last - self._t0) * 1000.0
            if self._t0 is not None and self._t_last > self._t0
            else 0.0
        )
        busy = (self.device_ms / wall_ms) if wall_ms > 0 else 0.0
        ms = self.per_flush_ms
        st = self.per_flush_states
        out = {
            "gpu_busy_fraction": round(busy, 4),
            "gpu_idle_fraction": round(1.0 - busy, 4) if wall_ms > 0 else None,
            "device_busy_ms": round(self.device_ms, 2),
            "wall_window_ms": round(wall_ms, 2),
            "measured_flushes": self.flushes,
            "measured_states": self.states,
            "mean_batch": round(self.states / self.flushes, 1) if self.flushes else 0.0,
            "mean_forward_ms": round(statistics.mean(ms), 3) if ms else 0.0,
            "mean_ms_per_state": (
                round(self.device_ms / self.states, 4) if self.states else 0.0
            ),
        }
        if len(ms) >= 2:
            out["median_forward_ms"] = round(statistics.median(ms), 3)
        if len(st) >= 2:
            out["median_batch"] = round(statistics.median(st), 1)
        return out


def plan_groups(sizes) -> list[tuple[int, int, int]]:
    """Padding-aware grouping over rows sorted descending by size. Returns
    (start, end, pad_to) groups. pad_to is the QUANT_NODES-quantized anchor
    (largest row in the group), so pad_to >= every row in the group. A group
    stops extending when (a) the pair ceiling would be exceeded or (b) the next
    row is smaller than the anchor pad by more than the waste bound (see
    WASTE_FRACTION)."""
    n = len(sizes)
    groups: list[tuple[int, int, int]] = []
    start = 0
    while start < n:
        pad_to = _ceil_quant(int(sizes[start]))
        floor = pad_to - max(QUANT_NODES, int(WASTE_FRACTION * pad_to))
        end = start + 1
        while end < n:
            if (end - start + 1) * (pad_to + NUM_TOKENS) ** 2 > PAIR_CEILING:
                break
            if int(sizes[end]) < floor:  # exceeds padding-waste bound -> split
                break
            end += 1
        groups.append((start, end, pad_to))
        start = end
    return groups


class _GraphCache:
    """CUDA-graph capture/replay for the serve forward (SHRIMP_CUDA_GRAPHS).

    The serve became kernel-LAUNCH bound: a flush of ~20 groups enqueues ~1000+
    kernels, saturating the CUDA launch queue so submit_payload blocks inside
    cudaLaunchKernel for roughly the whole device time. A captured graph
    replays a group's entire forward as ONE launch.

    Keyed by (B_bucket, Npad, request_moves_left): group batch sizes are
    rounded up a fixed ladder and the pad rows use the model's ordinary
    pad-row convention (zero feats, nbr=Npad, mask False), so outputs for the
    real rows are unchanged and the pad rows are sliced off after replay.
    Every graph shares one memory pool. A key that fails capture falls back to
    the regular compiled path permanently."""

    # Multiples of 4: mean pad waste ~1.5 rows/group. More keys than a coarse
    # ladder, but each capture is cheap and the set is bounded by the pair
    # ceiling (B <= ~260).
    MAX_B = 260

    def __init__(self, fwd, autocast_on: bool, device: torch.device) -> None:
        self._fwd = fwd
        self._autocast_on = autocast_on
        self._device = device
        self._pool = torch.cuda.graph_pool_handle()
        self._graphs: dict = {}
        self._failed: set = set()

    def bucket(self, g: int) -> int | None:
        b = max(4, -(-g // 4) * 4)
        return b if b <= self.MAX_B else None

    def _capture(self, key, npad: int, request_ml: bool):
        bb = key[0]
        dev = self._device
        static_in = {
            "feats": torch.zeros(bb, npad, NUM_FEATURES, dtype=torch.float16, device=dev),
            "nbr": torch.full((bb, npad, 6), npad, dtype=torch.int64, device=dev),
            "mask": torch.zeros(bb, npad, dtype=torch.bool, device=dev),
            "coords": torch.zeros(bb, npad, 2, dtype=torch.int64, device=dev),
        }

        def run():
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self._autocast_on
            ):
                return self._fwd(
                    static_in["feats"], static_in["nbr"], static_in["mask"],
                    static_in["coords"], request_moves_left=request_ml,
                )

        # Warm up on a side stream (allocator + dynamo guards + flex compiles
        # all settle), then capture on the default stream.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_out = run()
        # use_evt: recorded on the compute stream after every replay of this
        # entry, so a copy-stream refill of its statics waits only on THIS
        # key's previous use (not on all enqueued compute).
        return {
            "graph": graph, "in": static_in, "out": static_out,
            "use_evt": torch.cuda.Event(),
        }

    def entry_for(self, g: int, npad: int, request_ml: bool):
        """Get-or-capture the graph entry for a (bucketed) group shape, or None
        when the shape cannot/failed to capture (caller falls back)."""
        bb = self.bucket(g)
        if bb is None:
            return None
        key = (bb, npad, request_ml)
        if key in self._failed:
            return None
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(key, npad, request_ml)
            except Exception:
                self._failed.add(key)
                return None
            self._graphs[key] = entry
        return entry

    @staticmethod
    def replay(entry, g: int):
        """Replay with the static inputs already filled for rows [:g] (pad rows
        [g:] must already carry the pad convention — reset_pad handles it) and
        return the out dict sliced to the true rows, cloned off the statics."""
        entry["graph"].replay()
        out = {k: v[:g].clone() for k, v in entry["out"].items()}
        entry["use_evt"].record(torch.cuda.current_stream())
        return out

    @staticmethod
    def reset_pad(entry, g: int, npad: int) -> None:
        si = entry["in"]
        bb = si["feats"].shape[0]
        if bb > g:
            si["feats"][g:].zero_()
            si["nbr"][g:].fill_(npad)
            si["mask"][g:].zero_()
            si["coords"][g:].zero_()

    def run_group(self, d_feats, d_nbr, d_mask, d_coords, g: int, request_ml: bool):
        """Device-tensor entry point (non-copy-stream callers): D2D into the
        statics, replay, slice+clone."""
        npad = d_feats.shape[1]
        entry = self.entry_for(g, npad, request_ml)
        if entry is None:
            return None
        try:
            si = entry["in"]
            si["feats"][:g].copy_(d_feats)
            si["nbr"][:g].copy_(d_nbr)
            si["mask"][:g].copy_(d_mask)
            si["coords"][:g].copy_(d_coords)
            self.reset_pad(entry, g, npad)
            return self.replay(entry, g)
        except Exception:
            self._failed.add((self.bucket(g), npad, request_ml))
            return None


class ShrimpEvaluator:
    def __init__(self, model: ShrimpNet, device: torch.device | str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        # Serve forward compile (CUDA only; eval/self-play path, not training).
        # torch.compile fuses the many small elementwise/gather kernels of the
        # rel-pos bias machinery.
        #
        # A single dynamic compile serves every shape: both varying dims — batch
        # (dim 0) and cell-count Npad (dim 1) — are marked dynamic (in
        # _run_forward) and compile() is invoked with dynamic=True, so Inductor
        # builds one graph parameterized by symbolic (B, Npad) on the first flush
        # and reuses it for all later shapes.
        # Opt out with SHRIMP_NO_COMPILE=1; falls back to eager on any error.
        self._raw_fpv = self.model.forward_policy_value
        self._compiled_fpv = self._raw_fpv
        # cuda: compile by default. xpu: opt-in only (SHRIMP_COMPILE_XPU=1) —
        # inductor-on-xpu is young and a JIT stall on this backend presents as
        # exactly the mid-move hang the showcase failover exists to catch.
        self._use_compile = (
            self.device.type == "cuda"
            and os.environ.get("SHRIMP_NO_COMPILE") != "1"
        ) or (
            self.device.type == "xpu"
            and os.environ.get("SHRIMP_COMPILE_XPU") == "1"
        )
        # When set, defer the per-group decode/softmax/gather (which carry two
        # device syncs) from submit_payload to result(), so submit only enqueues
        # forwards. Default off (SHRIMP_DEFER_DECODE=1 to enable).
        self._defer_decode = os.environ.get("SHRIMP_DEFER_DECODE") == "1"
        # Serve autocast (see serve_autocast): fp16 default on cuda, OPT-IN on
        # xpu (the first fp16 kernel JIT wedged the A310's kernel driver and
        # took the whole LXC host down — see serve_autocast's docstring).
        self._autocast = serve_autocast(self.device)
        # Keep feats f16 through pack+H2D only where the forward will consume
        # them under fp16 autocast (f16 inputs into fp32 modules would dtype-
        # error). cuda: default. xpu: follows the autocast opt-in.
        # SHRIMP_F32_FEATS=1 forces the f32 path regardless.
        self._f16_feats = (
            (self.device.type == "cuda" or (self.device.type == "xpu" and self._autocast))
            and os.environ.get("SHRIMP_F32_FEATS") != "1"
        )
        # Pinned + non_blocking H2D: standard on cuda; on xpu OPT-IN
        # (SHRIMP_XPU_PIN=1) with a capability probe — pinned allocations on
        # this backend are another first-touch driver surface we don't
        # exercise by default after the host wedge.
        self._pin_h2d = self.device.type == "cuda"
        if self.device.type == "xpu" and os.environ.get("SHRIMP_XPU_PIN") == "1":
            try:
                torch.empty(1).pin_memory()
                self._pin_h2d = True
            except Exception:
                self._pin_h2d = False
        # Pure-fp16 serve (SHRIMP_SERVE_HALF=1, CUDA only): run the forward on
        # an fp16 COPY of the net with autocast DISABLED. Under autocast the
        # norm ops run fp32, which keeps the residual stream fp32 and doubles
        # every conv-gather/LayerNorm/pointwise byte; a half module keeps the
        # stream fp16 end-to-end (LayerNorm still accumulates its statistics in
        # fp32 internally). The copy leaves the caller's fp32 master weights
        # untouched (self-play/eval construct a fresh evaluator per epoch, so
        # the copy tracks the trained weights). Requires the f16 feats path.
        self._serve_half = (
            self.device.type == "cuda"
            and self._f16_feats
            and os.environ.get("SHRIMP_SERVE_HALF") == "1"
        )
        if self._serve_half:
            import copy

            self.model = copy.deepcopy(self.model).half().eval()
            # Keep the scalar value/moves-left tops fp32 (their inputs are cast
            # up at the head boundary in forward_policy_value): the binned
            # value decode is the parity-sensitive output, and the fp16 top
            # alone pushes |dvalue| past the shipped 3e-3 serve tolerance.
            for mod in (
                self.model.value_reduction,
                self.model.value_head,
                self.model.ml_reduction,
                self.model.moves_left_head,
            ):
                mod.float()
            self._raw_fpv = self.model.forward_policy_value
            self._compiled_fpv = self._raw_fpv
        # Rust parallel serve-pack with zero-copy buffers (SHRIMP_RUST_PACK):
        # grouping + per-group padding + f16/int buffer assembly run in parallel
        # Rust; the resulting zero-copy buffers are consumed via torch.frombuffer
        # + .to(device), skipping the Python pack loop and astype.
        # Gated on _f16_feats: the Rust pack emits f16 feats only, so the f32
        # toggle forces a fall back to the CSR/Python pack.
        self._rust_pack = (
            self.device.type == "cuda"
            and self._f16_feats
            and os.environ.get("SHRIMP_RUST_PACK") == "1"
        )
        # Dedicated copy stream + pinned staging ring (SHRIMP_COPY_STREAM=1,
        # rust-pack path only): makes submit_payload a true async enqueue
        # instead of serializing with the GPU stream via pageable H2D. See
        # _PinnedRing.
        self._copy_stream = (
            torch.cuda.Stream()
            if self.device.type == "cuda"
            and self._rust_pack
            and os.environ.get("SHRIMP_COPY_STREAM") == "1"
            else None
        )
        self._pin_ring = _PinnedRing() if self._copy_stream is not None else None
        # CUDA-graph capture/replay per (B_bucket, Npad, ml) serve shape
        # (SHRIMP_CUDA_GRAPHS=1, CUDA only). Built lazily after the compile
        # setup below (needs the final fpv callable); see _GraphCache.
        self._use_graphs = (
            self.device.type == "cuda"
            and os.environ.get("SHRIMP_CUDA_GRAPHS") == "1"
        )
        self._graph_cache = None
        # cuda.Event GPU-busy instrument (bench-only). Inert (None) unless
        # SHRIMP_PERF_TRACE=1. See PerfTrace.
        self._perf = (
            PerfTrace() if os.environ.get("SHRIMP_PERF_TRACE") == "1" else None
        )
        if self._use_compile:
            # suppress_errors drops any shape that fails to compile to eager
            # rather than raising. automatic_dynamic stays on. cache_size_limit
            # covers the specializations (request_moves_left True/False and the
            # batch-size-1 guard).
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.automatic_dynamic_shapes = True
            torch._dynamo.config.cache_size_limit = max(
                64, torch._dynamo.config.cache_size_limit
            )
            try:
                self._compiled_fpv = torch.compile(self._raw_fpv, dynamic=True)
            except Exception:
                self._compiled_fpv = self._raw_fpv
        if self._use_graphs:
            self._graph_cache = _GraphCache(
                self._compiled_fpv,
                autocast_on=not self._serve_half,
                device=self.device,
            )

    def __call__(self, payload: dict) -> dict:
        return self.evaluate_payload(payload)

    @torch.no_grad()
    def evaluate_payload(self, payload: dict) -> dict:
        """Synchronous serve: enqueue the forward and immediately read it back
        (submit_payload followed by result)."""
        return self.result(self.submit_payload(payload))

    def perf_trace_report(self) -> dict | None:
        """GPU-busy report, or None if SHRIMP_PERF_TRACE is unset."""
        return self._perf.report() if self._perf is not None else None

    @torch.no_grad()
    def submit_payload(self, payload: dict) -> dict:
        """Phase 1 of the async serve split: parse the request and enqueue every
        forward group on the GPU without synchronizing. Decoded outputs stay
        on-device; no .cpu() runs here. Returns an opaque handle to pass to
        result(), which performs the device->host sync."""
        if int(payload["abi"]) != 1:
            raise ValueError(f"unsupported shrimp ABI {payload['abi']}")
        b, total_nodes = (int(x) for x in payload["shape"])
        offsets = np.asarray(payload["node_row_offsets"], dtype=np.int64)
        if offsets.shape[0] != b + 1 or int(offsets[-1]) != total_nodes:
            raise ValueError("node_row_offsets inconsistent with shape")
        legal_counts = np.frombuffer(payload["legal_counts"], dtype=np.int32)
        if legal_counts.shape[0] != b:
            raise ValueError("legal_counts byte count mismatch")
        request_ml = bool(payload.get("request_moves_left", False))
        # When set, emit raw pre-softmax policy logits (priors_logits_bytes)
        # alongside priors_bytes. Off by default (no extra reply column).
        request_logits = bool(payload.get("request_logits", False))

        if self._rust_pack:
            # Rust parallel serve-pack: grouping + per-group padding + f16/int
            # buffer assembly happen in parallel Rust; the zero-copy buffers are
            # consumed via torch.frombuffer + .to(device).
            return self._submit_rust_pack(
                payload, b, offsets, legal_counts, request_ml, request_logits
            )

        feats16 = np.frombuffer(payload["node_feats"], dtype=np.float16)
        if feats16.shape[0] != total_nodes * NUM_FEATURES:
            raise ValueError("node_feats byte count mismatch")
        # The wire feats are f16 and the serve forward runs f16 under autocast.
        # On CUDA keep them f16 (pack/H2D below build f16, no astype); use f32
        # only on the CPU path or when the F32_FEATS toggle is set.
        feats = (
            feats16.reshape(total_nodes, NUM_FEATURES)
            if self._f16_feats
            else feats16.astype(np.float32).reshape(total_nodes, NUM_FEATURES)
        )
        qr = np.frombuffer(payload["node_qr"], dtype=np.int16).reshape(total_nodes, 2)
        nbr = np.frombuffer(payload["nbr"], dtype=np.uint16).reshape(total_nodes, 6)

        sizes = (offsets[1:] - offsets[:-1]).astype(np.int64)
        # Every group appends GPU tensors to these buffers; the single .cpu()
        # sync happens later, in result(). gpu_priors holds one flat tensor per
        # group (the group's rows' legal-prefix priors concatenated row-major);
        # plan_groups emits groups in ascending row order, so concatenating them
        # yields the full row-order flat-priors layout.
        gpu_priors: list[torch.Tensor] = []
        gpu_values: list[torch.Tensor] = []
        gpu_ml: list[torch.Tensor] = []
        gpu_logits: list[torch.Tensor] = []
        # Defer mode: collect raw per-group outputs; decode them in result()
        # instead of here (see _run_forward).
        deferred: list | None = [] if self._defer_decode else None

        # Bracket the forward enqueues with cuda.Events on the forward stream.
        # Read in result() after the D2H sync.
        perf_events = self._perf.make_events() if self._perf is not None else None
        if perf_events is not None:
            self._perf.on_submit()
            perf_events[0].record()

        # Padding-aware grouping (rows arrive size-descending); see plan_groups.
        for start, end, pad_to in plan_groups(sizes):
            self._forward_group(
                feats, qr, nbr, offsets, sizes, legal_counts, start, end, pad_to,
                request_ml, request_logits, gpu_values, gpu_ml, gpu_priors,
                gpu_logits, deferred,
            )

        if perf_events is not None:
            perf_events[1].record()

        if self._defer_decode:
            # Raw forwards enqueued; the syncing decode happens in result().
            return {
                "b": b,
                "request_ml": request_ml,
                "request_logits": request_logits,
                "legal_counts": legal_counts,
                "deferred": deferred,
                "perf_events": perf_events,
            }
        # Concatenate on-GPU (no D2H); the syncs happen in result().
        return {
            "b": b,
            "request_ml": request_ml,
            "request_logits": request_logits,
            "legal_counts": legal_counts,
            "values_gpu": torch.cat(gpu_values),
            "ml_gpu": torch.cat(gpu_ml) if request_ml else None,
            "priors_gpu": torch.cat(gpu_priors),
            "logits_gpu": torch.cat(gpu_logits) if request_logits else None,
            "perf_events": perf_events,
        }

    @torch.no_grad()
    def _submit_rust_pack(
        self, payload, b, offsets, legal_counts, request_ml, request_logits
    ) -> dict:
        """Rust parallel serve-pack consumption (SHRIMP_RUST_PACK).

        Hands the CSR-flat wire bytes (f16 feats, i16 coords, u16 nbr) and the
        i64 row offsets to `_rust.build_serve_groups`, which runs the same
        plan_groups planner and assembles every group's padded buffers in
        parallel: feats (f16, pad=0), nbr (i32, fill=pad_to, sentinel->pad_to),
        mask (u8, 1 at real nodes), coords (i32, pad=0). Each group's four
        buffers come back as read-only zero-copy buffers; torch.frombuffer views
        them in place and .to(device) copies to the GPU. The int32 nbr/coords are
        cast to int64 on-device (the model's gather needs int64). The forward
        tail is the shared _run_forward."""
        from shrimp import _rust  # local import: only the rust-pack path needs it

        dev = self.device
        groups = _rust.build_serve_groups(
            payload["node_feats"],
            payload["node_qr"],
            payload["nbr"],
            offsets.tolist(),
        )

        gpu_priors: list[torch.Tensor] = []
        gpu_values: list[torch.Tensor] = []
        gpu_ml: list[torch.Tensor] = []
        gpu_logits: list[torch.Tensor] = []
        deferred: list | None = [] if self._defer_decode else None

        # Bracket the forward enqueues (see submit_payload).
        perf_events = self._perf.make_events() if self._perf is not None else None
        if perf_events is not None:
            self._perf.on_submit()
            perf_events[0].record()

        for grp in groups:
            start = grp["start"]
            end = grp["end"]
            gn = grp["g"]
            p = grp["pad_to"]
            if self._graph_cache is not None and self._copy_stream is not None:
                # Fused graphs+copy-stream path: H2D straight from the pinned
                # staging into the graph's static input buffers (feats direct;
                # nbr/coords via an int32 staging tensor cast by copy_), then
                # one replay. No intermediate device tensors, no D2D.
                entry = self._graph_cache.entry_for(gn, p, request_ml)
                if entry is not None:
                    slot = self._pin_ring.acquire()
                    h_feats = _PinnedRing.stage(slot, "feats", grp["feats"])
                    h_nbr = _PinnedRing.stage(slot, "nbr", grp["nbr"])
                    h_mask = _PinnedRing.stage(slot, "mask", grp["mask"])
                    h_coords = _PinnedRing.stage(slot, "coords", grp["coords"])
                    si = entry["in"]
                    cur = torch.cuda.current_stream()
                    with torch.cuda.stream(self._copy_stream):
                        # Statics are owned by the compute stream; the copy
                        # stream must not write them before THIS key's previous
                        # replay (and its output clones) are done.
                        self._copy_stream.wait_event(entry["use_evt"])
                        si["feats"][:gn].copy_(
                            h_feats.view(torch.float16).reshape(gn, p, NUM_FEATURES),
                            non_blocking=True,
                        )
                        d_nbr32 = (
                            h_nbr.view(torch.int32)
                            .reshape(gn, p, 6)
                            .to(dev, non_blocking=True)
                        )
                        si["nbr"][:gn].copy_(d_nbr32, non_blocking=True)
                        si["mask"][:gn].copy_(
                            h_mask.reshape(gn, p), non_blocking=True
                        )
                        d_coords32 = (
                            h_coords.view(torch.int32)
                            .reshape(gn, p, 2)
                            .to(dev, non_blocking=True)
                        )
                        si["coords"][:gn].copy_(d_coords32, non_blocking=True)
                        evt = torch.cuda.Event()
                        evt.record(self._copy_stream)
                    slot["event"] = evt
                    cur.wait_event(evt)
                    for t in (d_nbr32, d_coords32):
                        t.record_stream(cur)
                    _GraphCache.reset_pad(entry, gn, p)
                    out = _GraphCache.replay(entry, gn)
                    if deferred is not None:
                        deferred.append((out, start, end))
                    else:
                        value, ml, priors_flat, logits_flat = self._decode_group(
                            out, legal_counts, start, end, request_ml, request_logits
                        )
                        gpu_values.append(value)
                        if request_ml:
                            gpu_ml.append(ml)
                        gpu_priors.append(priors_flat)
                        if request_logits:
                            gpu_logits.append(logits_flat)
                    continue
            if self._copy_stream is not None:
                # Copy-stream path: stage through pinned buffers, issue the H2D
                # on the copy stream, and make the compute stream wait on a
                # per-group event. submit no longer serializes with the GPU.
                slot = self._pin_ring.acquire()
                h_feats = _PinnedRing.stage(slot, "feats", grp["feats"])
                h_nbr = _PinnedRing.stage(slot, "nbr", grp["nbr"])
                h_mask = _PinnedRing.stage(slot, "mask", grp["mask"])
                h_coords = _PinnedRing.stage(slot, "coords", grp["coords"])
                with torch.cuda.stream(self._copy_stream):
                    d_feats = (
                        h_feats.view(torch.float16)
                        .reshape(gn, p, NUM_FEATURES)
                        .to(dev, non_blocking=True)
                    )
                    d_nbr32 = (
                        h_nbr.view(torch.int32)
                        .reshape(gn, p, 6)
                        .to(dev, non_blocking=True)
                    )
                    d_mask8 = (
                        h_mask.reshape(gn, p).to(dev, non_blocking=True)
                    )
                    d_coords32 = (
                        h_coords.view(torch.int32)
                        .reshape(gn, p, 2)
                        .to(dev, non_blocking=True)
                    )
                    evt = torch.cuda.Event()
                    evt.record(self._copy_stream)
                slot["event"] = evt
                cur = torch.cuda.current_stream()
                cur.wait_event(evt)
                # The device tensors were ALLOCATED on the copy stream; record
                # their use on the compute stream so the caching allocator does
                # not hand their blocks back to the copy stream while the
                # forward still reads them (the classic cross-stream UAF).
                for t in (d_feats, d_nbr32, d_mask8, d_coords32):
                    t.record_stream(cur)
                # int64/bool casts run on the compute stream (post-wait).
                d_nbr = d_nbr32.to(torch.int64)
                d_mask = d_mask8.to(torch.bool)
                d_coords = d_coords32.to(torch.int64)
            else:
                # frombuffer views the zero-copy Rust buffer; .to(dev) copies it
                # to the GPU (pageable source -> synchronous H2D).
                d_feats = (
                    torch.frombuffer(grp["feats"], dtype=torch.float16)
                    .reshape(gn, p, NUM_FEATURES)
                    .to(dev, non_blocking=True)
                )
                d_nbr = (
                    torch.frombuffer(grp["nbr"], dtype=torch.int32)
                    .reshape(gn, p, 6)
                    .to(dev, non_blocking=True)
                    .to(torch.int64)
                )
                d_mask = (
                    torch.frombuffer(grp["mask"], dtype=torch.uint8)
                    .reshape(gn, p)
                    .to(dev, non_blocking=True)
                    .to(torch.bool)
                )
                d_coords = (
                    torch.frombuffer(grp["coords"], dtype=torch.int32)
                    .reshape(gn, p, 2)
                    .to(dev, non_blocking=True)
                    .to(torch.int64)
                )
            self._run_forward(
                d_feats, d_nbr, d_mask, d_coords, gn, request_ml, request_logits,
                legal_counts, start, end, gpu_values, gpu_ml, gpu_priors,
                gpu_logits, deferred,
            )

        if perf_events is not None:
            perf_events[1].record()

        if self._defer_decode:
            return {
                "b": b,
                "request_ml": request_ml,
                "request_logits": request_logits,
                "legal_counts": legal_counts,
                "deferred": deferred,
                "perf_events": perf_events,
            }
        return {
            "b": b,
            "request_ml": request_ml,
            "request_logits": request_logits,
            "legal_counts": legal_counts,
            "values_gpu": torch.cat(gpu_values),
            "ml_gpu": torch.cat(gpu_ml) if request_ml else None,
            "priors_gpu": torch.cat(gpu_priors),
            "logits_gpu": torch.cat(gpu_logits) if request_logits else None,
            "perf_events": perf_events,
        }

    @torch.no_grad()
    def result(self, handle: dict) -> dict:
        """Phase 2: drain a submit_payload() handle. The .cpu() calls here are the
        single device->host sync for the whole flush."""
        b = handle["b"]
        request_ml = handle["request_ml"]
        request_logits = handle.get("request_logits", False)
        legal_counts = handle["legal_counts"]

        # Defer mode: the per-group decode/softmax/gather was held out of submit.
        # Do it now (before the D2H below), then fall through to the concat+.cpu()
        # path.
        if "deferred" in handle:
            gpu_values, gpu_ml, gpu_priors, gpu_logits = [], [], [], []
            for out, start, end in handle["deferred"]:
                value, ml, priors_flat, logits_flat = self._decode_group(
                    out, legal_counts, start, end, request_ml, request_logits
                )
                gpu_values.append(value)
                if request_ml:
                    gpu_ml.append(ml)
                gpu_priors.append(priors_flat)
                if request_logits:
                    gpu_logits.append(logits_flat)
            handle["values_gpu"] = torch.cat(gpu_values)
            handle["priors_gpu"] = torch.cat(gpu_priors)
            handle["ml_gpu"] = torch.cat(gpu_ml) if request_ml else None
            handle["logits_gpu"] = torch.cat(gpu_logits) if request_logits else None

        values_out = handle["values_gpu"].cpu().numpy().astype(np.float32, copy=False)
        # priors_gpu is torch.cat over the per-row legal-prefix priors in row
        # order, so flat_priors is the sum(legal_counts) positional layout the
        # Rust parser walks. Emitted as one contiguous f32 buffer; the row split
        # happens Rust-side from legal_counts.
        flat_priors = np.ascontiguousarray(
            handle["priors_gpu"].cpu().numpy(), dtype=np.float32
        )

        reply = {
            "values_bytes": values_out.tobytes(),
            "priors_bytes": flat_priors.tobytes(),
        }
        if request_ml:
            reply["moves_left_bytes"] = (
                handle["ml_gpu"].cpu().numpy().astype(np.float32, copy=False).tobytes()
            )
        if request_logits:
            # Raw pre-softmax logits, same positional layout as priors_bytes
            # (per-row legal prefix, row order).
            flat_logits = np.ascontiguousarray(
                handle["logits_gpu"].cpu().numpy(), dtype=np.float32
            )
            reply["priors_logits_bytes"] = flat_logits.tobytes()
        # The .cpu() syncs above guarantee this flush's forward events are
        # complete; read their elapsed device time now. Bench-only.
        if self._perf is not None:
            self._perf.on_result(handle.get("perf_events"), int(b))
        return reply

    def _forward_group(
        self, feats, qr, nbr, offsets, sizes, legal_counts, start, end, pad_to,
        request_ml, request_logits, gpu_values, gpu_ml, gpu_priors, gpu_logits,
        deferred=None,
    ) -> None:
        g = end - start
        # Host pack: build the padded (g, pad_to, *) numpy buffers one pass per
        # field, then one from_numpy + .to(device) per field. feats f16 on accelerators
        # (CPU path stays f32; numpy upcasts the f16 source on assignment there),
        # nbr sentinel remapped to pad_to, int64 coords, bool mask.
        feat_dtype = np.float16 if self._f16_feats else np.float32
        np_feats = np.zeros((g, pad_to, NUM_FEATURES), dtype=feat_dtype)
        np_nbr = np.full((g, pad_to, 6), pad_to, dtype=np.int64)
        np_mask = np.zeros((g, pad_to), dtype=np.bool_)
        np_coords = np.zeros((g, pad_to, 2), dtype=np.int64)
        for k in range(g):
            row = start + k
            n = int(sizes[row])
            o = int(offsets[row])
            np_feats[k, :n] = feats[o : o + n]
            row_nbr = nbr[o : o + n].astype(np.int64)
            np_nbr[k, :n] = np.where(row_nbr == NBR_SENTINEL, pad_to, row_nbr)
            np_mask[k, :n] = True
            np_coords[k, :n] = qr[o : o + n].astype(np.int64)
        batch_feats = torch.from_numpy(np_feats)
        batch_nbr = torch.from_numpy(np_nbr)
        batch_mask = torch.from_numpy(np_mask)
        batch_coords = torch.from_numpy(np_coords)

        device = self.device
        # Pinned + non_blocking H2D where the runtime supports it (cuda always;
        # xpu probed at init): page-lock each host buffer so the driver can DMA
        # it asynchronously and overlap the copies with queued GPU work. The
        # consuming forward runs on the same stream, so ordering holds. CPU (and
        # an xpu runtime without pinning) uses the plain blocking copy.

        def _h2d(t):
            return t.pin_memory().to(device, non_blocking=True) if self._pin_h2d else t.to(device)

        d_feats = _h2d(batch_feats)
        d_nbr = _h2d(batch_nbr)
        d_mask = _h2d(batch_mask)
        d_coords = _h2d(batch_coords)
        self._run_forward(
            d_feats, d_nbr, d_mask, d_coords, g, request_ml, request_logits,
            legal_counts, start, end, gpu_values, gpu_ml, gpu_priors, gpu_logits,
            deferred,
        )

    def _run_forward(
        self, d_feats, d_nbr, d_mask, d_coords, g, request_ml, request_logits,
        legal_counts, start, end, gpu_values, gpu_ml, gpu_priors, gpu_logits,
        deferred,
    ) -> None:
        """Shared forward tail for both the CSR (_forward_group) and Rust-pack
        (_submit_rust_pack) packers. Takes the four device tensors already in
        their final dtypes (feats f16/f32, nbr int64, mask bool, coords int64)
        and runs the compiled/eager forward: the batch-1 -> batch-2 duplication,
        mark_dynamic, autocast, and the defer-or-decode path. The two packers
        differ only in how these four device tensors are produced."""
        device = self.device
        # One dynamic graph for every shape (see __init__): mark both varying
        # dims dynamic — batch (dim 0) and cell-count Npad (dim 1).
        #
        # dynamo specializes a concrete batch of 1, leaving Npad the sole free
        # symbol, which trips Inductor's CantSplit on the attention head-merge
        # reshape. With batch >= 2 the graph compiles for every Npad, so a size-1
        # group is duplicated to batch 2 (each row is computed independently, so
        # row 0's outputs are unchanged) and the twin is sliced off after.
        # CUDA-graph fast path: one replay per group (see _GraphCache). Falls
        # through to the regular compiled path on capture failure / oversize.
        if self._graph_cache is not None:
            out = self._graph_cache.run_group(
                d_feats, d_nbr, d_mask, d_coords, g, request_ml
            )
            if out is not None:
                if deferred is not None:
                    deferred.append((out, start, end))
                    return
                value, ml, priors_flat, logits_flat = self._decode_group(
                    out, legal_counts, start, end, request_ml, request_logits
                )
                gpu_values.append(value)
                if request_ml:
                    gpu_ml.append(ml)
                gpu_priors.append(priors_flat)
                if request_logits:
                    gpu_logits.append(logits_flat)
                return
        use_compiled = self._use_compile and self._compiled_fpv is not self._raw_fpv
        fpv = self._compiled_fpv if use_compiled else self._raw_fpv
        pad_batch = use_compiled and g == 1
        if pad_batch:
            d_feats = d_feats.repeat(2, 1, 1)
            d_nbr = d_nbr.repeat(2, 1, 1)
            d_mask = d_mask.repeat(2, 1)
            d_coords = d_coords.repeat(2, 1, 1)
        if use_compiled:
            for t in (d_feats, d_nbr, d_mask, d_coords):
                torch._dynamo.mark_dynamic(t, 0)  # batch (>= 2 here) dynamic
                torch._dynamo.mark_dynamic(t, 1)  # Npad dynamic
        # serve-half runs the fp16 module natively: autocast must stay OFF, else
        # its fp32 norm policy re-upcasts the residual stream (see __init__).
        autocast_on = self._autocast and not self._serve_half
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_on):
            out = fpv(
                d_feats,
                d_nbr,
                d_mask,
                d_coords,
                request_moves_left=request_ml,
            )
        if pad_batch:  # drop the duplicated twin row -> back to the true g == 1
            out = {k: v[:g] for k, v in out.items()}
        # Defer mode (SHRIMP_DEFER_DECODE): stash the raw forward outputs and
        # run the per-group decode/softmax/gather later, in result(). The decode
        # carries two device syncs (the group_counts H2D and the priors[legal]
        # boolean gather's nonzero); running it here would make submit_payload
        # block on each group's forward. Deferring lets submit only enqueue the
        # forwards.
        if deferred is not None:
            deferred.append((out, start, end))
            return
        value, ml, priors_flat, logits_flat = self._decode_group(
            out, legal_counts, start, end, request_ml, request_logits
        )
        gpu_values.append(value)
        if request_ml:
            gpu_ml.append(ml)
        gpu_priors.append(priors_flat)
        if request_logits:
            gpu_logits.append(logits_flat)

    def _decode_group(self, out, legal_counts, start, end, request_ml, request_logits=False):
        """Per-group serve decode: binned value, moves-left, and the flattened
        legal-prefix prior gather. Sync-free: the legal mask comes from a
        host-side WRITABLE int64 copy of the counts (one small H2D enqueue per
        group; a copy of the read-only frombuffer view, which torch.from_numpy
        cannot take without the undefined-behavior warning), and the flat
        legal-prefix gather is a host-built index_select — unlike the boolean
        `priors[legal]` advanced index, whose nonzero() forces a device->host
        sync per group. Decoded value/ml are (g,) GPU tensors; the flat gather
        takes each row's first legal_counts[row] entries in row order (l==0
        rows contribute nothing)."""
        value = decode_binned_value(out["value"].float())
        ml = decode_moves_left(out["moves_left"].float()) if request_ml else None
        logits = out["policy"].float()
        npad = logits.shape[1]
        counts_host = np.array(legal_counts[start:end], dtype=np.int64)  # writable copy
        group_counts = torch.from_numpy(counts_host).to(
            logits.device, dtype=torch.long, non_blocking=True
        )
        col_idx = torch.arange(npad, device=logits.device)
        # Set columns at index >= the row's legal count to -inf before one
        # batched softmax. The model mask-zeroes (not -inf) those logits, so a
        # bare slice softmax would let the zeros enter the denominator. The -inf
        # columns contribute exp(-inf)=0, so each [:l] slice equals
        # torch.softmax(logits[k, :l]).
        legal = col_idx.unsqueeze(0) < group_counts.unsqueeze(1)  # (g, Npad)
        masked = logits.masked_fill(~legal, float("-inf"))
        priors = torch.softmax(masked, dim=1)  # fp32, GPU; rows with l==0 -> NaN
        # Row-major flat indices of every legal slot, built host-side from the
        # same counts (no device nonzero): row k contributes k*Npad + [0, l_k).
        flat_idx = torch.from_numpy(
            np.concatenate(
                [np.arange(c, dtype=np.int64) + k * npad for k, c in enumerate(counts_host)]
            )
            if counts_host.size
            else np.empty(0, dtype=np.int64)
        ).to(logits.device, non_blocking=True)
        priors_flat = priors.reshape(-1).index_select(0, flat_idx)
        # When requested, gather the raw (pre-softmax, un-masked) logits over
        # the same indices, so the flat layout is positionally identical to
        # priors_bytes. Skipped otherwise.
        logits_flat = (
            logits.reshape(-1).index_select(0, flat_idx) if request_logits else None
        )
        return value, ml, priors_flat, logits_flat
