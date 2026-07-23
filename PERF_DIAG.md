# hexfield_eq main_5 ep35 serve diagnosis

## Verdict

The 512-simulation latency is not explained by TSS alone. The existing public
measurements are consistent with wide-board network evaluation being the
dominant term:

- The reported two-stone turn is about 34 s, or about 17 s per 512-simulation
  search.
- A single wide 256-simulation search is 5–9 s. Near-linear visit scaling
  predicts 10–18 s at 512, which already spans the observed 17 s/search without
  requiring a park timeout.
- Latency rises with board spread. That is the signature of the five
  full-attention blocks' O(B·S²) work, not of a fixed 5 s timer.

TSS can still be a material tail-latency contributor. The current evidence does
not numerically separate it from evaluation, so the benchmark added in this
change reads the Rust searcher's existing evaluator and TSS telemetry and
performs causal TSS/park/all-leaves ablations. Those ablations intentionally
change search behavior and are diagnosis-only.

## What the XPU worker actually runs

ep35 has a `CCACCACCACCACCA` trunk: ten convolution blocks and five global
attention blocks. On XPU the evaluator is fp32 with no autocast. The reference
attention path:

1. builds a shared pair-index tensor;
2. materializes a per-block `(B, heads, S, S)` fp16 relative-position bias;
3. converts that bias to fp32 to match fp32 Q/K/V;
4. calls scaled-dot-product attention.

The attention arithmetic and bias traffic therefore grow quadratically in
support size. Far-flung stones increase the union of legal cells and halo, so a
wide board can cost much more than a compact board with the same ply and visit
count. `eval_virtual_batch_size=32` caps the requested leaf batch, but it does
not cap `S`; at 512 visits there are at least sixteen ideal full batches and
usually more chunks because a single-root tree cannot always expose 32 unique
ready leaves.

The benchmark reports `unique_states / evaluator_chunks` as `avgB`. A low
`avgB` means launch/packing overhead and poor XPU occupancy. A high `avgB` on a
wide board increases `B·S²` and VRAM pressure. Testing virtual batches 16/32/64
is therefore useful, but changing this value changes virtual-loss/search
scheduling and is not in the behavior-preserving core.

## The prime_serve_env clue

`prime_serve_env()` sets five import-time flags:

- `HEXFIELD_SERVE_FLEX`
- `HEXFIELD_FLEX_PAIR`
- `HEXFIELD_TRITON_CONV`
- `HEXFIELD_TRITON_ATTN`
- `HEXFIELD_TRITON_CONV_LN`

`model.py` reads them into module globals exactly once. Calling the function
after `hexfield_eq.model` was imported cannot change constructed nets or branch
selection.

The showcase sequence was:

1. `bots.py` imports the lightweight family registry.
2. `_WorkerRuntime` calls `HexfieldEqFamily.prepare_process`, which correctly
   installs checkpoint architecture globals.
3. `family.load_net` calls `_load_hexfield_net`, importing
   `hexfield_eq.model` with kernel flags still off.
4. Only later, `family.build_evaluator` calls `build_serve_evaluator`.
   `apply_serve_env_profile()` can still enable evaluator-time flags, but it is
   too late for model import-time flags.

This change adds a family hook between steps 2 and 3.

### CUDA versus XPU viability

The three repository custom kernels are not XPU kernels:

- fused conv requires `x.is_cuda`;
- fused conv+LayerNorm requires `x.is_cuda`;
- fused attention requires `q.is_cuda` and fp16.

The evaluator's half-module, compile, CUDA graph, pinned copy-stream, and
original Rust-pack gates were also CUDA-specific. Blindly calling the old
all-on `prime_serve_env()` on XPU therefore does not unlock those kernels.

Recent PyTorch FlexAttention accepts XPU tensors, so the two flex gates are
technically plausible on torch 2.12.1+xpu. That does not prove this model's
custom score modifier compiles on the installed Intel backend, runs within 4 GB
on an A310, is faster than XPU SDPA, or preserves the search policy. The worker
therefore:

- primes the full profile automatically on CUDA;
- leaves CUDA-only gates off on XPU;
- primes only Flex + FlexPair on XPU when `HEXFIELD_XPU_FLEX=1`.

The benchmark's `--xpu-flex on` option is the authoritative A310 viability,
speed, and memory probe. Flex remains experimental and is not part of the
default optimization.

## TSS: what the 5 s value means

`tss_solver_park_timeout_ms=5000` is not an unconditional five-second sleep per
leaf. A selected leaf is parked after an accepted async solve enqueue. Polling
can then:

- consume a verified hard result;
- release an unknown result to normal net evaluation; or
- bail to net evaluation after the 5 s upper bound.

Six solver workers overlap one another and XPU evaluation. Consequently
`park_wait_ms_sum` is overlapping per-leaf time and may exceed wall time; it
must not be subtracted from wall time. `park_wait_ms_max` shows whether the cap
was approached. The causal park cost is the same-position `live` versus
`park-off` wall delta, supported by `deep_calls`, `deep_nodes`, and park
counters.

`tss_solver_all_leaves=true` increases the leaves eligible for deep solving,
while the 500-node cap bounds each attempt. Solver CPU work can contend with
the Python/Rust packer and search thread on a seven-CPU container. It can be
nearly free on tactically quiet positions (quick unknowns that overlap XPU), or
dominate a tail when parked leaves repeatedly approach timeout. The A310 table
will distinguish those cases.

One constraint in the task is impossible literally under the existing live
profile: `tss_async.rs` explicitly documents that the visit which first sees a
proof is wall-clock dependent, so fixed-seed async-TSS searches are not
bit-reproducible even before this change. The parity harness therefore uses:

- exact evaluator reply bytes as the arithmetic/wire invariant;
- a fixed-seed TSS-off search as the deterministic action and visit-policy
  invariant;
- an optional baseline-repeat/live-TSS comparison to expose existing scheduler
  variance rather than mislabel it as numeric drift.

## Numeric attribution produced by the harness

For every row the harness prints:

- `wall_ms`: complete `session.search`;
- `eval_ms`: Rust-measured evaluator time, including pack, H2D, fp32 forward,
  decode, and D2H;
- `enc_ms` / `parse`: Rust feature encoding and evaluator-reply parsing;
- `other_ms = wall - eval - encode - parse`;
- TSS park sum/max, deep calls/nodes, evaluator chunks, and average batch.

On `tss-off`, `other_ms` is the measured MCTS/tree/control remainder. On
`live`, it also includes critical-path TSS coordination. The attribution to
record for each 512 row is:

| Component | Measurement |
|---|---|
| Network/evaluator | `eval_ms` |
| Deep TSS + park | `live.wall_ms - tss-off.wall_ms`, checked against park/deep counters |
| Park specifically | `live.wall_ms - park-off.wall_ms`; `park_sum/max` are supporting telemetry |
| MCTS/tree/control | `tss-off.other_ms` |
| Encoding + reply parse | `enc_ms + parse` |
| Batching efficiency | `avgB`, plus the 16/32/64 wall/eval comparison |

The live and ablated searches can explore different trees, so deltas are causal
configuration comparisons, not perfectly additive accounting. The direct
timers are the additive accounting.

## Behavior-preserving optimization

The default XPU path now has four independently toggleable changes:

| Change | Toggle (0 disables) | Hypothesis |
|---|---|---|
| Rust padded batch assembly and compact i32/f16 H2D, with lossless XPU fp32 widening | `HEXFIELD_RUST_PACK` | remove Python per-row copies and host fp16→fp32 expansion |
| Decode after all group forwards are queued | `HEXFIELD_DEFER_DECODE` | avoid serializing XPU forward groups on pageable count transfers |
| Copy the exact full fp32 softmax/logit rows and slice legal prefixes on CPU | `HEXFIELD_HOST_LEGAL_GATHER` | remove a dynamic-shape device gather/synchronization per group |
| Cache shape-only `arange` tensors | `HEXFIELD_DECODE_CACHE` | remove one allocation/kernel launch per group |

They do not alter weights, features, padding, fp32 network operations, softmax,
or wire order. The host gather selects already-computed fp32 values; it does no
host arithmetic. Expected gain is workload-dependent: it should be larger on
compact/under-filled batches where host synchronization is a visible share,
and smaller on pathological wide boards where O(S²) attention dominates.
Deployment requires the A310 parity gate and a measured positive wall-time
result; no numeric speedup is claimed from this non-A310 development machine.

## Non-core levers

The following are deliberately not deployed as behavior-preserving changes:

- disabling park, all-leaves, TSS, or changing solver mode/node cap;
- changing `eval_virtual_batch_size`;
- changing TSS async/Rayon thread counts;
- enabling XPU FlexAttention.

They can change tree timing, visits, or floating-point reduction order. The
benchmark exposes them so a separate, explicitly behavioral rollout can make an
informed latency/strength trade.
