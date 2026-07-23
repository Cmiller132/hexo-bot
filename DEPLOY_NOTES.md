# A310 deployment and verification

## Build

Build the XPU image from this commit. The image now includes both harnesses
under `/app/scripts`.

```bash
cd apps/showcase
docker compose -f docker-compose.yml -f docker-compose.xpu.yml build app
```

Do not set `HEXFIELD_XPU_FLEX` for the core rollout. The default remains
materialized fp32 attention; the custom Triton kernels are CUDA-only.

## Pre-deploy A310 gate

Run parity in a one-off container from the newly built image if practical, or
temporarily start the rebuilt app and run:

```bash
docker exec -w /app hexo-showcase-app-1 \
  python scripts/parity_hexfield_eq_main5_serve.py \
  --checkpoint /models/hexfield_eq_main5_ep35_infer.pt \
  --config /app/configs/hexfield_eq_main_5.toml \
  --device xpu --visits 128 --live-tss-check
```

Required result:

- `PASS evaluator reply bytes`, with every maximum delta `0`;
- `PASS compact deterministic search`;
- `PASS wide deterministic search`;
- final `HARD PARITY GATE: PASS`.

The live-TSS section is advisory. If `baseline_repeat=VARIED`, that confirms the
pre-existing async solver timing behavior documented in `tss_async.rs`. Do not
weaken the hard gate because of it.

Measure the current baseline and optimized core in separate fresh processes:

```bash
docker exec -w /app hexo-showcase-app-1 \
  python scripts/bench_hexfield_eq_main5_serve.py \
  --checkpoint /models/hexfield_eq_main5_ep35_infer.pt \
  --serve-path baseline --cases live --visits 64,128,256,512 \
  --batch-sizes 32

docker exec -w /app hexo-showcase-app-1 \
  python scripts/bench_hexfield_eq_main5_serve.py \
  --checkpoint /models/hexfield_eq_main5_ep35_infer.pt \
  --serve-path optimized --cases live --visits 64,128,256,512 \
  --batch-sizes 32
```

Expected banner difference:

```text
baseline:  rust_pack=False defer=False host_gather=False
optimized: rust_pack=True  defer=True  host_gather=True
```

Acceptance:

- identical `action` for the deterministic parity gate;
- optimized `wall_ms` and `eval_ms` no worse after warmup on both boards;
- preferably at least 10% lower `eval_ms` on compact 256/512. A smaller gain is
  plausible on wide boards because unchanged O(S²) fp32 attention dominates.
- no XPU OOM/device abort and no increase beyond the 60 s move deadline.

The benchmark warms once internally. If results are noisy, run each command
three times and compare medians; never compare a cold first process against a
warm second process.

## Attribute TSS and batching

These runs change search behavior and are diagnostics only:

```bash
docker exec -w /app hexo-showcase-app-1 \
  python scripts/bench_hexfield_eq_main5_serve.py \
  --checkpoint /models/hexfield_eq_main5_ep35_infer.pt \
  --serve-path optimized --cases live,tss-off,park-off,leaves-off \
  --visits 512 --batch-sizes 32

docker exec -w /app hexo-showcase-app-1 \
  python scripts/bench_hexfield_eq_main5_serve.py \
  --checkpoint /models/hexfield_eq_main5_ep35_infer.pt \
  --serve-path optimized --cases live --visits 512 \
  --batch-sizes 16,32,64
```

Copy the tables into `PERF_DIAG.md` or the deployment record. Interpret:

- net/evaluator = `eval_ms`;
- TSS total = live minus tss-off wall, with deep counters;
- park = live minus park-off wall, supported by park sum/max;
- MCTS/tree/control = tss-off `other_ms`;
- batching = wall/eval versus `avgB` across 16/32/64.

Do not deploy TSS or virtual-batch changes as part of this core patch.

## Optional FlexAttention experiment

PyTorch 2.12.1 exposes an XPU FlexAttention backend, but the A310 result is
unknown. Probe it only in a fresh process because model gates are import-time:

```bash
docker exec -w /app -e HEXFIELD_XPU_FLEX=1 hexo-showcase-app-1 \
  python scripts/bench_hexfield_eq_main5_serve.py \
  --checkpoint /models/hexfield_eq_main5_ep35_infer.pt \
  --serve-path optimized --xpu-flex on --cases live \
  --visits 64,128 --batch-sizes 32
```

Stop the experiment on compile failure, OOM, parity drift, or regression. Do
not add `HEXFIELD_XPU_FLEX=1` to compose merely because the probe launches.
Before any rollout it needs a materialized-versus-flex fixed-position parity
record and 512-simulation compact/wide timing. The custom Triton flags must
remain off on XPU.

## Deploy

After the hard parity gate and positive A310 timings:

```bash
docker compose -f docker-compose.yml -f docker-compose.xpu.yml up -d app
docker compose -f docker-compose.yml -f docker-compose.xpu.yml logs -f --tail=200 app
```

Expected worker log lines include:

```text
hexfield_eq XPU uses materialized fp32 attention; CUDA-only Triton gates remain off
hexfield_eq evaluator: device=xpu rust_pack=True defer_decode=True host_legal_gather=True decode_cache=True
showcase worker ready: device=xpu ... torch_threads=7
```

The old “call prime_serve_env” mismatch warning should be gone on XPU because
CUDA-only gates being off is intentional.

## Rollback

The core optimizations are independently disabled without changing the image:

```yaml
environment:
  HEXFIELD_RUST_PACK: "0"
  HEXFIELD_DEFER_DECODE: "0"
  HEXFIELD_HOST_LEGAL_GATHER: "0"
  HEXFIELD_DECODE_CACHE: "0"
```

Recreate the app after editing compose. Leave `HEXFIELD_XPU_FLEX` unset.

## Validation completed off-box

- Python syntax compilation for modified modules and both harnesses: pass.
- `git diff --check`: pass.
- `cargo test`: pass.
- `cargo test --features python`: 76 passed.
- `cargo build --release`: pass.

This PC had no torch environment, and its GPU was intentionally untouched.
Performance and full Python parity remain authoritative only on the A310
container via the commands above.
