# Intel Arc A310 `main_5` ep35 serve-path rollout

Verified on 2026-07-24 against the 4 GB Intel Arc A310 in LXC 121. The
optimized path remains default-off and has not been enabled on the live site.
The measured rebuilt image was
`sha256:21b7ba47ac561349b6f0de20babc9643425ecfaa4f298a7d75e3aaaa2b608ade`.

## Final A310 gate

The hard parity gate ran in a one-off container with the live app stopped and
the selected profile set explicitly:

```bash
docker compose -f docker-compose.yml -f docker-compose.xpu.yml run \
  --rm --no-deps -T app \
  python scripts/parity_hexfield_eq_main5_serve.py \
  --checkpoint /models/hexfield_eq_main5_ep35_infer.pt \
  --config /app/configs/hexfield_eq_main_5.toml \
  --device xpu --visits 128 --virtual-batch-size 32 \
  --rust-pack off --defer-decode off \
  --host-legal-gather off --decode-cache off \
  --lean-bias on --attn-head-split on --ray-coeff-lut on \
  --torch-threads 7 --live-tss-check
```

Result: `HARD PARITY GATE: PASS`.

- Pair-index rewrite: exact.
- Compact and wide B=2 evaluator replies: byte-identical for
  `moves_left`, `priors`, `priors_logits`, and `values`; every byte delta was
  zero and the two feature rows were confirmed distinct.
- Compact deterministic search: action, node IDs, visit counts, Q values, and
  root value all identical (`max_q_delta=0`, `root_delta=0`).
- Wide deterministic search: action, node IDs, visit counts, Q values, and root
  value all identical (`max_q_delta=0`, `root_delta=0`).
- The advisory live-TSS compact repeat was stable. Wide live-TSS retained the
  same action, IDs, and visits with only the documented asynchronous scheduling
  noise (`max_q_delta=8.94e-08`, `root_delta<=5.96e-08`); this is outside the
  deterministic hard gate.

## Final benchmark

Each profile ran in three fresh processes, alternating baseline then optimized.
The table contains the median of three `live` searches at virtual batch size
32. `peak_MiB` is `torch.xpu.max_memory_allocated`, reset immediately before
each measured search after warmup.

| Board | Visits | Baseline wall_ms | Optimized wall_ms | Wall delta | Baseline eval_ms | Optimized eval_ms | Baseline peak_MiB | Optimized peak_MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Compact | 256 | 6401.4 | 3254.4 | -49.2% | 6335.3 | 3181.3 | 246.881 | 180.059 |
| Compact | 512 | 10004.7 | 4560.6 | -54.4% | 9931.3 | 4485.0 | 434.894 | 308.170 |
| Wide | 256 | 48546.2 | 19151.0 | -60.6% | 48480.3 | 19082.5 | 1618.737 | 995.363 |
| Wide | 512 | 96587.8 | 37209.7 | -61.5% | 96454.3 | 37067.5 | 1618.737 | 994.124 |

Optimized `eval_ms` improved by 49.8%, 54.8%, 60.6%, and 61.6% in
table order. Peak allocation fell by 27.1%, 29.1%, 38.5%, and 38.6%.
All optimized rows were below the 60 s move deadline; the slowest was the
37.210 s wide 512-visit case. Both profiles stayed below the 4 GB device limit,
and optimized peak allocation was lower in every comparison. The baseline
wide 512-visit fixture itself exceeded 60 s, at a 96.588 s median.

## Toggle decision

Median-of-three component ablations were also run in fresh processes:

| Profile | Compact 256 wall_ms | Compact 512 wall_ms | Wide 256 wall_ms | Wide 512 wall_ms | Wide 512 peak_MiB |
|---|---:|---:|---:|---:|---:|
| Baseline, all seven off | 6401.4 | 10004.7 | 48546.2 | 96587.8 | 1618.737 |
| Ray LUT only | 3062.1 | 4463.1 | 21266.9 | 41955.5 | 1623.938 |
| Lean bias + head split only | 6593.9 | 10274.4 | 46519.6 | 90092.0 | 1100.174 |
| Final combined profile | 3254.4 | 4560.6 | 19151.0 | 37209.7 | 994.124 |

Keep:

- `HEXFIELD_XPU_RAY_COEFF_LUT=1`. It is independently byte-identical and is
  the dominant compute win on both board shapes.
- `HEXFIELD_XPU_LEAN_BIAS=1` and
  `HEXFIELD_XPU_ATTN_HEAD_SPLIT=1`, always together. The pair has a small
  compact-board cost in isolation, but improves wide latency, cuts wide peak
  allocation by roughly one third, and, when combined with the ray LUT,
  improves wide wall time by another 10-11% while reducing peak allocation
  from about 1.62 GiB to about 0.99 GiB.

Do not retain:

- `HEXFIELD_RUST_PACK`
- `HEXFIELD_DEFER_DECODE`
- `HEXFIELD_HOST_LEGAL_GATHER`
- `HEXFIELD_DECODE_CACHE`

The earlier four-toggle profile produced deterministic-tree divergence, and
the Rust-packed wide path exceeded the A310 memory budget. The final verified
profile is faster and memory-safe without any of these four, so all remain
off. `HEXFIELD_XPU_FLEX` and the CUDA-only Triton gates also remain off.

## Operator-approved go-live

Go-live is deliberately not part of this commit. On the already measured GPU
host, edit the site-local
`/opt/hexo-bot/apps/showcase/docker-compose.local.yml` so the app environment
contains exactly:

```yaml
services:
  app:
    environment:
      HEXFIELD_RUST_PACK: "0"
      HEXFIELD_DEFER_DECODE: "0"
      HEXFIELD_HOST_LEGAL_GATHER: "0"
      HEXFIELD_DECODE_CACHE: "0"
      HEXFIELD_XPU_LEAN_BIAS: "1"
      HEXFIELD_XPU_ATTN_HEAD_SPLIT: "1"
      HEXFIELD_XPU_RAY_COEFF_LUT: "1"
```

Then deploy the exact prebuilt and measured image; do not add `--build`:

```bash
cd /opt/hexo-bot/apps/showcase
docker compose \
  -f docker-compose.yml \
  -f docker-compose.xpu.yml \
  -f docker-compose.local.yml \
  up -d --no-deps app

docker compose \
  -f docker-compose.yml \
  -f docker-compose.xpu.yml \
  -f docker-compose.local.yml \
  logs --since=5m app \
  | grep -E 'hexfield_eq evaluator:|worker ready|SELF-CHECK FAILED|FALLING BACK'

curl -fsS https://hexo.blueshrimp.uk/healthz
```

The evaluator banner must show the original four flags false and
`lean_bias=True attn_head_split=True ray_coeff_lut=True`. Require no self-check
failure or fallback, a worker-ready line with 12 checkpoints, and
`{"ok":true,"checkpoints":12,...}` from `/healthz`.

## Rollback

Set all seven environment toggles above to `"0"` and recreate only the app:

```bash
cd /opt/hexo-bot/apps/showcase
docker compose \
  -f docker-compose.yml \
  -f docker-compose.xpu.yml \
  -f docker-compose.local.yml \
  up -d --no-deps --force-recreate app
```

Repeat the filtered log and health checks from the go-live procedure. This
keeps the rebuilt image but returns it to its default-off, numerically identical
path.

## End-of-window service state

After measurement, the untouched original container
`hexo-showcase-app-1` was restored with `docker start`. It is running on image
`sha256:6cded5384b69b408d7af7bf6605f4c9aa19402b721c5a8be12db6f9c656ac06a`,
Docker reports it healthy, and its in-container health check returned:

```json
{"ok":true,"checkpoints":12,"active_games":0}
```
