# llama.cpp — qwen4exp + MTP + GPU-resident LRU expert cache

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that adds native MTP
(multi-token prediction) speculative decoding for **Qwen3.8-Flash-Next** (Alibaba's
`qwen4exp` MoE architecture), plus an asynchronous, device-side GPU-resident LRU cache for
CPU-offloaded MoE expert weights.

Base: upstream commit `88ddbf0a1` (the commit that merged `qwen4exp` architecture support,
[PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)).

This repo is published as a squashed snapshot (one commit, no incremental history) rather
than the full commit-by-commit history against that base — `git log` here won't show
upstream's history or the intermediate steps that produced this fork. The code is the real,
built-and-measured artifact either way; only the trail of how it was written is missing.

## What's different from upstream

1. **MTP draft-head support** — native speculative decoding for `qwen4exp` (`nextn`/
   `hc_head` tensors, draft-head-only GGUF loading via `-md`).
2. **A GPU-resident LRU expert cache** (`--moe-expert-cache-experts`) — see below.

## The expert cache

`--n-cpu-moe N` keeps the first `N` MoE layers' expert weights in host RAM to fit large MoE
models in limited VRAM, computing those layers' `mul_mat_id` on the CPU every decode step.
Decode on those layers is bound by host RAM bandwidth: every token streams its routed
experts' weights out of system RAM.

This adds a fixed-size pool of frequently-used experts kept resident in VRAM in front of
those CPU-offloaded tensors:

```
--moe-expert-cache-experts N      # slots per cached layer (0 = disabled, default)
--moe-expert-cache-inserts N      # max uploads per cached layer per decode step (default: 2)
```

How it works:

- A cache miss this step just runs the **normal CPU path** for that expert — exactly what
  would happen with the cache off, so the cache only ever removes work, never adds a stall.
  Decode never blocks waiting on a cache miss.
- A background worker thread fills the cache **off the critical path**. The new mapping is
  only published (table swapped) once the upload actually completes, at a *later*
  `llama_moe_cache_step()` call — a running graph can never observe a torn slot.
- Uploads are throttled (`--moe-expert-cache-inserts` per layer per step) so a cold cache
  can't saturate the host↔GPU link.

## Correctness

Verified via greedy (temp=0) decoding on low-entropy prompts (e.g. "list the first 20 prime
numbers") — far more sensitive to a data bug than stochastic sampling or open-ended
creative/technical prompts — against the plain `--n-cpu-moe` (no cache) build:
**bit-identical** output, standalone and with MTP.

If you're validating this on your own box: run the same prompt at `temp=0` (greedy) with the
cache on and off and diff the output byte-for-byte. Greedy decoding surfaces a caching bug
immediately — any wrong expert weight changes the argmax token somewhere in the sequence,
whereas stochastic sampling can mask a bug behind sampling noise for a long time before it's
visible.

## Benchmarks

Dual-GPU box: AMD Radeon AI PRO R9700 32GB (gfx1201, `main-gpu`) + AMD Radeon RX 9070 16GB
(gfx1201, on a 4-lane PCIe 3.0 link), Qwen3.8-Flash-Next UD-Q4_K_XL, `--ctx-size 262144`,
`--tensor-split 87,13`, `--n-cpu-moe 40`. Figures are pooled averages across multiple trials
of three varied prompts each; run-to-run noise on this box is real (~2-2.5 t/s band) — treat
any single number as ± that, not exact.

| Config | Speed |
|---|---|
| Cache disabled | 10.9 t/s |
| Cache enabled | ~20.8 t/s |

That's **+57%** at matched settings otherwise. A few other things measured along the way,
all reflected in the flags below:

- `HSA_ENABLE_SDMA=1` (not `=0`) — re-enables ROCm's dedicated copy engines for cross-device
  traffic. Measured **+8.9%** on this dual-GPU box; the *opposite* direction on a single-GPU
  box (no cross-device traffic to unblock), so don't carry this over blindly.
- `--spec-type ngram-mod,draft-mtp` instead of just `draft-mtp` — a priority-ordered chain
  (cheap exact n-gram matches get first shot each step, falling back to MTP's learned draft).
  +2.5% measured on novel/non-repetitive content, much larger on repetitive or templated
  content, no measured downside.
- GPU performance level: leave it at **`auto`**, don't lock it to `high`. Counterintuitive,
  but measured **~10% slower** locked to `high` in a repeated A/B on this workload — this
  offload-heavy decode pattern has enough idle gaps between GPU bursts that a locked-high
  clock loses to the driver's own dynamic boosting.
- `--threads`: more isn't better past your physical core count. On this 6-core/12-thread
  CPU, 6 threads beat 4, 8, and 10; **12 threads (the full logical count) hung the server
  outright** rather than just running slower. Don't default to `nproc`.
- `--moe-expert-cache-experts 96` / `--n-cpu-moe 40` / `--moe-expert-cache-inserts 4` are
  this box's measured sweep optimum, not defaults to copy blindly — see "Tuning" below.

Actual speedup depends heavily on model, quant, hardware, and `--n-cpu-moe`/context-size
tuning. The numbers above are one box's measurements, not a general guarantee.

## Building

Standard llama.cpp build process — see [docs/build.md](docs/build.md) for the full matrix.
ROCm/HIP example (used for the numbers above):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1201 -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/lib/cmake" \
    -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=ON -S .
cmake --build build --config Release -j "$(nproc)"
```

Swap `-DAMDGPU_TARGETS=gfx1201` for your GPU's target. CUDA builds should work the same way
with `-DGGML_CUDA=ON` in place of `-DGGML_HIP=ON` — the cache's tensor-set/synchronize calls
go through the standard `ggml-backend` async API, nothing HIP-specific — but this hasn't
been tested on real NVIDIA hardware.

## Running

```bash
./build/bin/llama-server \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf \
  --spec-type ngram-mod,draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 --spec-draft-ngl 999 \
  --moe-expert-cache-experts 96 --moe-expert-cache-inserts 4 \
  --ctx-size 262144 --n-gpu-layers 999 --split-mode layer --tensor-split 87,13 --main-gpu 0 \
  --n-cpu-moe 40 --override-tensor "per_layer_token_embd.*=CPU" \
  --flash-attn on --threads 6 --jinja
```

`HSA_ENABLE_SDMA=1` should already be your ROCm default; only worth setting explicitly if
something else in your environment disables it.

## Tuning notes

Numbers you should re-derive on your own hardware, not copy — but the *shape* of these
findings likely generalizes:

- **`--moe-expert-cache-experts` (cache size) has a real ceiling, and it isn't gentle.** On
  this box, sweeping 80/96/112/128/144 at fixed `--n-cpu-moe 40`: 96 was the local optimum,
  128 OOM'd outright at startup, and 144 loaded but ran at roughly a third of the
  hit-rate-appropriate speed (VRAM too tight, thrashing) — going too big is worse than going
  too small.
- **VRAM on a "minority" tensor-split card fills up fast.** At `--tensor-split 87,13` and
  `--ctx-size 262144`, the 16GB card was already ~97.5% used with the cache's whole capacity
  landing on the other (main) GPU. Shifting the split to push more cache capacity onto the
  smaller card OOM'd immediately at every ratio tried (80/20, 75/25, 70/30) — there was no
  headroom to redistribute without cutting context size first.
- **`--moe-expert-cache-inserts`**: swept 1/2/4/8 at fixed cache size — 4 (the point where
  this repo's default already sits, if you pass no flag it's 2) was the clear local optimum
  on this box; 1 starves the cache, 8 adds enough background upload traffic to cost more
  than it buys.
- **`--spec-draft-p-min`**: swept 0.50/0.65/0.75/0.85 at fixed n-max=3 — 0.65-0.75 is a flat
  plateau, both ends of that range measured worse.

## License

MIT, same as upstream llama.cpp — see [LICENSE](LICENSE).

---

For everything else (supported models, general server/CLI docs, the broader llama.cpp
project), see [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
