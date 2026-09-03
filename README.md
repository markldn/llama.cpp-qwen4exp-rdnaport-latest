# llama.cpp — qwen4exp + MTP + async GPU-resident LRU expert cache

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that adds native MTP
(multi-token prediction) speculative decoding for **Qwen3.8-Flash-Next** (Alibaba's
`qwen4exp` MoE architecture), plus an **asynchronous, device-side GPU-resident LRU cache**
for CPU-offloaded MoE expert weights.

This is the second, async design. The original synchronous version lives at
[markldn/llama.cpp-qwen4exp-lru-cache](https://github.com/markldn/llama.cpp-qwen4exp-lru-cache) —
**read "Which one should I use" below before picking one**, the answer depends on your GPU setup
and isn't the same for everyone.

## Which one should I use?

| Your setup | Use |
|---|---|
| Single GPU | [the synchronous fork](https://github.com/markldn/llama.cpp-qwen4exp-lru-cache) — measured **21% faster** than this one in a same-settings A/B on this box (19.75 vs 16.30 t/s pooled) |
| Multiple GPUs (tensor-split) | This fork — measured **+6.6%** over the synchronous design at matched settings (19.25 vs 18.06 t/s pooled), and the full tuning pass in this repo's history took one dual-GPU box from ~17.5 t/s to ~20.8 t/s end to end |

Why the split: on a single GPU, a cache miss under the synchronous design pays a blocking
copy but then computes that expert on the (fast) GPU. This async design skips the blocking
copy but computes misses on the CPU instead. That trade only wins when the GPU side is under
real cross-device contention — true with a multi-GPU tensor-split, not true with one GPU
sitting idle waiting for its own copy. Both were measured directly on the same hardware, not
assumed.

Base: upstream commit `88ddbf0a1` (the commit that merged `qwen4exp` architecture support,
[PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)).

This repo is published as a squashed snapshot (one commit, no incremental history) rather
than the full commit-by-commit history against that base — `git log` here won't show
upstream's history or the intermediate steps that produced this fork. The code is the real,
built-and-measured artifact either way; only the trail of how it was written is missing.

## What's different from upstream

1. **MTP draft-head support** — native speculative decoding for `qwen4exp` (`nextn`/
   `hc_head` tensors, draft-head-only GGUF loading via `-md`).
2. **An async GPU-resident LRU expert cache** (`--moe-expert-cache-experts`) — see below.

## The async expert cache

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

### Why "async", and what came before

The first version of this cache (the sibling repo linked above) fetched a miss **in-graph**:
every decode step that touched an uncached expert blocked on a synchronous PCIe copy before
it could continue. Profiling with `rocprofv3 --hip-trace` showed that copy dominating the
decode window under real multi-GPU load.

This version never blocks decode on a miss:

- A cache miss this step just runs the **normal CPU path** for that expert — exactly what
  would happen with the cache off, so the cache only ever removes work, never adds a stall.
- A background worker thread fills the cache **off the critical path**. The new mapping is
  only published (table swapped) once the upload actually completes, at a *later*
  `llama_moe_cache_step()` call — a running graph can never observe a torn slot.
- Uploads are throttled (`--moe-expert-cache-inserts` per layer per step) so a cold cache
  can't saturate the host↔GPU link.

Two real bugs turned up while building this and are worth knowing about if you're reading
the code (see `src/llama-moe-expert-cache.cpp` and `src/llama-graph.cpp`):

- `ggml_get_rows` requires `a->ne[2] == b->ne[1]` exactly — no implicit batch broadcast. The
  shared per-layer expert→slot lookup table has to be flattened to one column per call
  instead of attempted per-token batching, or it crashes on MTP's multi-token verify batches.
- The CUDA/HIP backend's plain `ggml_backend_tensor_set()` does a `cudaMemcpyAsync`
  immediately followed by a full `cudaStreamSynchronize` — i.e. every "async" upload was
  actually blocking, one at a time, with zero pipelining. Confirmed via
  `rocprofv3 --hip-trace`: `hipStreamSynchronize` alone was over half of a 20-second decode
  window. Fixed by switching to `ggml_backend_tensor_set_async` and draining the whole
  pending batch before one `ggml_backend_synchronize()` call.

### MTP draft window: n-max=4 crash, root-caused and fixed

`--spec-draft-n-max` above 3 used to crash the server with this cache enabled (verify
batches are `n-max + 1` tokens; the cache's graph-building code capped at 4 tokens). That's
now fixed for n-max up to 4 (5-token batches) — the fix, and how it was actually found, is
worth documenting since it's a real bug in shared CUDA/HIP kernel code, not something
specific to this fork's own logic.

**The bug.** A live `gdb` backtrace on the crash showed `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
— a genuine GPU out-of-bounds write — inside `quantize_mmq_q8_1`'s scatter path
(`ggml/src/ggml-cuda/quantize.cu`), reachable only with the cache active. Tracing it back:
`mm_ids_helper` (`mmid.cu`) assumes a token's `n_expert_used` routed expert ids are
distinct — true for ordinary top-k routing, but not for this cache, which remaps every
currently-uncached expert to the same dummy slot id. When a token routes to that slot more
than once (multiple simultaneously-uncached experts), the helper's per-(token,expert)
compaction only records *one* of the duplicates, leaving the rest of its output buffer
(`ids_src1`) uninitialized for that token. The scatter-quantize kernel then reads those
slots unconditionally as destination row indices and writes to them — an unbounded write on
whatever garbage happened to be sitting in that pool-allocated memory. (This turned out to
match a bug independently found and reported against the exact upstream mechanism this cache
is ported from — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861) — by someone
extending it to small batches with `compute-sanitizer`; credit to that report for confirming
the diagnosis and pointing at the actual fix shape.)

**The fix** (`ggml/src/ggml-cuda/mmq.cu`, `quantize.cu`): memset `ids_src1` to a `-1`
sentinel before the helper runs, and skip the scatter write when a `-1` is read back. This is
provably safe, not just defensive — the real MMQ matmul kernel downstream bounds all its
reads to `expert_bounds`' properly-compacted ranges, so it never reads a duplicate-shadowed
slot anyway; skipping the write just avoids touching memory nothing will consume.

**Verification**: 600+ verify batches at `n_tokens=5` (n-max=4) with no crash, and
**bit-identical** output vs. the no-cache path on the same low-entropy correctness prompts
used everywhere else in this README — both at n-max=3 (regression check) and n-max=4 (the
previously-crashing case). One thing worth knowing if you go looking yourself: on *creative*
prompts, plain MTP at n-max=4 (even **with the cache off entirely**) already diverges from a
non-speculative reference at temp=0 — that's a pre-existing characteristic of deeper verify
windows in this fork (almost certainly batch-size-dependent floating-point differences in the
attention/softmax kernels, a known class of behavior in this kind of software, not something
this cache causes) — so don't use a creative-writing diff as your correctness test here; use
a low-entropy one, as described below.

**Should you actually raise it?** Probably not, on its own — measured back-to-back on this
box (a lower-noise-floor session than the headline benchmark table below, so treat the
absolute numbers as internal to this comparison, not a contradiction of it), n-max=4 was
within noise of n-max=3 (19.13 vs 19.18 t/s pooled), so the shipped config here still uses
n-max=3. The value of this fix is a real stability/correctness fix that happens to also
open the door if your hardware or workload responds differently to a wider window — you'd
need to measure that yourself. The cap now sits at `n_tokens <= 5` (n-max <= 4); if you want
to push further, re-run the same verification (crash-free over many verify batches +
bit-identical low-entropy diff) rather than just bumping the number — this class of bug is
specifically about duplicate ids within a token's routed set, which doesn't get less likely
as the batch grows.

### Correctness

Verified via greedy (temp=0) decoding on low-entropy prompts (e.g. "list the first 20 prime
numbers") — far more sensitive to a data bug than stochastic sampling or open-ended
creative/technical prompts — against the plain `--n-cpu-moe` (no cache) build:
**bit-identical** output, standalone and with MTP at both n-max=3 and n-max=4.

## Benchmarks

Dual-GPU box: AMD Radeon AI PRO R9700 32GB (gfx1201, `main-gpu`) + AMD Radeon RX 9070 16GB
(gfx1201, on a 4-lane PCIe 3.0 link), Qwen3.8-Flash-Next UD-Q4_K_XL, `--ctx-size 262144`,
`--tensor-split 87,13`, `--n-cpu-moe 40`. Figures are pooled averages across multiple trials
of three varied prompts each; run-to-run noise on this box is real (~2-2.5 t/s band) — treat
any single number as ± that, not exact.

| Config | Speed |
|---|---|
| Synchronous cache (sibling repo), n-max=3 | 18.1 t/s |
| **This (async) cache, n-max=3** | **~20.8 t/s** |

That's a **+6.6%** async-vs-sync delta measured as a same-conditions A/B (both cache
designs, identical flags otherwise). Cache-off vs cache-on is a bigger win still — a
separate profiling run (under `rocprofv3`, so not directly comparable to the unprofiled
numbers above) measured 10.9 t/s with the cache off vs 17.1 t/s on, **+57%**, at otherwise
matched settings.

Getting from where this box started the tuning session (~17.5 t/s, synchronous cache) to
the ~20.8 t/s figure above wasn't just the cache redesign — it's stacked with a few other
fixes found along the way, all reflected in the flags below:

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

Multi-GPU (the setup this fork is for):

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

Single GPU: use [the synchronous fork](https://github.com/markldn/llama.cpp-qwen4exp-lru-cache)
instead — see "Which one should I use" above.

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

## Correctness / verification methodology

If you're validating this on your own box: run the same prompt at `temp=0` (greedy) with the
cache on and off and diff the output byte-for-byte. Greedy decoding surfaces a caching bug
immediately — any wrong expert weight changes the argmax token somewhere in the sequence,
whereas stochastic sampling can mask a bug behind sampling noise for a long time before it's
visible.

## License

MIT, same as upstream llama.cpp — see [LICENSE](LICENSE).

---

For everything else (supported models, general server/CLI docs, the broader llama.cpp
project), see [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
