# herbert-cuda-bf16 — Qwen3.5 in bf16 on NVIDIA GPUs, every kernel emitted as PTX

The CUDA counterpart of [herbert-vk-bf16](https://github.com/xigh/herbert-vk-bf16) (Vulkan, SPIR-V emitted) and [herbert-jit-bf16](https://github.com/xigh/herbert-jit-bf16) (CPU): the same model (Qwen3.5, text), the same numerical contract (bf16 weights, bf16 activations, bf16 KV cache, f32 accumulators, one rounding to nearest even at the output of each kernel). No CUDA C, no nvcc, no NVRTC, no cuBLAS: every compute kernel is PTX emitted at start-up by Rust (`src/ptx`) and handed to the driver, which is itself loaded at run time (`libcuda.so.1`, `nvcuda.dll` on Windows). Nothing of the CUDA toolkit is needed to build or to run.

The engine runs Qwen3.5-0.8B and Qwen3.5-4B: block prefill on the tensor cores (`mma`, `ldmatrix`, `cp.async`), a decode step replayed as a CUDA graph, and per-kernel profiling.

Author: Philippe Anel. License: MIT, see `LICENSE`.

## Build and run

```sh
cargo check                                   # anywhere, no CUDA SDK needed
cargo build --release --locked                # on the GPU machine
cargo build --release --locked --features serve   # adds herbert-cuda-serve
```

The binaries (`target/release/`):

```sh
herbert-cuda-probe                            # what the device declares, then measurements with emitted kernels
herbert-cuda-probe --emit /tmp/ptx            # no device: every kernel of the probe written as /tmp/ptx/<name>.ptx
herbert-cuda-chat ~/models/Qwen3.5-0.8B --no-system --no-think --ask 'What is the capital of France?'
herbert-cuda-chat ~/models/Qwen3.5-4B --ctx 61440 --no-system --no-think --max-tokens 128 --ask-file long-prompt.txt
herbert-cuda-serve ~/models/Qwen3.5-4B --ctx 4608 --port 8080      # feature serve
herbert-cuda-bench ~/models/Qwen3.5-4B --cases cases.json --out run.json   # shared-ID request timing
```

`MODEL_DIR` is a Hugging Face checkpoint directory of Qwen3.5 (`config.json`, `tokenizer.json`, the bf16 safetensors).

`herbert-cuda-chat` options: `--device N` (CUDA ordinal, default 0), `--ctx N` (positions reserved, default 4096), `--no-think`, `--max-tokens N` (default 1024), `--system TEXT | --no-system`, `--ask TEXT` and `--ask-file PATH` (repeatable, in order; a prompt over 128 KiB does not fit in one argument on Linux), `--temperature T` (sampling instead of greedy), `--dump-ptx DIR`, `--tune KEY=VALUE` (repeatable), `--profile` (every launch timed on the GPU, no graph, a table per kernel after each turn), `--warmup N`, `--pause MS`. On stderr, per turn: `[ids]` (the generated ids, in the format of `herbert-bf16-chat`, the CPU oracle, for a token-by-token comparison), `[decoded]` and `[turn]` (counts and speeds).

`herbert-cuda-serve` (feature `serve`) answers the Anthropic Messages API (`POST /v1/messages`, streamed or not), one request at a time, every request from an empty cache: the bench times every engine through the same HTTP client with it. Options: `--device`, `--ctx` (default 4608), `--tune`, `--no-think`, `--logits`, `--host` (default 127.0.0.1), `--port` (default 8080), `--name`.

`herbert-cuda-bench` times greedy requests on pretokenized ids, excluding model loading, tokenization and detokenization: `--prepare --cases FILE --prompts DIR --labels A,B` writes the cases, then `--cases FILE --out FILE [--outputs N,…] [--rounds N] [--warmup N] [--ctx N] [--tune …]` times them; `--variants-file` runs several `--tune` sets interleaved on the same loaded weights.

The emitted PTX targets sm_89 (PTX 8.5); the driver compiles it forward for newer GPUs, and on sm_80 … sm_88 the target line is lowered to the device's (the kernels use nothing past sm_80). Below sm_80 there is no bf16 `mma`. To check the emitted modules with the toolkit's assembler:

```sh
herbert-cuda-probe --emit /tmp/ptx && for f in /tmp/ptx/*.ptx; do ptxas -arch=sm_89 -o /dev/null $f; done
```

## Measured on

An RTX 4080 16 GB (sm_89, 76 SM) in a Ryzen 9 9900X machine under Linux, NVIDIA driver 595.99.02, SM clock locked at 2 805 MHz and memory at 11 201 MHz, the card held in P0 under CUDA by the application profile `CudaNoStablePerfLimit` (without it the driver keeps the card in P2 during any CUDA context). Built with the stable Rust toolchain (1.98), release profile `opt-level=3`, no LTO. On macOS the crate only compiles (`cargo check`).

The defaults below were chosen on that card. Other GPUs are untested; the automatic GEMM policy and the per-model prefill block apply only to sm_89 with 76 SM, elsewhere the engine falls back to policy 1 and blocks of 1 024 rows.

## Tuning

Every kernel family has its parameters in `src/cuda/tuning.rs`, set with `--tune key=value`; the load prints the whole set.

**Attention.** Since 2026-10-21 the defaults are `sdpa.gqa=1` (the decode attention in one pass per GQA group, K and V read once for the group), `sdpa.split=64` (blocks per KV head of the decode attention) and `sdpa.pipe=1` (the prefill attention's copies of K and V overlap its arithmetic; same results). The former attention, the one measured in phases 1 to 5 of the [#003 campaign](https://github.com/xigh/herbert-cuda-bench), is

```sh
--tune sdpa.gqa=0 --tune sdpa.split=8 --tune sdpa.pipe=0
```

Other attention switches: `sdpa.rows=16|32` (query rows per block of the prefill attention; 32 puts 8 warps on each K and V tile), `sdpa.bk` (keys per step of the prefill attention), `sdpa.softmax=reg|shared`, `sdpa.maxnreg` (register cap, an ablation).

**Prefill.** Since 2026-10-19 the default prefill is the measured candidate: `gemm.auto=3`, `gemm.layout=xor`, `prefill.block=auto`. The former default (64×64 GEMM, K=32, three stages, padded layout, 1 024-row blocks) is `--tune gemm.auto=0 --tune gemm.layout=pad --tune prefill.block=1024`.

`gemm.auto=1` selects 64×64 K=64/two stages at M≤256, 64×64 K=32/three stages at M=257–511, and 64×128 K=32/three stages at M≥512. `gemm.auto=3` selects the measured candidates on the 76-SM sm_89 GPU. For the 4B (hidden=2560), M=129–255 uses 64×64, warps 1×4, K=64/two stages; exactly M=256 with the padded shared layout uses 128×128, warps 4×4, K=64/two stages on gate_up and padded DeltaNet projection, the original 64×64 K=32/three stages on residual outputs, and 64×64 K=64/two stages on QKV. M≥512 uses 64×128, warps 2×2, K=32/two stages. For the 0.8B (hidden=1024), policy 1 is retained below M=1024; larger blocks use 64×128, warps 2×2, K=32/two stages. Other hardware or hidden sizes fall back to policy 1. Kernel and tile are selected together. DeltaNet projections are padded to 128 columns in all automatic modes.

`prefill.block=auto` resolves to 2 048 rows for the 4B on that GPU, 4 096 for the 0.8B, and 1 024 elsewhere. Numeric block sizes remain available; `1` feeds the prompt token by token through the decode step. The DeltaNet state is rounded to bf16 at block boundaries, so changing the block size can change numerical results: the 4B 4 001-input/256-output case first differs from auto 1 at index 145 with 2 048-row blocks, and block 1 024 preserves the baseline sequence.

Manual `gemm.tile` supports `32`, `32w1x1`, `64`, `64x32`, `64w4x1`, `64w1x4`, `64x128`, `64x128w4x2`, `64x128w2x2`, `64x256`, `128`, `128w4x4` and `128w4x2`. `gemm.bk` accepts 32/64 and `gemm.stages` accepts 2/3. Automatic modes, including the default `gemm.auto=3`, reject manual tile/K/stage overrides: add `gemm.auto=0`. `gemm.layout=pad` or `xor` (default) controls shared addressing; XOR removes row padding and applies the same 16-byte permutation to copies and fragment loads. `gemm.carveout=driver` (default) or 0..100 sets the preferred shared/L1 split. Resource reports describe a calculated bound, not achieved occupancy.

Experimental, off by default: `gemm.schedule=0..5` (order of independent fragments, each accumulator's K order preserved), `gemm.pipeline=0..3` (copy scheduling, ready/free stage barriers), `gemm.down_short` (explicit tiles for the 4B residual down projection at exactly M=128/256), `gemm.ldmatrix=0`, `gemm.copy=sync`, `gemm.wmma=1`; `gemm.auto=2`, `4`, `5` and `6` preserve matrix and large-tile experiments. They are not faster settings in general.

**Decode and the rest.** `matvec.rows`, `matvec.warps`, `matvec.depth`, `matvec.cache` (ca, nc, nol1, cg, cs), `graph=0|1` (the decode step as a CUDA graph), `act=bf16|f32` (the DeltaNet's activations and states), `rec.tc=0|1` (the prefill's delta rule chunked on the tensor cores), `ld.scope=plain|sm|gpu` (an ablation of the global loads' scope).

## The code

| module | what it does |
|---|---|
| `core` | the model's `config.json`, bf16 conversions, the error type, the sampler |
| `model` | the safetensors checkpoint, read into bf16 row-major tensors |
| `chat` | the Qwen3.5 template, tokens, streaming detokenization, the session |
| `ptx` | the PTX emitter every kernel is written with |
| `cuda/driver` | the CUDA driver API, loaded at run time |
| `cuda/gpu` | the device, modules, launches, graphs |
| `cuda/kernels`, `cuda/decode` | the emitted kernels of one decode step |
| `cuda/prefill`, `cuda/gemm`, `cuda/dntc` | the emitted kernels of a prefill block: GEMMs, attention, the chunked delta rule |
| `cuda/tuning` | the parameters of the kernel families, `--tune` |
| `cuda/engine` | the model on the device, the recorded step |
| `cuda/probe` | the device report and the measurements |
| `serve` | the Anthropic Messages API over HTTP (feature `serve`) |

## Bench scripts

`bench/compare.sh MODEL OUTDIR DEVICE…` runs the prompts of `bench/prompts/` on the CPU oracle (`herbert-bf16-chat` of `herbert-jit-bf16`, expected in `~/herbert-jit-bf16`) and on each CUDA device, greedy, and keeps the full outputs; `bench/_diff.py` reports the first divergence of each run. `VARIANTS="name:--tune a=1;other:…"` runs each variant. The prefill fixtures named 512 and 4000 contain 513 and 4 001 tokens once the template is applied.

`bench/speed.sh`, `bench/vs.sh` and `bench/prefill-vs.sh` run interleaved speed campaigns against herbert-vk and vLLM (`_speed.py`, `_vs.py`, `_pp.py`, `_dec.py` read their logs). `bench/request-vs.sh OUTDIR [ROUNDS]` compares the same pretokenized ids, greedy, fixed output counts, with vLLM before and after the Herbert window (`_requests.py`, `_gemm_requests.py`, `_gemm_campaign.py` summarize). `bench/trace-requests.sh` (Nsight Systems) and `bench/ncu-requests.sh` (Nsight Compute, needs `RmProfilingAdminOnly=0`) capture real requests; `_nsys.py`, `_ncu.py` and the `_gemm_*.py` audits read the captures. Models are looked up in `$MODELS` (default `~/models`).

The campaign that compares this engine with llama.cpp, vLLM, SGLang and transformers (protocol, predictions written before measuring, scripts) is [herbert-cuda-bench](https://github.com/xigh/herbert-cuda-bench).
