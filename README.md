# Gyro DX

[![CI](https://github.com/mattneel/gyro-dx/actions/workflows/ci.yml/badge.svg)](https://github.com/mattneel/gyro-dx/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Zig inference engine for Qwen3.8-Flash-Next on Windows.
The GPU path uses Direct3D 12 compute and DirectStorage. It does not use CUDA.
The model adapter targets the AgentionAI Gyro-Coder GGUF files.

The engine runs the complete Qwen4Exp graph.
That graph includes hyper-connections, gated delta recurrence, block-sparse attention, the learned PLE n-gram module, and 256 routed Gyro experts.
Routed experts stay in their packed 2-bit trellis form on the GPU.
They are either fully resident or streamed on demand through DirectStorage.

## Measured results

The measurements used one RTX 5090 Laptop GPU with 24 GB of memory, driver 32.0.16.2101, and Windows 10.0.26200.
The model was Gyro-Coder `TQ2_0` at revision `b7d23f50`.
Decoding was greedy, with all experts resident unless stated otherwise.

| Workload | Prefill | Decode | Decode latency p50 / p95 |
| --- | ---: | ---: | ---: |
| GPU, 1,024-token prompt, 128 generated | 247 tok/s | 54.6 tok/s | 17.6 ms |
| GPU, 512-token prompt, 256 generated, before LinAlg prefill | 80 tok/s | 56.9 tok/s | 17.5 / 18.5 ms |
| GPU, 4,096-token prompt, 128 generated | 76 tok/s | 54.2 tok/s | 18.4 / 19.4 ms |
| GPU, 4 GiB expert cache, prefetch on | 31 tok/s | 20.0 tok/s | 47.7 / 74.3 ms |
| GPU, 4 GiB expert cache, prefetch off | 33 tok/s | 18.7 tok/s | 50.6 / 83.0 ms |
| GPU, 10 GiB expert cache, prefetch on | 39 tok/s | 27.7 tok/s | 34.5 / 47.0 ms |
| GPU, 10 GiB expert cache, prefetch off | 37 tok/s | 26.5 tok/s | 34.6 / 53.7 ms |
| CPU, Ryzen 9 9955HX3D, 32 threads | 11 tok/s | 3.6 tok/s | 273 ms |

Cache rows use a 128-token prompt and 128 generated tokens.
Prefetch reduces demand misses and storage waits, but wrong predictions add traffic.
With the 4 GiB cache, total expert transfer rose from 24.6 GB to 42.9 GB over the run.
Resident mode allocates 20.5 GiB of video memory in total, including 15.3 GB of packed experts.
[docs/VALIDATION.md](docs/VALIDATION.md) records the commands, cache statistics, per-phase GPU times, and parity measurements.

## Correctness evidence

- The CPU engine and the pinned reference runtime produce identical 40-token greedy traces from the same prompt.
- The GPU engine reproduces the same greedy trace. Its final logits match the CPU engine with 0.95% relative RMS error and cosine similarity 0.99996.
- Cache mode produces bit-identical logits to resident mode.
- The GPU and reference runtimes both retrieve the same needle from a 6,430-token prompt that uses the sparse indexer.
- The CPU decoder matches 65,536 values from an independent C oracle bit for bit.
- The GPU scalar decoder matches the CPU decoder bit for bit for all eight Gyro types.
- The tokenizer matches the official Hugging Face tokenizer on 480 strings.
  The chat template matches Jinja rendering on 15 conversations.

## Components

| Component | Source | Behavior |
| --- | --- | --- |
| GGUF v3 reader | `src/gguf.zig` | Bounded metadata reads, checked arithmetic, tensor ranges, rotation metadata |
| Model binding | `src/model.zig` | Memory-mapped weights, validated shapes, and Hadamard sign vectors |
| CPU engine | `src/engine.zig`, `src/kernels.zig` | Complete graph on a thread pool. It is the parity baseline for the GPU |
| GPU engine | `src/gpu_engine.zig`, `shaders/engine/*.hlsl` | Bindless SM 6.6 kernels for the same graph |
| Native bridge | `native/gyro_native.cpp` | C ABI for resources, pipelines, dispatch, timestamps, and DirectStorage |
| Expert cache | `src/cache.zig` | Transactional admission, typed storage and compute fences, pins, LRU eviction |
| Expert scheduler | `src/gpu_engine.zig` | Route readback, batched demand loads, next-layer speculative prefetch |
| Tokenizer and chat | `src/tokenizer.zig`, `src/chat.zig` | Byte-level BPE with the `qwen35` pre-tokenizer, and the model's chat template |
| Sampling and runtime | `src/sampler.zig`, `src/runtime.zig` | Temperature, top-k, top-p, min-p, repeat penalty, and prompt-prefix reuse |
| HTTP API | `src/server.zig` | OpenAI-compatible chat and completion endpoints, including streaming |
| PLE addresses | `src/ple.zig` | Exact n-gram hash rows and coalesced page requests |
| Planning tools | `src/plan.zig`, `src/trace.zig` | Memory planner and cache trace replay |

Zig owns the runtime logic.
The C++ bridge isolates the Microsoft headers and COM interfaces behind a small C ABI.
HLSL implements GPU work.
The Zig ABI declarations use explicit `extern` types, because the pinned Zig master removed `@cImport`.

## Pinned dependencies

| Dependency | Version |
| --- | --- |
| Zig master | `0.18.0-dev.738+ae9a8acc7` |
| Agility SDK | `1.722.0-preview` |
| DirectStorage | `1.3.0` |
| Preview DXC | `v1.10.2605.37` |
| Inference shader target | `cs_6_6` |
| LinAlg shader target | `cs_6_10` |
| Scalar Gyro shader target | `cs_6_0` |

The scripts verify archive hashes before extraction.
The native build uses the shipped SDK headers as the ABI authority.

## Portable build procedure

The portable build contains the CPU engine, the tokenizer, the HTTP API, and every inspection command.

```sh
python scripts/bootstrap_zig.py   # optional: installs the pinned compiler under .tools/
zig build test
zig build -Doptimize=ReleaseFast
python tests/integration.py zig-out/bin/gyro-dx
```

## Windows build procedure

1. Open a Developer PowerShell with Visual Studio Build Tools, the Windows SDK, and CMake.
2. Enable Windows Developer Mode for experimental shader support.
3. Run the Windows build script.
4. Inspect the adapter capabilities and run the kernel validation commands.

```powershell
.\scripts\build_windows.ps1 -Zig zig
.\zig-out\bin\gyro-dx.exe probe
.\zig-out\bin\gyro-dx.exe linalg .\zig-out\bin\linalg_smoke.dxil
.\zig-out\bin\gyro-dx.exe gyro .\zig-out\bin\gyro_decode.dxil
```

The build script places the runtime DLLs, the `D3D12` Agility directory, and the compiled `kernels` directory beside the executable.
The executable exports `D3D12SDKVersion=722` and `D3D12SDKPath`, so Windows loads the Agility runtime instead of the in-box runtime.
Without those exports, `probe` reports Shader Model 6.8 and no experimental features.

The inference kernels require Shader Model 6.6.
The `linalg` command additionally requires the experimental LinAlg feature and a compatible driver.

## Inference procedure

```powershell
$model = "C:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf"
.\zig-out\bin\gyro-dx.exe generate $model --backend gpu --prompt "Write a binary search in Zig." --stats
.\zig-out\bin\gyro-dx.exe chat $model --backend gpu --no-think
.\zig-out\bin\gyro-dx.exe serve $model --backend gpu --port 8080
.\zig-out\bin\gyro-dx.exe bench $model --backend gpu --prompt-tokens 512 --gen 256 --profile
```

`--backend cpu` is the default and needs no GPU.
Prompts use the GGUF chat template unless `--raw` is given.
`--no-think` disables the template's reasoning block.
Sampling defaults to temperature 0.6, top-k 20, and top-p 0.95. `--temp 0` selects greedy decoding.

### Expert residency

`--experts auto` keeps every expert resident when the adapter budget allows it.
Otherwise it switches to a demand cache.
`--experts cache --cache-gib G` forces a cache of G GiB.
`--reserve-gib` keeps video memory free for other allocations; the default is 1.5 GiB.

In cache mode, each layer reads its routed expert IDs back to the host.
It admits the experts as one cache transaction, loads the misses with one DirectStorage batch per arena, and verifies the storage status before compute uses the slots.
During the same readback, the next layer's router runs on the current hidden state.
Its predicted experts load speculatively while the GPU continues.
`--no-prefetch` disables the prediction.

### HTTP API

`serve` exposes these endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Requests accept `stream`, `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, and `chat_template_kwargs.enable_thinking`.
Chat responses separate `reasoning_content` from `content`.
The server runs one sequence and serves requests in order.
A request whose prompt extends the previous conversation reuses the computed prefix.

## Model inspection procedure

```powershell
.\zig-out\bin\gyro-dx.exe inspect $model
.\zig-out\bin\gyro-dx.exe plan $model 24 2
.\zig-out\bin\gyro-dx.exe ranges $model 0 17
.\zig-out\bin\gyro-dx.exe stream $model 0 17
.\zig-out\bin\gyro-dx.exe replay $model routes.txt 4096
.\zig-out\bin\gyro-dx.exe ple-ranges $model 12345 456 789
.\zig-out\bin\gyro-dx.exe tokenize $model --prompt "hello" --raw
```

The `24 2` arguments mean 24 GiB of available memory and a 2 GiB reserve for non-weight allocations.
`replay` simulates cache hits and misses for recorded routes. It does not model timing.
It accepts uniform expert sizes and rejects Coder-L's mixed sizes explicitly.

## Parity procedure

`dump` writes graph tensors in the same file format as the reference dump tool.
`tests/parity.py` compares two dump directories.

```powershell
.\zig-out\bin\gyro-dx.exe dump $model out-gpu --backend gpu --tokens 248045,846,198 --gen 40
.\zig-out\bin\gyro-dx.exe dump $model out-cpu --filter "l_last-*,ffn_moe_topk-*" --tokens 248045,846,198 --gen 40
python tests/parity.py out-cpu out-gpu
```

The reference dump tool is documented in [docs/VALIDATION.md](docs/VALIDATION.md).
It is a small program built in the pinned `agentionai/llama.cpp` tree.
The CPU engine emits per-layer tensors under the reference callback names.
The GPU engine emits logits and greedy traces.

## Known limits

- Gyro-Coder-L uses three pair-trellis expert types with different sizes per layer.
  The CPU decoder covers those types, and the GPU decoder matches it bit for bit.
  No end-to-end Coder-L run was performed, because the 84 GB file was not downloaded.
- The engine serves one sequence at a time.
  Batched multi-user decoding and multi-token prediction are not implemented.
- Image and video inputs are not supported. The PLE image-token path is unused.
- Prefill uses LinAlg wave matrices for Q8_0 products in passes of 64 tokens.
  The activations enter the tiles as FP16.
  Final logits then differ from the CPU engine by 4.9% relative RMS, against 0.95% for the matrix-vector path, and greedy traces stay identical.
  Routed experts still use the per-slot fused kernel during prefill.
  Adapters without LinAlg fall back to matrix-vector kernels.
- Logits match the reference runtime with cosine similarity 0.979. Greedy traces are identical.
  [INFERENCE] The likely cause is that the reference CPU path quantizes activations before quantized products. This engine keeps activations in float32.

## Design and source references

- [Architecture](docs/ARCHITECTURE.md)
- [Verified model layouts and source pins](docs/MODEL.md)
- [Native bridge contracts](native/README.md)
- [Validation record](docs/VALIDATION.md)

## License

Gyro DX is licensed under the [Apache License 2.0](LICENSE).
The third-party Gyro reference decoder in `third_party/agention-gyro` retains its own Apache 2.0 license and notice.
The repository contains no model weights. The model retains the license stated by its publisher.

## Contributing and support

- [CONTRIBUTING.md](CONTRIBUTING.md) describes the required checks and parity evidence.
- [SECURITY.md](SECURITY.md) describes private vulnerability reporting.
- Sponsorship: [github.com/sponsors/mattneel](https://github.com/sponsors/mattneel).
