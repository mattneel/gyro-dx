# Validation record

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 10.0.26200, Developer Mode enabled |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 24 GB, driver 32.0.16.2101 |
| CPU | AMD Ryzen 9 9955HX3D, 16 cores and 32 threads, 61.6 GiB RAM |
| Storage | Samsung MZVLC2T0HBLD NVMe, 2 TB |
| Compiler | Zig `0.18.0-dev.738+ae9a8acc7`, MSVC 19.51 for the native bridge |
| Model | `Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf`, revision `b7d23f50`, 75,796,732,256 bytes |
| Reference | `agentionai/llama.cpp` at `9aa51ce4`, built with CUDA 13.1 for SM 120 |

## Automated tests

| Verification | Result | Command |
| --- | --- | --- |
| Core module tests | 43 passed | `zig build test` or `zig test src/root.zig` |
| Independent Gyro oracle test | Passed | 512 encoded blocks, 65,536 float32 comparisons |
| Tokenizer | Passed | 480 strings against Hugging Face `tokenizers`, plus round trips |
| Chat template | Passed | 15 conversations against Jinja rendering of the GGUF template |
| CLI integration | 27 invocations passed | `python tests/integration.py zig-out/bin/gyro-dx.exe` |
| Portable builds | Built | Windows native, Windows without the bridge, and Linux x64 cross-compilation |

The tokenizer and chat tests read the real model header from `$GYRO_HEADER` or `C:/src/gyro-work/coder-header.bin`.
They skip when neither exists.
`scripts/fetch_header.py coder FILE` downloads that header with SHA-256 verification.

## Hardware kernel validation

| Command | Result |
| --- | --- |
| `gyro-dx probe` | Shader Model 6.10 (`0x6a`), LinAlg tier 16, wave 16×16×16 FP16→FP32 supported, experimental features enabled |
| `gyro-dx linalg linalg_smoke.dxil` | Passed, maximum absolute error 0 |
| `gyro-dx gyro gyro_decode.dxil` | All eight Gyro types bit-identical to the CPU decoder |
| `gyro-dx stream MODEL 0 17` | 1,305,600 DirectStorage bytes equal ordinary file reads |

The executable must export `D3D12SDKVersion` and `D3D12SDKPath`.
Before `build.zig` set `rdynamic` for native builds, the in-box runtime loaded instead.
That runtime reported Shader Model 6.8, no experimental features, and LinAlg tier 0.

## Model parity

### Reference dump tool

`tools/reference/gyro-dump.cpp` is a small program for the pinned reference tree.
It copies into `examples/gyro-dump`, and `examples/CMakeLists.txt` adds the subdirectory.
It writes every graph tensor that passes `--gyro-filter`, plus the final logits and an optional greedy trace.
Each file holds four little-endian `int64` dimensions followed by float32 values.
A tensor name that repeats within one step receives a `#N` suffix.

```powershell
llama-gyro-dump -m $model -ngl 0 -c 4096 --gyro-out ref --gyro-tokens 248045,846,198 --gyro-filter "l_last-*" --gyro-gen 40
gyro-dx dump $model ours --tokens 248045,846,198 --filter "l_last-*" --gen 40
python tests/parity.py ref ours
```

### Results

The prompt was the chat-rendered request "Write a Python function that returns the n-th Fibonacci number iteratively." with thinking disabled, which is 26 tokens.

| Comparison | Result |
| --- | --- |
| CPU engine vs reference, 40-token greedy trace | Identical |
| CPU engine vs reference, layer-0 hyper-connection norm | Relative RMS 1.1e-7 |
| CPU engine vs reference, layer-0 linear-attention output | Relative RMS 0.016 |
| CPU engine vs reference, final residual stream, layer 22 | Relative RMS 0.036, cosine 0.9994 |
| CPU engine vs reference, final logits | Relative RMS 0.21, cosine 0.979 |
| GPU engine vs CPU engine, final logits | Relative RMS 0.0095, cosine 0.99996 |
| GPU engine vs CPU engine, 40-token greedy trace | Identical |
| GPU cache mode vs GPU resident mode, logits | Bit-identical |
| GPU engine vs reference, 6,430-token needle prompt | Both answer `PURPLE-ELEPHANT-42`, identical 10-token traces |

An exact recomputation in float64 of the layer-0 hyper-connection gate showed the source of the reference difference.
The CPU engine matched it with relative error 2.4e-7.
The reference runtime differed by 7.4e-3.
That difference grows across 48 layers. The routed expert sets still agree for most tokens and layers.
[INFERENCE] The likely cause is reduced-precision activation quantization in the reference CPU products.

The needle prompt places one password among 260 synthetic records.
At that length the sparse indexer selects 2,051 cells per query, so the test exercises the block selection path.

A spot check found a model error that the reference shares.
Asked "What does HTTP stand for?", the reference runtime, the CPU engine, and the GPU engine all answer "Hyper Transfer Protocol".
The reproduced error is evidence of parity, not an engine fault.

## Performance

All runs used greedy decoding, the default 8,192-token context allocation, and `gyro-dx bench`.
Prompts repeat a fixed English sentence. Results vary by about 5% between runs.

### Resident experts

| Prompt | Generated | Prefill | Decode | p50 / p95 / max latency |
| ---: | ---: | ---: | ---: | --- |
| 512 | 256 | 80 tok/s | 56.9 tok/s | 17.5 / 18.5 / 21.0 ms |
| 4,096 | 128 | 76 tok/s | 54.2 tok/s | 18.4 / 19.4 ms |

Memory: 22,017,523,968 bytes allocated on the device; adapter usage 21.2 GB of a 24.0 GB budget.
The expert arenas hold 12,288 bundles in 15.3 GB.

GPU time per token, averaged over 512 prefill and 256 decode tokens with `--profile`:

| Phase | ms per token |
| --- | ---: |
| Linear-attention projections | 3.56 |
| Routed experts | 2.57 |
| Hyper-connection mixers | 2.32 |
| Full attention and indexer | 2.08 |
| Linear-attention output projection | 1.45 |
| Router and shared expert | 1.08 |
| Output projection | 0.32 |
| Convolution and recurrence | 0.36 |
| PLE, combination, and other | 0.27 |

Timestamps between phases include barrier drain time.
Total GPU time therefore exceeds the work in each kernel.

### Demand expert cache

The runs used a 128-token prompt and 128 generated tokens.

| Cache | Prefetch | Prefill | Decode | p50 / p95 | Demand hits / misses | Bytes loaded | Storage wait |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 4 GiB | on | 31 tok/s | 20.0 tok/s | 47.7 / 74.3 ms | 112,257 / 10,623 | 42.9 GB | 3.65 s |
| 4 GiB | off | 33 tok/s | 18.7 tok/s | 50.6 / 83.0 ms | 104,054 / 18,826 | 24.6 GB | 4.68 s |
| 10 GiB | on | 39 tok/s | 27.7 tok/s | 34.5 / 47.0 ms | 119,723 / 3,157 | 12.5 GB | 1.22 s |
| 10 GiB | off | 37 tok/s | 26.5 tok/s | 34.6 / 53.7 ms | 114,345 / 8,535 | 11.1 GB | 2.20 s |

Without prefetch, the 4 GiB run transferred 24.6 GB during 4.68 s of storage wait.
That is an effective 5.3 GB/s for demand misses.
[INFERENCE] Some reads may come from the Windows file cache: the system has 61.6 GiB of RAM, and earlier runs had read the same file.

Prefetch issues loads for the next layer's predicted experts.
It removes 44% of demand misses at 4 GiB and 63% at 10 GiB.
Wrong predictions add transfer volume.

### CPU engine

| Prompt | Generated | Prefill | Decode | p50 latency |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 11.3 tok/s | 3.6 tok/s | 273 ms |

## Remaining verification

- Gyro-Coder-L end-to-end inference. Its 84 GB file was not downloaded.
- Long-context runs beyond 8,192 tokens. `--ctx` raises the limit; each full-attention layer stores 2,304 bytes per token: binary16 keys and values plus raw indexer keys.
- D3D12 debug-layer runs of the inference kernels. `--debug-layer` enables the layer, but no run was recorded.
- Quality benchmarks beyond trace parity and the needle test.
