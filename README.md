# Gyro DX

A Zig master foundation for a Windows Qwen3.8-Flash-Next inference engine.
The native bridge uses Direct3D 12 LinAlg and DirectStorage.
The model adapter targets the supplied AgentionAI Gyro-Coder GGUF files.

**The package does not yet generate text.**
It supplies real format, storage, cache, and kernel validation components.
The complete Qwen4Exp execution graph remains unfinished.
GPU execution also requires validation on a Windows machine.

## Implemented components

| Component | Behavior |
| --- | --- |
| GGUF v3 reader | Bounded metadata reads, checked arithmetic, known tensor sizes, expert ranges, and metadata validation |
| Gyro CPU decoder | All eight custom tensor types, exact codebook, and separate Hadamard transforms |
| PLE address calculator | Sixteen row addresses from token history, EOS behavior, and page coalescing |
| Expert cache policy | Transactional admission, independent fence domains, pins, generations, and LRU eviction |
| Trace replay | Exact cache request and byte counts for supplied expert routes |
| Memory planner | Fixed weights, explicit reserve, per-layer resident buffers, and conservative spill slots |
| DirectStorage bridge | Raw file ranges into GPU buffers, status validation, fences, and byte readback |
| LinAlg validation | A real SM 6.10 wave matrix product with FP16 inputs and FP32 output |
| Gyro GPU reference | A scalar HLSL decoder with a native comparison path against the CPU decoder |

The main runtime logic uses Zig.
The C++ bridge isolates Microsoft headers and COM interfaces behind a small C ABI.
HLSL implements GPU work.
The Zig ABI declarations use explicit `extern` types.
The pinned Zig master removed `@cImport`.

## Pinned dependencies

| Dependency | Version |
| --- | --- |
| Zig master | `0.18.0-dev.738+ae9a8acc7` |
| Agility SDK | `1.722.0-preview` |
| DirectStorage | `1.3.0` |
| Preview DXC | `v1.10.2605.37` |
| LinAlg shader target | `cs_6_10` |
| Scalar Gyro shader target | `cs_6_0` |

The SDK pins reflect the sources available on October 29, 2026.
The scripts verify archive hashes before extraction.
The native build uses the shipped SDK headers as the ABI authority.
The live specification contains some newer draft fields.

## Portable build procedure

1. Install the pinned Zig compiler.
2. Run the core tests.
3. Build the command line tool.
4. Run the integration test.

```sh
zig build test
zig build -Doptimize=ReleaseSafe
python tests/integration.py zig-out/bin/gyro-dx
```

### Optional compiler bootstrap

The bootstrap requires Python 3.12 or later on Windows x64 or Linux x64.
It installs the compiler within the project directory.

```sh
python scripts/bootstrap_zig.py
```

## Windows build procedure

1. Open a Developer PowerShell with Visual Studio Build Tools, the Windows SDK, and CMake.
2. Enable Windows Developer Mode for experimental shader support.
3. Run the Windows build script.
4. Inspect the adapter capabilities.
5. Run both kernel validation commands.

```powershell
.\scripts\build_windows.ps1 -Zig zig
.\zig-out\bin\gyro-dx.exe probe
.\zig-out\bin\gyro-dx.exe linalg .\zig-out\bin\linalg_smoke.dxil
.\zig-out\bin\gyro-dx.exe gyro .\zig-out\bin\gyro_decode.dxil
```

The build script deploys the runtime DLLs beside the executable.
The `D3D12` subdirectory contains the Agility runtime.
The executable exports `D3D12SDKVersion=722` and the runtime path.

The LinAlg command requires a compatible experimental driver and the selected matrix configuration.
A GPU model name does not establish that support.
An unsupported configuration returns an error.
The scalar Gyro command does not require LinAlg support.

## Model inspection procedure

1. Inspect the actual model file.
2. Review the memory plan with an explicit non-weight reserve.
3. Print the three source ranges for one expert.
4. Verify the expert transfer on Windows.

```powershell
$model = "C:\models\Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf"
.\zig-out\bin\gyro-dx.exe inspect $model
.\zig-out\bin\gyro-dx.exe plan $model 24 2
.\zig-out\bin\gyro-dx.exe ranges $model 0 17
.\zig-out\bin\gyro-dx.exe stream $model 0 17
```

The `24 2` arguments mean 24 GiB of available local memory and a 2 GiB reserve.
This example is a planning assumption, not a guaranteed allocation on a 24 GB adapter.
The reserve covers attention state, recurrence state, temporary matrices, PLE pages, staging, and allocator overhead.
The adapter budget and current usage appear in `probe`.
Other processes and the desktop can reduce the usable memory.

The planner first evaluates resident buffers with each layer's actual expert size.
If all experts fit, `cache_slot_bytes` is null because those buffers have different possible sizes.
Otherwise, the planner uses uniform spill slots with the largest expert size.
Both strategies account for 64 KiB allocation granularity per expert buffer.
The fixed-weight figure counts payload bytes, so the reserve also covers their allocation padding.

The `stream` command compares DirectStorage output against ordinary file reads.
It executes no expert matrix operation.
The transfer benchmark therefore establishes byte correctness, without an inference speed claim.

## Trace replay procedure

1. Export actual model routes into a text file.
2. Put one token and layer pair on each line.
3. Append that layer's local expert IDs.
4. Run the replay command with a cache slot count.

```text
# token layer expert IDs
0 0 4 19 23 26 33 46 52 77 128 240
0 1 1 17 31 35 61 84 93 124 168 191
1 0 4 19 23 26 33 46 52 77 128 240
```

```powershell
.\zig-out\bin\gyro-dx.exe replay $model routes.txt 4096
```

Replay counts demand hits, misses, and transferred bytes.
It assumes that each line completes before the next line starts.
It does not model asynchronous timing, speculation, or driver throughput.
The current replay command accepts uniform expert bundle sizes, as in Gyro-Coder.
It rejects Coder-L's mixed sizes explicitly.
The memory planner and range commands support both released files.

## PLE inspection procedure

1. Supply accepted token IDs in sequence order.
2. Inspect the useful rows and merged page ranges.

```powershell
.\zig-out\bin\gyro-dx.exe ple-ranges $model 12345 456 789
```

The command reads the model's EOS value and hash parameters.
It does not tokenize text.
It prints sixteen useful row ranges and their coalesced 4 KiB page requests for each token.

## Verification evidence

The CPU decoder matched 65,536 decoded values from 512 independent C oracle blocks, bit for bit.
The fixtures cover all eight Gyro tensor types and finite binary16 scale edge cases.
Hadamard transforms and inverse transforms also matched the oracle at widths 640 and 2,560.

The cache tests cover unsuccessful admission, pending reads, failed reads, stale leases, and incomplete GPU work.
The PLE tests cover EOS boundaries, integer wrap, page overlap, and file bounds.
The CLI integration test uses an independent synthetic GGUF fixture.

Both released GGUF headers passed the reader and memory analysis.
The test fetched metadata prefixes at a fixed model revision.
No complete model file or tensor payload was necessary for that analysis.
Header hashes identify metadata, not the full weight payload.

The Windows x64 native DLL and smoke executable passed cross-compilation and link checks.
The shaders passed compilation with the pinned DXC.
Windows driver execution and model-level parity remain unverified.
The package contains no measured tokens-per-second result.

## Remaining engine work

- Complete Qwen4Exp graph execution and state management.
- GPU Hadamard transforms and production Gyro matrix kernels.
- Gated delta recurrence, sparse attention, and hyper-connection kernels.
- Learned PLE projection and convolution execution.
- A real asynchronous expert scheduler with router readback and compute submissions.
- Tokenization, chat templates, sampling, and an inference API.
- Layer-by-layer parity against the pinned reference runtime.
- Hardware latency, bandwidth, memory, and end-to-end quality measurements.

## Design and source references

- [Architecture and implementation order](docs/ARCHITECTURE.md)
- [Verified model layouts and source pins](docs/MODEL.md)
- [Native bridge contracts](native/README.md)
- [Validation record](docs/VALIDATION.md)
- [Zig master download metadata](https://ziglang.org/download/index.json)
- [Agility SDK releases](https://devblogs.microsoft.com/directx/directx12agility/)
- [DirectStorage 1.3](https://devblogs.microsoft.com/directx/directstorage-1-3-is-now-available/)

The package contains no model weights.
The third-party reference decoder retains its Apache 2.0 license and notice.
The model retains the license stated by its publisher.
