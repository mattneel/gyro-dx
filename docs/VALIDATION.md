# Validation record

## Environment and scope

The development host ran Linux x64.
The compiler was Zig `0.18.0-dev.738+ae9a8acc7`.
Windows artifacts passed cross-compilation and link verification.
The environment supplied no Windows GPU or DirectStorage runtime execution.
The package therefore makes no hardware performance or complete inference claim.

## Completed verification

| Verification | Result | Evidence |
| --- | --- | --- |
| Core module tests | 29 passed | `zig build test` |
| Independent Gyro oracle test | Passed | 512 encoded blocks and 65,536 float32 comparisons |
| Hadamard oracle comparison | Passed | Rotation and inverse at widths 640 and 2,560 |
| PLE address cases | Passed | History, EOS, wrap behavior, and page bounds |
| Linux executable | Built | ReleaseSafe mode |
| CLI integration | 22 invocations passed | 14 expected error paths and independent synthetic GGUF encoding |
| Actual Coder header | Passed | 1,224 tensors, 77 metadata entries, pinned header SHA-256 |
| Actual Coder-L header | Passed | 1,224 tensors, 83 metadata entries, pinned header SHA-256 |
| Windows native DLL | Compiled and linked | Exact Agility 722 and DirectStorage 1.3 headers |
| Windows native smoke executable | Compiled and linked | C++20 through Zig's C++ compiler |
| Windows Zig executable | Compiled and linked | All native entry points, including GPU Gyro validation |
| LinAlg shader | Compiled | Preview DXC `v1.10.2605.37`, `cs_6_10` |
| Gyro reference shader | Compiled | Same DXC, `cs_6_0`, strict arithmetic |
| Native ABI layout | Passed | Shared C and Zig assertions for sizes and field offsets |

The test runner sometimes reports the oracle executable as cached.
That executable contains a separate test from the 29 core tests.
The preserved fixture hash is `8f739c3eb9cd3099ec7ccd1f46f60fb115cf2ec87d2ac6c373f2a49ed032818c`.

## CLI coverage

The integration script creates and removes a synthetic file within a temporary directory.
Its fixture has two layers, four experts per layer, and two selected experts.
Its independently calculated expert bundle size is 13,056 bytes.
The corresponding committed buffer allocation is 65,536 bytes.

The script verifies these commands:

- `inspect` and `inspect-prefix`
- `manifest`
- `plan`, in resident and spill modes
- `ranges`
- `replay`
- `ple-ranges`

It also verifies exact error behavior for malformed files and invalid requests.
Those cases include truncation, overlap, alignment, dimensions, unknown tensor types, and invalid expert IDs.

## Actual model evidence

The analysis used repository revision `b7d23f5036e9134b3b84e20e7040ba47bc5e0b4c`.
Only metadata prefixes were necessary.
The parser used the complete file lengths from the HTTP range responses.

- `coder-inspection.json` contains the exact Coder result.
- `coder-l-inspection.json` contains the exact Coder-L result.
- `MODEL.md` records the model and format sources.
- `scripts/fetch_header.py` reproduces the bounded download with SHA-256 verification.

Header verification covers dimensions, metadata, and calculated byte ranges.
It does not verify complete weight contents or inference outputs.

## Hardware verification that remains

The following gates require a Windows system with the intended adapter and driver:

- DirectStorage file-to-buffer equality through the actual device.
- LinAlg capability and matrix validation.
- GPU Gyro bit parity, including signed zero and binary16 subnormal scales.
- Resource validation under the D3D12 debug layer.
- Transfer latency, queue overlap, memory pressure, and cache reuse measurements.
- Complete Qwen graph parity and end-to-end generation.

The CMake and PowerShell build path did not execute on this Linux host.
Cross-compilation verified the native source and executable links.
The supplied Windows procedures cover the remaining build and execution checks.

## Reproduction procedure

1. Select the pinned Zig compiler.
2. Run the portable test suite.
3. Build the portable executable.
4. Run the CLI integration test.

```sh
zig build test
zig build -Doptimize=ReleaseSafe
python tests/integration.py zig-out/bin/gyro-dx
```

### Metadata reproduction procedure

1. Fetch the Coder header.
2. Inspect the header with its complete file size.

```sh
python scripts/fetch_header.py coder coder-header.bin
zig-out/bin/gyro-dx inspect-prefix coder-header.bin 75796732256
```

The bounded downloader rejects a full-file response.
It also verifies the exact header range and checksum.
It never downloads the model's tensor payloads.
