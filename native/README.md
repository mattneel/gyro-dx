# Windows native backend

## Status

This directory contains a C ABI bridge for Direct3D 12 and DirectStorage.
The bridge compiles against the actual Agility SDK 1.722.0-preview headers.
The Windows x64 DLL and smoke executable also passed a link check with Zig's C++ compiler.
The LinAlg shader compiles with preview DXC v1.10.2605.37.
The scalar Gyro decoder compiles with the same compiler under `cs_6_0`.
The transfer, LinAlg, and Gyro validation commands pass on an RTX 5090 Laptop GPU with driver 32.0.16.2101.
The bindless compute API below runs the Qwen4Exp inference kernels.

## Dependencies

| Component | Pinned version |
|---|---|
| Windows target | x64 |
| Agility SDK | 1.722.0-preview |
| `D3D12SDKVersion` export | 722 |
| DirectStorage | 1.3.0 |
| DXC | v1.10.2605.37 |
| Shader profile | `cs_6_10` |
| Native language | C++20 |

The build requires CMake and Visual Studio Build Tools with the Windows SDK.
The build script verifies SHA-256 hashes for each downloaded package.
The shader requires the preview compiler, experimental shader support, and a compatible GPU driver.
GPU names alone do not establish LinAlg support.

## Build procedure

1. Open a Developer PowerShell for Visual Studio.
2. Enable Windows Developer Mode for experimental shader support.
3. Run the build script.

```powershell
.\native\build.ps1
```

## Verification procedure

1. Select a local file for the transfer test.
2. Run the transfer test.

```powershell
.\native\build\Release\gyro-native-smoke.exe C:\models\model.gguf --bytes 1048576
```

3. Run the LinAlg test with the compiled shader.

```powershell
.\native\build\Release\gyro-native-smoke.exe C:\models\model.gguf --bytes 1048576 --shader .\native\build\linalg_smoke.dxil --debug
```

The transfer test compares file bytes against GPU readback bytes across two submissions.
The LinAlg test compares a 16×16×16 FP16/FP32 matrix product against a CPU reference.
The shader uses a 32-thread wave.
Unsupported configurations return `DXGI_ERROR_UNSUPPORTED`.
The test does not estimate inference speed.

## C ABI

The public header is `include/gyro_native.h`.
All public structs use fixed-width integers and opaque handles.
All results use signed HRESULT values.
Each context requires serialized calls.
The caller retains a handle after a timeout or failed release.

The executable owns the `D3D12SDKVersion` and `D3D12SDKPath` exports.
The DLL does not own these exports.
The smoke executable supplies both exports.
The build places the Agility runtime under `D3D12` beside the executable.

## Transfer contract

Each cache slot requires its own buffer resource.
The bridge treats a whole buffer as one resource-state domain.
It does not permit simultaneous writes and compute reads on different portions of one buffer.
Each raw request fits the configured staging buffer.
The default staging size is 32 MiB.

The bridge validates these conditions before it enqueues a batch:

- Every source range fits the file.
- Every destination range fits the buffer.
- Destination ranges do not overlap.
- An earlier storage batch no longer owns the buffer.
- Every request fits the staging buffer.

`gd_read_ranges` uses `IDStorageQueue3::EnqueueRequests`.
Each batch waits for the buffer's last compute fence before GPU work.
DirectStorage writes buffers in `COMMON` state.
The bridge returns each buffer to `COMMON` after compute access.
Storage and compute use separate fences.

A status array records every storage batch result.
Fence completion alone does not establish successful data transfer.
The bridge returns a pending status instead of success if a completed fence lacks a completed status.
A failed batch prevents later data consumption through the context.
The bridge retains resources after a compute timeout until its fence completes.

`gd_readback` uses a CPU wait before it consumes storage data.
That wait permits a status check before GPU work.
The bridge uses one reusable compute allocator and command list.
This conservative path supports hardware validation.
A production inference scheduler requires a pool of compute command lists and asynchronous validation policy.

## Gyro boundary

The transfer path uses `DSTORAGE_COMPRESSION_FORMAT_NONE`.
Gyro trellis quantization is not GDeflate compression.
DirectStorage transports the packed bytes without a format change.
The LinAlg shader accepts ordinary FP16 matrices.

The runtime must decode Gyro tiles before a LinAlg operation consumes their numeric values.
Persistent dense FP16 expansion defeats the model's memory advantage.
An engine requires bounded temporary tiles or a fused decoder and matrix kernel.

## Gyro conformance path

`gd_validate_gyro` runs the scalar decoder in `shaders/gyro_decode.hlsl`.
The caller supplies packed blocks, the normative binary16 codebook, and CPU reference values.
The method accepts all eight Gyro type IDs from 144 through 151.
It requires exactly 128 expected values per block.
The maximum request contains 16,777,216 values.

The method validates all byte counts before a GPU dispatch.
Raw descriptor tables supply buffer sizes for shader bounds checks.
The output starts with NaN sentinels so an absent shader write fails validation.
The comparison uses float32 bit patterns, including signed zero.
Maximum absolute error remains available as a diagnostic.

The decoder produces values in the stored basis.
It does not apply the Hadamard transform.
The conformance path requires SM 6.0 and does not require LinAlg support.
The hardware test passed for all eight types on the validation machine.

## Bindless compute contract

`gd_create` builds the bindless state when the adapter reports Shader Model 6.6 or later.
That state consists of one shader-visible descriptor heap with 65,536 entries and one root signature holding 60 root constants.
The root signature sets `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`.

- `gd_create_resource` allocates a buffer with a raw SRV and a raw UAV.
  `GD_RESOURCE_HOST` buffers are persistently mapped write-back memory that shaders can also access.
- `gd_create_pipeline` builds a compute pipeline from DXIL compiled for `cs_6_6` against that root signature.
- `gd_begin`, `gd_dispatch`, `gd_barrier`, `gd_timestamp`, and `gd_submit` record work on a ring of three command lists.
  `gd_barrier` inserts a global UAV barrier.
- `gd_read_ranges_after` writes file ranges into any buffer after a named compute ticket.
  The expert scheduler uses it for slot arenas.
- `gd_wait_storage` verifies the status of every batch up to a ticket.

Within one submission, a shader reads a buffer either through its SRV or through its UAV, never both.
Bindless dispatches do not record which buffers they touch.
`gd_release_buffer` therefore waits for the last bindless submission.

## Sources

- [Agility SDK release table](https://devblogs.microsoft.com/directx/directx12agility/)
- [Agility SDK 1.722 release](https://devblogs.microsoft.com/directx/agilitysdk-1-620-retail-and-1-722-preview/)
- [LinAlg launch and capability queries](https://devblogs.microsoft.com/directx/d3d12-linalg-preview/)
- [LinAlg runtime specification](https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html)
- [Preview DXC release](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.10.2605.37)
- [DirectStorage 1.3 release](https://devblogs.microsoft.com/directx/directstorage-1-3-is-now-available/)
- [DirectStorage developer guidance](https://github.com/microsoft/DirectStorage/blob/main/Docs/DeveloperGuidance.md)

The current specification is a draft and differs from the released SDK in some structure fields.
This code uses the released headers as its ABI authority.
