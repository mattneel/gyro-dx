# Contributing

Issues and pull requests are welcome.

## Before you open a pull request

1. Install the pinned compiler with `python scripts/bootstrap_zig.py`.
   The required version is in `.zig-version`.
2. Run the portable checks.

   ```sh
   zig build test
   zig build -Doptimize=ReleaseSafe
   python tests/integration.py zig-out/bin/gyro-dx
   ```

3. For changes to `native/`, `shaders/`, or `src/gpu*.zig`, run the Windows build and the hardware checks.
   The README lists them under "Windows build procedure".
4. For changes to the model graph or the kernels, compare tensors against a parity baseline.
   The README describes the procedure under "Parity procedure".
   Report the `tests/parity.py` output, including the greedy-trace comparison.

## Expectations

- Correctness first. Changes to numerics need parity evidence against the CPU engine or the pinned reference runtime.
- Performance claims need `gyro-dx bench` output and the hardware name.
- Keep the C ABI in `native/include/gyro_native.h` and `src/native_abi.zig` in sync.
  Both files assert the struct layouts.
- Match the style of the surrounding code. Do not reformat unrelated code.
- The repository must not contain model weights.

## Reporting bugs

Include the following:

- The `gyro-dx probe` output for GPU issues.
- The model file name and revision.
- The exact command.
- The full error output.
