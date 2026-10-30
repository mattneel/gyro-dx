# AgentionAI Gyro reference

This directory contains unmodified files from the AgentionAI Gyro reference repository.

| Field | Value |
| --- | --- |
| Repository | https://gitlab.com/agentionai-group/agention-gyro |
| Revision | `587a5269848eb07996c9c1433f9ad638604eb4d8` |
| Specification | Version 1.1 |
| License | Apache License 2.0 |
| Attribution | AgentionAI, Laurent Zuijdwijk |

The source package preserves `LICENSE` and `NOTICE`.
The C source serves as an independent oracle for the Zig decoder.
The source includes the upstream encoder because `tq_ref.c` contains both functions.
The inference code does not call the encoder.

The file `src/gyro_codebook.f16` contains the normative table as little-endian binary16 values.
Its SHA-256 is `ee481f151d8947ce8b08817fdf89d0a56d647b6435ceb3df0f079ea748e0e588`.

The generated fixtures contain C reference results for all eight trellis types.
They also contain transforms for widths 640 and 2560.
The script `tests/gyro_vectors_generate.py` creates these fixtures with a local C compiler.
