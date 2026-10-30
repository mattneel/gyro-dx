# Verified model and format

## Scope

This package implements the complete Qwen4Exp graph for the specified model on the CPU and on Direct3D 12.
The source package contains no model weights.

The measurements below come from the two GGUF headers at one pinned repository revision.
The byte totals exclude runtime buffers unless a row states otherwise.
The GPU memory figures in the model card describe the publisher's runtime.

## Source pins

| Source | Revision | Purpose |
| --- | --- | --- |
| [Gyro-Coder GGUF files](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF/tree/b7d23f5036e9134b3b84e20e7040ba47bc5e0b4c) | `b7d23f5036e9134b3b84e20e7040ba47bc5e0b4c` | Actual tensor metadata and file sizes |
| [Base configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Upstream architecture context |
| [Gyro format specification](https://gitlab.com/agentionai-group/agention-gyro/-/blob/587a5269848eb07996c9c1433f9ad638604eb4d8/SPEC.md) | `587a5269848eb07996c9c1433f9ad638604eb4d8` | Format, codebook, transforms, and C oracle |
| [Qwen4Exp runtime graph](https://gitlab.com/agentionai-group/llama.cpp/-/blob/9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f/src/models/qwen4exp.cpp) | `9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f` | Model operations and PLE hash semantics |
| [PLE disk reader](https://gitlab.com/agentionai-group/llama.cpp/-/blob/9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f/src/llama-ple-disk.cpp) | Same runtime revision | Existing row cache and I/O behavior |

The model repository replaced the Coder-L file under the same filename on October 29, 2026.
The pinned file declares version `v1.2`.
The filename alone does not identify the expert set.

## Header evidence

| Property | Gyro-Coder | Gyro-Coder-L v1.2 |
| --- | ---: | ---: |
| File bytes | 75,796,732,256 | 83,975,625,408 |
| Header bytes, including alignment | 11,042,656 | 11,043,008 |
| Tensors | 1,224 | 1,224 |
| Metadata entries | 77 | 83 |
| Expert tensors | 144 | 144 |
| N-gram table bytes | 54,400,261,120 | 54,400,261,120 |
| Expert weight bytes | 16,043,212,800 | 24,222,105,600 |
| Other weight bytes | 5,342,215,680 | 5,342,215,680 |
| Weights without the n-gram table | 19.9167 GiB | 27.5339 GiB |

Both header requests returned HTTP 206 with the requested byte range.
The parser used the complete file size from `Content-Range`.
It did not infer the file size from the downloaded prefix.

The GGUF tensors contain 116,514,464,640 elements, including 51,200,245,760 elements in the n-gram table.
The remaining tensors contain 65,314,218,880 elements.
These counts explain the difference between the Hub's 117B summary and the card's 65B resident subset.

| Header | SHA-256 |
| --- | --- |
| Coder | `586e30b73a4e362ae5d15c72d30796303253b116d88c1dca7b970b9d115f9eb8` |
| Coder-L | `5b14b64c3a371f0a45ac2fc90125a1488fa13a5723e0ce162a3f3ab54da4e997` |

## Architecture

The GGUF architecture name is `qwen4exp`.
The model is not a conventional decoder with only dense attention and MoE blocks.
The runtime graph also contains hyper-connections and a PLE module.

| Feature | Verified value |
| --- | --- |
| Main layers | 48 |
| Hidden width | 2,560 |
| Local experts per layer | 256 |
| Active experts per token per layer | 10 |
| Expert intermediate width | 640 |
| Shared expert intermediate width | 640 |
| Linear attention layers | 36 |
| Full attention layers | 12, every fourth layer |
| Full attention heads | 24 |
| Full attention KV heads | 2 |
| Full attention head width | 256 |
| Rotary width | 64 |
| Linear attention key heads | 16 |
| Linear attention value heads | 48 |
| Linear attention head width | 128 |
| Linear attention convolution width | 4 |
| Hyper-connection streams | 4 |
| Hyper-connection low rank | 320 |
| Sparse indexer heads | 4 |
| Sparse indexer head width | 128 |
| Sparse indexer budget | 2,048 |
| Sparse indexer compression ratio | 4 on full attention layers |
| Vocabulary entries | 248,320 |
| PLE GGUF layer index | 1, zero-based |

Each main layer contains two hyper-connection modules.
The first surrounds attention.
The second surrounds the expert block.
The final hyper-connection module supplies the output normalization and stream combination.

The routed expert block uses softmax routing and ten selected experts.
The runtime normalizes the selected weights.
The shared expert block uses a separate sigmoid gate.
The expert activation is SiLU.

The linear attention path includes a convolution and a gated delta recurrence.
The full attention path includes a separate sparse indexer.
These operations require additional kernels beyond matrix multiplication.
The complete graph also requires tokenization, sampling, cache semantics, and output formatting.

## Expert tensor layout

The GGUF innermost dimension appears first in the following table.
The expert index occupies the third dimension.
Each expert slice contains contiguous rows within one projection tensor.

| Tensor pattern | Dimensions | Matrix interpretation |
| --- | --- | --- |
| `blk.L.ffn_gate_exps.weight` | `[2560, 640, 256]` | 640 output features by 2,560 input features |
| `blk.L.ffn_up_exps.weight` | `[2560, 640, 256]` | 640 output features by 2,560 input features |
| `blk.L.ffn_down_exps.weight` | `[640, 2560, 256]` | 2,560 output features by 640 input features |
| `blk.L.ffn_gate_inp.weight` | `[2560, 256]` | Router for the 256 retained experts |

The three projection tensors occupy separate file ranges.
A complete expert therefore requires three source ranges.
The original file does not store a combined gate, up, and down record for each expert.

Coder uses `TQ2_T` for all 144 expert tensors.
One projection slice occupies 435,200 bytes.
One complete expert occupies 1,305,600 bytes.
The slice calculation uses quantization block bytes and dimensions.

Coder-L uses three pair-trellis types across its layers.
Its layer allocation is shown below.

| Zero-based layers | Type | Bytes per projection per expert | Bytes per complete expert |
| --- | --- | ---: | ---: |
| 0–15, 24–31 | `TQP6` | 640,000 | 1,920,000 |
| 16–23, 40–47 | `TQP7` | 742,400 | 2,227,200 |
| 32–39 | `TQP5` | 537,600 | 1,612,800 |

The runtime uses local expert IDs from zero through 255.
The header does not supply a map to the original 512 expert IDs.
The file does not contain the removed experts.
An engine cannot recover them through a cache policy.

## Gyro encoding

The model filenames use size-class labels.
`TQ2_0` and `TQ3_0` in those filenames do not specify the tensor decoder.
The actual tensor IDs determine the format.

| Format | GGUF type ID | Weights per block | Block bytes |
| --- | ---: | ---: | ---: |
| `TQ2_T` | 144 | 128 | 34 |
| `TQK6` | 145 | 128 | 26 |
| `TQK7` | 146 | 128 | 30 |
| `TQP5` | 147 | 128 | 42 |
| `TQP6` | 148 | 128 | 50 |
| `TQP7` | 149 | 128 | 58 |
| `TQK5` | 150 | 128 | 22 |
| `TQK4` | 151 | 128 | 18 |

Each block starts with one binary16 scale.
The remaining bytes form a circular trellis stream.
A 15-bit state and a fixed multiplicative hash select values from the normative codebook.
The reference decoded value is one binary32 product of the scale and codebook value.

The codebook occupies 8,192 bytes.
Its SHA-256 is `ee481f151d8947ce8b08817fdf89d0a56d647b6435ceb3df0f079ea748e0e588`.
All eight types require `agention.trellis.state_bits = 15`.
The C reference source and codebook use Apache License 2.0.
The source package preserves the required notice.

The Zig decoder matches 65,536 finite-scale values from 512 independent C oracle blocks, bit for bit.
The fixtures cover every supported type.
They include negative scales and binary16 subnormal scales.
The comparison does not claim bit parity for NaN payload propagation across platforms.

The scalar HLSL reference kernel uses the same packed codebook and trellis definitions.
It emits 128 float32 values per block and leaves rotation separate.
The kernel compiled with DXC for Shader Model 6.0.
On the RTX 5090 Laptop GPU, the scalar kernel matches the CPU decoder bit for bit for all eight types.
The inference engine uses a fused decoder and matrix-vector kernel in `shaders/engine/moe.hlsl`.

## Rotation

Rotation is a separate per-tensor property.
The `prism.hadamard.weight_names` array identifies the tensors with a rotated basis.
Unlisted tensors retain their plain basis, even when their format is a trellis type.

Both verified files list all 144 expert tensors.
They use a normalized Sylvester-Walsh-Hadamard transform with blocks of 128 input features.
Explicit sign vectors cover widths 640 and 2,560.
The transform operates along GGUF dimension `ne[0]`.

For a sign matrix `D` and normalized block Hadamard matrix `H`, the stored weight matrix is `W D H`.
The corresponding activation is `H D x`.
Their product equals `W x` because `H H = I` and `D D = I`.
An alternative decoder can apply the inverse transform to each weight row.

The gate and up projections share one transformed input.
The down projection needs a separate transform after the nonlinear expert activation.
The transform cannot move through SiLU or the elementwise gate product.
The decoder therefore leaves values in their stored basis until the caller applies the required transform.

## PLE n-gram table

The tensor `per_layer_token_embd.weight` has dimensions `[160, 320001536]`.
It uses `Q8_0`, with 34 bytes per block of 32 values.
Each row therefore occupies 170 bytes.
The complete table occupies 50.6642 GiB.

Each token selects eight 2-gram rows and eight 3-gram rows.
The useful read volume is 2,720 bytes per token before page expansion.
The hash uses the current token and two predecessors.
It does not depend on the hidden activation or the expert router.

The hash for length `n` is:

```text
mixed = (token[0] * multiplier[0]) XOR ... XOR (token[n-1] * multiplier[n-1])
row[head] = mixed % head_vocab_size[head] + head_offset[head]
```

All products wrap as unsigned 64-bit arithmetic.
The multipliers and head ranges come from GGUF metadata.
A missing predecessor acts as the configured PLE EOS token.
An EOS predecessor also replaces every older predecessor with EOS.
An EOS current token retains its own earlier context for this lookup.

The PLE EOS token is 248044 in the verified files.
The engine must not substitute its chat stop token for this value.
The GGUF PLE layer index is authoritative for this artifact.

After row lookup, the graph applies learned key and value projections.
It combines the result with the hyper-connection state through grouped normalization and a gate.
It also applies a causal convolution with width four and dilation three.
The n-gram table is therefore part of the learned model graph.
It is not a speculative text cache.

The address module supports the verified 3-gram, eight-head configuration.
It rejects other configurations explicitly.
Its page helper expands useful ranges and merges adjacent or overlapping pages.
It bounds the final page by the actual file size.

Accepted token IDs expose future PLE reads before the GPU reaches the PLE layer.
This gives PLE prefetch a stronger dependency guarantee than expert prediction.
A page cache can serve repeated token windows.
Random row access still requires latency measurements on the target NVMe device.

## Storage limits

Coder requires 626,688,000 expert bytes per generated token when every selected expert misses the cache.
Coder-L requires 946,176,000 bytes under the same assumption.
These totals include all 48 layers and ten active experts per layer.
They exclude page expansion, metadata, and repeated reads.

At 7 GB/s of sustained useful expert traffic, the ideal storage ceilings are approximately 11.2 and 7.4 tokens per second.
These figures are calculations, not measured engine speeds.
Latency and serialized layer dependencies reduce the achievable rate.
Cache hits reduce the required expert traffic.

The compact Coder weights already fit the publisher's 24 GB configuration.
Full expert residency is therefore a relevant baseline.
Expert eviction targets smaller cards, larger contexts, or a larger model variant.
DirectStorage does not remove the cost of a cache miss.

The raw compressed expert cache preserves the model's memory benefit.
A full FP16 expansion defeats that benefit.
LinAlg execution needs a supported numeric operand format, so decoded tiles require a bounded scratch allocation.
Tile decode, transform, matrix execution, and I/O overlap require target-device measurements.
