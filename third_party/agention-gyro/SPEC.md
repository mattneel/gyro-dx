# Gyro tensor format: TQ2_T, TQK4..TQK7, TQP5..TQP7 and block-Hadamard rotation

Specification version 1.1. This document is normative for the byte layout, the decoded
values, and the rotation metadata. `src/tq_ref.c` (`tq_dequantize_row`) is the executable form of it; where
this text and that function disagree, report it as a bug.

Tensor format: trellis-coded quantization with Hadamard rotation, in the family of QTIP
(Tseng et al., 2024). Format, encoder and kernels: AgentionAI (Laurent Zuijdwijk).

## 1. Overview

All eight types quantize a tensor row in **blocks of 128 consecutive weights**. A block is a
binary16 scale `d` followed by a bitstream that drives a **tail-biting bitshift trellis**:

- the 128 weights are 32 **steps** of 4 weights (TQ2_T, TQK: V = 4), or 64 steps of 2
  weights (TQP, the "pair trellis": V = 2);
- each step consumes `K` new bits and has a 16-bit **state** (a window of the stream);
- the low **15** bits of the state select points of a fixed 2048-entry 2-D codebook
  through a multiplicative hash: two points (4 values) per step for V = 4, one point
  (2 values) per step for V = 2;
- the weight is `d * codebook value`.

Weights are normally stored in a rotated basis (section 8); the rotation is declared in the
GGUF metadata per tensor and is separate from the block encoding.

Every possible byte string is a valid block. There is no per-weight sign, offset, or
sub-block scale.

| Type  | GGUF / ggml type id | K (bits per step) | Block bytes | Bits per weight |
|-------|--------------------:|------------------:|------------:|----------------:|
| TQK4  | 151 | 4 | 18 | 1.125 |
| TQK5  | 150 | 5 | 22 | 1.375 |
| TQK6  | 145 | 6 | 26 | 1.625 |
| TQK7  | 146 | 7 | 30 | 1.875 |
| TQ2_T | 144 | 8 | 34 | 2.125 |
| TQP5  | 147 | 5 | 42 | 2.625 |
| TQP6  | 148 | 6 | 50 | 3.125 |
| TQP7  | 149 | 7 | 58 | 3.625 |

Block bytes = `2 + steps*K/8`: `2 + 4*K` for TQ2_T / TQK (32 steps), `2 + 8*K` for TQP
(64 steps). ggml block size (`blck_size`) is 128 for all eight; a row length must be a
multiple of 128.

Informative: the `general.file_type` ("mostly X") values used by the reference runtime
are TQ2_T = 130, TQK6 = 131, TQK7 = 132, TQP5 = 133, TQP6 = 134, TQP7 = 135, TQK5 = 136,
TQK4 = 137.

## 2. Block layout

All multi-byte fields are little-endian.

```
offset  size     field
0       2        d   : IEEE-754 binary16 scale
2       S*K/8    qs  : trellis stream (S steps x K bits; S = 32, or 64 for TQP)
```

C declarations (the reference runtime uses the same):

```c
typedef struct { uint16_t d; uint8_t qs[32]; } block_tq2_t;   // K = 8, 34 bytes
typedef struct { uint16_t d; uint8_t qs[28]; } block_tqk7;    // K = 7, 30 bytes
typedef struct { uint16_t d; uint8_t qs[24]; } block_tqk6;    // K = 6, 26 bytes
typedef struct { uint16_t d; uint8_t qs[20]; } block_tqk5;    // K = 5, 22 bytes
typedef struct { uint16_t d; uint8_t qs[16]; } block_tqk4;    // K = 4, 18 bytes
typedef struct { uint16_t d; uint8_t qs[40]; } block_tqp5;    // K = 5, 42 bytes
typedef struct { uint16_t d; uint8_t qs[48]; } block_tqp6;    // K = 6, 50 bytes
typedef struct { uint16_t d; uint8_t qs[56]; } block_tqp7;    // K = 7, 58 bytes
```

There is no padding. Blocks of a row are stored one after another; rows one after another.

## 3. States

For TQ2_T and TQK, step `t` (t = 0..31) covers weights `4t, 4t+1, 4t+2, 4t+3` of the block;
for TQP, step `t` (t = 0..63) covers weights `2t, 2t+1`. The 16-bit state `s_t` is defined
per type family. In every case only `s_t & 0x7FFF` (the low 15 bits, `TQ_STATE_BITS = 15`)
affects the decoded values.

### 3.1 TQ2_T (K = 8): one byte per step

```
s_t = (qs[(t + 31) % 32] << 8) | qs[t]
```

Byte `qs[t]` is step `t`'s new 8 bits; the high byte is the previous step's byte, with
step 0 wrapping to `qs[31]` (tail-biting).

### 3.2 TQK4..TQK7 (K = 4..7): circular bitstream

`qs` is a circular stream of `N = 32*K` bits, numbered LSB-first:

```
bit(i) = (qs[i >> 3] >> (i & 7)) & 1,   0 <= i < N
```

The state of step `t` is the 16 stream bits starting at bit `off_t = (31 - t) * K`, read
upward and wrapping modulo `N`:

```
s_t = sum_{b=0..15} bit((off_t + b) mod N) << b
```

Because `N` is a multiple of 8 the wrap is a byte wrap, so an implementation can read three
bytes:

```c
uint32_t tqk_state(const uint8_t *qs, int K, int t) {
    uint32_t nbytes = 4*K, off = (31 - t) * K, i0 = off >> 3;
    uint32_t w = qs[i0] | qs[(i0 + 1) % nbytes] << 8 | qs[(i0 + 2) % nbytes] << 16;
    return (w >> (off & 7)) & 0xFFFF;
}
```

Equivalently, step `t`'s new bits ("chunk t") are stream bits `(31-t)*K .. (31-t)*K + K-1`,
so chunk 0 sits at the top of the stream and chunk 31 at bit 0. Moving from step `t` to
`t+1` the window slides down by `K` bits.

### 3.3 TQP5..TQP7 (K = 5..7): circular bitstream, 64 steps

Exactly as 3.2 with 64 steps: `qs` is a circular stream of `N = 64*K` bits (LSB-first as
above) and the state of step `t` (t = 0..63) is the 16 bits starting at
`off_t = (63 - t) * K`, wrapping modulo `N`:

```c
uint32_t tqp_state(const uint8_t *qs, int K, int t) {
    uint32_t nbytes = 8*K, off = (63 - t) * K, i0 = off >> 3;
    uint32_t w = qs[i0] | qs[(i0 + 1) % nbytes] << 8 | qs[(i0 + 2) % nbytes] << 16;
    return (w >> (off & 7)) & 0xFFFF;
}
```

### 3.4 Trellis view (all families)

In all families

```
s_t & 0x7FFF == ((s_{t-1} << K) | v_t) & 0x7FFF,     s_{-1} := s_{S-1}
```

where `v_t` is step `t`'s K new bits (`qs[t]` for TQ2_T, chunk `t` at stream bit
`(S-1-t)*K` for TQK and TQP) and `S` the number of steps. This is a
bitshift trellis with 2^15 effective states, `2^K` branches per state, and a tail-biting
constraint: the state before step 0 is the state of the last step. A decoder never needs this
view; an encoder does (section 7).

## 4. Codebook

The codebook is 2048 two-dimensional points stored as 4096 binary16 values,
`CB[2i], CB[2i+1]` = point `i`. **The table is normative** and is shipped as
`tq_codebook_f16` in `src/tq_codebook.c` (also embedded, base64, in `python/tq_ref.py`).
A decoder must use exactly these bit patterns. SHA-256 of the 8192-byte little-endian
table:

```
ee481f151d8947ce8b08817fdf89d0a56d647b6435ceb3df0f079ea748e0e588
```

(`tests/test_python_parity.py` checks it.)

Informative, origin: the points are a k-means (Lloyd) quantizer of the 2-D standard normal
distribution, computed once from a fixed pseudo-random sample set and rounded to binary16.
Their empirical moments are mean (−0.031, 0.037), variance (1.128, 1.169), range
[−3.98, 3.83]. Because the table is shipped, nothing depends on regenerating it.

## 5. Codebook index

For state `s` and pair `p` (0 = weights 4t, 4t+1; 1 = weights 4t+2, 4t+3):

```
x   = ((s & 0x7FFF) * 0x9E3779B1) mod 2^32      (unsigned 32-bit wrapping multiply)
idx = p == 0 ? x >> 21 : (x >> 10) & 2047
```

One multiply serves both pairs. The index is always in 0..2047.

## 6. Decoded values

```
d     = binary16_to_float32(block.d)
TQ2_T, TQK:                             # V = 4
  for t in 0..31:
      s = state(t)                      # section 3
      for j in 0..3:
          i = index(s, j >> 1)          # section 5
          w[4t + j] = d * binary16_to_float32(CB[2*i + (j & 1)])
TQP:                                    # V = 2: only pair 0 of each state is used
  for t in 0..63:
      s = tqp_state(t)
      i = index(s, 0)
      for j in 0..1:
          w[2t + j] = d * binary16_to_float32(CB[2*i + j])
```

The product is a single IEEE-754 binary32 multiplication (round to nearest even) of two
exactly converted binary16 values; with no fused or reassociated arithmetic this is
bit-exact across platforms. Runtimes that compute in binary16 or accumulate dot products
directly will of course differ in the last bits of their outputs; the definition above is
what "the decoded tensor" means.

`d` may be any binary16 value; encoders produce finite values (zero for an all-zero block).
A decoder should not special-case `d`.

## 7. Encoding (informative)

Any byte string decodes, so any search over the stream is a valid encoder. The reference
encoder (`tq_quantize_row_ref`) does, per block:

1. `sigma = sqrt(sum w_j x_j^2 / sum w_j)` with per-weight importances `w_j` (all 1 if none
   are given). An all-zero block is written as `d = 0`, all-zero stream.
2. For each candidate scale `d0 = sigma * g`, `g` in {0.70, 0.78, 0.86, 0.94, 1.02, 1.10,
   1.20, 1.32}, and one more at the best refitted scale so far:
   - target `y = x / d0`; branch cost of step `t` in 15-bit state `s` is
     `sum_j w_{Vt+j} (y_{Vt+j} - c_j(s))^2` over the step's V weights;
   - **tail-biting Viterbi** (section 3.4): one free-start pass; its 4 best final states
     `c` are each tried as a fixed start/end state (`s_{-1} = s_{S-1} = c`), and the cheapest
     of those valid tail-biting paths is kept;
   - refit `d` by weighted least squares on the chosen path, round it to binary16, and
     score the weighted error at the rounded scale.
3. Keep the best (path, d); write `v_t = s_t & (2^K - 1)` for each step.

The candidate-start method is a heuristic: it always returns a valid tail-biting path, but
not necessarily the optimal one. The importance vector is per weight, so an imatrix-style
per-column weight is passed by repeating it for every row.

## 8. Block-Hadamard rotation

Gyro files store selected weight matrices in a rotated basis, so that the quantized values
are close to Gaussian. The rotation is declared in GGUF metadata under the
`prism.hadamard.*` keys (the key namespace the reference runtime reads; the names are given
here exactly as they appear in files). It is independent of the block encoding: a rotated
tensor is quantized and decoded like any other, and the rotation applies to the decoded
values.

### 8.1 The transform

Let `b` be the block size and `s` a ±1 sign vector of length `n` (the tensor's input
width). For a vector `v` of length `n` (a multiple of `b`),

```
T v = H_b (s ⊙ v)        applied independently to each slice v[k*b .. k*b + b-1]
H_b[r][c] = (-1)^popcount(r & c) / sqrt(b)       0 <= r, c < b
```

`H_b` is the normalised Sylvester-Walsh-Hadamard matrix. It is symmetric and orthogonal
(`H_b H_b = I`), so `T^-1 v = s ⊙ (H_b v)` and `(T w) · (T x) = w · x`. `H_b v` can be
computed with the standard in-place fast Walsh-Hadamard butterfly followed by a multiply by
`1/sqrt(b)`; `tq_hadamard_rotate` / `tq_hadamard_unrotate` in `src/tq_ref.c` do this.

### 8.2 What is stored, and what a runtime does

The axis is the GGUF innermost dimension `ne[0]`, i.e. the input features of the matmul
(`"input-last-dimension"`). For a weight `W` with rows `w` of length `ne[0]` (for 3-D
expert tensors: every row of every expert), a listed tensor stores, row by row,

```
w_stored = T w = H_b (s ⊙ w)                 (equivalently W_stored = W D H, D = diag(s))
```

A runtime has two equivalent options:

- **rotate the input** (what the reference runtime does): before the matmul with a listed
  weight, compute `x' = T x = H_b (s ⊙ x)` (multiply the activation elementwise by the sign
  vector for that weight's width, then apply `H_b` blockwise along the feature dimension),
  and use `W_stored x'`, which equals `W x`; the same `x'` serves every listed weight that
  reads the same `x` (for example gate and up projections);
- **unrotate the weights** once at load time: `w = T^-1 w_stored = s ⊙ (H_b w_stored)` per
  row, then use the weights as plain weights.

Rotation is per tensor: tensors not listed are stored in the plain basis.

### 8.3 Metadata keys

| Key | GGUF type | Value |
|-----|-----------|-------|
| `prism.hadamard.version` | uint32 | `1` (see 8.4 for `2`) |
| `prism.hadamard.block_size` | uint32 | `b`, a power of two (Gyro files: 128); must divide `ne[0]` of every listed tensor |
| `prism.hadamard.transform` | string | `"normalized-sylvester-walsh-hadamard"` (only value) |
| `prism.hadamard.axis` | string | `"input-last-dimension"` (only value) |
| `prism.hadamard.sign_mode` | string | `"explicit"` (Gyro files), or `"identity"`: `s` = all +1 and the sign keys are absent |
| `prism.hadamard.sign_widths` | array of int32 | the distinct input widths that have a sign vector, each a positive multiple of `b` |
| `prism.hadamard.sign_values` | array of int32 | the sign vectors concatenated in the order of `sign_widths`; every value is +1 or -1; total length = sum of `sign_widths` |
| `prism.hadamard.weight_names` | array of string | names of the tensors stored rotated (8.2); non-empty |

The sign vector of a listed tensor is the one whose width equals the tensor's `ne[0]`
(one vector per width, shared by every tensor of that width); with `"explicit"` a listed
tensor whose width has no vector is an error. Readers should reject other values of
`version`, `transform`, `axis` and `sign_mode`, duplicate widths, and sign values other
than ±1.

Example (header of a released Gyro file, abbreviated):

```
prism.hadamard.version      = 1
prism.hadamard.block_size   = 128
prism.hadamard.transform    = "normalized-sylvester-walsh-hadamard"
prism.hadamard.axis         = "input-last-dimension"
prism.hadamard.sign_mode    = "explicit"
prism.hadamard.sign_widths  = [640, 2560]
prism.hadamard.sign_values  = [-1, -1, 1, 1, 1, -1, ...]          (3200 values)
prism.hadamard.weight_names = ["blk.0.ffn_down_exps.weight",
                               "blk.0.ffn_gate_exps.weight", ...]  (144 names)
```

Here the down-projection experts (input width 640) use the 640-wide vector and the gate /
up experts (input width 2560) the 2560-wide one.

### 8.4 Further keys read by the reference runtime

Current Gyro releases do not use these; they are listed so that a reader can recognise
(or reject) them.

- `prism.hadamard.inverse_weight_names` (array of string, optional): tables read by row
  lookup and stored rotated; only `"token_embd.weight"` is accepted. The runtime applies
  the inverse after the lookup: `h = s ⊙ (H_b z)` for a looked-up row `z`.
- `prism.hadamard.tied_output` (bool, optional): with `version = 2` (and only then), the
  output head is tied to a rotated token embedding listed in `inverse_weight_names`, and
  `output.weight` must be absent; the head then takes the rotated input like any listed
  weight.
- `prism.hadamard.gdn_v_grouped` (bool, optional): for listed `ssm_out` tensors of
  gated-delta-net layers, the input features are permuted before the signs and `H_b`:
  with `hd = ne[0] / n_v`, `nk` = number of key groups and `rep = n_v / nk`, the feature at
  `a + hd*(i_k + nk*i_r)` moves to `a + hd*(i_r + rep*i_k)` (tiled `[hd, nk, rep]` to
  grouped `[hd, rep, nk]` order). The reference code here does not implement this.

## 9. GGUF metadata

A GGUF file containing any of the trellis types sets

```
agention.trellis.state_bits : uint32 = 15
```

The reference runtime refuses TQ tensors when this key is missing or has another value.
It records the 15-bit state width of section 3; no other value is defined.

## 10. Worked example

Produced by `build/tq_tool example`; values are printed to 6 decimals (they are exact
binary16 codebook values times `d`).

### 10.1 TQK4 block, d = 1.0

```
bytes: 00 3c 07 18 29 3a 4b 5c 6d 7e 8f a0 b1 c2 d3 e4 f5 06
       \d=1.0/ \---------------- qs[0..15] ------------------/
```

Step 0: `off = 31*4 = 124`, `i0 = 15`, `w = qs[15] | qs[0]<<8 | qs[1]<<16 = 0x180706`,
`s_0 = (0x180706 >> 4) & 0xFFFF = 0x8070`, `s & 0x7FFF = 0x0070`,
`x = 0x70 * 0x9E3779B1 mod 2^32 = 0x38453D70`, `idx0 = x >> 21 = 450`,
`idx1 = (x >> 10) & 2047 = 335`, so weights 0..3 are `CB[900], CB[901], CB[670], CB[671]`.

```
  t  state  &0x7fff  idx0  idx1   w[4t..4t+3]
  0  0x8070  0x0070   450   335    0.986816  0.455322 -0.211914  0.653809
  1  0x0706  0x0706   461    44   -0.363281 -0.655762  1.682617  0.349121
  2  0x706f  0x706f  1786   716   -1.328125 -1.093750  1.550781  0.284668
  3  0x06f5  0x06f5  1471  1127   -0.833008 -0.468506 -0.644531  3.388672
  4  0x6f5e  0x6f5e   305   155   -0.148438 -1.610352 -0.192505  1.069336
  5  0xf5e4  0x75e4   544   401    0.129639  0.837402  2.464844 -1.177734
  6  0x5e4d  0x5e4d  1962  1958   -0.071106  0.053070  0.496826  0.335449
  7  0xe4d3  0x64d3   154   156    1.415039 -1.458008  2.027344 -2.001953
  8  0x4d3c  0x4d3c  1572  1712   -0.819824 -0.303955  0.279785  0.704590
  9  0xd3c2  0x53c2  1811  1958   -0.999023 -1.158203  0.496826  0.335449
 10  0x3c2b  0x3c2b  1182  1467    1.833984  1.565430  3.314453  1.412109
 11  0xc2b1  0x42b1  1421  1713   -0.385742 -2.082031 -0.431641 -0.634277
 12  0x2b1a  0x2b1a   792  1221   -1.070312 -0.020416 -1.464844 -0.658203
 13  0xb1a0  0x31a0  1031  1467   -0.682129 -1.320312  3.314453  1.412109
 14  0x1a08  0x1a08  1184  1521   -0.704102  1.626953  0.290527 -1.017578
 15  0xa08f  0x208f   641  1222    0.681152 -0.107117  2.207031  0.443115
 16  0x08f7  0x08f7   794  1276   -0.064697 -0.915039 -0.574707 -0.947754
 17  0x8f7e  0x0f7e   251   977   -1.401367 -0.163696 -3.265625 -0.301270
 18  0xf7e6  0x77e6  1915   551   -0.219971 -0.269775 -0.038788 -1.349609
 19  0x7e6d  0x7e6d  1372   251    1.355469  1.356445 -1.401367 -0.163696
 20  0xe6d5  0x66d5  1525   305    0.908691  0.213745 -0.148438 -1.610352
 21  0x6d5c  0x6d5c   982     6    0.138916  0.248901 -0.645996  1.152344
 22  0xd5c4  0x55c4  1135    60   -0.567383  0.310791  0.065613 -1.361328
 23  0x5c4b  0x5c4b   591  1808    0.915527 -1.282227 -1.073242  0.343506
 24  0xc4b3  0x44b3   744  1862    1.282227 -0.937988  1.164062 -0.139893
 25  0x4b3a  0x4b3a   201  1563    0.275635  2.220703  0.523926  0.680176
 26  0xb3a2  0x33a2   354  1617    0.055969 -0.081543  1.324219 -0.149170
 27  0x3a29  0x3a29  1859  1317   -2.154297 -0.731934  2.232422  0.005775
 28  0xa291  0x2291  2012  1371    1.078125 -2.488281  0.814941 -0.323730
 29  0x2918  0x2918  1469  1072    0.457275  0.086792  0.069519 -0.147217
 30  0x9180  0x1180  1622  1126   -1.035156 -0.640625  1.830078  0.432617
 31  0x1807  0x1807  1079   826    1.384766  2.029297  0.327148 -1.681641
```

Note step 31: `off = 0`, so its state is simply `qs[0] | qs[1] << 8 = 0x1807`, and step 0's
window (bits 124..139) wraps from the top byte into `qs[0]`, `qs[1]`.

### 10.2 TQ2_T block, d = 0.5

```
bytes: 00 38 | 0b 30 55 7a 9f c4 e9 0e 33 58 7d a2 c7 ec 11 36
               5b 80 a5 ca ef 14 39 5e 83 a8 cd f2 17 3c 61 86
```

```
  t  state  &0x7fff  idx0  idx1   w[4t..4t+3]
  0  0x860b  0x060b   201  1816    0.137817  1.110352 -0.016190  0.334961
  1  0x0b30  0x0b30   101    91    0.541016 -0.452148 -0.084961 -0.065308
  2  0x3055  0x3055  1913  1829    0.097473  0.123352 -0.097412 -0.225952
  ...
 30  0x3c61  0x3c61  1948   678    0.111755  1.028320  0.513184 -0.323975
 31  0x6186  0x6186  1713   368   -0.215820 -0.317139  1.118164 -0.260742
```

Step 0's high byte is `qs[31] = 0x86` (tail-biting); its top bit is masked off.

### 10.3 TQP5 block, d = -2.0

```
bytes: 00 c0 03 68 cd 32 97 fc 61 c6 2b 90 f5 5a bf 24 89 ee 53 b8 1d 82 e7 4c b1 16 7b e0 45 aa 0f 74 d9 3e a3 08 6d d2 37 9c 01 66
```

Step 0: `off = 63*5 = 315`, `i0 = 39`, `w = qs[39] | qs[0]<<8 | qs[1]<<16 = 0x680366`,
`s_0 = (0x680366 >> 3) & 0xFFFF = 0x006C`; only `idx0` is used.

```
  t  state  &0x7fff  idx0   w[2t..2t+1]
  0  0x006c  0x006c  1531    0.792969 -2.251953
  1  0x0d98  0x0d98  1552    2.339844  0.539062
  2  0xb300  0x3300   105   -1.109375  2.826172
  ...
 62  0x6b40  0x6b40  1517   -0.012222 -2.412109
 63  0x6803  0x6803   800    2.078125 -1.284180
```

### 10.4 Rotation

For `b = 4`, `s = (1, -1, 1, 1)` and `v = (1, 2, 3, 4)`: `s ⊙ v = (1, -2, 3, 4)`,
`H_4 (s ⊙ v) = (1-2+3+4, 1+2+3-4, 1-2-3-4, 1+2-3+4) / 2 = (3, 1, -4, 2)`. Unrotating:
`H_4 (3, 1, -4, 2) = (3+1-4+2, 3-1-4-2, 3+1+4-2, 3-1+4+2)/2 = (1, -2, 3, 4)`, times `s`
gives `(1, 2, 3, 4)`.

## 11. Conformance

An implementation conforms if, for every block, it produces the values of section 6, and
applies or undoes the rotation of section 8 for every tensor listed in
`prism.hadamard.weight_names`. The tests in this repository check the C decoder, the
Python decoder and the reference runtime's decoder against each other bit for bit on
random blocks (random stream bytes, random finite scales including negative and subnormal
ones) and on encoder output, and check the rotation against the dense definition of
`H_b`, for round-trip identity, and for dot-product invariance.

## 12. Design notes (informative)

Why the format looks the way it does. Gyro and EXL3 both descend from QTIP (Tseng et al.,
2024); Gyro is an independent design for GGUF runtimes (llama.cpp and engines that read
GGUF) on a wide range of hardware: AMD and NVIDIA GPUs, unified-memory machines and CPU
offload of mixture-of-experts layers.

- **15-bit state.** The trellis state is 15 bits rather than QTIP's 16. Halving the state
  space (32,768 states) keeps the per-state decode tables small: a full state-to-four-weights
  table is 32,768 x 4 int16 (256 KiB), and the GPU paths need only the 8 KiB codebook. In
  our kernels this made decoding noticeably faster for a small quality cost, which is the
  trade we wanted for bandwidth-bound decoding.
- **Explicit 2,048-point codebook.** A weight pair is a table lookup selected by a
  multiplicative hash of the state (section 5), not a value computed per weight. One 32-bit
  multiply serves both pairs of a step, and the table fits in shared memory or L1. Fixing
  the table in the format (with its SHA-256) makes every decoder bit-exact.
- **128-weight blocks with one fp16 scale.** Small enough to follow local scale changes in
  expert rows, large enough that the scale costs 0.125 bits per weight. Rows of every common
  hidden width divide into whole blocks, and blocks are independent: a block can be decoded
  without its neighbours, which suits row-parallel GPU kernels and CPU vector dots alike.
- **Tail-biting streams.** The state window wraps around the block's own stream, so no bits
  are spent on a start state and every byte string is a valid block.
- **One family, many rates.** TQK4..TQK7, TQ2_T and the pair-trellis TQP5..TQP7 share the
  codebook, hash and stream rules and differ only in K (and V for TQP). A runtime that
  implements one implements all eight with a template parameter, and a model can mix rates
  per tensor to spend bits where they matter.
- **Rotation declared, not baked in.** The block-Hadamard rotation is metadata (section 8):
  a runtime rotates the matmul input or unrotates the weights once, and tensors that do not
  benefit are simply not listed.
