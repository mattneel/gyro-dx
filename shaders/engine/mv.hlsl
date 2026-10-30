// Matrix-vector products for up to 8 activation rows: y[t][r] = sum_i W[r][i] * x[t][i].
// One wave computes one output row for every token, so each weight row is read once.
#include "common.hlsli"

cbuffer C : register(b0) {
    uint w_srv; uint w_off; uint rows; uint cols;
    uint x_uav; uint x_off; uint x_stride; uint n_tok;
    uint y_uav; uint y_off; uint y_stride; uint mode;
    uint act; float scale; uint row_bytes; uint rows_per_wave;
};

#define MAX_TOK 8
#define SHARED_X 8192

// Activations staged once per group when they fit; rows of the group then read groupshared.
groupshared float gx[SHARED_X];

bool stage_x(uint tid) {
    uint total = n_tok * cols;
    if (total > SHARED_X) return false;
    RWByteAddressBuffer x = uav(x_uav);
    for (uint i = tid; i < total; i += 256) {
        uint t = i / cols;
        gx[i] = ldf(x, x_off + t * x_stride + (i - t * cols));
    }
    GroupMemoryBarrierWithGroupSync();
    return true;
}

float4 x4(RWByteAddressBuffer x, bool staged, uint t, uint c) {
    if (staged) {
        uint i = t * cols + c;
        return float4(gx[i], gx[i + 1], gx[i + 2], gx[i + 3]);
    }
    return asfloat(x.Load4((x_off + t * x_stride + c) * 4));
}

void finish(uint row, uint lane, float acc[MAX_TOK]) {
    RWByteAddressBuffer y = uav(y_uav);
    [unroll] for (uint t = 0; t < MAX_TOK; ++t) {
        if (t < n_tok) {
            float total = WaveActiveSum(acc[t]);
            if (lane == 0) {
                uint idx = y_off + t * y_stride + row;
                float v = epilogue(total, act, scale);
                if (mode == 1) v += ldf(y, idx);
                stf(y, idx, v);
            }
        }
    }
}

// Q8_0 in the repacked device layout: each row holds `cols` int8 values followed by cols / 32
// binary16 scales (see q8_repack). A lane reads 16 consecutive quants per step, so a wave
// reads 512 contiguous bytes.
[WaveSize(32)]
[numthreads(256, 1, 1)]
void mv_q8(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint lane = WaveGetLaneIndex();
    uint wave = tid.x / WAVE;
    ByteAddressBuffer w = srv(w_srv);
    RWByteAddressBuffer x = uav(x_uav);
    bool staged = stage_x(tid.x);
    for (uint rr = 0; rr < rows_per_wave; ++rr) {
        uint row = (gid.x * 8 + wave) * rows_per_wave + rr;
        if (row >= rows) return;
        float acc[MAX_TOK];
        [unroll] for (uint t0 = 0; t0 < MAX_TOK; ++t0) acc[t0] = 0;
        uint base = w_off + row * row_bytes;
        for (uint c = lane * 16; c < cols; c += WAVE * 16) {
            uint4 q = w.Load4(base + c);
            float d = load_half(w, base + cols + (c / 32) * 2);
            float v[16];
            [unroll] for (uint k = 0; k < 4; ++k) {
                uint word = q[k];
                v[k * 4 + 0] = d * float(sext8(word & 0xff));
                v[k * 4 + 1] = d * float(sext8((word >> 8) & 0xff));
                v[k * 4 + 2] = d * float(sext8((word >> 16) & 0xff));
                v[k * 4 + 3] = d * float(sext8(word >> 24));
            }
            [unroll] for (uint t = 0; t < MAX_TOK; ++t) {
                if (t < n_tok) {
                    float sum = 0;
                    [unroll] for (uint k2 = 0; k2 < 4; ++k2) {
                        float4 xv = x4(x, staged, t, c + k2 * 4);
                        sum += v[k2 * 4] * xv.x + v[k2 * 4 + 1] * xv.y + v[k2 * 4 + 2] * xv.z + v[k2 * 4 + 3] * xv.w;
                    }
                    acc[t] += sum;
                }
            }
        }
        finish(row, lane, acc);
    }
}

// Converts GGUF Q8_0 rows (34-byte blocks) into the repacked layout read by mv_q8.
// Thread per block. Constants reuse the matrix-vector layout: w_srv source, y_uav destination.
[numthreads(256, 1, 1)]
void q8_repack(uint3 dtid : SV_DispatchThreadID) {
    uint blocks_per_row = cols / 32;
    uint index = dtid.x + dtid.y * 65535 * 256;
    if (index >= rows * blocks_per_row) return;
    uint row = index / blocks_per_row;
    uint b = index % blocks_per_row;
    ByteAddressBuffer src = srv(w_srv);
    RWByteAddressBuffer dst = uav(y_uav);
    uint so = w_off + row * blocks_per_row * 34 + b * 34;
    uint dro = y_off + row * row_bytes;
    [unroll] for (uint k = 0; k < 8; ++k) {
        uint word = load_byte(src, so + 2 + k * 4) | (load_byte(src, so + 3 + k * 4) << 8) |
            (load_byte(src, so + 4 + k * 4) << 16) | (load_byte(src, so + 5 + k * 4) << 24);
        dst.Store(dro + b * 32 + k * 4, word);
    }
    // Two scales share a dword; the even block writes both.
    if ((b & 1) == 0) {
        uint lo = load_byte(src, so) | (load_byte(src, so + 1) << 8);
        uint hi = b + 1 < blocks_per_row ? (load_byte(src, so + 34) | (load_byte(src, so + 35) << 8)) : 0;
        dst.Store(dro + cols + b * 2, lo | (hi << 16));
    }
}

// BF16 rows: a lane reads eight values (16 bytes) per step.
[WaveSize(32)]
[numthreads(256, 1, 1)]
void mv_bf16(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint lane = WaveGetLaneIndex();
    uint wave = tid.x / WAVE;
    ByteAddressBuffer w = srv(w_srv);
    RWByteAddressBuffer x = uav(x_uav);
    bool staged = stage_x(tid.x);
    for (uint rr = 0; rr < rows_per_wave; ++rr) {
        uint row = (gid.x * 8 + wave) * rows_per_wave + rr;
        if (row >= rows) return;
        float acc[MAX_TOK];
        [unroll] for (uint t0 = 0; t0 < MAX_TOK; ++t0) acc[t0] = 0;
        uint base = w_off + row * row_bytes;
        for (uint c = lane * 8; c < cols; c += WAVE * 8) {
            uint4 raw = w.Load4(base + c * 2);
            float v[8];
            v[0] = asfloat(raw.x << 16); v[1] = asfloat(raw.x & 0xffff0000);
            v[2] = asfloat(raw.y << 16); v[3] = asfloat(raw.y & 0xffff0000);
            v[4] = asfloat(raw.z << 16); v[5] = asfloat(raw.z & 0xffff0000);
            v[6] = asfloat(raw.w << 16); v[7] = asfloat(raw.w & 0xffff0000);
            [unroll] for (uint t = 0; t < MAX_TOK; ++t) {
                if (t < n_tok) {
                    float4 xa = x4(x, staged, t, c), xb = x4(x, staged, t, c + 4);
                    acc[t] += v[0] * xa.x + v[1] * xa.y + v[2] * xa.z + v[3] * xa.w +
                              v[4] * xb.x + v[5] * xb.y + v[6] * xb.z + v[7] * xb.w;
                }
            }
        }
        finish(row, lane, acc);
    }
}

// F32 rows: a lane reads four values per step.
[WaveSize(32)]
[numthreads(256, 1, 1)]
void mv_f32(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint lane = WaveGetLaneIndex();
    uint wave = tid.x / WAVE;
    ByteAddressBuffer w = srv(w_srv);
    RWByteAddressBuffer x = uav(x_uav);
    bool staged = stage_x(tid.x);
    for (uint rr = 0; rr < rows_per_wave; ++rr) {
        uint row = (gid.x * 8 + wave) * rows_per_wave + rr;
        if (row >= rows) return;
        float acc[MAX_TOK];
        [unroll] for (uint t0 = 0; t0 < MAX_TOK; ++t0) acc[t0] = 0;
        uint base = w_off + row * row_bytes;
        for (uint c = lane * 4; c < cols; c += WAVE * 4) {
            float4 v = asfloat(w.Load4(base + c * 4));
            [unroll] for (uint t = 0; t < MAX_TOK; ++t) {
                if (t < n_tok) {
                    float4 xv = x4(x, staged, t, c);
                    acc[t] += dot(v, xv);
                }
            }
        }
        finish(row, lane, acc);
    }
}
