// Prefill matrix product on LinAlg wave matrices (Shader Model 6.10, experimental):
// y[t][r] (+)= act(scale * sum_c x[t][c] * W[r][c]) for a block of tokens.
// W uses the repacked Q8_0 layout (row = int8 quants, then one binary16 scale per 32 values).
// A group computes 16 tokens x 64 rows. Each K step stages 16 x 64 activations and
// 64 x 64 dequantized weights as FP16 in groupshared memory; four waves each multiply a
// 16 x 16 output tile with FP32 accumulation.
#include <hlsl/dx/linalg.h>
#include "common.hlsli"

using namespace dx::linalg;
using TileA = Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave>;
using TileB = Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave>;
using TileC = Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave>;

cbuffer C : register(b0) {
    uint w_srv; uint w_off; uint rows; uint cols;
    uint x_uav; uint x_off; uint x_stride; uint n_tok;
    uint y_uav; uint y_off; uint y_stride; uint mode;
    uint act; float scale; uint row_bytes; uint pad0;
};

#define KT 64
groupshared half ga[16 * KT];      // tokens x k, row-major
groupshared half gb[64 * KT];      // rows x k, row-major (read as column-major K x N)
groupshared float gc[4 * 16 * 16];  // per-wave output tiles

[WaveSize(32)]
[numthreads(128, 1, 1)]
void gemm_q8(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint t0 = gid.y * 16;
    uint r0 = gid.x * 64;
    uint wave = tid.x / WAVE;
    ByteAddressBuffer w = srv(w_srv);
    RWByteAddressBuffer x = uav(x_uav);
    TileC acc = TileC::Splat(0.0f);
    for (uint k0 = 0; k0 < cols; k0 += KT) {
        // Activations: 16 x 64 values, 8 per thread.
        for (uint i = tid.x; i < 16 * KT; i += 128) {
            uint t = i / KT, k = i % KT;
            float v = (t0 + t < n_tok) ? ldf(x, x_off + (t0 + t) * x_stride + k0 + k) : 0.0f;
            ga[i] = (half)v;
        }
        // Weights: 64 rows x 64 values; each thread dequantizes one row half (32 values).
        {
            uint r = tid.x / 2;
            uint half_idx = tid.x % 2;
            uint row = r0 + r;
            uint kk = k0 + half_idx * 32;
            if (row < rows) {
                uint base = w_off + row * row_bytes;
                float d = load_half(w, base + cols + (kk / 32) * 2);
                [unroll] for (uint q = 0; q < 8; ++q) {
                    uint word = w.Load(base + kk + q * 4);
                    uint o = r * KT + half_idx * 32 + q * 4;
                    gb[o + 0] = (half)(d * float(sext8(word & 0xff)));
                    gb[o + 1] = (half)(d * float(sext8((word >> 8) & 0xff)));
                    gb[o + 2] = (half)(d * float(sext8((word >> 16) & 0xff)));
                    gb[o + 3] = (half)(d * float(sext8(word >> 24)));
                }
            } else {
                [unroll] for (uint q2 = 0; q2 < 32; ++q2) gb[r * KT + half_idx * 32 + q2] = (half)0;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint ks = 0; ks < KT; ks += 16) {
            TileA a = TileA::Load(ga, ks, KT, MatrixLayout::RowMajor);
            // B[k][n] = W[r0 + wave * 16 + n][k0 + ks + k]: column-major with stride KT.
            TileB b = TileB::Load(gb, wave * 16 * KT + ks, KT, MatrixLayout::ColMajor);
            acc.MultiplyAccumulate(a, b);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    acc.Store(gc, wave * 256, 16, MatrixLayout::RowMajor);
    GroupMemoryBarrierWithGroupSync();
    RWByteAddressBuffer y = uav(y_uav);
    uint lane = tid.x % WAVE;
    for (uint e = lane; e < 256; e += WAVE) {
        uint t = e / 16, n = e % 16;
        uint row = r0 + wave * 16 + n;
        if (t0 + t >= n_tok || row >= rows) continue;
        uint idx = y_off + (t0 + t) * y_stride + row;
        float v = epilogue(gc[wave * 256 + e], act, scale);
        if (mode == 1) v += ldf(y, idx);
        stf(y, idx, v);
    }
}
