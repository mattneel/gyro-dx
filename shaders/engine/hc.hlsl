// Fused hyper-connection mixer: grouped RMS norm, low-rank down projection with SiLU, the
// injection weights, the up projection gate, and the stream mean. Weights use the repacked Q8_0
// layout (row = int8 quants, then binary16 scales per 32 values).
#include "common.hlsli"

cbuffer PreC : register(b0) {
    uint p_x_uav; uint p_x_off; uint p_wn_srv; uint p_wn_off;
    uint p_down_srv; uint p_down_row; uint p_inj_srv; uint p_inj_row;
    uint p_rank; uint p_hc; uint p_dim; uint p_lo_uav;
    uint p_inj_uav; uint p_s_uav; float p_eps; uint p_has_inject;
    uint p_tok_stride; uint p_pad0; uint p_pad1; uint p_pad2;
};

groupshared float pre_red[8][8];

// Grid: x = output row (rank rows of down, then hc rows of inject), y = token.
// xn[c][i] = x[c][i] * s_c * w[c][i] with s_c = rsqrt(mean(x[c]^2) + eps).
[WaveSize(32)]
[numthreads(256, 1, 1)]
void hc_pre(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint r = gid.x, t = gid.y;
    bool inject = r >= p_rank;
    ByteAddressBuffer w = inject ? srv(p_inj_srv) : srv(p_down_srv);
    uint row_bytes = inject ? p_inj_row : p_down_row;
    uint base = (inject ? r - p_rank : r) * row_bytes;
    uint cols = p_dim * p_hc;
    RWByteAddressBuffer x = uav(p_x_uav);
    ByteAddressBuffer wn = srv(p_wn_srv);
    uint xo = p_x_off + t * p_tok_stride;
    float part[4] = { 0, 0, 0, 0 };
    float sq[4] = { 0, 0, 0, 0 };
    for (uint c = tid.x * 16; c < cols; c += 256 * 16) {
        uint stream = c / p_dim;
        uint4 q = w.Load4(base + c);
        float d = load_half(w, base + cols + (c / 32) * 2);
        float p = 0, s = 0;
        [unroll] for (uint k = 0; k < 4; ++k) {
            float4 xv = asfloat(x.Load4((xo + c + k * 4) * 4));
            float4 gv = asfloat(wn.Load4((p_wn_off + c + k * 4) * 4));
            uint word = q[k];
            float4 qv = float4(sext8(word & 0xff), sext8((word >> 8) & 0xff), sext8((word >> 16) & 0xff), sext8(word >> 24));
            p += dot(qv, xv * gv);
            s += dot(xv, xv);
        }
        part[stream] += d * p;
        sq[stream] += s;
    }
    uint lane = WaveGetLaneIndex(), wave = tid.x / WAVE;
    [unroll] for (uint k2 = 0; k2 < 4; ++k2) {
        float a = WaveActiveSum(part[k2]);
        float b = WaveActiveSum(sq[k2]);
        if (lane == 0) { pre_red[wave][k2] = a; pre_red[wave][4 + k2] = b; }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0) {
        float result = 0;
        for (uint c2 = 0; c2 < p_hc; ++c2) {
            float a = 0, b = 0;
            [unroll] for (uint wv = 0; wv < 8; ++wv) { a += pre_red[wv][c2]; b += pre_red[wv][4 + c2]; }
            float sc = rsqrt(b / float(p_dim) + p_eps);
            result += sc * a;
            if (r == 0) stf(uav(p_s_uav), t * p_hc + c2, sc);
        }
        if (inject) stf(uav(p_inj_uav), t * p_hc + (r - p_rank), result);
        else stf(uav(p_lo_uav), t * p_rank + r, silu(result / float(p_hc)));
    }
}

cbuffer PostC : register(b0) {
    uint q_up_srv; uint q_up_row; uint q_lo_uav; uint q_x_uav;
    uint q_x_off; uint q_wn_srv; uint q_wn_off; uint q_s_uav;
    uint q_out_uav; uint q_dim; uint q_hc; uint q_rank;
    uint q_tok_stride; uint q_pad0; uint q_pad1; uint q_pad2;
};

groupshared float post_lo[512];

// Grid: x = element blocks of 8 (one wave per element), y = token.
// gate[c][i] = up[c * dim + i] . lo; out[i] = mean_c xn[c][i] * sigmoid(gate[c][i]).
[WaveSize(32)]
[numthreads(256, 1, 1)]
void hc_post(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint t = gid.y;
    RWByteAddressBuffer lo = uav(q_lo_uav);
    for (uint k = tid.x; k < q_rank; k += 256) post_lo[k] = ldf(lo, t * q_rank + k);
    GroupMemoryBarrierWithGroupSync();
    uint lane = WaveGetLaneIndex();
    uint i = gid.x * 8 + tid.x / WAVE;
    if (i >= q_dim) return;
    ByteAddressBuffer up = srv(q_up_srv);
    RWByteAddressBuffer x = uav(q_x_uav);
    RWByteAddressBuffer svals = uav(q_s_uav);
    ByteAddressBuffer wn = srv(q_wn_srv);
    float acc = 0;
    for (uint c = 0; c < q_hc; ++c) {
        uint base = (c * q_dim + i) * q_up_row;
        float g = 0;
        for (uint col = lane * 16; col < q_rank; col += WAVE * 16) {
            uint4 q = up.Load4(base + col);
            float d = load_half(up, base + q_rank + (col / 32) * 2);
            float p = 0;
            [unroll] for (uint k2 = 0; k2 < 4; ++k2) {
                uint word = q[k2];
                uint o = col + k2 * 4;
                p += float(sext8(word & 0xff)) * post_lo[o] + float(sext8((word >> 8) & 0xff)) * post_lo[o + 1] +
                     float(sext8((word >> 16) & 0xff)) * post_lo[o + 2] + float(sext8(word >> 24)) * post_lo[o + 3];
            }
            g += d * p;
        }
        g = WaveActiveSum(g);
        uint e = c * q_dim + i;
        float xn = ldf(x, q_x_off + t * q_tok_stride + e) * ldf(svals, t * q_hc + c) * ldf_ro(wn, q_wn_off + e);
        acc += xn * sigmoid(g);
    }
    if (lane == 0) stf(uav(q_out_uav), t * q_dim + i, acc / float(q_hc));
}
