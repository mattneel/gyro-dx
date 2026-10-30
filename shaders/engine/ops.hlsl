// Normalization, hyper-connection, and elementwise kernels.
#include "common.hlsli"

// ---- segment normalization ----
// Segment s starts at x_off + (s / per_tok) * tok_stride + (s % per_tok) * seg_stride.
// mode 0: RMS norm (mean of squares). mode 1: L2 norm as build_gdn_l2_norm.
// w_mode 0: no weight. 1: weight row (s % per_tok). 2: one shared weight row.
cbuffer NormC : register(b0) {
    uint n_x_uav; uint n_x_off; uint n_y_uav; uint n_y_off;
    uint n_len; uint n_per_tok; uint n_seg_stride; uint n_tok_stride;
    uint n_w_srv; uint n_w_off; uint n_w_mode; uint n_mode;
    float n_eps; uint n_pad0; uint n_pad1; uint n_pad2;
};

groupshared float norm_red[8];

[WaveSize(32)]
[numthreads(256, 1, 1)]
void norm_rows(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint s = gid.x + gid.y * 65535;
    uint t = s / n_per_tok;
    uint within = s % n_per_tok;
    uint off = t * n_tok_stride + within * n_seg_stride;
    RWByteAddressBuffer x = uav(n_x_uav);
    float ss = 0;
    for (uint i = tid.x; i < n_len; i += 256) {
        float v = ldf(x, n_x_off + off + i);
        ss += v * v;
    }
    ss = WaveActiveSum(ss);
    if (WaveGetLaneIndex() == 0) norm_red[tid.x / WAVE] = ss;
    GroupMemoryBarrierWithGroupSync();
    float total = 0;
    [unroll] for (uint k = 0; k < 8; ++k) total += norm_red[k];
    float n = float(n_len);
    float scale = n_mode == 1 ? rsqrt(total / n + n_eps / n) / sqrt(n) : rsqrt(total / n + n_eps);
    RWByteAddressBuffer y = uav(n_y_uav);
    ByteAddressBuffer w = srv(n_w_srv);
    uint woff = n_w_off + (n_w_mode == 1 ? within * n_len : 0);
    for (uint i2 = tid.x; i2 < n_len; i2 += 256) {
        float v = ldf(x, n_x_off + off + i2) * scale;
        if (n_w_mode != 0) v *= ldf_ro(w, woff + i2);
        stf(y, n_y_off + off + i2, v);
    }
}

// ---- hyper-connections ----

cbuffer HcC : register(b0) {
    uint h_a_uav; uint h_a_off; uint h_b_uav; uint h_b_off;
    uint h_c_uav; uint h_c_off; uint h_dim; uint h_hc;
    uint h_n_tok; uint h_pad0; uint h_pad1; uint h_pad2;
};

// out[t][i] = mean_c xn[t][c][i] * sigmoid(gate[t][c][i]); a = xn, b = gate, c = out.
[numthreads(256, 1, 1)]
void hc_mix(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= h_dim * h_n_tok) return;
    uint t = idx / h_dim;
    uint i = idx % h_dim;
    RWByteAddressBuffer xn = uav(h_a_uav);
    RWByteAddressBuffer gate = uav(h_b_uav);
    float acc = 0;
    for (uint c = 0; c < h_hc; ++c) {
        uint o = (t * h_hc + c) * h_dim + i;
        acc += ldf(xn, h_a_off + o) * sigmoid(ldf(gate, h_b_off + o));
    }
    stf(uav(h_c_uav), h_c_off + idx, acc / float(h_hc));
}

// res[t][c][i] += block[t][i] * 2 * sigmoid(inject[t][c] / hc); a = res, b = block, c = inject.
[numthreads(256, 1, 1)]
void hc_combine(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= h_dim * h_n_tok * h_hc) return;
    uint i = idx % h_dim;
    uint tc = idx / h_dim;
    uint t = tc / h_hc;
    RWByteAddressBuffer res = uav(h_a_uav);
    float w = 2.0f * sigmoid(ldf(uav(h_c_uav), h_c_off + tc) / float(h_hc));
    float v = ldf(res, h_a_off + idx) + ldf(uav(h_b_uav), h_b_off + t * h_dim + i) * w;
    stf(res, h_a_off + idx, v);
}

// res[t][c][i] = emb[t][i]; a = res, b = emb.
[numthreads(256, 1, 1)]
void hc_init(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= h_dim * h_n_tok * h_hc) return;
    uint i = idx % h_dim;
    uint t = idx / (h_dim * h_hc);
    stf(uav(h_a_uav), h_a_off + idx, ldf(uav(h_b_uav), h_b_off + t * h_dim + i));
}

// ---- elementwise ----
// y[r][i] = op(a[r][i], b[r][i]) with row strides. op 0: silu(a) * b. 1: a * sigmoid(b).
// 2: a + b. 3: copy a. 4: a * sigmoid(b[r]) (row scalar). 5: bit-exact copy of a.
cbuffer EwC : register(b0) {
    uint e_a_uav; uint e_a_off; uint e_a_stride; uint e_b_uav;
    uint e_b_off; uint e_b_stride; uint e_y_uav; uint e_y_off;
    uint e_y_stride; uint e_n; uint e_rows; uint e_op;
};

[numthreads(256, 1, 1)]
void ew(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= e_n * e_rows) return;
    uint r = idx / e_n;
    uint i = idx % e_n;
    if (e_op == 5) {
        // Bit-exact 32-bit copy for packed data such as the codebook.
        uav(e_y_uav).Store((e_y_off + r * e_y_stride + i) * 4, uav(e_a_uav).Load((e_a_off + r * e_a_stride + i) * 4));
        return;
    }
    float a = ldf(uav(e_a_uav), e_a_off + r * e_a_stride + i);
    float v;
    if (e_op == 3) {
        v = a;
    } else {
        float b = e_op == 4 ? ldf(uav(e_b_uav), e_b_off + r * e_b_stride)
                            : ldf(uav(e_b_uav), e_b_off + r * e_b_stride + i);
        if (e_op == 0) v = silu(a) * b;
        else if (e_op == 1 || e_op == 4) v = a * sigmoid(b);
        else v = a + b;
    }
    stf(uav(e_y_uav), e_y_off + r * e_y_stride + i, v);
}
