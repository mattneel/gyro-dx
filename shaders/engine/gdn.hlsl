// Linear attention: causal depthwise convolution and the gated delta rule recurrence.
#include "common.hlsli"

cbuffer ConvC : register(b0) {
    uint c_x_uav; uint c_x_off; uint c_y_uav; uint c_y_off;
    uint c_state_uav; uint c_state_off; uint c_w_srv; uint c_w_off;
    uint c_channels; uint c_kernel; uint c_n_tok; uint c_dilation;
    uint c_res_uav; uint c_res_off; uint c_gated_uav; uint c_gated_off;
};

// Thread per channel. The state holds the previous (kernel - 1) * dilation inputs, oldest first.
// y[t][ch] = silu(sum_k w[ch][k] * x[t - (kernel - 1 - k) * dilation][ch]).
// When c_res_uav is valid (PLE path), res[t][ch] += gated[t][ch] + y instead of storing y.
[numthreads(256, 1, 1)]
void conv_causal(uint3 dtid : SV_DispatchThreadID) {
    uint ch = dtid.x;
    if (ch >= c_channels) return;
    uint hist = (c_kernel - 1) * c_dilation;
    RWByteAddressBuffer x = uav(c_x_uav);
    RWByteAddressBuffer state = uav(c_state_uav);
    ByteAddressBuffer w = srv(c_w_srv);
    float window[16];
    for (uint j = 0; j < hist; ++j) window[j] = ldf(state, c_state_off + j * c_channels + ch);
    float wk[4];
    for (uint k = 0; k < c_kernel; ++k) wk[k] = ldf_ro(w, c_w_off + ch * c_kernel + k);
    for (uint t = 0; t < c_n_tok; ++t) {
        float cur = ldf(x, c_x_off + t * c_channels + ch);
        float acc = wk[c_kernel - 1] * cur;
        for (uint k2 = 0; k2 + 1 < c_kernel; ++k2) {
            uint back = (c_kernel - 1 - k2) * c_dilation;
            acc += wk[k2] * window[hist - back];
        }
        float y = silu(acc);
        if (c_res_uav != 0xffffffffu) {
            RWByteAddressBuffer res = uav(c_res_uav);
            uint ri = c_res_off + t * c_channels + ch;
            stf(res, ri, ldf(res, ri) + ldf(uav(c_gated_uav), c_gated_off + t * c_channels + ch) + y);
        } else {
            stf(uav(c_y_uav), c_y_off + t * c_channels + ch, y);
        }
        for (uint j2 = 0; j2 + 1 < hist; ++j2) window[j2] = window[j2 + 1];
        window[hist - 1] = cur;
    }
    for (uint j3 = 0; j3 < hist; ++j3) stf(state, c_state_off + j3 * c_channels + ch, window[j3]);
}

cbuffer GateC : register(b0) {
    uint q_beta_uav; uint q_beta_off; uint q_alpha_uav; uint q_alpha_off;
    uint q_dt_srv; uint q_dt_off; uint q_a_srv; uint q_a_off;
    uint q_heads; uint q_n_tok; uint q_pad0; uint q_pad1;
};

// beta = sigmoid(beta); alpha becomes the per-step decay exp(softplus(alpha + dt) * a).
[numthreads(64, 1, 1)]
void gdn_gates(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= q_heads * q_n_tok) return;
    uint h = idx % q_heads;
    RWByteAddressBuffer beta = uav(q_beta_uav);
    RWByteAddressBuffer alpha = uav(q_alpha_uav);
    stf(beta, q_beta_off + idx, sigmoid(ldf(beta, q_beta_off + idx)));
    float g = softplus(ldf(alpha, q_alpha_off + idx) + ldf_ro(srv(q_dt_srv), q_dt_off + h)) * ldf_ro(srv(q_a_srv), q_a_off + h);
    stf(alpha, q_alpha_off + idx, exp(g));
}

cbuffer StepC : register(b0) {
    uint s_state_uav; uint s_state_off; uint s_qkv_uav; uint s_qkv_off;
    uint s_channels; uint s_k_heads; uint s_v_heads; uint s_n_tok;
    uint s_beta_uav; uint s_beta_off; uint s_decay_uav; uint s_decay_off;
    uint s_z_uav; uint s_z_off; uint s_out_uav; uint s_out_off;
    uint s_norm_srv; uint s_norm_off; float s_eps; uint s_pad;
};

// One wave per (value head, column j); lane l owns state rows i = 4l..4l+3 of column j, stored
// transposed as s[j * 128 + i] so a wave reads one contiguous 512-byte column.
// S = decay * S; delta = (v - S^T k) * beta; S += k delta^T; o = S^T q / sqrt(128).
// The column state stays in registers across the tokens of one pass. The gated output norm
// runs afterwards (norm_rows and ew).
[WaveSize(32)]
[numthreads(256, 1, 1)]
void gdn_step(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint lane = WaveGetLaneIndex();
    uint column = gid.x * 8 + tid.x / WAVE;
    uint hv = column / 128;
    uint j = column % 128;
    if (hv >= s_v_heads) return;
    uint hk = hv % s_k_heads;
    RWByteAddressBuffer state = uav(s_state_uav);
    RWByteAddressBuffer qkv = uav(s_qkv_uav);
    uint sa = (s_state_off + hv * 128 * 128 + j * 128 + lane * 4) * 4;
    float4 sv = asfloat(state.Load4(sa));
    for (uint t = 0; t < s_n_tok; ++t) {
        uint row = s_qkv_off + t * s_channels;
        float4 q = asfloat(qkv.Load4((row + hk * 128 + lane * 4) * 4));
        float4 k = asfloat(qkv.Load4((row + s_k_heads * 128 + hk * 128 + lane * 4) * 4));
        float v = ldf(qkv, row + 2 * s_k_heads * 128 + hv * 128 + j);
        float beta = ldf(uav(s_beta_uav), s_beta_off + t * s_v_heads + hv);
        float decay = ldf(uav(s_decay_uav), s_decay_off + t * s_v_heads + hv);
        sv *= decay;
        float kv = WaveActiveSum(dot(sv, k));
        float delta = (v - kv) * beta;
        sv += k * delta;
        float o = WaveActiveSum(dot(sv, q)) * 0.08838834764831845f;
        if (lane == 0) stf(uav(s_out_uav), s_out_off + t * s_v_heads * 128 + hv * 128 + j, o);
    }
    state.Store4(sa, asuint(sv));
}
