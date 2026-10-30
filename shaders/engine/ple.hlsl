// PLE gate: per-stream grouped norms, signed-root sigmoid gate, gated value, and its grouped norm.
#include "common.hlsli"

cbuffer PleC : register(b0) {
    uint p_key_uav; uint p_key_off; uint p_res_uav; uint p_res_off;
    uint p_value_uav; uint p_value_off; uint p_gated_uav; uint p_gated_off;
    uint p_normed_uav; uint p_normed_off; uint p_w_srv; uint p_wk_off;
    uint p_wq_off; uint p_wc_off; uint p_dim; uint p_hc;
    float p_eps; uint p_pad0; uint p_pad1; uint p_pad2;
};

groupshared float ple_red[3][8];

float3 group_sum3(float3 v, uint tid) {
    v.x = WaveActiveSum(v.x);
    v.y = WaveActiveSum(v.y);
    v.z = WaveActiveSum(v.z);
    if (WaveGetLaneIndex() == 0) {
        ple_red[0][tid / WAVE] = v.x;
        ple_red[1][tid / WAVE] = v.y;
        ple_red[2][tid / WAVE] = v.z;
    }
    GroupMemoryBarrierWithGroupSync();
    float3 total = 0;
    [unroll] for (uint k = 0; k < 8; ++k) total += float3(ple_red[0][k], ple_red[1][k], ple_red[2][k]);
    GroupMemoryBarrierWithGroupSync();
    return total;
}

// Grid: x = stream, y = token.
[WaveSize(32)]
[numthreads(256, 1, 1)]
void ple_mix(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint c = gid.x, t = gid.y;
    uint E = p_dim;
    uint seg = (t * p_hc + c) * E;
    RWByteAddressBuffer key = uav(p_key_uav);
    RWByteAddressBuffer res = uav(p_res_uav);
    RWByteAddressBuffer value = uav(p_value_uav);
    ByteAddressBuffer w = srv(p_w_srv);
    // Pass 1: squares of key, hidden, value.
    float3 acc = 0;
    for (uint i = tid.x; i < E; i += 256) {
        float k = ldf(key, p_key_off + seg + i);
        float h = ldf(res, p_res_off + seg + i);
        float v = ldf(value, p_value_off + t * E + i);
        acc += float3(k * k, h * h, v * v);
    }
    float3 sq = group_sum3(acc, tid.x);
    float sk = rsqrt(sq.x / float(E) + p_eps);
    float sh = rsqrt(sq.y / float(E) + p_eps);
    // Pass 2: dot of the two normalized, weighted vectors.
    float dot = 0;
    for (uint i2 = tid.x; i2 < E; i2 += 256) {
        float k = ldf(key, p_key_off + seg + i2) * sk * ldf_ro(w, p_wk_off + c * E + i2);
        float h = ldf(res, p_res_off + seg + i2) * sh * ldf_ro(w, p_wq_off + c * E + i2);
        dot += k * h;
    }
    float3 d3 = group_sum3(float3(dot, 0, 0), tid.x);
    float s = d3.x * rsqrt(float(E));
    float mag = sqrt(max(abs(s), 1e-6f));
    float gate = sigmoid(sign(s) * mag);
    // gated = value * gate; its RMS is |gate| * rms(value).
    float sg = rsqrt(gate * gate * sq.z / float(E) + p_eps);
    RWByteAddressBuffer gated = uav(p_gated_uav);
    RWByteAddressBuffer normed = uav(p_normed_uav);
    for (uint i3 = tid.x; i3 < E; i3 += 256) {
        float g = ldf(value, p_value_off + t * E + i3) * gate;
        stf(gated, p_gated_off + seg + i3, g);
        stf(normed, p_normed_off + seg + i3, g * sg * ldf_ro(w, p_wc_off + c * E + i3));
    }
}
