// Routed experts: softmax routing, fused Gyro decode with matrix-vector products, and combination.
// Expert weights stay packed. A slot table names the buffer and byte offset of each selected
// expert projection, so resident and cached experts share the same kernels.
#include "common.hlsli"

// ---- routing ----

cbuffer RouteC : register(b0) {
    uint r_logits_uav; uint r_logits_off; uint r_n_expert; uint r_n_used;
    uint r_ids_uav; uint r_ids_off; uint r_weights_uav; uint r_weights_off;
    uint r_host_uav; uint r_host_off; float r_scale; uint r_n_tok;
};

groupshared float route_p[512];
groupshared float route_red[8];
groupshared uint route_redi[8];

// One group per token. Softmax over the router logits, top-k with lower-index ties,
// then weights normalized by their sum (clamped as in the reference).
[WaveSize(32)]
[numthreads(256, 1, 1)]
void moe_route(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint t = gid.x;
    uint i = tid.x;
    uint lane = WaveGetLaneIndex();
    uint wave = i / WAVE;
    RWByteAddressBuffer logits = uav(r_logits_uav);
    float v0 = i < r_n_expert ? ldf(logits, r_logits_off + t * r_n_expert + i) : -1e30f;
    float v1 = i + 256 < r_n_expert ? ldf(logits, r_logits_off + t * r_n_expert + i + 256) : -1e30f;
    float m = WaveActiveMax(max(v0, v1));
    if (lane == 0) route_red[wave] = m;
    GroupMemoryBarrierWithGroupSync();
    m = route_red[0];
    [unroll] for (uint k = 1; k < 8; ++k) m = max(m, route_red[k]);
    GroupMemoryBarrierWithGroupSync();
    float e0 = i < r_n_expert ? exp(v0 - m) : 0;
    float e1 = i + 256 < r_n_expert ? exp(v1 - m) : 0;
    float s = WaveActiveSum(e0 + e1);
    if (lane == 0) route_red[wave] = s;
    GroupMemoryBarrierWithGroupSync();
    s = 0;
    [unroll] for (uint k2 = 0; k2 < 8; ++k2) s += route_red[k2];
    route_p[i] = e0 / s;
    route_p[i + 256] = e1 / s;
    GroupMemoryBarrierWithGroupSync();

    RWByteAddressBuffer ids = uav(r_ids_uav);
    RWByteAddressBuffer weights = uav(r_weights_uav);
    float total = 0;
    for (uint k3 = 0; k3 < r_n_used; ++k3) {
        float a = route_p[i], b = route_p[i + 256];
        float best = max(a, b);
        uint best_i = a >= b ? i : i + 256;
        float wbest = WaveActiveMax(best);
        uint wi = WaveActiveMin(best == wbest ? best_i : 0xffffffffu);
        if (lane == 0) { route_red[wave] = wbest; route_redi[wave] = wi; }
        GroupMemoryBarrierWithGroupSync();
        float gb = route_red[0];
        uint gi = route_redi[0];
        [unroll] for (uint q = 1; q < 8; ++q) {
            if (route_red[q] > gb || (route_red[q] == gb && route_redi[q] < gi)) { gb = route_red[q]; gi = route_redi[q]; }
        }
        total += gb;
        if (i == 0) {
            ids.Store((r_ids_off + t * r_n_used + k3) * 4, gi);
            stf(weights, r_weights_off + t * r_n_used + k3, gb);
            if (r_host_uav != 0xffffffffu) uav(r_host_uav).Store((r_host_off + t * r_n_used + k3) * 4, gi);
        }
        GroupMemoryBarrierWithGroupSync();
        if (i == gi) route_p[i] = -1;
        if (i + 256 == gi) route_p[i + 256] = -1;
        GroupMemoryBarrierWithGroupSync();
    }
    if (i < r_n_used) {
        float w = ldf(weights, r_weights_off + t * r_n_used + i);
        w = w / max(total, 6.103515625e-5f);
        if (r_scale != 0 && r_scale != 1) w *= r_scale;
        stf(weights, r_weights_off + t * r_n_used + i, w);
    }
}

// Slot table entry (6 uints): gate srv, gate offset, up srv, up offset, down srv, down offset.
// A slot holds one expert bundle [gate | up | down] inside an arena buffer. lookup maps
// (layer * n_expert + expert) to its slot for fully resident experts.
cbuffer ResolveC : register(b0) {
    uint s_ids_uav; uint s_ids_off; uint s_count; uint s_table_uav;
    uint s_lookup_uav; uint s_lookup_off; uint s_per_arena; uint s_slot_stride;
    uint s_gate_bytes; uint s_n_arenas; uint s_pad0; uint s_pad1;
    uint4 s_arena_lo; uint4 s_arena_hi;
};

[numthreads(64, 1, 1)]
void moe_resolve(uint3 dtid : SV_DispatchThreadID) {
    uint k = dtid.x;
    if (k >= s_count) return;
    uint e = uav(s_ids_uav).Load((s_ids_off + k) * 4);
    uint slot = uav(s_lookup_uav).Load((s_lookup_off + e) * 4);
    uint a = slot / s_per_arena;
    uint arena = a < 4 ? s_arena_lo[a] : s_arena_hi[a - 4];
    uint o = (slot % s_per_arena) * s_slot_stride;
    RWByteAddressBuffer table = uav(s_table_uav);
    table.Store3(k * 24, uint3(arena, o, arena));
    table.Store3(k * 24 + 12, uint3(o + s_gate_bytes, arena, o + 2 * s_gate_bytes));
}

// ---- fused Gyro decode and matrix-vector product ----

cbuffer GyroC : register(b0) {
    uint g_table_uav; uint g_codebook_srv; uint g_x_uav; uint g_x_off;
    uint g_x_stride; uint g_y_uav; uint g_y_off; uint g_rows;
    uint g_cols; uint g_type; uint g_block_bytes; uint g_proj;
    uint g_n_used; uint g_x_per_slot; uint g_pad0; uint g_pad1;
};

groupshared uint gs_book[2048];
groupshared float gs_x[2560];

uint stream_byte(ByteAddressBuffer w, uint block_offset, uint index) {
    return load_byte(w, block_offset + 2 + index);
}

// Generic k-bit state for the pair-trellis and k-bit types (SPEC.md trellis definition).
uint generic_state(ByteAddressBuffer w, uint block_offset, uint stream_bytes, uint steps, uint k, uint step) {
    uint offset = (steps - 1 - step) * k;
    uint byte = offset >> 3;
    uint word = stream_byte(w, block_offset, byte) |
        (stream_byte(w, block_offset, (byte + 1) % stream_bytes) << 8) |
        (stream_byte(w, block_offset, (byte + 2) % stream_bytes) << 16);
    return (word >> (offset & 7)) & 0xffff;
}

void load_codebook(uint tid) {
    ByteAddressBuffer book = srv(g_codebook_srv);
    for (uint i = tid; i < 2048; i += 256) gs_book[i] = book.Load(i * 4);
}

// One row of one expert projection, reduced across a wave. Lanes take trellis steps.
float gyro_row(ByteAddressBuffer w, uint row_offset, uint lane) {
    uint blocks = g_cols / 128;
    float acc = 0;
    if (g_type == 144) {
        for (uint b = 0; b < blocks; ++b) {
            uint bo = row_offset + b * 34;
            uint mine = stream_byte(w, bo, lane);
            uint prev = WaveReadLaneAt(mine, (lane + 31) & 31);
            uint state = (prev << 8) | mine;
            uint h = (state & 0x7fff) * 0x9e3779b1u;
            uint pa = gs_book[h >> 21];
            uint pb = gs_book[(h >> 10) & 2047];
            uint xi = b * 128 + lane * 4;
            float part = f16tof32(pa & 0xffff) * gs_x[xi] + f16tof32(pa >> 16) * gs_x[xi + 1] +
                         f16tof32(pb & 0xffff) * gs_x[xi + 2] + f16tof32(pb >> 16) * gs_x[xi + 3];
            acc += load_half(w, bo) * part;
        }
    } else {
        // k-bit types: steps 32 (two pairs per step) or 64 (one pair per step).
        uint k = g_type == 145 ? 6 : g_type == 146 ? 7 : g_type == 147 ? 5 : g_type == 148 ? 6 :
                 g_type == 149 ? 7 : g_type == 150 ? 5 : 4;
        uint steps = (g_type >= 147 && g_type <= 149) ? 64 : 32;
        uint stream_bytes = g_block_bytes - 2;
        for (uint b = 0; b < blocks; ++b) {
            uint bo = row_offset + b * g_block_bytes;
            float part = 0;
            for (uint step = lane; step < steps; step += WAVE) {
                uint state = generic_state(w, bo, stream_bytes, steps, k, step);
                uint h = (state & 0x7fff) * 0x9e3779b1u;
                uint pa = gs_book[h >> 21];
                if (steps == 64) {
                    uint xi = b * 128 + step * 2;
                    part += f16tof32(pa & 0xffff) * gs_x[xi] + f16tof32(pa >> 16) * gs_x[xi + 1];
                } else {
                    uint pb = gs_book[(h >> 10) & 2047];
                    uint xi = b * 128 + step * 4;
                    part += f16tof32(pa & 0xffff) * gs_x[xi] + f16tof32(pa >> 16) * gs_x[xi + 1] +
                            f16tof32(pb & 0xffff) * gs_x[xi + 2] + f16tof32(pb >> 16) * gs_x[xi + 3];
                }
            }
            acc += load_half(w, bo) * part;
        }
    }
    return WaveActiveSum(acc);
}

// Grid: x = row blocks of 32 rows, y = selected slot (token * n_used + k), z = projection pass.
// g_proj selects 0 = gate (z adds up as 1), 2 = down. Output: y[slot * rows + row].
[WaveSize(32)]
[numthreads(256, 1, 1)]
void moe_gyro(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint lane = WaveGetLaneIndex();
    uint wave = tid.x / WAVE;
    uint slot = gid.y;
    uint proj = g_proj + gid.z;
    load_codebook(tid.x);
    RWByteAddressBuffer x = uav(g_x_uav);
    uint xbase = g_x_off + (g_x_per_slot ? slot : slot / g_n_used) * g_x_stride;
    for (uint i = tid.x; i < g_cols; i += 256) gs_x[i] = ldf(x, xbase + i);
    GroupMemoryBarrierWithGroupSync();
    RWByteAddressBuffer table = uav(g_table_uav);
    uint2 entry = table.Load2(slot * 24 + proj * 8);
    ByteAddressBuffer w = srv(entry.x);
    uint row_bytes = g_cols / 128 * g_block_bytes;
    RWByteAddressBuffer y = uav(g_y_uav);
    // Gate and up write [slot][2][rows]; down writes [slot][rows].
    uint out_base = g_proj == 2 ? g_y_off + slot * g_rows : g_y_off + (slot * 2 + gid.z) * g_rows;
    for (uint r = 0; r < 4; ++r) {
        uint row = gid.x * 32 + wave * 4 + r;
        if (row >= g_rows) break;
        float v = gyro_row(w, entry.y + row * row_bytes, lane);
        if (lane == 0) stf(y, out_base + row, v);
    }
}

// ---- activation, rotation, and combination ----

cbuffer ActC : register(b0) {
    uint a_gu_uav; uint a_gu_off; uint a_out_uav; uint a_out_off;
    uint a_signs_srv; uint a_signs_off; uint a_ff; uint a_rotate;
};

groupshared float act_buf[1024];

// One group per slot: out = H D (silu(gate) * up) for the down projection input.
[numthreads(128, 1, 1)]
void moe_act(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint slot = gid.x;
    RWByteAddressBuffer gu = uav(a_gu_uav);
    uint base = a_gu_off + slot * a_ff * 2;
    for (uint i = tid.x; i < a_ff; i += 128) {
        float g = ldf(gu, base + i);
        float u = ldf(gu, base + a_ff + i);
        float v = silu(g) * u;
        if (a_rotate) v *= ldf_ro(srv(a_signs_srv), a_signs_off + i);
        act_buf[i] = v;
    }
    GroupMemoryBarrierWithGroupSync();
    if (a_rotate) {
        // Normalized Walsh-Hadamard transform within each 128-value block.
        for (uint width = 1; width < 128; width *= 2) {
            for (uint p = tid.x; p < a_ff / 2; p += 128) {
                uint blk = p / 64;
                uint j = p % 64;
                uint lo = (j / width) * (2 * width) + (j % width);
                uint a = blk * 128 + lo;
                float x0 = act_buf[a], x1 = act_buf[a + width];
                act_buf[a] = x0 + x1;
                act_buf[a + width] = x0 - x1;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    RWByteAddressBuffer o = uav(a_out_uav);
    float norm = a_rotate ? 0.08838834764831845f : 1.0f; // 1 / sqrt(128)
    for (uint i2 = tid.x; i2 < a_ff; i2 += 128) stf(o, a_out_off + slot * a_ff + i2, act_buf[i2] * norm);
}

cbuffer RotC : register(b0) {
    uint t_x_uav; uint t_x_off; uint t_y_uav; uint t_y_off;
    uint t_signs_srv; uint t_signs_off; uint t_n; uint t_stride;
};

groupshared float rot_buf[128];

// y = H (signs * x) per 128-value block. Grid: x = blocks, y = tokens.
[numthreads(64, 1, 1)]
void rotate128(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint base = gid.y * t_stride + gid.x * 128;
    RWByteAddressBuffer x = uav(t_x_uav);
    ByteAddressBuffer signs = srv(t_signs_srv);
    for (uint i = tid.x; i < 128; i += 64)
        rot_buf[i] = ldf(x, t_x_off + base + i) * ldf_ro(signs, t_signs_off + gid.x * 128 + i);
    GroupMemoryBarrierWithGroupSync();
    for (uint width = 1; width < 128; width *= 2) {
        uint j = tid.x;
        uint lo = (j / width) * (2 * width) + (j % width);
        float x0 = rot_buf[lo], x1 = rot_buf[lo + width];
        GroupMemoryBarrierWithGroupSync();
        rot_buf[lo] = x0 + x1;
        rot_buf[lo + width] = x0 - x1;
        GroupMemoryBarrierWithGroupSync();
    }
    RWByteAddressBuffer y = uav(t_y_uav);
    for (uint i2 = tid.x; i2 < 128; i2 += 64) stf(y, t_y_off + base + i2, rot_buf[i2] * 0.08838834764831845f);
}

cbuffer CombineC : register(b0) {
    uint c_down_uav; uint c_down_off; uint c_w_uav; uint c_w_off;
    uint c_out_uav; uint c_out_off; uint c_n_used; uint c_dim;
    uint c_sh_uav; uint c_sh_off; uint c_shg_uav; uint c_shg_off;
};

// out[t][i] = sum_k w[t][k] * down[t*U + k][i] + sigmoid(shared_gate[t]) * shared[t][i]
[numthreads(256, 1, 1)]
void moe_combine(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint t = gid.y;
    uint i = gid.x * 256 + tid.x;
    if (i >= c_dim) return;
    RWByteAddressBuffer down = uav(c_down_uav);
    RWByteAddressBuffer w = uav(c_w_uav);
    float acc = 0;
    for (uint k = 0; k < c_n_used; ++k)
        acc += ldf(w, c_w_off + t * c_n_used + k) * ldf(down, c_down_off + (t * c_n_used + k) * c_dim + i);
    float gate = sigmoid(ldf(uav(c_shg_uav), c_shg_off + t));
    acc += gate * ldf(uav(c_sh_uav), c_sh_off + t * c_dim + i);
    stf(uav(c_out_uav), c_out_off + t * c_dim + i, acc);
}
