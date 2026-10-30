// Full attention: rotary embedding, binary16 KV store, chunked attention with merge, and the
// block-sparse indexer (QSA) selection.
#include "common.hlsli"

// ---- NeoX rotary embedding on segment heads ----
// Segment s starts at x_off + (s / per_tok) * tok_stride + (s % per_tok) * seg_stride.
// Position = pos0 + (s / per_tok) * pos_mul.
cbuffer RopeC : register(b0) {
    uint r_x_uav; uint r_x_off; uint r_per_tok; uint r_seg_stride;
    uint r_tok_stride; uint r_n_seg; uint r_n_rot; uint r_pos0;
    uint r_pos_mul; float r_base; uint r_pad0; uint r_pad1;
};

[numthreads(64, 1, 1)]
void rope_neox(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint s = gid.x;
    uint half_rot = r_n_rot / 2;
    uint i = tid.x;
    if (i >= half_rot || s >= r_n_seg) return;
    uint t = s / r_per_tok;
    uint off = r_x_off + t * r_tok_stride + (s % r_per_tok) * r_seg_stride;
    float pos = float(r_pos0 + t * r_pos_mul);
    float theta = pos * pow(r_base, -2.0f * float(i) / float(r_n_rot));
    float c = cos(theta), sn = sin(theta);
    RWByteAddressBuffer x = uav(r_x_uav);
    float a = ldf(x, off + i), b = ldf(x, off + i + half_rot);
    stf(x, off + i, a * c - b * sn);
    stf(x, off + i + half_rot, a * sn + b * c);
}

// ---- binary16 cache store: dst[(pos0 + t) * width + i] = half(src[t * src_stride + i]) ----
cbuffer StoreC : register(b0) {
    uint k_src_uav; uint k_src_off; uint k_src_stride; uint k_dst_uav;
    uint k_width; uint k_pos0; uint k_n_tok; uint k_pad;
};

[numthreads(256, 1, 1)]
void store_f16(uint3 dtid : SV_DispatchThreadID) {
    uint pair = dtid.x;
    uint pairs = k_width / 2;
    if (pair >= pairs * k_n_tok) return;
    uint t = pair / pairs;
    uint i = (pair % pairs) * 2;
    RWByteAddressBuffer src = uav(k_src_uav);
    float a = ldf(src, k_src_off + t * k_src_stride + i);
    float b = ldf(src, k_src_off + t * k_src_stride + i + 1);
    uint packed = f32tof16(a) | (f32tof16(b) << 16);
    uav(k_dst_uav).Store(((k_pos0 + t) * k_width + i) * 2, packed);
}

float load_f16(RWByteAddressBuffer b, uint element) {
    uint word = b.Load((element & ~1u) * 2);
    return f16tof32((element & 1u) ? (word >> 16) : (word & 0xffffu));
}

// ---- chunked attention ----
// Grid: x = chunk of 256 cells, y = query head, z = token. Dense mode visits cells
// 0..pos0+t; sparse mode visits sel[t][0..len[t]). Writes (m, l, o[256]) per chunk.
cbuffer AttnC : register(b0) {
    uint a_q_uav; uint a_q_off; uint a_q_tok_stride; uint a_q_head_stride;
    uint a_k_uav; uint a_v_uav; uint a_kv_width; uint a_head_dim;
    uint a_group; uint a_pos0; uint a_sparse; uint a_sel_uav;
    uint a_sel_stride; uint a_len_uav; uint a_part_uav; uint a_chunks;
    float a_scale; uint a_n_head; uint a_pad0; uint a_pad1;
};

groupshared float at_q[256];
groupshared float at_p[256];
groupshared uint at_cell[256];
groupshared float at_red[8];

[WaveSize(32)]
[numthreads(256, 1, 1)]
void attn_chunk(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint chunk = gid.x, head = gid.y, t = gid.z;
    uint c = tid.x;
    uint lane = WaveGetLaneIndex();
    uint wave = c / WAVE;
    uint D = a_head_dim;
    uint kvh = head / a_group;
    RWByteAddressBuffer q = uav(a_q_uav);
    for (uint d = c; d < D; d += 256) at_q[d] = ldf(q, a_q_off + t * a_q_tok_stride + head * a_q_head_stride + d);
    uint visible = a_sparse ? uav(a_len_uav).Load(t * 4) : a_pos0 + t + 1;
    uint idx = chunk * 256 + c;
    bool live = idx < visible;
    uint cell = live ? (a_sparse ? uav(a_sel_uav).Load((t * a_sel_stride + idx) * 4) : idx) : 0;
    at_cell[c] = cell;
    GroupMemoryBarrierWithGroupSync();
    float score = -1e30f;
    if (live) {
        RWByteAddressBuffer kc = uav(a_k_uav);
        uint kb = (cell * a_kv_width + kvh * D) * 2;
        float dot = 0;
        for (uint d2 = 0; d2 < D; d2 += 8) {
            uint4 w = kc.Load4(kb + d2 * 2);
            dot += f16tof32(w.x & 0xffff) * at_q[d2] + f16tof32(w.x >> 16) * at_q[d2 + 1] +
                   f16tof32(w.y & 0xffff) * at_q[d2 + 2] + f16tof32(w.y >> 16) * at_q[d2 + 3] +
                   f16tof32(w.z & 0xffff) * at_q[d2 + 4] + f16tof32(w.z >> 16) * at_q[d2 + 5] +
                   f16tof32(w.w & 0xffff) * at_q[d2 + 6] + f16tof32(w.w >> 16) * at_q[d2 + 7];
        }
        score = dot * a_scale;
    }
    float m = WaveActiveMax(score);
    if (lane == 0) at_red[wave] = m;
    GroupMemoryBarrierWithGroupSync();
    m = at_red[0];
    [unroll] for (uint k = 1; k < 8; ++k) m = max(m, at_red[k]);
    GroupMemoryBarrierWithGroupSync();
    float p = live ? exp(score - m) : 0;
    at_p[c] = p;
    float l = WaveActiveSum(p);
    if (lane == 0) at_red[wave] = l;
    GroupMemoryBarrierWithGroupSync();
    l = 0;
    [unroll] for (uint k2 = 0; k2 < 8; ++k2) l += at_red[k2];
    uint count = min(256u, visible > chunk * 256 ? visible - chunk * 256 : 0u);
    RWByteAddressBuffer vc = uav(a_v_uav);
    RWByteAddressBuffer part = uav(a_part_uav);
    uint pbase = ((t * a_n_head + head) * a_chunks + chunk) * (D + 2);
    for (uint d3 = c; d3 < D; d3 += 256) {
        float acc = 0;
        for (uint j = 0; j < count; ++j) acc += at_p[j] * load_f16(vc, at_cell[j] * a_kv_width + kvh * D + d3);
        stf(part, pbase + 2 + d3, acc);
    }
    if (c == 0) {
        stf(part, pbase, m);
        stf(part, pbase + 1, l);
    }
}

// Merges chunk partials and applies the sigmoid output gate stored after each query head.
cbuffer MergeC : register(b0) {
    uint g_part_uav; uint g_chunks; uint g_head_dim; uint g_n_head;
    uint g_q_uav; uint g_q_off; uint g_q_tok_stride; uint g_q_head_stride;
    uint g_out_uav; uint g_out_off; uint g_pos0; uint g_sparse;
    uint g_len_uav; uint g_pad0; uint g_pad1; uint g_pad2;
};

[numthreads(256, 1, 1)]
void attn_merge(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint head = gid.x, t = gid.y;
    uint D = g_head_dim;
    uint visible = g_sparse ? uav(g_len_uav).Load(t * 4) : g_pos0 + t + 1;
    uint used = min(g_chunks, (visible + 255) / 256);
    RWByteAddressBuffer part = uav(g_part_uav);
    uint base = (t * g_n_head + head) * g_chunks * (D + 2);
    float M = -1e30f;
    for (uint k = 0; k < used; ++k) M = max(M, ldf(part, base + k * (D + 2)));
    float L = 0;
    for (uint k2 = 0; k2 < used; ++k2) L += ldf(part, base + k2 * (D + 2) + 1) * exp(ldf(part, base + k2 * (D + 2)) - M);
    RWByteAddressBuffer q = uav(g_q_uav);
    for (uint d = tid.x; d < D; d += 256) {
        float o = 0;
        for (uint k3 = 0; k3 < used; ++k3)
            o += ldf(part, base + k3 * (D + 2) + 2 + d) * exp(ldf(part, base + k3 * (D + 2)) - M);
        o /= L;
        float gate = ldf(q, g_q_off + t * g_q_tok_stride + head * g_q_head_stride + D + d);
        stf(uav(g_out_uav), g_out_off + (t * g_n_head + head) * D + d, o * sigmoid(gate));
    }
}

// ---- QSA indexer ----

// Pooled block keys: mean of `ratio` raw binary16 keys, RMS norm with weight, NeoX rotation at
// the block's first position. One group per block, 128 threads (indexer key width).
cbuffer PoolC : register(b0) {
    uint p_raw_uav; uint p_pool_uav; uint p_first_block; uint p_ratio;
    uint p_dim; uint p_w_srv; uint p_w_off; uint p_n_rot;
    float p_base; float p_eps; uint p_pad0; uint p_pad1;
};

groupshared float pool_v[128];
groupshared float pool_red[4];

[WaveSize(32)]
[numthreads(128, 1, 1)]
void qsa_pool(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint b = p_first_block + gid.x;
    uint i = tid.x;
    RWByteAddressBuffer raw = uav(p_raw_uav);
    float v = 0;
    for (uint j = 0; j < p_ratio; ++j) v += load_f16(raw, (b * p_ratio + j) * p_dim + i);
    v /= float(p_ratio);
    float ss = WaveActiveSum(v * v);
    if (WaveGetLaneIndex() == 0) pool_red[i / WAVE] = ss;
    GroupMemoryBarrierWithGroupSync();
    float total = pool_red[0] + pool_red[1] + pool_red[2] + pool_red[3];
    v = v * rsqrt(total / float(p_dim) + p_eps) * ldf_ro(srv(p_w_srv), p_w_off + i);
    pool_v[i] = v;
    GroupMemoryBarrierWithGroupSync();
    uint half_rot = p_n_rot / 2;
    float outv = v;
    if (i < p_n_rot) {
        uint pi = i % half_rot;
        float theta = float(b * p_ratio) * pow(p_base, -2.0f * float(pi) / float(p_n_rot));
        float c = cos(theta), sn = sin(theta);
        float a = pool_v[pi], bb = pool_v[pi + half_rot];
        outv = i < half_rot ? a * c - bb * sn : a * sn + bb * c;
    }
    stf(uav(p_pool_uav), b * p_dim + i, outv);
}

// Block scores: score[t][b] = sum_h relu(q[t][h] . pooled[b]). Thread per block.
cbuffer ScoreC : register(b0) {
    uint sc_q_uav; uint sc_q_off; uint sc_pool_uav; uint sc_score_uav;
    uint sc_heads; uint sc_dim; uint sc_n_blocks_max; uint sc_pos0;
    uint sc_ratio; uint sc_pad0; uint sc_pad1; uint sc_pad2;
};

[numthreads(256, 1, 1)]
void qsa_score(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint t = gid.y;
    uint b = gid.x * 256 + tid.x;
    uint q_pos = sc_pos0 + t;
    uint n_blocks = (q_pos + 1) / sc_ratio; // complete blocks before the tail
    if (b >= n_blocks) return;
    RWByteAddressBuffer q = uav(sc_q_uav);
    RWByteAddressBuffer pool = uav(sc_pool_uav);
    float total = 0;
    for (uint h = 0; h < sc_heads; ++h) {
        float sum = 0;
        for (uint i = 0; i < sc_dim; i += 4) {
            float4 kv = asfloat(pool.Load4((b * sc_dim + i) * 4));
            float4 qv = asfloat(q.Load4((sc_q_off + (t * sc_heads + h) * sc_dim + i) * 4));
            sum += dot(kv, qv);
        }
        total += max(sum, 0.0f);
    }
    stf(uav(sc_score_uav), t * sc_n_blocks_max + b, total);
}

// Cell selection for one token, one group of 1024 threads. Keeps the tail cells, then whole
// blocks in (score descending, index ascending) order until `width` cells; the last block may
// contribute part of its cells. Matches the CPU engine's tie rule.
cbuffer SelectC : register(b0) {
    uint se_score_uav; uint se_sel_uav; uint se_len_uav; uint se_n_blocks_max;
    uint se_sel_stride; uint se_width; uint se_pos0; uint se_ratio;
};

groupshared uint sel_hist[256];
groupshared uint sel_prefix;
groupshared uint sel_need;
groupshared uint sel_count;
groupshared uint sel_eq_taken;

uint sortable(float f) {
    uint u = asuint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

[numthreads(1024, 1, 1)]
void qsa_select(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint t = gid.x;
    uint i = tid.x;
    uint q_pos = se_pos0 + t;
    uint visible = q_pos + 1;
    RWByteAddressBuffer sel = uav(se_sel_uav);
    RWByteAddressBuffer len = uav(se_len_uav);
    uint sbase = t * se_sel_stride;
    if (visible <= se_width) {
        for (uint j = i; j < visible; j += 1024) sel.Store((sbase + j) * 4, j);
        if (i == 0) len.Store(t * 4, visible);
        return;
    }
    uint tail_start = (q_pos + 1) / se_ratio * se_ratio;
    uint n_blocks = tail_start / se_ratio;
    uint tail = visible - tail_start;
    uint need_cells = se_width - tail;
    uint need_blocks = (need_cells + se_ratio - 1) / se_ratio;
    RWByteAddressBuffer score = uav(se_score_uav);
    uint soff = t * se_n_blocks_max;
    // Radix select of the need_blocks-th largest sortable key, 8 bits per pass.
    if (i == 0) { sel_prefix = 0; sel_need = need_blocks; }
    GroupMemoryBarrierWithGroupSync();
    for (int pass = 3; pass >= 0; --pass) {
        if (i < 256) sel_hist[i] = 0;
        GroupMemoryBarrierWithGroupSync();
        uint shift = uint(pass) * 8;
        uint mask_hi = pass == 3 ? 0u : (0xffffffffu << (shift + 8));
        for (uint b = i; b < n_blocks; b += 1024) {
            uint key = sortable(ldf(score, soff + b));
            if ((key & mask_hi) == (sel_prefix & mask_hi)) InterlockedAdd(sel_hist[(key >> shift) & 255], 1);
        }
        GroupMemoryBarrierWithGroupSync();
        if (i == 0) {
            uint need = sel_need;
            uint acc = 0;
            for (int digit = 255; digit >= 0; --digit) {
                uint h = sel_hist[digit];
                if (acc + h >= need) {
                    sel_prefix |= uint(digit) << shift;
                    sel_need = need - acc;
                    break;
                }
                acc += h;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint threshold = sel_prefix;
    uint eq_needed = sel_need; // blocks equal to the threshold that are taken, lowest index first
    // Tail cells first.
    for (uint j2 = i; j2 < tail; j2 += 1024) sel.Store((sbase + j2) * 4, tail_start + j2);
    if (i == 0) { sel_count = tail; sel_eq_taken = 0; }
    GroupMemoryBarrierWithGroupSync();
    // Blocks above the threshold, then equal blocks in index order. The partial last block is
    // the highest-index equal block taken.
    uint full_blocks_cells = need_blocks * se_ratio;
    uint partial = full_blocks_cells - need_cells; // cells dropped from the last-ranked block
    for (uint base = 0; base < n_blocks; base += 1024) {
        uint b2 = base + i;
        uint key = b2 < n_blocks ? sortable(ldf(score, soff + b2)) : 0;
        bool above = b2 < n_blocks && key > threshold;
        bool equal = b2 < n_blocks && key == threshold;
        // Order equal blocks by index within this chunk with a prefix count.
        uint eq_before = WavePrefixCountBits(equal);
        uint wave_eq = WaveActiveCountBits(equal);
        if (WaveGetLaneIndex() == 0) sel_hist[i / WAVE] = wave_eq;
        GroupMemoryBarrierWithGroupSync();
        uint offset = 0;
        for (uint w2 = 0; w2 < i / WAVE; ++w2) offset += sel_hist[w2];
        uint rank = sel_eq_taken + offset + eq_before;
        bool take_eq = equal && rank < eq_needed;
        if (above || take_eq) {
            bool last = take_eq && rank == eq_needed - 1;
            uint cells = last ? se_ratio - partial : se_ratio;
            uint at;
            InterlockedAdd(sel_count, cells, at);
            for (uint r = 0; r < cells; ++r) sel.Store((sbase + at + r) * 4, b2 * se_ratio + r);
        }
        GroupMemoryBarrierWithGroupSync();
        if (i == 0) {
            uint total_eq = 0;
            for (uint w3 = 0; w3 < 32; ++w3) total_eq += sel_hist[w3];
            sel_eq_taken += total_eq;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (i == 0) len.Store(t * 4, sel_count);
}
