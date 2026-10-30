//! CPU execution of the complete Qwen4Exp graph for one sequence.
//! The operation order follows agentionai/llama.cpp src/models/qwen4exp.cpp at 9aa51ce4.
const std = @import("std");
const kernels = @import("kernels.zig");
const model_mod = @import("model.zig");
const ple_mod = @import("ple.zig");
const Pool = @import("pool.zig").Pool;
const Allocator = std.mem.Allocator;
const Matrix = kernels.Matrix;
const Model = model_mod.Model;
const Layer = model_mod.Layer;

/// Optional observer for parity dumps. Names follow the reference graph callbacks.
pub const Observer = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, name: []const u8, layer: i32, values: []const f32, tokens: usize) void,
    fn emit(o: ?Observer, name: []const u8, layer: i32, values: []const f32, tokens: usize) void {
        if (o) |obs| obs.func(obs.ctx, name, layer, values, tokens);
    }
};

const LinearState = struct {
    conv: []f32, // (ssm_conv - 1) * channels, oldest first
    ssm: []f32, // v_heads * state * state, transposed per head: s[j * S + i] = S[i][j]
};

const FullState = struct {
    k: []u16, // ctx * kv_heads * head_dim, binary16
    v: []u16,
    idx_k: []u16, // ctx * idx_dim raw indexer keys, binary16
    pooled: []f32, // (ctx / ratio) * idx_dim normalized and rotated block keys
    pooled_valid: usize,
};

pub const Session = struct {
    allocator: Allocator,
    m: *const Model,
    pool: *Pool,
    max_ctx: u32,
    pos: u32 = 0,
    linear: []LinearState,
    full: []FullState,
    layer_slot: []u32,
    ple_history: ple_mod.History = .{},
    ple_conv: []f32, // (kernel - 1) * dilation rows of hc_dim, oldest first
    observer: ?Observer = null,
    scratch: Scratch,

    pub fn init(allocator: Allocator, m: *const Model, pool: *Pool, max_ctx: u32) !Session {
        const h = m.h;
        var s = Session{
            .allocator = allocator, .m = m, .pool = pool, .max_ctx = max_ctx,
            .linear = &.{}, .full = &.{}, .layer_slot = &.{}, .ple_conv = &.{},
            .scratch = .{},
        };
        errdefer s.deinit();
        var n_linear: usize = 0;
        var n_full: usize = 0;
        for (m.layers) |l| switch (l.mixer) {
            .linear => n_linear += 1,
            .full => n_full += 1,
        };
        s.layer_slot = try allocator.alloc(u32, m.layers.len);
        s.linear = try allocator.alloc(LinearState, n_linear);
        for (s.linear) |*ls| ls.* = .{ .conv = &.{}, .ssm = &.{} };
        s.full = try allocator.alloc(FullState, n_full);
        for (s.full) |*fs| fs.* = .{ .k = &.{}, .v = &.{}, .idx_k = &.{}, .pooled = &.{}, .pooled_valid = 0 };
        var li: u32 = 0;
        var fi: u32 = 0;
        for (m.layers, 0..) |l, i| switch (l.mixer) {
            .linear => {
                s.layer_slot[i] = li;
                s.linear[li].conv = try allocator.alloc(f32, (h.ssm_conv - 1) * h.convChannels());
                s.linear[li].ssm = try allocator.alloc(f32, h.ssm_v_heads * h.ssm_state * h.ssm_state);
                li += 1;
            },
            .full => |f| {
                s.layer_slot[i] = fi;
                const kv = @as(usize, max_ctx) * h.n_head_kv * h.head_dim;
                s.full[fi].k = try allocator.alloc(u16, kv);
                s.full[fi].v = try allocator.alloc(u16, kv);
                s.full[fi].idx_k = try allocator.alloc(u16, @as(usize, max_ctx) * h.idx_dim);
                s.full[fi].pooled = try allocator.alloc(f32, (@as(usize, max_ctx) / f.compress_ratio + 1) * h.idx_dim);
                fi += 1;
            },
        };
        s.ple_conv = try allocator.alloc(f32, (h.ple_conv - 1) * h.ple_dilation * h.hcDim());
        s.reset();
        return s;
    }

    pub fn deinit(s: *Session) void {
        for (s.linear) |ls| {
            s.allocator.free(ls.conv);
            s.allocator.free(ls.ssm);
        }
        for (s.full) |fs| {
            s.allocator.free(fs.k);
            s.allocator.free(fs.v);
            s.allocator.free(fs.idx_k);
            s.allocator.free(fs.pooled);
        }
        s.allocator.free(s.linear);
        s.allocator.free(s.full);
        s.allocator.free(s.layer_slot);
        s.allocator.free(s.ple_conv);
        s.scratch.deinit(s.allocator);
    }

    /// Clears all request state. Weights and allocations remain.
    pub fn reset(s: *Session) void {
        s.pos = 0;
        for (s.linear) |ls| {
            @memset(ls.conv, 0);
            @memset(ls.ssm, 0);
        }
        for (s.full) |*fs| fs.pooled_valid = 0;
        @memset(s.ple_conv, 0);
        s.ple_history.reset();
    }

    /// Runs the graph for `tokens` at positions pos..pos+n. Writes logits of the last token when
    /// `logits` is non-null (length n_vocab), or logits for every token when `all_logits` is set.
    pub fn forward(s: *Session, tokens: []const u32, logits: ?[]f32, all_logits: bool) !void {
        const h = s.m.h;
        const n = tokens.len;
        if (n == 0) return;
        if (@as(usize, s.pos) + n > s.max_ctx) return error.ContextFull;
        for (tokens) |t| if (t >= h.n_vocab) return error.InvalidToken;
        try s.scratch.ensure(s.allocator, h, n, s.max_ctx);
        const sc = &s.scratch;
        const E = h.n_embd;
        const HD = h.hcDim();

        // Embedding rows and the hc identical residual copies.
        for (tokens, 0..) |t, i| {
            const row = sc.cur[i * E ..][0..E];
            kernels.decodeRow(s.m.token_embd.type_id, s.m.token_embd.row(t), row);
            for (0..h.hc) |c| @memcpy(sc.res[i * HD + c * E ..][0..E], row);
        }
        Observer.emit(s.observer, "hc_init", -1, sc.res[0 .. n * HD], n);

        // PLE rows depend only on token history, so gather them before the layers.
        for (tokens, 0..) |t, i| {
            const rows = try s.ple_history.accept(s.m.ple_config, t);
            for (rows, 0..) |r, head| {
                kernels.decodeRow(s.m.ple_table.type_id, s.m.ple_table.row(@intCast(r.row)),
                    sc.ple_emb[i * E + head * h.ple_head_dim ..][0..h.ple_head_dim]);
            }
        }

        for (s.m.layers, 0..) |*l, il| {
            const ilayer: i32 = @intCast(il);
            if (l.ple) |*p| try s.pleBlock(p, n, ilayer);
            s.hcMix(&l.hc_attn, n, sc.cur, sc.inject, ilayer);
            switch (l.mixer) {
                .linear => |*lin| s.linearAttention(lin, s.layer_slot[il], n, ilayer),
                .full => |*f| try s.fullAttention(f, s.layer_slot[il], n, ilayer),
            }
            Observer.emit(s.observer, if (l.mixer == .linear) "linear_attn_out" else "attn_output", ilayer, sc.block_out[0 .. n * E], n);
            s.hcCombine(n, sc.block_out, sc.inject);
            s.hcMix(&l.hc_ffn, n, sc.cur, sc.inject, ilayer);
            s.ffn(l, n, ilayer);
            Observer.emit(s.observer, "ffn_out", ilayer, sc.block_out[0 .. n * E], n);
            s.hcCombine(n, sc.block_out, sc.inject);
            Observer.emit(s.observer, "l_last", ilayer, sc.res[0 .. n * HD], n);
        }

        s.pos += @intCast(n);
        if (logits) |out| {
            const first: usize = if (all_logits) 0 else n - 1;
            const count = n - first;
            // Only the output rows need the final mixer.
            @memmove(sc.res[0 .. count * HD], sc.res[first * HD ..][0 .. count * HD]);
            s.hcMix(&s.m.hc_head, count, sc.cur, null, -1);
            Observer.emit(s.observer, "result_norm", -1, sc.cur[0 .. count * E], count);
            if (out.len < count * h.n_vocab) return error.OutputTooSmall;
            kernels.matmul(s.pool, s.m.output, sc.cur[0 .. count * E], out, count);
        }
    }

    // ---- hyper-connections ----

    fn hcMix(s: *Session, w: *const model_mod.Hc, n: usize, out: []f32, inject: ?[]f32, il: i32) void {
        const h = s.m.h;
        const sc = &s.scratch;
        const E = h.n_embd;
        const HD = h.hcDim();
        const hcf: f32 = @floatFromInt(h.hc);
        for (0..n) |t| {
            for (0..h.hc) |c| {
                const o = t * HD + c * E;
                kernels.rmsNorm(sc.xn[o..][0..E], sc.res[o..][0..E], w.norm[c * E ..][0..E], h.eps);
            }
        }
        Observer.emit(s.observer, "hc_norm", il, sc.xn[0 .. n * HD], n);
        kernels.matmul(s.pool, w.down, sc.xn[0 .. n * HD], sc.lo, n);
        for (sc.lo[0 .. n * h.hc_rank]) |*v| v.* = kernels.silu(v.* / hcf);
        kernels.matmul(s.pool, w.up, sc.lo[0 .. n * h.hc_rank], sc.gate, n);
        Observer.emit(s.observer, "hc_gate", il, sc.gate[0 .. n * HD], n);
        for (0..n) |t| {
            const dst = out[t * E ..][0..E];
            @memset(dst, 0);
            for (0..h.hc) |c| {
                const o = t * HD + c * E;
                for (dst, sc.xn[o..][0..E], sc.gate[o..][0..E]) |*d, x, g| d.* += x * kernels.sigmoid(g);
            }
            for (dst) |*d| d.* /= hcf;
        }
        Observer.emit(s.observer, "hc_mixed", il, out[0 .. n * E], n);
        if (inject) |inj| {
            kernels.matmul(s.pool, w.inject.?, sc.xn[0 .. n * HD], inj, n);
            Observer.emit(s.observer, "hc_inject", il, inj[0 .. n * h.hc], n);
        }
    }

    fn hcCombine(s: *Session, n: usize, block_out: []const f32, inject: []const f32) void {
        const h = s.m.h;
        const E = h.n_embd;
        const hcf: f32 = @floatFromInt(h.hc);
        for (0..n) |t| {
            for (0..h.hc) |c| {
                const wgt = 2.0 * kernels.sigmoid(inject[t * h.hc + c] / hcf);
                const dst = s.scratch.res[t * h.hcDim() + c * E ..][0..E];
                for (dst, block_out[t * E ..][0..E]) |*d, b| d.* += b * wgt;
            }
        }
    }

    // ---- PLE ----

    fn pleBlock(s: *Session, p: *const model_mod.Ple, n: usize, il: i32) !void {
        const h = s.m.h;
        const sc = &s.scratch;
        const E = h.n_embd;
        const HD = h.hcDim();
        Observer.emit(s.observer, "ple_embd", -1, sc.ple_emb[0 .. n * E], n);
        kernels.matmul(s.pool, p.key, sc.ple_emb[0 .. n * E], sc.gate, n); // key: n * HD
        kernels.matmul(s.pool, p.value, sc.ple_emb[0 .. n * E], sc.block_out, n); // value: n * E
        const hist_rows = (h.ple_conv - 1) * h.ple_dilation;
        const inv_sqrt_e = 1.0 / @sqrt(@as(f32, @floatFromInt(E)));
        var gates: [16]f32 = undefined;
        for (0..n) |t| {
            const key = sc.gate[t * HD ..][0..HD];
            const hidden = sc.res[t * HD ..][0..HD];
            const value = sc.block_out[t * E ..][0..E];
            const gated = sc.xn[t * HD ..][0..HD];
            for (0..h.hc) |c| {
                var kn: [8192]f32 = undefined;
                var qn: [8192]f32 = undefined;
                kernels.rmsNorm(kn[0..E], key[c * E ..][0..E], p.norm_key[c * E ..][0..E], h.eps);
                kernels.rmsNorm(qn[0..E], hidden[c * E ..][0..E], p.norm_query[c * E ..][0..E], h.eps);
                const score = kernels.dotDense(kn[0..E], qn[0..E]) * inv_sqrt_e;
                const mag = @sqrt(@max(@abs(score), 1e-6));
                const sgn: f32 = if (score > 0) 1 else if (score < 0) -1 else 0;
                gates[c] = kernels.sigmoid(sgn * mag);
                for (gated[c * E ..][0..E], value) |*g, v| g.* = v * gates[c];
            }
            Observer.emit(s.observer, "ple_gate", il, gates[0..h.hc], 1);
            // Grouped norm of the gated value, then the dilated causal convolution with history.
            var normed = sc.lo_big[0..HD];
            for (0..h.hc) |c| kernels.rmsNorm(normed[c * E ..][0..E], gated[c * E ..][0..E], p.norm_conv[c * E ..][0..E], h.eps);
            const K = h.ple_conv;
            for (0..HD) |ch| {
                var acc: f32 = 0;
                for (0..K) |k| {
                    const back = (K - 1 - k) * h.ple_dilation;
                    const x = if (back == 0) normed[ch] else s.ple_conv[(hist_rows - back) * HD + ch];
                    acc += p.conv[ch * K + k] * x;
                }
                hidden[ch] += gated[ch] + kernels.silu(acc);
            }
            // Shift history: drop the oldest row, append this token's normalized row.
            std.mem.copyForwards(f32, s.ple_conv[0 .. (hist_rows - 1) * HD], s.ple_conv[HD..][0 .. (hist_rows - 1) * HD]);
            @memcpy(s.ple_conv[(hist_rows - 1) * HD ..][0..HD], normed);
        }
    }

    // ---- linear attention (gated delta net) ----

    fn linearAttention(s: *Session, w: *const model_mod.Linear, slot: u32, n: usize, il: i32) void {
        const h = s.m.h;
        const sc = &s.scratch;
        const st = &s.linear[slot];
        const E = h.n_embd;
        const CH = h.convChannels();
        const S = h.ssm_state;
        const HV = h.ssm_v_heads;
        const HK = h.ssm_k_heads;
        const K = h.ssm_conv;
        kernels.matmul(s.pool, w.qkv, sc.cur[0 .. n * E], sc.qkv, n);
        kernels.matmul(s.pool, w.z, sc.cur[0 .. n * E], sc.z, n);
        kernels.matmul(s.pool, w.beta, sc.cur[0 .. n * E], sc.beta, n);
        kernels.matmul(s.pool, w.alpha, sc.cur[0 .. n * E], sc.alpha, n);
        Observer.emit(s.observer, "linear_attn_qkv_mixed", il, sc.qkv[0 .. n * CH], n);
        for (0..n) |t| {
            for (0..HV) |hh| {
                sc.beta[t * HV + hh] = kernels.sigmoid(sc.beta[t * HV + hh]);
                sc.alpha[t * HV + hh] = kernels.softplus(sc.alpha[t * HV + hh] + w.dt_bias[hh]) * w.a[hh];
            }
            // Causal depthwise convolution over [state..., x_t].
            const x = sc.qkv[t * CH ..][0..CH];
            const conv_out = sc.conv_out[t * CH ..][0..CH];
            for (0..CH) |ch| {
                var acc: f32 = 0;
                for (0..K - 1) |k| acc += w.conv[ch * K + k] * st.conv[k * CH + ch];
                acc += w.conv[ch * K + K - 1] * x[ch];
                conv_out[ch] = kernels.silu(acc);
            }
            std.mem.copyForwards(f32, st.conv[0 .. (K - 2) * CH], st.conv[CH..][0 .. (K - 2) * CH]);
            @memcpy(st.conv[(K - 2) * CH ..][0..CH], x);
        }
        Observer.emit(s.observer, "conv_output_silu", il, sc.conv_out[0 .. n * CH], n);
        for (0..n) |t| {
            const conv_out = sc.conv_out[t * CH ..][0..CH];
            for (0..HK) |hk| {
                kernels.l2Norm(conv_out[hk * S ..][0..S], h.eps);
                kernels.l2Norm(conv_out[HK * S + hk * S ..][0..S], h.eps);
            }
        }
        const Ctx = struct { s: *Session, st: *LinearState, n: usize };
        const ctx = Ctx{ .s = s, .st = st, .n = n };
        s.pool.run(HV, 1, &ctx, struct {
            fn f(c: *const Ctx, start: usize, end: usize) void {
                const hp = c.s.m.h;
                const sc2 = &c.s.scratch;
                const Sd = hp.ssm_state;
                const CHd = hp.convChannels();
                const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(Sd)));
                for (start..end) |hv| {
                    const hk = hv % hp.ssm_k_heads;
                    const state = c.st.ssm[hv * Sd * Sd ..][0 .. Sd * Sd];
                    for (0..c.n) |t| {
                        const row = sc2.conv_out[t * CHd ..][0..CHd];
                        const q = row[hk * Sd ..][0..Sd];
                        const k = row[hp.ssm_k_heads * Sd + hk * Sd ..][0..Sd];
                        const v = row[2 * hp.ssm_k_heads * Sd + hv * Sd ..][0..Sd];
                        const decay = @exp(sc2.alpha[t * hp.ssm_v_heads + hv]);
                        const beta = sc2.beta[t * hp.ssm_v_heads + hv];
                        const out = sc2.gdn_out[t * hp.ssm_inner + hv * Sd ..][0..Sd];
                        gdnStep(state, q, k, v, decay, beta, scale, out);
                    }
                }
            }
        }.f);
        Observer.emit(s.observer, "attn_output", il, sc.gdn_out[0 .. n * h.ssm_inner], n);
        for (0..n) |t| {
            for (0..HV) |hv| {
                const o = sc.gdn_out[t * h.ssm_inner + hv * S ..][0..S];
                kernels.rmsNorm(o, o, w.norm, h.eps);
                for (o, sc.z[t * h.ssm_inner + hv * S ..][0..S]) |*v, zz| v.* *= kernels.sigmoid(zz);
            }
        }
        kernels.matmul(s.pool, w.out, sc.gdn_out[0 .. n * h.ssm_inner], sc.block_out, n);
    }

    // ---- full attention with the sparse indexer ----

    fn fullAttention(s: *Session, w: *const model_mod.Full, slot: u32, n: usize, il: i32) !void {
        const h = s.m.h;
        const sc = &s.scratch;
        const st = &s.full[slot];
        const E = h.n_embd;
        const D = h.head_dim;
        const KVW = h.n_head_kv * D;
        const ID = h.idx_dim;
        const r = w.compress_ratio;

        // Indexer keys: raw keys enter the cache; complete blocks are pooled, normalized and rotated.
        kernels.matmul(s.pool, w.idx_k, sc.cur[0 .. n * E], sc.idx_k, n);
        kernels.matmul(s.pool, w.idx_q, sc.cur[0 .. n * E], sc.idx_q, n);
        for (0..n) |t| {
            const p = s.pos + t;
            for (0..ID) |i| st.idx_k[p * ID + i] = kernels.f32ToF16(sc.idx_k[t * ID + i]);
            for (0..h.idx_heads) |hh| {
                const q = sc.idx_q[t * h.idx_heads * ID + hh * ID ..][0..ID];
                kernels.rmsNorm(q, q, w.idx_q_norm, h.eps);
                kernels.ropeNeox(q, h.n_rot, @floatFromInt(p), h.rope_base);
            }
        }
        const total = s.pos + n;
        while ((st.pooled_valid + 1) * r <= total) : (st.pooled_valid += 1) {
            const b = st.pooled_valid;
            const dst = st.pooled[b * ID ..][0..ID];
            @memset(dst, 0);
            for (0..r) |j| {
                for (0..ID) |i| dst[i] += kernels.f16ToF32(st.idx_k[(b * r + j) * ID + i]);
            }
            for (dst) |*v| v.* /= @floatFromInt(r);
            kernels.rmsNorm(dst, dst, w.idx_k_norm, h.eps);
            kernels.ropeNeox(dst, h.n_rot, @floatFromInt(b * r), h.rope_base);
        }

        // Projections, norms, rotary embedding, and the cache store.
        kernels.matmul(s.pool, w.wq, sc.cur[0 .. n * E], sc.qfull, n);
        kernels.matmul(s.pool, w.wk, sc.cur[0 .. n * E], sc.kcur, n);
        kernels.matmul(s.pool, w.wv, sc.cur[0 .. n * E], sc.vcur, n);
        for (0..n) |t| {
            const p = s.pos + t;
            const pf: f32 = @floatFromInt(p);
            for (0..h.n_head) |hh| {
                const q = sc.qfull[t * h.n_head * 2 * D + hh * 2 * D ..][0..D];
                kernels.rmsNorm(q, q, w.q_norm, h.eps);
                kernels.ropeNeox(q, h.n_rot, pf, h.rope_base);
            }
            for (0..h.n_head_kv) |hh| {
                const k = sc.kcur[t * KVW + hh * D ..][0..D];
                kernels.rmsNorm(k, k, w.k_norm, h.eps);
                kernels.ropeNeox(k, h.n_rot, pf, h.rope_base);
            }
            for (0..KVW) |i| {
                st.k[p * KVW + i] = kernels.f32ToF16(sc.kcur[t * KVW + i]);
                st.v[p * KVW + i] = kernels.f32ToF16(sc.vcur[t * KVW + i]);
            }
        }

        // Cell selection per query token.
        const width = h.idx_top_k + r - 1;
        for (0..n) |t| {
            const q_pos = s.pos + t;
            const visible = q_pos + 1;
            const sel = sc.selected[t * sc.sel_stride ..][0..sc.sel_stride];
            if (visible <= width) {
                for (0..visible) |j| sel[j] = @intCast(j);
                sc.selected_len[t] = @intCast(visible);
                continue;
            }
            const tail_start = (q_pos + 1) / r * r;
            const n_blocks = tail_start / r;
            const q = sc.idx_q[t * h.idx_heads * ID ..][0 .. h.idx_heads * ID];
            const scores = sc.block_scores[0..n_blocks];
            for (0..n_blocks) |b| {
                const key = st.pooled[b * ID ..][0..ID];
                var sum: f32 = 0;
                for (0..h.idx_heads) |hh| sum += @max(kernels.dotDense(q[hh * ID ..][0..ID], key), 0);
                scores[b] = sum;
                sc.block_order[b] = @intCast(b);
            }
            const Order = struct {
                scores: []const f32,
                fn less(ctx: @This(), a: u32, b: u32) bool {
                    const sa = ctx.scores[a];
                    const sb = ctx.scores[b];
                    return if (sa != sb) sa > sb else a < b;
                }
            };
            std.mem.sort(u32, sc.block_order[0..n_blocks], Order{ .scores = scores }, Order.less);
            var count: usize = 0;
            for (tail_start..visible) |j| {
                sel[count] = @intCast(j);
                count += 1;
            }
            var bi: usize = 0;
            while (count < width and bi < n_blocks) : (bi += 1) {
                const b = sc.block_order[bi];
                var j: usize = 0;
                while (j < r and count < width) : (j += 1) {
                    sel[count] = b * r + @as(u32, @intCast(j));
                    count += 1;
                }
            }
            sc.selected_len[t] = @intCast(count);
        }

        // Attention over the selected cells, one task per query head and token.
        const Ctx = struct { s: *Session, st: *FullState, n: usize };
        const ctx = Ctx{ .s = s, .st = st, .n = n };
        s.pool.run(n * h.n_head, 1, &ctx, struct {
            fn f(c: *const Ctx, start: usize, end: usize) void {
                const hp = c.s.m.h;
                const sc2 = &c.s.scratch;
                const Dd = hp.head_dim;
                const kvw = hp.n_head_kv * Dd;
                const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(Dd)));
                var weights_buf: [8192]f32 = undefined;
                var kbuf: [512]f32 = undefined;
                for (start..end) |task| {
                    const t = task / hp.n_head;
                    const hh = task % hp.n_head;
                    const kvh = hh / (hp.n_head / hp.n_head_kv);
                    const q = sc2.qfull[t * hp.n_head * 2 * Dd + hh * 2 * Dd ..][0..Dd];
                    const sel = sc2.selected[t * sc2.sel_stride ..][0..sc2.selected_len[t]];
                    const out = sc2.attn[t * hp.n_head * Dd + hh * Dd ..][0..Dd];
                    const weights = weights_buf[0..sel.len];
                    var max: f32 = -std.math.inf(f32);
                    for (sel, 0..) |cell, i| {
                        const krow = c.st.k[cell * kvw + kvh * Dd ..][0..Dd];
                        for (0..Dd) |d| kbuf[d] = kernels.f16ToF32(krow[d]);
                        const sdot = kernels.dotDense(q, kbuf[0..Dd]) * scale;
                        weights[i] = sdot;
                        max = @max(max, sdot);
                    }
                    var sum: f32 = 0;
                    for (weights) |*wv| {
                        wv.* = @exp(wv.* - max);
                        sum += wv.*;
                    }
                    @memset(out, 0);
                    for (sel, weights) |cell, wv| {
                        const vrow = c.st.v[cell * kvw + kvh * Dd ..][0..Dd];
                        const a = wv / sum;
                        for (out, vrow) |*o, vv| o.* += a * kernels.f16ToF32(vv);
                    }
                }
            }
        }.f);
        // Output gate, interleaved after each query head in wq.
        for (0..n) |t| {
            for (0..h.n_head) |hh| {
                const gate = sc.qfull[t * h.n_head * 2 * D + hh * 2 * D + D ..][0..D];
                for (sc.attn[t * h.n_head * D + hh * D ..][0..D], gate) |*o, g| o.* *= kernels.sigmoid(g);
            }
        }
        Observer.emit(s.observer, "attn_gated", il, sc.attn[0 .. n * h.n_head * D], n);
        kernels.matmul(s.pool, w.wo, sc.attn[0 .. n * h.n_head * D], sc.block_out, n);
    }

    // ---- routed and shared experts ----

    fn ffn(s: *Session, l: *const Layer, n: usize, il: i32) void {
        const h = s.m.h;
        const sc = &s.scratch;
        const E = h.n_embd;
        const F = h.n_ff_exp;
        const U = h.n_expert_used;
        kernels.matmul(s.pool, l.router, sc.cur[0 .. n * E], sc.router, n);
        for (0..n) |t| {
            const probs = sc.router[t * h.n_expert ..][0..h.n_expert];
            kernels.softmaxInPlace(probs);
            const ids = sc.route_ids[t * U ..][0..U];
            const wts = sc.route_w[t * U ..][0..U];
            topK(probs, ids, wts);
            var sum: f32 = 0;
            for (wts) |v| sum += v;
            sum = @max(sum, 6.103515625e-5);
            for (wts) |*v| {
                v.* /= sum;
                if (h.expert_weights_scale != 0 and h.expert_weights_scale != 1) v.* *= h.expert_weights_scale;
            }
        }
        if (s.observer) |_| {
            var ids_f: [1024]f32 = undefined;
            for (0..@min(n * U, 1024)) |i| ids_f[i] = @floatFromInt(sc.route_ids[i]);
            Observer.emit(s.observer, "ffn_moe_topk", il, ids_f[0..@min(n * U, 1024)], n);
        }

        // Rotated expert input, shared by gate and up.
        @memcpy(sc.xh[0 .. n * E], sc.cur[0 .. n * E]);
        if (l.gate_signs) |signs| {
            for (0..n) |t| kernels.rotate(sc.xh[t * E ..][0..E], 128, signs);
        }

        // Group token rows by expert so a batch decodes each expert row once.
        @memset(sc.block_out[0 .. n * E], 0);
        var expert_count: [512]u32 = @splat(0);
        for (sc.route_ids[0 .. n * U]) |e| expert_count[e] += 1;
        var order_start: [513]u32 = @splat(0);
        for (0..h.n_expert) |e| order_start[e + 1] = order_start[e] + expert_count[e];
        var fill = order_start;
        for (0..n * U) |i| {
            const e = sc.route_ids[i];
            sc.route_order[fill[e]] = @intCast(i);
            fill[e] += 1;
        }
        const Ctx = struct { s: *Session, l: *const Layer, n: usize, starts: *const [513]u32 };
        const ctx = Ctx{ .s = s, .l = l, .n = n, .starts = &order_start };
        // Phase 1: gate and up rows for every (expert, row) pair.
        s.pool.run(h.n_expert * F, 16, &ctx, struct {
            fn f(c: *const Ctx, start: usize, end: usize) void {
                const hp = c.s.m.h;
                const sc2 = &c.s.scratch;
                const Fd = hp.n_ff_exp;
                const Ed = hp.n_embd;
                var wbuf: [kernels.max_cols]f32 align(64) = undefined;
                for (start..end) |task| {
                    const e = task / Fd;
                    const row = task % Fd;
                    const first = c.starts[e];
                    const last = c.starts[e + 1];
                    if (first == last) continue;
                    const g = c.l.expertGate(hp, e);
                    const u = c.l.expertUp(hp, e);
                    if (last - first == 1) {
                        const slot = sc2.route_order[first];
                        const x = sc2.xh[(slot / hp.n_expert_used) * Ed ..][0..Ed];
                        sc2.act[slot * Fd + row] = kernels.dotRow(g.type_id, g.row(row), x);
                        sc2.act_up[slot * Fd + row] = kernels.dotRow(u.type_id, u.row(row), x);
                        continue;
                    }
                    kernels.decodeRow(g.type_id, g.row(row), wbuf[0..Ed]);
                    for (sc2.route_order[first..last]) |slot| {
                        const x = sc2.xh[(slot / hp.n_expert_used) * Ed ..][0..Ed];
                        sc2.act[slot * Fd + row] = kernels.dotDense(wbuf[0..Ed], x);
                    }
                    kernels.decodeRow(u.type_id, u.row(row), wbuf[0..Ed]);
                    for (sc2.route_order[first..last]) |slot| {
                        const x = sc2.xh[(slot / hp.n_expert_used) * Ed ..][0..Ed];
                        sc2.act_up[slot * Fd + row] = kernels.dotDense(wbuf[0..Ed], x);
                    }
                }
            }
        }.f);
        for (0..n * U) |slot| {
            const a = sc.act[slot * F ..][0..F];
            for (a, sc.act_up[slot * F ..][0..F]) |*g, u| g.* = kernels.silu(g.*) * u;
            if (l.down_signs) |signs| kernels.rotate(a, 128, signs);
        }
        // Phase 2: down rows; each task owns one output row of one expert.
        s.pool.run(h.n_expert * E, 64, &ctx, struct {
            fn f(c: *const Ctx, start: usize, end: usize) void {
                const hp = c.s.m.h;
                const sc2 = &c.s.scratch;
                const Fd = hp.n_ff_exp;
                const Ed = hp.n_embd;
                var wbuf: [kernels.max_cols]f32 align(64) = undefined;
                for (start..end) |task| {
                    const e = task / Ed;
                    const row = task % Ed;
                    const first = c.starts[e];
                    const last = c.starts[e + 1];
                    if (first == last) continue;
                    const d = c.l.expertDown(hp, e);
                    if (last - first == 1) {
                        const slot = sc2.route_order[first];
                        sc2.down[slot * Ed + row] = kernels.dotRow(d.type_id, d.row(row), sc2.act[slot * Fd ..][0..Fd]);
                        continue;
                    }
                    kernels.decodeRow(d.type_id, d.row(row), wbuf[0..Fd]);
                    for (sc2.route_order[first..last]) |slot|
                        sc2.down[slot * Ed + row] = kernels.dotDense(wbuf[0..Fd], sc2.act[slot * Fd ..][0..Fd]);
                }
            }
        }.f);
        for (0..n) |t| {
            const out = sc.block_out[t * E ..][0..E];
            for (0..U) |k| {
                const slot = t * U + k;
                const wgt = sc.route_w[slot];
                for (out, sc.down[slot * E ..][0..E]) |*o, v| o.* += wgt * v;
            }
        }
        Observer.emit(s.observer, "ffn_moe_out", il, sc.block_out[0 .. n * E], n);

        // Shared expert with its sigmoid gate.
        const SF = h.n_ff_shexp;
        kernels.matmul(s.pool, l.sh_gate, sc.cur[0 .. n * E], sc.act, n);
        kernels.matmul(s.pool, l.sh_up, sc.cur[0 .. n * E], sc.act_up, n);
        for (sc.act[0 .. n * SF], sc.act_up[0 .. n * SF]) |*g, u| g.* = kernels.silu(g.*) * u;
        kernels.matmul(s.pool, l.sh_down, sc.act[0 .. n * SF], sc.down, n);
        for (0..n) |t| {
            const gate = kernels.sigmoid(kernels.dotDense(l.sh_gate_inp, sc.cur[t * E ..][0..E]));
            for (sc.block_out[t * E ..][0..E], sc.down[t * E ..][0..E]) |*o, v| o.* += v * gate;
        }
    }
};

/// One gated delta rule step on a transposed state: s[j * S + i] = S[i][j].
pub fn gdnStep(state: []f32, q: []const f32, k: []const f32, v: []const f32, decay: f32, beta: f32, scale: f32, out: []f32) void {
    @setFloatMode(.optimized);
    const S = q.len;
    for (state) |*x| x.* *= decay;
    for (0..S) |j| {
        const row = state[j * S ..][0..S];
        const delta = (v[j] - kernels.dotDense(row, k)) * beta;
        for (row, k) |*x, kk| x.* += kk * delta;
        out[j] = kernels.dotDense(row, q) * scale;
    }
}

/// Selects the largest values; ties prefer the lower index.
fn topK(values: []const f32, ids: []u32, weights: []f32) void {
    for (ids, weights) |*id, *w| {
        id.* = std.math.maxInt(u32);
        w.* = -std.math.inf(f32);
    }
    for (values, 0..) |v, i| {
        if (v <= weights[weights.len - 1] and ids[ids.len - 1] != std.math.maxInt(u32)) continue;
        var pos = weights.len - 1;
        while (pos > 0 and (v > weights[pos - 1] or ids[pos - 1] == std.math.maxInt(u32))) : (pos -= 1) {
            weights[pos] = weights[pos - 1];
            ids[pos] = ids[pos - 1];
        }
        weights[pos] = v;
        ids[pos] = @intCast(i);
    }
}

const Scratch = struct {
    capacity: usize = 0,
    sel_stride: usize = 0,
    res: []f32 = &.{},
    cur: []f32 = &.{},
    xn: []f32 = &.{},
    lo: []f32 = &.{},
    lo_big: []f32 = &.{},
    gate: []f32 = &.{},
    inject: []f32 = &.{},
    block_out: []f32 = &.{},
    ple_emb: []f32 = &.{},
    qkv: []f32 = &.{},
    conv_out: []f32 = &.{},
    z: []f32 = &.{},
    beta: []f32 = &.{},
    alpha: []f32 = &.{},
    gdn_out: []f32 = &.{},
    idx_k: []f32 = &.{},
    idx_q: []f32 = &.{},
    qfull: []f32 = &.{},
    kcur: []f32 = &.{},
    vcur: []f32 = &.{},
    attn: []f32 = &.{},
    selected: []u32 = &.{},
    selected_len: []u32 = &.{},
    block_scores: []f32 = &.{},
    block_order: []u32 = &.{},
    router: []f32 = &.{},
    route_ids: []u32 = &.{},
    route_w: []f32 = &.{},
    route_order: []u32 = &.{},
    xh: []f32 = &.{},
    act: []f32 = &.{},
    act_up: []f32 = &.{},
    down: []f32 = &.{},

    fn deinit(sc: *Scratch, allocator: Allocator) void {
        const info = @typeInfo(Scratch).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == []f32 or T == []u32) allocator.free(@field(sc, name));
        }
        sc.* = .{};
    }

    /// Sizes every buffer for n tokens per forward call. Existing buffers are kept when large enough.
    fn ensure(sc: *Scratch, allocator: Allocator, h: model_mod.Hparams, n: usize, max_ctx: usize) !void {
        if (n <= sc.capacity) return;
        sc.deinit(allocator);
        const E: usize = h.n_embd;
        const HD: usize = h.hcDim();
        const CH: usize = h.convChannels();
        const U: usize = h.n_expert_used;
        const ffw: usize = @max(U * h.n_ff_exp, h.n_ff_shexp);
        sc.sel_stride = h.idx_top_k + 64;
        const blocks = max_ctx / 2 + 2;
        inline for (.{
            .{ "res", n * HD }, .{ "cur", n * E }, .{ "xn", n * HD }, .{ "lo", n * h.hc_rank },
            .{ "lo_big", HD }, .{ "gate", n * HD }, .{ "inject", n * h.hc }, .{ "block_out", n * E },
            .{ "ple_emb", n * E }, .{ "qkv", n * CH }, .{ "conv_out", n * CH }, .{ "z", n * h.ssm_inner },
            .{ "beta", n * h.ssm_v_heads }, .{ "alpha", n * h.ssm_v_heads }, .{ "gdn_out", n * h.ssm_inner },
            .{ "idx_k", n * h.idx_dim }, .{ "idx_q", n * h.idx_heads * h.idx_dim },
            .{ "qfull", n * h.n_head * 2 * h.head_dim }, .{ "kcur", n * h.n_head_kv * h.head_dim },
            .{ "vcur", n * h.n_head_kv * h.head_dim }, .{ "attn", n * h.n_head * h.head_dim },
            .{ "selected", n * sc.sel_stride }, .{ "selected_len", n }, .{ "block_scores", blocks },
            .{ "block_order", blocks }, .{ "router", n * h.n_expert }, .{ "route_ids", n * U },
            .{ "route_w", n * U }, .{ "route_order", n * U }, .{ "xh", n * E }, .{ "act", n * ffw },
            .{ "act_up", n * ffw }, .{ "down", n * U * E },
        }) |entry| {
            const T = @typeInfo(@TypeOf(@field(sc, entry[0]))).pointer.child;
            @field(sc, entry[0]) = try allocator.alloc(T, entry[1]);
        }
        sc.capacity = n;
    }
};
