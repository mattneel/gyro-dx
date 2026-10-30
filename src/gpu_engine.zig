//! Direct3D 12 execution of the Qwen4Exp graph. The schedule mirrors engine.zig (CPU).
//! Weights stay packed in video memory. Routed experts live in slot arenas filled by
//! DirectStorage, either fully resident or as a demand cache driven by router readback.
const std = @import("std");
const core = @import("gyro_dx");
const gpu = @import("gpu.zig");
const c = @import("native_abi.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Model = core.model.Model;
const kernels = core.kernels;
const cache_mod = core.cache;

pub const ExpertMode = enum { auto, resident, cache };

pub const Options = struct {
    adapter: u32 = 0,
    debug: bool = false,
    kernel_dir: []const u8,
    model_path: []const u8,
    expert_mode: ExpertMode = .auto,
    /// Expert cache bytes in cache mode. Null derives the size from the adapter budget.
    cache_bytes: ?u64 = null,
    /// Video memory kept free for other allocations and the desktop.
    reserve_bytes: u64 = 1536 << 20,
    profile: bool = false,
    /// Cache mode: predict the next layer's experts from the current router input and load early.
    prefetch: bool = true,
};

/// Activation rows per GPU pass. Larger prompts run in passes of this size.
pub const max_tok = 8;

const Kernel = enum {
    mv_q8, q8_repack, mv_bf16, mv_f32, moe_route, moe_resolve, moe_gyro, moe_act, rotate128, moe_combine,
    norm_rows, hc_mix, hc_combine, hc_init, ew, conv_causal, gdn_gates, gdn_step,
    rope_neox, store_f16, attn_chunk, attn_merge, qsa_pool, qsa_score, qsa_select, ple_mix, hc_pre, hc_post,
};

const Weight = struct { srv: u32, off: u32, type_id: u32, rows: u32, cols: u32, row_bytes: u32 };
const Vec = struct { srv: u32, off: u32 };

const HcW = struct { norm: Vec, down: Weight, up: Weight, inject: ?Weight };
const LinW = struct { qkv: Weight, z: Weight, beta: Weight, alpha: Weight, out: Weight, conv: Vec, dt: Vec, a: Vec, norm: Vec };
const FullW = struct {
    wq: Weight, wk: Weight, wv: Weight, wo: Weight, q_norm: Vec, k_norm: Vec,
    idx_q: Weight, idx_k: Weight, idx_q_norm: Vec, idx_k_norm: Vec, ratio: u32,
};
const PleW = struct { key: Weight, value: Weight, norm_key: Vec, norm_query: Vec, norm_conv: Vec, conv: Vec };

const LayerW = struct {
    hc_attn: HcW,
    hc_ffn: HcW,
    lin: ?LinW,
    full: ?FullW,
    ple: ?PleW,
    router: Weight,
    sh_gate: Weight,
    sh_up: Weight,
    sh_down: Weight,
    sh_gate_inp: Weight,
    gate_signs: ?Vec,
    down_signs: ?Vec,
    expert_type: u32,
    gate_bytes: u32,
    bundle_bytes: u32,
    gate_tensor: core.gguf.Tensor,
    up_tensor: core.gguf.Tensor,
    down_tensor: core.gguf.Tensor,
};

const LinState = struct { conv: gpu.Buffer, ssm: gpu.Buffer };
const FullState = struct { k: gpu.Buffer, v: gpu.Buffer, idx_raw: gpu.Buffer, pooled: gpu.Buffer, pooled_valid: u32 = 0 };

pub const Phase = enum { inputs, ple, hc_mix, lin_proj, lin_conv, lin_step, linear_attn, full_attn, hc_combine, router_shared, experts, head, output };

pub const Stats = struct {
    phase_ns: [@typeInfo(Phase).@"enum".field_names.len]u64 = @splat(0),
    tokens: u64 = 0,
    expert_requests: u64 = 0,
    expert_hits: u64 = 0,
    expert_misses: u64 = 0,
    bytes_loaded: u64 = 0,
    sync_points: u64 = 0,
    storage_wait_ns: u64 = 0,
    prefetch_loads: u64 = 0,
    prefetch_used: u64 = 0,
};

pub const Engine = struct {
    allocator: Allocator,
    io: Io,
    m: *Model,
    dev: gpu.Device,
    file: *c.gd_file,
    pipes: std.EnumArray(Kernel, *c.gd_pipeline),
    owned: std.ArrayList(gpu.Buffer),
    aux: gpu.Buffer,
    codebook: gpu.Buffer,
    layers: []LayerW,
    head: HcW,
    output: Weight,
    // activations
    res: gpu.Buffer, cur: gpu.Buffer, xn: gpu.Buffer, gate: gpu.Buffer, lo: gpu.Buffer, inject: gpu.Buffer,
    block: gpu.Buffer, qkv: gpu.Buffer, conv_out: gpu.Buffer, z: gpu.Buffer, beta: gpu.Buffer, alpha: gpu.Buffer,
    gdn_out: gpu.Buffer, qfull: gpu.Buffer, kcur: gpu.Buffer, vcur: gpu.Buffer, attn: gpu.Buffer,
    idx_q: gpu.Buffer, idx_k: gpu.Buffer, part: gpu.Buffer, sel: gpu.Buffer, sel_len: gpu.Buffer, scores: gpu.Buffer,
    hc_scale: gpu.Buffer, router: gpu.Buffer, router2: gpu.Buffer, ids2: gpu.Buffer, weights2: gpu.Buffer, ids: gpu.Buffer, weights: gpu.Buffer, table: gpu.Buffer, xh: gpu.Buffer, gu: gpu.Buffer,
    act: gpu.Buffer, down: gpu.Buffer, sh: gpu.Buffer, sh_act: gpu.Buffer, sh_out: gpu.Buffer, shg: gpu.Buffer,
    // host-visible
    emb_in: gpu.Buffer, ple_in: gpu.Buffer, logits: gpu.Buffer, ids_host: gpu.Buffer, table_host: gpu.Buffer,
    // state
    lin_state: []LinState,
    full_state: []FullState,
    layer_slot: []u32,
    ple_hist: gpu.Buffer,
    ple_history: core.ple.History = .{},
    pos: u32 = 0,
    max_ctx: u32,
    sel_stride: u32,
    chunks: u32,
    blocks_max: u32,
    // experts
    arenas: []gpu.Buffer,
    per_arena: u32,
    slot_stride: u32,
    total_slots: u32,
    resident: bool,
    lookup: gpu.Buffer,
    cache: cache_mod.Cache,
    slot_of: []u32,
    pending: std.ArrayList(cache_mod.Lease),
    prefetches: std.ArrayList(struct { fence: cache_mod.LoadFence, last: u64 }),
    last_submitted: u64 = 0,
    prefetch: bool = true,
    load_ticket: u64 = 0,
    compute_done: u64 = 0,
    stats: Stats = .{},
    options: Options,
    scratch: ?gpu.Buffer = null,
    marks: [4096]Phase = undefined,
    mark_count: u32 = 0,

    /// Records a GPU timestamp closing the given phase when profiling.
    fn mark(e: *Engine, phase: Phase) !void {
        if (!e.options.profile) return;
        const index = try e.dev.timestamp();
        e.marks[index] = phase;
        e.mark_count = index + 1;
    }

    fn collectMarks(e: *Engine) !void {
        if (!e.options.profile or e.mark_count < 2) return;
        var ticks: [4096]u64 = undefined;
        const freq = try e.dev.timestamps(ticks[0..e.mark_count]);
        for (1..e.mark_count) |i| {
            const dt = ticks[i] -% ticks[i - 1];
            e.stats.phase_ns[@intFromEnum(e.marks[i])] += dt * 1_000_000_000 / freq;
        }
        e.mark_count = 0;
    }

    // ---------------- creation ----------------

    pub fn create(io: Io, allocator: Allocator, m: *Model, max_ctx: u32, options: Options) !*Engine {
        const e = try allocator.create(Engine);
        errdefer allocator.destroy(e);
        e.* = undefined;
        e.allocator = allocator;
        e.io = io;
        e.m = m;
        e.options = options;
        e.max_ctx = max_ctx;
        e.owned = .empty;
        e.pending = .empty;
        e.prefetches = .empty;
        e.last_submitted = 0;
        e.prefetch = options.prefetch;
        e.ple_history = .{};
        e.pos = 0;
        e.stats = .{};
        e.load_ticket = 0;
        e.compute_done = 0;
        e.scratch = null;
        e.mark_count = 0;
        e.dev = try gpu.Device.init(options.adapter, options.debug);
        errdefer e.dev.deinit();
        const path_z = try allocator.dupeSentinel(u8, options.model_path, 0);
        defer allocator.free(path_z);
        e.file = try e.dev.openFile(path_z);
        try e.loadKernels();
        try e.loadWeights();
        if (e.scratch) |b| e.dev.release(b);
        e.scratch = null;
        try e.allocateActivations();
        try e.allocateState();
        try e.setupExperts();
        return e;
    }

    pub fn destroy(e: *Engine) void {
        const allocator = e.allocator;
        if (e.resident == false) e.cache.deinit();
        allocator.free(e.slot_of);
        allocator.free(e.arenas);
        allocator.free(e.layers);
        allocator.free(e.lin_state);
        allocator.free(e.full_state);
        allocator.free(e.layer_slot);
        e.pending.deinit(allocator);
        e.prefetches.deinit(allocator);
        e.owned.deinit(allocator);
        e.dev.deinit();
        allocator.destroy(e);
    }

    fn own(e: *Engine, bytes: u64, kind: gpu.Kind) !gpu.Buffer {
        const b = try e.dev.buffer(bytes, kind);
        try e.owned.append(e.allocator, b);
        return b;
    }

    fn loadKernels(e: *Engine) !void {
        var dir = try Io.Dir.cwd().openDir(e.io, e.options.kernel_dir, .{});
        defer dir.close(e.io);
        inline for (@typeInfo(Kernel).@"enum".field_names) |name| {
            const bytes = dir.readFileAlloc(e.io, name ++ ".dxil", e.allocator, .limited(4 << 20)) catch |err| {
                std.log.err("kernel {s}.dxil: {s}", .{ name, @errorName(err) });
                return err;
            };
            defer e.allocator.free(bytes);
            e.pipes.set(@field(Kernel, name), try e.dev.pipeline(bytes));
        }
    }

    // ---------------- weights ----------------

    const AuxPlan = struct {
        bytes: u64 = 0,
        items: std.ArrayList(struct { tensor: core.gguf.Tensor, off: u64 }) = .empty,
    };

    fn tensor(e: *Engine, name: []const u8) !core.gguf.Tensor {
        return e.m.meta.tensor(name) orelse {
            std.log.err("missing tensor {s}", .{name});
            return error.MissingTensor;
        };
    }

    /// Reads a file range into a buffer through DirectStorage, in staging-sized pieces.
    fn streamInto(e: *Engine, b: gpu.Buffer, buffer_offset: u64, file_offset: u64, bytes: u64) !u64 {
        var ranges: [256]c.gd_range = undefined;
        var n: usize = 0;
        var done: u64 = 0;
        var ticket: u64 = 0;
        const piece: u64 = 16 << 20;
        while (done < bytes) {
            const len = @min(piece, bytes - done);
            ranges[n] = .{ .file_offset = file_offset + done, .buffer_offset = buffer_offset + done, .bytes = @intCast(len), .reserved = 0 };
            n += 1;
            done += len;
            if (n == ranges.len or done == bytes) {
                ticket = try e.dev.readRanges(e.file, b, ranges[0..n], 0);
                n = 0;
            }
        }
        return ticket;
    }

    fn matrix(e: *Engine, name: []const u8, cols: usize, rows: usize) !Weight {
        const t = try e.tensor(name);
        const row_bytes = try kernels.rowBytes(t.type_id, cols);
        if (t.byte_length.? != row_bytes * rows) return error.InvalidTensorShape;
        const b = try e.own(t.byte_length.?, .device);
        if (t.type_id != kernels.type_q8_0) {
            _ = try e.streamInto(b, 0, t.file_offset, t.byte_length.?);
            return .{ .srv = b.srv, .off = 0, .type_id = t.type_id, .rows = @intCast(rows), .cols = @intCast(cols), .row_bytes = @intCast(row_bytes) };
        }
        // Q8_0 goes through a scratch buffer and is repacked into aligned quants plus scales.
        if (cols % 64 != 0) return error.UnsupportedTensorShape;
        if (e.scratch == null or e.scratch.?.bytes < t.byte_length.?) {
            if (e.scratch) |old| e.dev.release(old);
            e.scratch = try e.dev.buffer(@max(t.byte_length.?, 256 << 20), .device);
        }
        const scratch = e.scratch.?;
        try e.dev.waitStorage(try e.streamInto(scratch, 0, t.file_offset, t.byte_length.?));
        const packed_row: u32 = @intCast(cols + cols / 16);
        const blocks: u32 = @intCast(rows * (cols / 32));
        const groups = (blocks + 255) / 256;
        const cst = [16]u32{ scratch.srv, 0, @intCast(rows), @intCast(cols), 0, 0, 0, 0, b.uav, 0, 0, 0, 0, 0, packed_row, 0 };
        try e.dev.begin();
        try e.dispatch(.q8_repack, &cst, @min(groups, 65535), (groups + 65534) / 65535, 1);
        try e.dev.wait(try e.dev.submit());
        return .{ .srv = b.srv, .off = 0, .type_id = t.type_id, .rows = @intCast(rows), .cols = @intCast(cols), .row_bytes = packed_row };
    }

    fn vec(e: *Engine, plan: *AuxPlan, name: []const u8) !Vec {
        const t = try e.tensor(name);
        if (t.type_id != kernels.type_f32) return error.UnsupportedTensorType;
        const off = plan.bytes;
        try plan.items.append(e.allocator, .{ .tensor = t, .off = off });
        plan.bytes += std.mem.alignForward(u64, t.byte_length.?, 256);
        return .{ .srv = 0, .off = @intCast(off / 4) };
    }

    fn fixVec(e: *Engine, v: *Vec) void {
        v.srv = e.aux.srv;
    }

    fn loadWeights(e: *Engine) !void {
        const h = e.m.h;
        const HD = h.hcDim();
        var plan = AuxPlan{};
        defer plan.items.deinit(e.allocator);
        // Sign vectors and the F32 vectors live in one auxiliary buffer.
        const sign_count = e.m.sign_store.items.len;
        plan.bytes = std.mem.alignForward(u64, sign_count * 4, 256);
        e.layers = try e.allocator.alloc(LayerW, h.n_layer);
        var buf: [128]u8 = undefined;
        const name = struct {
            fn f(b: []u8, il: usize, s: []const u8) []const u8 {
                return std.fmt.bufPrint(b, "blk.{d}.{s}", .{ il, s }) catch unreachable;
            }
        }.f;
        e.head = .{
            .norm = try e.vec(&plan, "output_hc_norm.weight"),
            .down = try e.matrix("output_hc_down.weight", HD, h.hc_rank),
            .up = try e.matrix("output_hc_up.weight", h.hc_rank, HD),
            .inject = null,
        };
        e.output = try e.matrix(if (e.m.meta.tensor("output.weight") != null) "output.weight" else "token_embd.weight", h.n_embd, h.n_vocab);
        for (e.layers, e.m.layers, 0..) |*lw, *ml, il| {
            lw.hc_attn = .{ .norm = try e.vec(&plan, name(&buf, il, "hc_attn_norm.weight")),
                .down = try e.matrix(name(&buf, il, "hc_attn_down.weight"), HD, h.hc_rank),
                .up = try e.matrix(name(&buf, il, "hc_attn_up.weight"), h.hc_rank, HD),
                .inject = try e.matrix(name(&buf, il, "hc_attn_inject.weight"), HD, h.hc) };
            lw.hc_ffn = .{ .norm = try e.vec(&plan, name(&buf, il, "hc_ffn_norm.weight")),
                .down = try e.matrix(name(&buf, il, "hc_ffn_down.weight"), HD, h.hc_rank),
                .up = try e.matrix(name(&buf, il, "hc_ffn_up.weight"), h.hc_rank, HD),
                .inject = try e.matrix(name(&buf, il, "hc_ffn_inject.weight"), HD, h.hc) };
            lw.lin = null;
            lw.full = null;
            switch (ml.mixer) {
                .linear => {
                    const CH = h.convChannels();
                    lw.lin = .{
                        .qkv = try e.matrix(name(&buf, il, "attn_qkv.weight"), h.n_embd, CH),
                        .z = try e.matrix(name(&buf, il, "attn_gate.weight"), h.n_embd, h.ssm_inner),
                        .beta = try e.matrix(name(&buf, il, "ssm_beta.weight"), h.n_embd, h.ssm_v_heads),
                        .alpha = try e.matrix(name(&buf, il, "ssm_alpha.weight"), h.n_embd, h.ssm_v_heads),
                        .out = try e.matrix(name(&buf, il, "ssm_out.weight"), h.ssm_inner, h.n_embd),
                        .conv = try e.vec(&plan, name(&buf, il, "ssm_conv1d.weight")),
                        .dt = try e.vec(&plan, name(&buf, il, "ssm_dt.bias")),
                        .a = try e.vec(&plan, name(&buf, il, "ssm_a")),
                        .norm = try e.vec(&plan, name(&buf, il, "ssm_norm.weight")),
                    };
                },
                .full => |f| {
                    lw.full = .{
                        .wq = try e.matrix(name(&buf, il, "attn_q.weight"), h.n_embd, h.head_dim * h.n_head * 2),
                        .wk = try e.matrix(name(&buf, il, "attn_k.weight"), h.n_embd, h.head_dim * h.n_head_kv),
                        .wv = try e.matrix(name(&buf, il, "attn_v.weight"), h.n_embd, h.head_dim * h.n_head_kv),
                        .wo = try e.matrix(name(&buf, il, "attn_output.weight"), h.head_dim * h.n_head, h.n_embd),
                        .q_norm = try e.vec(&plan, name(&buf, il, "attn_q_norm.weight")),
                        .k_norm = try e.vec(&plan, name(&buf, il, "attn_k_norm.weight")),
                        .idx_q = try e.matrix(name(&buf, il, "indexer.q_proj.weight"), h.n_embd, h.idx_heads * h.idx_dim),
                        .idx_k = try e.matrix(name(&buf, il, "indexer.k_proj.weight"), h.n_embd, h.idx_dim),
                        .idx_q_norm = try e.vec(&plan, name(&buf, il, "indexer.q_norm.weight")),
                        .idx_k_norm = try e.vec(&plan, name(&buf, il, "indexer.k_norm.weight")),
                        .ratio = f.compress_ratio,
                    };
                },
            }
            lw.ple = if (ml.ple != null) .{
                .key = try e.matrix(name(&buf, il, "ple_key.weight"), h.n_embd, HD),
                .value = try e.matrix(name(&buf, il, "ple_value.weight"), h.n_embd, h.n_embd),
                .norm_key = try e.vec(&plan, name(&buf, il, "ple_norm_key.weight")),
                .norm_query = try e.vec(&plan, name(&buf, il, "ple_norm_query.weight")),
                .norm_conv = try e.vec(&plan, name(&buf, il, "ple_norm_conv.weight")),
                .conv = try e.vec(&plan, name(&buf, il, "ple_conv1d.weight")),
            } else null;
            lw.router = try e.matrix(name(&buf, il, "ffn_gate_inp.weight"), h.n_embd, h.n_expert);
            lw.sh_gate = try e.matrix(name(&buf, il, "ffn_gate_shexp.weight"), h.n_embd, h.n_ff_shexp);
            lw.sh_up = try e.matrix(name(&buf, il, "ffn_up_shexp.weight"), h.n_embd, h.n_ff_shexp);
            lw.sh_down = try e.matrix(name(&buf, il, "ffn_down_shexp.weight"), h.n_ff_shexp, h.n_embd);
            const shv = try e.vec(&plan, name(&buf, il, "ffn_gate_inp_shexp.weight"));
            lw.sh_gate_inp = .{ .srv = 0, .off = shv.off * 4, .type_id = kernels.type_f32, .rows = 1, .cols = h.n_embd, .row_bytes = h.n_embd * 4 };
            lw.gate_signs = if (ml.gate_signs) |s| .{ .srv = 0, .off = @intCast((s.ptr - e.m.sign_store.items.ptr)) } else null;
            lw.down_signs = if (ml.down_signs) |s| .{ .srv = 0, .off = @intCast((s.ptr - e.m.sign_store.items.ptr)) } else null;
            if (ml.up_signs != null and (ml.gate_signs == null or ml.up_signs.?.ptr != ml.gate_signs.?.ptr)) return error.UnsupportedConfiguration;
            lw.gate_tensor = try e.tensor(name(&buf, il, "ffn_gate_exps.weight"));
            lw.up_tensor = try e.tensor(name(&buf, il, "ffn_up_exps.weight"));
            lw.down_tensor = try e.tensor(name(&buf, il, "ffn_down_exps.weight"));
            lw.expert_type = lw.gate_tensor.type_id;
            if (lw.up_tensor.type_id != lw.expert_type or lw.down_tensor.type_id != lw.expert_type) return error.UnsupportedConfiguration;
            lw.gate_bytes = @intCast(lw.gate_tensor.byte_length.? / h.n_expert);
            const down_bytes = lw.down_tensor.byte_length.? / h.n_expert;
            if (lw.up_tensor.byte_length.? / h.n_expert != lw.gate_bytes or down_bytes != lw.gate_bytes) return error.UnsupportedConfiguration;
            lw.bundle_bytes = lw.gate_bytes * 3;
        }
        // Auxiliary buffer: signs first, then every F32 vector.
        e.aux = try e.own(plan.bytes, .device);
        var last: u64 = 0;
        for (plan.items.items) |item| last = try e.streamInto(e.aux, item.off, item.tensor.file_offset, item.tensor.byte_length.?);
        if (last != 0) try e.dev.waitStorage(last);
        // Fix vector buffer indices now that the aux buffer exists.
        e.fixVec(&e.head.norm);
        for (e.layers) |*lw| {
            e.fixVec(&lw.hc_attn.norm);
            e.fixVec(&lw.hc_ffn.norm);
            if (lw.lin) |*l| {
                e.fixVec(&l.conv);
                e.fixVec(&l.dt);
                e.fixVec(&l.a);
                e.fixVec(&l.norm);
            }
            if (lw.full) |*f| {
                e.fixVec(&f.q_norm);
                e.fixVec(&f.k_norm);
                e.fixVec(&f.idx_q_norm);
                e.fixVec(&f.idx_k_norm);
            }
            if (lw.ple) |*p| {
                e.fixVec(&p.norm_key);
                e.fixVec(&p.norm_query);
                e.fixVec(&p.norm_conv);
                e.fixVec(&p.conv);
            }
            lw.sh_gate_inp.srv = e.aux.srv;
            if (lw.gate_signs) |*s| e.fixVec(s);
            if (lw.down_signs) |*s| e.fixVec(s);
        }
        // Signs and the codebook: host staging copied on the GPU.
        const staging = try e.dev.buffer(sign_count * 4 + 8192, .host);
        defer e.dev.release(staging);
        @memcpy(staging.floats()[0..sign_count], e.m.sign_store.items);
        @memcpy(staging.mapped.?[sign_count * 4 ..][0..8192], core.gyro.codebook_bytes);
        e.codebook = try e.own(8192, .device);
        try e.dev.begin();
        if (sign_count > 0) try e.ew(staging.uav, 0, 0, staging.uav, 0, 0, e.aux.uav, 0, 0, @intCast(sign_count), 1, 3);
        try e.ew(staging.uav, @intCast(sign_count), 0, staging.uav, 0, 0, e.codebook.uav, 0, 0, 2048, 1, 5);
        try e.dev.wait(try e.dev.submit());
        // Matrices were enqueued without waits; drain the storage queue once.
        try e.dev.waitStorage(e.dev.next_storage);
    }


    // ---------------- buffers ----------------

    fn allocateActivations(e: *Engine) !void {
        const h = e.m.h;
        const T: u64 = max_tok;
        const E: u64 = h.n_embd;
        const HD: u64 = h.hcDim();
        const CH: u64 = h.convChannels();
        const U: u64 = h.n_expert_used;
        e.sel_stride = h.idx_top_k + 64;
        e.chunks = (e.sel_stride + 255) / 256;
        e.blocks_max = e.max_ctx / 2 + 2;
        const f = 4;
        e.res = try e.own(T * HD * f, .device);
        e.cur = try e.own(T * E * f, .device);
        e.xn = try e.own(T * HD * f, .device);
        e.gate = try e.own(T * HD * f, .device);
        e.lo = try e.own(T * h.hc_rank * f, .device);
        e.inject = try e.own(T * h.hc * f, .device);
        e.hc_scale = try e.own(T * h.hc * f, .device);
        e.block = try e.own(T * E * f, .device);
        e.qkv = try e.own(T * @max(CH, HD) * f, .device);
        e.conv_out = try e.own(T * CH * f, .device);
        e.z = try e.own(T * h.ssm_inner * f, .device);
        e.beta = try e.own(T * h.ssm_v_heads * f, .device);
        e.alpha = try e.own(T * h.ssm_v_heads * f, .device);
        e.gdn_out = try e.own(T * h.ssm_inner * f, .device);
        e.qfull = try e.own(T * h.n_head * 2 * h.head_dim * f, .device);
        e.kcur = try e.own(T * h.n_head_kv * h.head_dim * f, .device);
        e.vcur = try e.own(T * h.n_head_kv * h.head_dim * f, .device);
        e.attn = try e.own(T * h.n_head * h.head_dim * f, .device);
        e.idx_q = try e.own(T * h.idx_heads * h.idx_dim * f, .device);
        e.idx_k = try e.own(T * h.idx_dim * f, .device);
        e.part = try e.own(T * h.n_head * e.chunks * (h.head_dim + 2) * f, .device);
        e.sel = try e.own(T * e.sel_stride * 4, .device);
        e.sel_len = try e.own(T * 4, .device);
        e.scores = try e.own(T * e.blocks_max * f, .device);
        e.router = try e.own(T * h.n_expert * f, .device);
        e.router2 = try e.own(T * h.n_expert * f, .device);
        e.ids2 = try e.own(T * U * 4, .device);
        e.weights2 = try e.own(T * U * f, .device);
        e.ids = try e.own(T * U * 4, .device);
        e.weights = try e.own(T * U * f, .device);
        e.table = try e.own(T * U * 24, .device);
        e.xh = try e.own(T * E * f, .device);
        e.gu = try e.own(T * U * 2 * h.n_ff_exp * f, .device);
        e.act = try e.own(T * U * h.n_ff_exp * f, .device);
        e.down = try e.own(T * U * E * f, .device);
        e.sh = try e.own(T * 2 * h.n_ff_shexp * f, .device);
        e.sh_act = try e.own(T * h.n_ff_shexp * f, .device);
        e.sh_out = try e.own(T * E * f, .device);
        e.shg = try e.own(T * f, .device);
        e.emb_in = try e.own(T * E * f, .host);
        e.ple_in = try e.own(T * E * f, .host);
        e.logits = try e.own(@as(u64, h.n_vocab) * f, .host);
        e.ids_host = try e.own(2 * T * U * 4, .host);
        e.table_host = try e.own(T * U * 24, .host);
    }

    fn allocateState(e: *Engine) !void {
        const h = e.m.h;
        const ctx: u64 = e.max_ctx;
        var n_lin: usize = 0;
        var n_full: usize = 0;
        for (e.layers) |lw| {
            if (lw.lin != null) n_lin += 1;
            if (lw.full != null) n_full += 1;
        }
        e.lin_state = try e.allocator.alloc(LinState, n_lin);
        e.full_state = try e.allocator.alloc(FullState, n_full);
        e.layer_slot = try e.allocator.alloc(u32, e.layers.len);
        var li: u32 = 0;
        var fi: u32 = 0;
        for (e.layers, 0..) |lw, il| {
            if (lw.lin != null) {
                e.layer_slot[il] = li;
                e.lin_state[li] = .{
                    .conv = try e.own(@as(u64, h.ssm_conv - 1) * h.convChannels() * 4, .device),
                    .ssm = try e.own(@as(u64, h.ssm_v_heads) * h.ssm_state * h.ssm_state * 4, .device),
                };
                li += 1;
            } else {
                e.layer_slot[il] = fi;
                const ratio = lw.full.?.ratio;
                e.full_state[fi] = .{
                    .k = try e.own(ctx * h.n_head_kv * h.head_dim * 2, .device),
                    .v = try e.own(ctx * h.n_head_kv * h.head_dim * 2, .device),
                    .idx_raw = try e.own(ctx * h.idx_dim * 2, .device),
                    .pooled = try e.own((ctx / ratio + 1) * h.idx_dim * 4, .device),
                };
                fi += 1;
            }
        }
        e.ple_hist = try e.own(@as(u64, (h.ple_conv - 1) * h.ple_dilation) * h.hcDim() * 4, .device);
        try e.clearState();
    }

    /// Zeroes recurrent state on the GPU.
    fn clearState(e: *Engine) !void {
        try e.dev.begin();
        for (e.lin_state) |ls| {
            try e.fill(ls.conv);
            try e.fill(ls.ssm);
        }
        try e.fill(e.ple_hist);
        try e.dev.wait(try e.dev.submit());
        for (e.full_state) |*fs| fs.pooled_valid = 0;
        e.pos = 0;
        e.ple_history.reset();
    }

    fn fill(e: *Engine, b: gpu.Buffer) !void {
        // y = a + b with a = b = zero row is not available; copy from the zero-initialized host logits
        // would race. Use the elementwise copy of a zero host buffer region instead.
        const n: u32 = @intCast(b.bytes / 4);
        var done: u32 = 0;
        while (done < n) {
            const chunk = @min(n - done, @as(u32, @intCast(e.zeros().bytes / 4)));
            try e.ew(e.zeros().uav, 0, 0, e.zeros().uav, 0, 0, b.uav, done, 0, chunk, 1, 5);
            done += chunk;
        }
        try e.dev.barrier();
    }

    fn zeros(e: *Engine) gpu.Buffer {
        // The PLE input host buffer stays zero outside forward passes.
        return e.ple_in;
    }

    // ---------------- experts ----------------

    fn setupExperts(e: *Engine) !void {
        const h = e.m.h;
        var max_bundle: u32 = 0;
        for (e.layers) |lw| max_bundle = @max(max_bundle, lw.bundle_bytes);
        e.slot_stride = @intCast(std.mem.alignForward(u64, max_bundle, 256));
        const all_slots: u32 = h.n_layer * h.n_expert;
        const arena_limit: u64 = 2 << 30;
        e.per_arena = @intCast(arena_limit / e.slot_stride);
        const mem = try e.dev.budget();
        const free = mem.budget -| mem.usage;
        const want_all: u64 = @as(u64, all_slots) * e.slot_stride;
        var slots: u32 = all_slots;
        switch (e.options.expert_mode) {
            .resident => {},
            .cache => slots = @intCast(@min(all_slots, (e.options.cache_bytes orelse (free -| e.options.reserve_bytes)) / e.slot_stride)),
            .auto => if (want_all + e.options.reserve_bytes > free) {
                slots = @intCast(@min(all_slots, (e.options.cache_bytes orelse (free -| e.options.reserve_bytes)) / e.slot_stride));
            },
        }
        // A cached layer needs room for the experts of one pass.
        const min_slots = h.n_expert_used * max_tok * 2;
        if (slots < min_slots) return error.InsufficientVideoMemory;
        e.resident = slots == all_slots;
        const n_arenas = (slots + e.per_arena - 1) / e.per_arena;
        if (n_arenas > 8) return error.TooManyArenas;
        e.arenas = try e.allocator.alloc(gpu.Buffer, n_arenas);
        var remaining = slots;
        for (e.arenas) |*a| {
            const here = @min(remaining, e.per_arena);
            a.* = try e.own(@as(u64, here) * e.slot_stride, .device);
            remaining -= here;
        }
        e.total_slots = slots;
        e.slot_of = try e.allocator.alloc(u32, all_slots);
        @memset(e.slot_of, std.math.maxInt(u32));
        e.lookup = try e.own(@as(u64, all_slots) * 4, .host);
        if (e.resident) {
            // Every expert receives a fixed slot; bulk-load all bundles.
            var slot: u32 = 0;
            var last: u64 = 0;
            for (e.layers, 0..) |lw, il| {
                for (0..h.n_expert) |ex| {
                    e.slot_of[il * h.n_expert + ex] = slot;
                    last = try e.loadBundle(lw, @intCast(ex), slot, 0);
                    slot += 1;
                }
            }
            try e.dev.waitStorage(last);
            @memcpy(e.lookup.words()[0..all_slots], e.slot_of);
            e.stats.bytes_loaded += want_all;
        } else {
            e.cache = try cache_mod.Cache.init(e.allocator, slots, e.slot_stride);
        }
        std.log.info("experts: {s}, {d} of {d} slots ({d} MiB in {d} arenas)", .{
            if (e.resident) "resident" else "demand cache", slots, all_slots,
            (@as(u64, slots) * e.slot_stride) >> 20, n_arenas,
        });
    }

    fn slotLocation(e: *const Engine, slot: u32) struct { arena: gpu.Buffer, offset: u32 } {
        return .{ .arena = e.arenas[slot / e.per_arena], .offset = (slot % e.per_arena) * e.slot_stride };
    }

    /// Enqueues the three projection ranges of one expert into a slot.
    fn loadBundle(e: *Engine, lw: LayerW, expert: u32, slot: u32, after_compute: u64) !u64 {
        const loc = e.slotLocation(slot);
        const g = try lw.gate_tensor.expertRange(expert);
        const u = try lw.up_tensor.expertRange(expert);
        const d = try lw.down_tensor.expertRange(expert);
        const ranges = [3]c.gd_range{
            .{ .file_offset = g.file_offset, .buffer_offset = loc.offset, .bytes = @intCast(g.bytes), .reserved = 0 },
            .{ .file_offset = u.file_offset, .buffer_offset = loc.offset + lw.gate_bytes, .bytes = @intCast(u.bytes), .reserved = 0 },
            .{ .file_offset = d.file_offset, .buffer_offset = loc.offset + 2 * lw.gate_bytes, .bytes = @intCast(d.bytes), .reserved = 0 },
        };
        return e.dev.readRanges(e.file, loc.arena, &ranges, after_compute);
    }

    /// Demand path for one layer: reads the routed IDs, admits them as one transaction, loads
    /// misses with one storage batch per arena, verifies the status, and writes the slot table.
    /// Slots that an earlier prefetch is still loading are awaited and resolved first.
    fn admitLayer(e: *Engine, il: usize, n_tok: usize) !void {
        const h = e.m.h;
        const U = h.n_expert_used;
        const lw = e.layers[il];
        var leases: [max_tok * 16]cache_mod.Lease = undefined;
        const n = n_tok * U;
        const ids = e.ids_host.words()[0..n];
        e.stats.expert_requests += n;
        const issued = try e.enqueueLoads(il, ids, &leases);
        // Resolve prefetched slots this layer depends on.
        for (leases[0..n]) |lease| {
            const slot = try e.cache.get(lease);
            if (slot.state != .loading) continue;
            if (issued.misses > 0 and slot.load_fence.value == issued.fence.value) continue;
            try e.finishPrefetch(slot.load_fence);
        }
        if (issued.misses > 0) {
            const start = Io.Clock.awake.now(e.io);
            const ok = if (e.dev.waitStorage(issued.last)) true else |_| false;
            e.stats.storage_wait_ns += @intCast(start.durationTo(Io.Clock.awake.now(e.io)).nanoseconds);
            try e.cache.completeLoad(issued.fence, .{ .value = issued.last }, ok);
            if (!ok) return error.ExpertLoadFailed;
        }
        for (leases[0..n]) |lease| if ((try e.cache.get(lease)).state != .ready) return error.ExpertLoadFailed;
        e.stats.expert_misses += issued.misses;
        e.stats.expert_hits += n - issued.misses;
        const table = e.table_host.words();
        for (leases[0..n], 0..) |lease, k| {
            const loc = e.slotLocation(lease.slot);
            const o = loc.offset;
            table[k * 6 ..][0..6].* = .{ loc.arena.srv, o, loc.arena.srv, o + lw.gate_bytes, loc.arena.srv, o + 2 * lw.gate_bytes };
            try e.pending.append(e.allocator, lease);
        }
    }

    const Issued = struct { fence: cache_mod.LoadFence, last: u64, misses: u32 };

    /// Admits expert IDs for one layer and enqueues storage reads for the misses without waiting.
    /// Storage tickets are monotonic: the first batch names the cache fence, and a successful
    /// wait on the last batch proves that every earlier batch completed with success.
    fn enqueueLoads(e: *Engine, il: usize, ids: []const u32, leases: []cache_mod.Lease) !Issued {
        const h = e.m.h;
        const lw = e.layers[il];
        var requests: [max_tok * 16]cache_mod.Request = undefined;
        for (ids, 0..) |ex, k| {
            if (ex >= h.n_expert) return error.InvalidExpertId;
            requests[k] = .{ .key = .{ .layer = @intCast(il), .expert = ex }, .bytes = lw.bundle_bytes };
        }
        const fence = cache_mod.LoadFence{ .value = e.dev.next_storage + 1 };
        try e.cache.observeCompute(.{ .value = e.compute_done });
        try e.cache.acquireBatch(requests[0..ids.len], fence, leases[0..ids.len]);
        var ranges: [8][max_tok * 16 * 3]c.gd_range = undefined;
        var counts: [8]usize = @splat(0);
        var misses: u32 = 0;
        for (leases[0..ids.len], 0..) |lease, k| {
            const slot = try e.cache.get(lease);
            if (slot.state != .loading or slot.load_fence.value != fence.value) continue;
            var first = true;
            for (leases[0..k]) |prev| if (prev.slot == lease.slot) {
                first = false;
            };
            if (!first) continue;
            if (e.slotKeyIndex(lease.slot)) |o| e.slot_of[o] = std.math.maxInt(u32);
            const ex = requests[k].key.expert;
            e.slot_of[il * h.n_expert + ex] = lease.slot;
            const loc = e.slotLocation(lease.slot);
            const ai = lease.slot / e.per_arena;
            const parts = [3]core.gguf.Range{ try lw.gate_tensor.expertRange(ex), try lw.up_tensor.expertRange(ex), try lw.down_tensor.expertRange(ex) };
            for (parts, 0..) |r, j| {
                ranges[ai][counts[ai]] = .{ .file_offset = r.file_offset, .buffer_offset = loc.offset + @as(u32, @intCast(j)) * lw.gate_bytes,
                    .bytes = @intCast(r.bytes), .reserved = 0 };
                counts[ai] += 1;
            }
            misses += 1;
        }
        var last: u64 = 0;
        for (counts, 0..) |count, ai| {
            if (count == 0) continue;
            // Writes wait for the last submitted compute work, the latest user of any evicted slot.
            last = try e.dev.readRanges(e.file, e.arenas[ai], ranges[ai][0..count], e.last_submitted);
        }
        if (misses > 0) {
            e.load_ticket = last;
            e.stats.bytes_loaded += @as(u64, misses) * lw.bundle_bytes;
        }
        return .{ .fence = fence, .last = last, .misses = misses };
    }

    /// Issues speculative loads for predicted experts of a later layer. Leases are released at
    /// once; loading slots cannot be evicted, and the demand path resolves them.
    fn prefetchLayer(e: *Engine, il: usize, ids: []const u32) !void {
        var leases: [max_tok * 16]cache_mod.Lease = undefined;
        const issued = e.enqueueLoads(il, ids, &leases) catch |err| switch (err) {
            error.CacheFull => return, // a pinned working set leaves no room to speculate
            else => return err,
        };
        for (leases[0..ids.len]) |lease| try e.cache.release(lease);
        if (issued.misses == 0) return;
        try e.prefetches.append(e.allocator, .{ .fence = issued.fence, .last = issued.last });
        e.stats.prefetch_loads += issued.misses;
    }

    fn finishPrefetch(e: *Engine, fence: cache_mod.LoadFence) !void {
        for (e.prefetches.items, 0..) |pf, i| {
            if (pf.fence.value != fence.value) continue;
            const start = Io.Clock.awake.now(e.io);
            const ok = if (e.dev.waitStorage(pf.last)) true else |_| false;
            e.stats.storage_wait_ns += @intCast(start.durationTo(Io.Clock.awake.now(e.io)).nanoseconds);
            try e.cache.completeLoad(pf.fence, .{ .value = pf.last }, ok);
            _ = e.prefetches.swapRemove(i);
            e.stats.prefetch_used += 1;
            return;
        }
        return error.UnknownLoadFence;
    }

    /// Resolves every finished prefetch so its slots become ready or reusable.
    fn pollPrefetches(e: *Engine) !void {
        var i: usize = 0;
        while (i < e.prefetches.items.len) {
            const pf = e.prefetches.items[i];
            if (pf.last <= e.dev.completedStorage()) {
                const ok = if (e.dev.waitStorage(pf.last)) true else |_| false;
                try e.cache.completeLoad(pf.fence, .{ .value = pf.last }, ok);
                _ = e.prefetches.swapRemove(i);
            } else i += 1;
        }
    }

    /// The expert key whose bundle currently occupies a slot, if any.
    fn slotKeyIndex(e: *Engine, slot: u32) ?usize {
        return std.mem.indexOfScalar(u32, e.slot_of, slot);
    }

    fn submitAndRetire(e: *Engine) !u64 {
        const ticket = try e.dev.submit();
        e.last_submitted = ticket;
        if (!e.resident) {
            for (e.pending.items) |lease| try e.cache.retire(lease, .{ .value = ticket });
            e.pending.clearRetainingCapacity();
        }
        return ticket;
    }

    fn syncPoint(e: *Engine) !void {
        const ticket = try e.submitAndRetire();
        try e.dev.wait(ticket);
        e.compute_done = ticket;
        e.stats.sync_points += 1;
        try e.collectMarks();
        try e.dev.begin();
        try e.mark(.router_shared);
    }

    // ---------------- kernel helpers ----------------

    fn dispatch(e: *Engine, k: Kernel, constants: []const u32, x: u32, y: u32, z: u32) !void {
        try e.dev.dispatch(e.pipes.get(k), constants, x, y, z);
    }

    fn bits(v: f32) u32 {
        return @bitCast(v);
    }

    /// y[t][r] (+)= act(scale * W x[t]) for n_tok rows.
    fn mv(e: *Engine, w: Weight, x_uav: u32, x_off: u32, x_stride: u32, y_uav: u32, y_off: u32, y_stride: u32,
        n_tok: u32, act: u32, scale: f32, accumulate: bool) !void
    {
        const k: Kernel = switch (w.type_id) {
            kernels.type_q8_0 => .mv_q8,
            kernels.type_bf16 => .mv_bf16,
            kernels.type_f32 => .mv_f32,
            else => return error.UnsupportedTensorType,
        };
        // Several rows per wave amortize the staged activations; keep enough groups to fill the GPU.
        var rows_per_wave: u32 = 1;
        while (w.rows / (8 * rows_per_wave * 2) >= 1024) rows_per_wave *= 2;
        const groups = (w.rows + 8 * rows_per_wave - 1) / (8 * rows_per_wave);
        var done: u32 = 0;
        while (done < n_tok) {
            const n = @min(n_tok - done, 8);
            const cst = [16]u32{ w.srv, w.off, w.rows, w.cols, x_uav, x_off + done * x_stride, x_stride, n,
                y_uav, y_off + done * y_stride, y_stride, @intFromBool(accumulate), act, bits(scale), w.row_bytes, rows_per_wave };
            try e.dispatch(k, &cst, groups, 1, 1);
            done += n;
        }
    }

    fn ew(e: *Engine, a_uav: u32, a_off: u32, a_stride: u32, b_uav: u32, b_off: u32, b_stride: u32,
        y_uav: u32, y_off: u32, y_stride: u32, n: u32, rows: u32, op: u32) !void
    {
        const cst = [12]u32{ a_uav, a_off, a_stride, b_uav, b_off, b_stride, y_uav, y_off, y_stride, n, rows, op };
        try e.dispatch(.ew, &cst, (n * rows + 255) / 256, 1, 1);
    }

    fn norm(e: *Engine, x_uav: u32, x_off: u32, y_uav: u32, y_off: u32, len: u32, per_tok: u32, seg_stride: u32,
        tok_stride: u32, n_tok: u32, w: ?Vec, w_mode: u32, l2: bool) !void
    {
        const segs = per_tok * n_tok;
        const cst = [16]u32{ x_uav, x_off, y_uav, y_off, len, per_tok, seg_stride, tok_stride,
            if (w) |v| v.srv else e.aux.srv, if (w) |v| v.off else 0, if (w == null) 0 else w_mode, @intFromBool(l2),
            bits(e.m.h.eps), 0, 0, 0 };
        try e.dispatch(.norm_rows, &cst, @min(segs, 65535), (segs + 65534) / 65535, 1);
    }

    fn rope(e: *Engine, x_uav: u32, x_off: u32, per_tok: u32, seg_stride: u32, tok_stride: u32, n_tok: u32, pos0: u32, pos_mul: u32) !void {
        const h = e.m.h;
        const segs = per_tok * n_tok;
        const cst = [12]u32{ x_uav, x_off, per_tok, seg_stride, tok_stride, segs, h.n_rot, pos0, pos_mul, bits(h.rope_base), 0, 0 };
        try e.dispatch(.rope_neox, &cst, segs, 1, 1);
    }

    /// Hyper-connection mixer in two fused passes: (norm, down + SiLU, inject), then (up, gate, mean).
    fn hcMix(e: *Engine, w: HcW, res_off: u32, n_tok: u32, with_inject: bool) !void {
        const h = e.m.h;
        const HD = h.hcDim();
        if (w.down.type_id != kernels.type_q8_0 or w.up.type_id != kernels.type_q8_0 or
            (with_inject and w.inject.?.type_id != kernels.type_q8_0)) return error.UnsupportedTensorType;
        const inj = w.inject orelse w.down;
        const pre = [20]u32{ e.res.uav, res_off, w.norm.srv, w.norm.off, w.down.srv, w.down.row_bytes, inj.srv, inj.row_bytes,
            h.hc_rank, h.hc, h.n_embd, e.lo.uav, e.inject.uav, e.hc_scale.uav, bits(h.eps), @intFromBool(with_inject), HD, 0, 0, 0 };
        try e.dispatch(.hc_pre, &pre, h.hc_rank + if (with_inject) h.hc else 0, n_tok, 1);
        try e.dev.barrier();
        const post = [16]u32{ w.up.srv, w.up.row_bytes, e.lo.uav, e.res.uav, res_off, w.norm.srv, w.norm.off, e.hc_scale.uav,
            e.cur.uav, h.n_embd, h.hc, h.hc_rank, HD, 0, 0, 0 };
        try e.dispatch(.hc_post, &post, (h.n_embd + 7) / 8, n_tok, 1);
        try e.dev.barrier();
    }

    fn hcCombine(e: *Engine, n_tok: u32) !void {
        const h = e.m.h;
        const cst = [12]u32{ e.res.uav, 0, e.block.uav, 0, e.inject.uav, 0, h.n_embd, h.hc, n_tok, 0, 0, 0 };
        try e.dispatch(.hc_combine, &cst, (h.n_embd * h.hc * n_tok + 255) / 256, 1, 1);
        try e.dev.barrier();
    }

    fn conv(e: *Engine, x_uav: u32, y_uav: u32, state: gpu.Buffer, w: Vec, channels: u32, kernel: u32, dilation: u32,
        n_tok: u32, res_uav: u32, gated_uav: u32) !void
    {
        const cst = [16]u32{ x_uav, 0, y_uav, 0, state.uav, 0, w.srv, w.off, channels, kernel, n_tok, dilation,
            res_uav, 0, gated_uav, 0 };
        try e.dispatch(.conv_causal, &cst, (channels + 255) / 256, 1, 1);
    }

    // ---------------- graph ----------------

    fn pleBlock(e: *Engine, p: PleW, n_tok: u32) !void {
        const h = e.m.h;
        const E = h.n_embd;
        const HD = h.hcDim();
        try e.mv(p.key, e.ple_in.uav, 0, E, e.gate.uav, 0, HD, n_tok, 0, 1.0, false);
        try e.mv(p.value, e.ple_in.uav, 0, E, e.block.uav, 0, E, n_tok, 0, 1.0, false);
        try e.dev.barrier();
        const cst = [20]u32{ e.gate.uav, 0, e.res.uav, 0, e.block.uav, 0, e.xn.uav, 0, e.qkv.uav, 0, e.aux.srv,
            p.norm_key.off, p.norm_query.off, p.norm_conv.off, E, h.hc, bits(h.eps), 0, 0, 0 };
        try e.dispatch(.ple_mix, &cst, h.hc, n_tok, 1);
        try e.dev.barrier();
        try e.conv(e.qkv.uav, e.qkv.uav, e.ple_hist, p.conv, HD, h.ple_conv, h.ple_dilation, n_tok, e.res.uav, e.xn.uav);
        try e.dev.barrier();
    }

    fn linearBlock(e: *Engine, w: LinW, st: LinState, n_tok: u32) !void {
        const h = e.m.h;
        const E = h.n_embd;
        const CH = h.convChannels();
        try e.mv(w.qkv, e.cur.uav, 0, E, e.qkv.uav, 0, CH, n_tok, 0, 1.0, false);
        try e.mv(w.z, e.cur.uav, 0, E, e.z.uav, 0, h.ssm_inner, n_tok, 0, 1.0, false);
        try e.mv(w.beta, e.cur.uav, 0, E, e.beta.uav, 0, h.ssm_v_heads, n_tok, 0, 1.0, false);
        try e.mv(w.alpha, e.cur.uav, 0, E, e.alpha.uav, 0, h.ssm_v_heads, n_tok, 0, 1.0, false);
        try e.dev.barrier();
        try e.mark(.lin_proj);
        const gc = [12]u32{ e.beta.uav, 0, e.alpha.uav, 0, w.dt.srv, w.dt.off, w.a.srv, w.a.off, h.ssm_v_heads, n_tok, 0, 0 };
        try e.dispatch(.gdn_gates, &gc, (h.ssm_v_heads * n_tok + 63) / 64, 1, 1);
        try e.conv(e.qkv.uav, e.conv_out.uav, st.conv, w.conv, CH, h.ssm_conv, 1, n_tok, std.math.maxInt(u32), 0);
        try e.dev.barrier();
        const S = h.ssm_state;
        try e.norm(e.conv_out.uav, 0, e.conv_out.uav, 0, S, h.ssm_k_heads, S, CH, n_tok, null, 0, true);
        try e.norm(e.conv_out.uav, h.ssm_k_heads * S, e.conv_out.uav, h.ssm_k_heads * S, S, h.ssm_k_heads, S, CH, n_tok, null, 0, true);
        try e.dev.barrier();
        try e.mark(.lin_conv);
        const sc = [20]u32{ st.ssm.uav, 0, e.conv_out.uav, 0, CH, h.ssm_k_heads, h.ssm_v_heads, n_tok,
            e.beta.uav, 0, e.alpha.uav, 0, e.z.uav, 0, e.gdn_out.uav, 0, w.norm.srv, w.norm.off, bits(h.eps), 0 };
        try e.dispatch(.gdn_step, &sc, h.ssm_v_heads * S / 8, 1, 1);
        try e.dev.barrier();
        // Gated output norm: rms_norm(o) * w * sigmoid(z), one 128-wide segment per head.
        try e.norm(e.gdn_out.uav, 0, e.gdn_out.uav, 0, S, h.ssm_v_heads, S, h.ssm_inner, n_tok, w.norm, 2, false);
        try e.dev.barrier();
        try e.ew(e.gdn_out.uav, 0, h.ssm_inner, e.z.uav, 0, h.ssm_inner, e.gdn_out.uav, 0, h.ssm_inner, h.ssm_inner, n_tok, 1);
        try e.dev.barrier();
        try e.mark(.lin_step);
        try e.mv(w.out, e.gdn_out.uav, 0, h.ssm_inner, e.block.uav, 0, E, n_tok, 0, 1.0, false);
        try e.dev.barrier();
    }

    fn fullBlock(e: *Engine, w: FullW, st: *FullState, n_tok: u32) !void {
        const h = e.m.h;
        const E = h.n_embd;
        const D = h.head_dim;
        const QW = h.n_head * 2 * D;
        const KVW = h.n_head_kv * D;
        const ID = h.idx_dim;
        // Indexer projections and the raw key cache.
        try e.mv(w.idx_k, e.cur.uav, 0, E, e.idx_k.uav, 0, ID, n_tok, 0, 1.0, false);
        try e.mv(w.idx_q, e.cur.uav, 0, E, e.idx_q.uav, 0, h.idx_heads * ID, n_tok, 0, 1.0, false);
        try e.mv(w.wq, e.cur.uav, 0, E, e.qfull.uav, 0, QW, n_tok, 0, 1.0, false);
        try e.mv(w.wk, e.cur.uav, 0, E, e.kcur.uav, 0, KVW, n_tok, 0, 1.0, false);
        try e.mv(w.wv, e.cur.uav, 0, E, e.vcur.uav, 0, KVW, n_tok, 0, 1.0, false);
        try e.dev.barrier();
        const si = [8]u32{ e.idx_k.uav, 0, ID, st.idx_raw.uav, ID, e.pos, n_tok, 0 };
        try e.dispatch(.store_f16, &si, (ID / 2 * n_tok + 255) / 256, 1, 1);
        try e.norm(e.idx_q.uav, 0, e.idx_q.uav, 0, ID, h.idx_heads, ID, h.idx_heads * ID, n_tok, w.idx_q_norm, 2, false);
        try e.norm(e.qfull.uav, 0, e.qfull.uav, 0, D, h.n_head, 2 * D, QW, n_tok, w.q_norm, 2, false);
        try e.norm(e.kcur.uav, 0, e.kcur.uav, 0, D, h.n_head_kv, D, KVW, n_tok, w.k_norm, 2, false);
        try e.dev.barrier();
        try e.rope(e.idx_q.uav, 0, h.idx_heads, ID, h.idx_heads * ID, n_tok, e.pos, 1);
        try e.rope(e.qfull.uav, 0, h.n_head, 2 * D, QW, n_tok, e.pos, 1);
        try e.rope(e.kcur.uav, 0, h.n_head_kv, D, KVW, n_tok, e.pos, 1);
        // Complete indexer blocks are pooled once.
        const total = e.pos + n_tok;
        const complete = total / w.ratio;
        if (complete > st.pooled_valid) {
            const pc = [12]u32{ st.idx_raw.uav, st.pooled.uav, st.pooled_valid, w.ratio, ID, w.idx_k_norm.srv, w.idx_k_norm.off,
                h.n_rot, bits(h.rope_base), bits(h.eps), 0, 0 };
            try e.dev.barrier();
            try e.dispatch(.qsa_pool, &pc, complete - st.pooled_valid, 1, 1);
            st.pooled_valid = complete;
        }
        try e.dev.barrier();
        const sk = [8]u32{ e.kcur.uav, 0, KVW, st.k.uav, KVW, e.pos, n_tok, 0 };
        try e.dispatch(.store_f16, &sk, (KVW / 2 * n_tok + 255) / 256, 1, 1);
        const sv = [8]u32{ e.vcur.uav, 0, KVW, st.v.uav, KVW, e.pos, n_tok, 0 };
        try e.dispatch(.store_f16, &sv, (KVW / 2 * n_tok + 255) / 256, 1, 1);
        // Sparse selection only once a query sees more cells than the budget.
        const width = h.idx_top_k + w.ratio - 1;
        const sparse = total > width;
        if (sparse) {
            const blocks_needed = total / w.ratio;
            const scst = [12]u32{ e.idx_q.uav, 0, st.pooled.uav, e.scores.uav, h.idx_heads, ID, e.blocks_max, e.pos, w.ratio, 0, 0, 0 };
            try e.dispatch(.qsa_score, &scst, (blocks_needed + 255) / 256, n_tok, 1);
            try e.dev.barrier();
            const sel = [8]u32{ e.scores.uav, e.sel.uav, e.sel_len.uav, e.blocks_max, e.sel_stride, width, e.pos, w.ratio };
            try e.dispatch(.qsa_select, &sel, n_tok, 1, 1);
        }
        try e.dev.barrier();
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        const ac = [20]u32{ e.qfull.uav, 0, QW, 2 * D, st.k.uav, st.v.uav, KVW, D, h.n_head / h.n_head_kv, e.pos,
            @intFromBool(sparse), e.sel.uav, e.sel_stride, e.sel_len.uav, e.part.uav, e.chunks, bits(scale), h.n_head, 0, 0 };
        try e.dispatch(.attn_chunk, &ac, e.chunks, h.n_head, n_tok);
        try e.dev.barrier();
        const mc = [16]u32{ e.part.uav, e.chunks, D, h.n_head, e.qfull.uav, 0, QW, 2 * D, e.attn.uav, 0, e.pos,
            @intFromBool(sparse), e.sel_len.uav, 0, 0, 0 };
        try e.dispatch(.attn_merge, &mc, h.n_head, n_tok, 1);
        try e.dev.barrier();
        try e.mv(w.wo, e.attn.uav, 0, h.n_head * D, e.block.uav, 0, E, n_tok, 0, 1.0, false);
        try e.dev.barrier();
    }

    fn ffn(e: *Engine, il: usize, n_tok: u32) !void {
        const h = e.m.h;
        const lw = e.layers[il];
        const E = h.n_embd;
        const U = h.n_expert_used;
        const F = h.n_ff_exp;
        const SF = h.n_ff_shexp;
        try e.mv(lw.router, e.cur.uav, 0, E, e.router.uav, 0, h.n_expert, n_tok, 0, 1.0, false);
        // The shared expert does not depend on routing, so it overlaps a demand load.
        try e.mv(lw.sh_gate, e.cur.uav, 0, E, e.sh.uav, 0, 2 * SF, n_tok, 0, 1.0, false);
        try e.mv(lw.sh_up, e.cur.uav, 0, E, e.sh.uav, SF, 2 * SF, n_tok, 0, 1.0, false);
        try e.mv(lw.sh_gate_inp, e.cur.uav, 0, E, e.shg.uav, 0, 1, n_tok, 0, 1.0, false);
        if (lw.gate_signs) |signs| {
            const rc = [8]u32{ e.cur.uav, 0, e.xh.uav, 0, signs.srv, signs.off, E, E };
            try e.dispatch(.rotate128, &rc, E / 128, n_tok, 1);
        } else try e.ew(e.cur.uav, 0, E, e.cur.uav, 0, E, e.xh.uav, 0, E, E, n_tok, 3);
        try e.dev.barrier();
        const route = [12]u32{ e.router.uav, 0, h.n_expert, U, e.ids.uav, 0, e.weights.uav, 0,
            if (e.resident) std.math.maxInt(u32) else e.ids_host.uav, 0, bits(h.expert_weights_scale), n_tok };
        try e.dispatch(.moe_route, &route, n_tok, 1, 1);
        try e.ew(e.sh.uav, 0, 2 * SF, e.sh.uav, SF, 2 * SF, e.sh_act.uav, 0, SF, SF, n_tok, 0);
        try e.dev.barrier();
        try e.mv(lw.sh_down, e.sh_act.uav, 0, SF, e.sh_out.uav, 0, E, n_tok, 0, 1.0, false);
        try e.mark(.router_shared);
        const n_slots = n_tok * U;
        var table_uav = e.table.uav;
        if (e.resident) {
            var arena_srv: [8]u32 = @splat(0);
            for (e.arenas, 0..) |a, i| arena_srv[i] = a.srv;
            const rs = [20]u32{ e.ids.uav, 0, n_slots, e.table.uav, e.lookup.uav, @intCast(il * h.n_expert), e.per_arena,
                e.slot_stride, lw.gate_bytes, @intCast(e.arenas.len), 0, 0,
                arena_srv[0], arena_srv[1], arena_srv[2], arena_srv[3], arena_srv[4], arena_srv[5], arena_srv[6], arena_srv[7] };
            try e.dispatch(.moe_resolve, &rs, (n_slots + 63) / 64, 1, 1);
        } else {
            // Predict the next layer's experts with its router on this layer's FFN input.
            const predict = e.prefetch and il + 1 < e.layers.len;
            if (predict) {
                try e.mv(e.layers[il + 1].router, e.cur.uav, 0, E, e.router2.uav, 0, h.n_expert, n_tok, 0, 1.0, false);
                try e.dev.barrier();
                const r2 = [12]u32{ e.router2.uav, 0, h.n_expert, U, e.ids2.uav, 0, e.weights2.uav, 0,
                    e.ids_host.uav, max_tok * U, bits(h.expert_weights_scale), n_tok };
                try e.dispatch(.moe_route, &r2, n_tok, 1, 1);
            }
            try e.syncPoint();
            try e.pollPrefetches();
            try e.admitLayer(il, n_tok);
            if (predict) try e.prefetchLayer(il + 1, e.ids_host.words()[max_tok * U ..][0 .. n_tok * U]);
            table_uav = e.table_host.uav;
        }
        try e.dev.barrier();
        const info = core.gyro.typeInfo(lw.expert_type) orelse return error.UnsupportedTensorType;
        const gu = [16]u32{ table_uav, e.codebook.srv, e.xh.uav, 0, E, e.gu.uav, 0, F, E, lw.expert_type, info.block_bytes,
            0, U, 0, 0, 0 };
        try e.dispatch(.moe_gyro, &gu, (F + 31) / 32, n_slots, 2);
        try e.dev.barrier();
        const ac = [8]u32{ e.gu.uav, 0, e.act.uav, 0, if (lw.down_signs) |s| s.srv else e.aux.srv,
            if (lw.down_signs) |s| s.off else 0, F, @intFromBool(lw.down_signs != null) };
        try e.dispatch(.moe_act, &ac, n_slots, 1, 1);
        try e.dev.barrier();
        const dn = [16]u32{ table_uav, e.codebook.srv, e.act.uav, 0, F, e.down.uav, 0, E, F, lw.expert_type, info.block_bytes,
            2, U, 1, 0, 0 };
        try e.dispatch(.moe_gyro, &dn, (E + 31) / 32, n_slots, 1);
        try e.dev.barrier();
        const cc = [12]u32{ e.down.uav, 0, e.weights.uav, 0, e.block.uav, 0, U, E, e.sh_out.uav, 0, e.shg.uav, 0 };
        try e.dispatch(.moe_combine, &cc, (E + 255) / 256, n_tok, 1);
        try e.dev.barrier();
        try e.mark(.experts);
    }

    /// One pass of up to max_tok tokens.
    fn pass(e: *Engine, tokens: []const u32, logits: ?[]f32) !void {
        const h = e.m.h;
        const E = h.n_embd;
        const n: u32 = @intCast(tokens.len);
        // Host inputs: embedding rows and PLE rows (exact address math on the CPU).
        const emb = e.emb_in.floats();
        const ple = e.ple_in.floats();
        for (tokens, 0..) |t, i| {
            kernels.decodeRow(e.m.token_embd.type_id, e.m.token_embd.row(t), emb[i * E ..][0..E]);
            const rows = try e.ple_history.accept(e.m.ple_config, t);
            for (rows, 0..) |r, head| kernels.decodeRow(e.m.ple_table.type_id, e.m.ple_table.row(@intCast(r.row)),
                ple[i * E + head * h.ple_head_dim ..][0..h.ple_head_dim]);
        }
        try e.dev.begin();
        try e.mark(.inputs);
        const ic = [12]u32{ e.res.uav, 0, e.emb_in.uav, 0, 0, 0, E, h.hc, n, 0, 0, 0 };
        try e.dispatch(.hc_init, &ic, (E * h.hc * n + 255) / 256, 1, 1);
        try e.dev.barrier();
        try e.mark(.inputs);
        for (e.layers, 0..) |lw, il| {
            if (lw.ple) |p| {
                try e.pleBlock(p, n);
                try e.mark(.ple);
            }
            try e.hcMix(lw.hc_attn, 0, n, true);
            try e.mark(.hc_mix);
            if (lw.lin) |l| {
                try e.linearBlock(l, e.lin_state[e.layer_slot[il]], n);
                try e.mark(.linear_attn);
            } else {
                try e.fullBlock(lw.full.?, &e.full_state[e.layer_slot[il]], n);
                try e.mark(.full_attn);
            }
            try e.hcCombine(n);
            try e.mark(.hc_combine);
            try e.hcMix(lw.hc_ffn, 0, n, true);
            try e.mark(.hc_mix);
            try e.ffn(il, n);
            try e.hcCombine(n);
            try e.mark(.hc_combine);
        }
        if (logits != null) {
            try e.hcMix(e.head, (n - 1) * h.hcDim(), 1, false);
            try e.mark(.head);
            try e.mv(e.output, e.cur.uav, 0, E, e.logits.uav, 0, h.n_vocab, 1, 0, 1.0, false);
            try e.mark(.output);
        }
        const ticket = try e.submitAndRetire();
        try e.dev.wait(ticket);
        e.compute_done = ticket;
        try e.collectMarks();
        // The zero source for state clears must stay zero.
        @memset(ple[0 .. tokens.len * E], 0);
        e.pos += n;
        e.stats.tokens += n;
        if (logits) |out| @memcpy(out, e.logits.floats()[0..h.n_vocab]);
    }

    pub fn forward(e: *Engine, tokens: []const u32, logits: ?[]f32) !void {
        if (@as(usize, e.pos) + tokens.len > e.max_ctx) return error.ContextFull;
        for (tokens) |t| if (t >= e.m.h.n_vocab) return error.InvalidToken;
        var i: usize = 0;
        while (i < tokens.len) {
            const end = @min(tokens.len, i + max_tok);
            try e.pass(tokens[i..end], if (end == tokens.len) logits else null);
            i = end;
        }
    }

    pub fn reset(e: *Engine) !void {
        try e.clearState();
    }

    // ---------------- runtime backend ----------------

    pub const vtable = core.runtime.Backend.VTable{
        .forward = vForward,
        .reset = vReset,
        .position = vPosition,
        .capacity = vCapacity,
        .destroy = vDestroy,
    };
    fn vForward(ctx: *anyopaque, tokens: []const u32, logits: ?[]f32) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        return e.forward(tokens, logits);
    }
    fn vReset(ctx: *anyopaque) void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        e.reset() catch |err| std.log.err("GPU state reset failed: {s}", .{@errorName(err)});
    }
    fn vPosition(ctx: *anyopaque) u32 {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        return e.pos;
    }
    fn vCapacity(ctx: *anyopaque) u32 {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        return e.max_ctx;
    }
    fn vDestroy(ctx: *anyopaque) void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        e.destroy();
    }

};

/// Runtime backend factory. The options must outlive the runtime.
pub fn backendFactory(options: *Options) core.runtime.BackendFactory {
    return .{ .ctx = options, .create = createBackend };
}

fn createBackend(ctx: ?*anyopaque, io: Io, allocator: Allocator, m: *Model, max_ctx: u32) anyerror!core.runtime.Backend {
    const options: *Options = @ptrCast(@alignCast(ctx.?));
    const e = try Engine.create(io, allocator, m, max_ctx, options.*);
    return .{ .ctx = e, .vtable = &Engine.vtable, .name = "d3d12" };
}

/// Writes engine statistics as JSON: memory, expert traffic, and per-phase GPU time.
pub fn report(e: *Engine, writer: *Io.Writer) !void {
    const mem = try e.dev.budget();
    var phases: [@typeInfo(Phase).@"enum".field_names.len]struct { phase: []const u8, ms_per_token: f64 } = undefined;
    const toks: f64 = @floatFromInt(@max(e.stats.tokens, 1));
    inline for (@typeInfo(Phase).@"enum".field_names, 0..) |name, i|
        phases[i] = .{ .phase = name, .ms_per_token = @as(f64, @floatFromInt(e.stats.phase_ns[i])) / 1e6 / toks };
    try writer.print("{f}\n", .{std.json.fmt(.{
        .adapter = e.dev.adapterName(),
        .expert_mode = if (e.resident) "resident" else "cache",
        .expert_slots = e.total_slots,
        .device_allocated_bytes = e.dev.allocated_device_bytes,
        .host_visible_bytes = e.dev.allocated_host_bytes,
        .adapter_budget_bytes = mem.budget,
        .adapter_usage_bytes = mem.usage,
        .tokens = e.stats.tokens,
        .expert_requests = e.stats.expert_requests,
        .expert_hits = e.stats.expert_hits,
        .expert_misses = e.stats.expert_misses,
        .expert_bytes_loaded = e.stats.bytes_loaded,
        .sync_points = e.stats.sync_points,
        .storage_wait_ms = @as(f64, @floatFromInt(e.stats.storage_wait_ns)) / 1e6,
        .prefetch_loads = e.stats.prefetch_loads,
        .prefetch_batches_used = e.stats.prefetch_used,
        .gpu_phase_ms_per_token = if (e.options.profile) phases[0..] else phases[0..0],
    }, .{ .whitespace = .indent_2 })});
}
