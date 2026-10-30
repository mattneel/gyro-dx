//! Qwen4Exp model binding: maps a GGUF file and resolves every tensor the graph reads.
const std = @import("std");
const gguf = @import("gguf.zig");
const ple = @import("ple.zig");
const kernels = @import("kernels.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Matrix = kernels.Matrix;

pub const Error = error{
    UnsupportedArchitecture, MissingTensor, InvalidTensorShape, UnsupportedTensorType,
    UnsupportedConfiguration, MisalignedTensor,
};

pub const Hparams = struct {
    n_layer: u32,
    n_embd: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    n_rot: u32,
    rope_base: f32,
    eps: f32,
    n_expert: u32,
    n_expert_used: u32,
    n_ff_exp: u32,
    n_ff_shexp: u32,
    ssm_conv: u32,
    ssm_state: u32,
    ssm_k_heads: u32,
    ssm_v_heads: u32,
    ssm_inner: u32,
    hc: u32,
    hc_rank: u32,
    idx_heads: u32,
    idx_dim: u32,
    idx_top_k: u32,
    ple_layer: u32,
    ple_head_dim: u32,
    ple_heads: u32,
    ple_conv: u32,
    ple_dilation: u32,
    n_vocab: u32,
    context_length: u32,
    expert_weights_scale: f32,

    pub fn hcDim(h: Hparams) u32 {
        return h.hc * h.n_embd;
    }
    pub fn convChannels(h: Hparams) u32 {
        return h.ssm_state * h.ssm_k_heads * 2 + h.ssm_state * h.ssm_v_heads;
    }
};

pub const Hc = struct {
    norm: []const f32,
    down: Matrix,
    up: Matrix,
    inject: ?Matrix,
};

pub const Linear = struct {
    qkv: Matrix,
    z: Matrix,
    beta: Matrix,
    alpha: Matrix,
    out: Matrix,
    conv: []const f32,
    dt_bias: []const f32,
    a: []const f32,
    norm: []const f32,
};

pub const Full = struct {
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    q_norm: []const f32,
    k_norm: []const f32,
    idx_q: Matrix,
    idx_k: Matrix,
    idx_q_norm: []const f32,
    idx_k_norm: []const f32,
    compress_ratio: u32,
};

pub const Ple = struct {
    key: Matrix,
    value: Matrix,
    norm_key: []const f32,
    norm_query: []const f32,
    norm_conv: []const f32,
    conv: []const f32,
};

pub const Layer = struct {
    index: u32,
    hc_attn: Hc,
    hc_ffn: Hc,
    mixer: union(enum) { linear: Linear, full: Full },
    ple: ?Ple,
    router: Matrix,
    gate_exps: Matrix,
    up_exps: Matrix,
    down_exps: Matrix,
    gate_signs: ?[]const f32,
    up_signs: ?[]const f32,
    down_signs: ?[]const f32,
    sh_gate: Matrix,
    sh_up: Matrix,
    sh_down: Matrix,
    sh_gate_inp: []const f32,

    pub fn expertGate(l: *const Layer, h: Hparams, e: usize) Matrix {
        return l.gate_exps.slice(e * h.n_ff_exp, h.n_ff_exp);
    }
    pub fn expertUp(l: *const Layer, h: Hparams, e: usize) Matrix {
        return l.up_exps.slice(e * h.n_ff_exp, h.n_ff_exp);
    }
    pub fn expertDown(l: *const Layer, h: Hparams, e: usize) Matrix {
        return l.down_exps.slice(e * h.n_embd, h.n_embd);
    }
};

pub const Model = struct {
    allocator: Allocator,
    io: Io,
    file: Io.File,
    map: Io.File.MemoryMap,
    bytes: []const u8,
    meta: gguf.Model,
    h: Hparams,
    ple_config: ple.Config,
    token_embd: Matrix,
    output: Matrix,
    hc_head: Hc,
    ple_table: Matrix,
    layers: []Layer,
    sign_store: std.ArrayList(f32),

    pub fn deinit(m: *Model) void {
        m.allocator.free(m.layers);
        m.sign_store.deinit(m.allocator);
        m.meta.deinit();
        m.map.destroy(m.io);
        m.file.close(m.io);
    }

    /// Returns the file bytes of a tensor.
    pub fn tensorBytes(m: *const Model, t: gguf.Tensor) []const u8 {
        return m.bytes[@intCast(t.file_offset)..][0..@intCast(t.byte_length.?)];
    }
};

fn f32Of(m: *const Model, name: []const u8, expected: usize) ![]const f32 {
    const t = m.meta.tensor(name) orelse {
        std.log.err("missing tensor {s}", .{name});
        return error.MissingTensor;
    };
    if (t.type_id != kernels.type_f32) return error.UnsupportedTensorType;
    var n: u64 = 1;
    for (t.dims[0..t.rank]) |d| n *= d;
    if (n != expected) {
        std.log.err("tensor {s} has {d} elements, expected {d}", .{ name, n, expected });
        return error.InvalidTensorShape;
    }
    const bytes = m.tensorBytes(t);
    if (@intFromPtr(bytes.ptr) % 4 != 0) return error.MisalignedTensor;
    return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0..expected];
}

fn matOf(m: *const Model, name: []const u8, cols: usize, rows: usize) !Matrix {
    const t = m.meta.tensor(name) orelse {
        std.log.err("missing tensor {s}", .{name});
        return error.MissingTensor;
    };
    var outer: u64 = 1;
    for (t.dims[1..t.rank]) |d| outer *= d;
    if (t.dims[0] != cols or outer != rows) {
        std.log.err("tensor {s} has shape {any}, expected [{d}, {d}]", .{ name, t.dims[0..t.rank], cols, rows });
        return error.InvalidTensorShape;
    }
    const row_bytes = try kernels.rowBytes(t.type_id, cols);
    const bytes = m.tensorBytes(t);
    if (bytes.len != row_bytes * rows) return error.InvalidTensorShape;
    return .{ .type_id = t.type_id, .cols = cols, .rows = rows, .row_bytes = row_bytes, .data = bytes };
}

fn layerName(buf: []u8, il: u32, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ il, suffix });
}

fn num(meta: gguf.Model, key: []const u8) !u32 {
    return @intCast(try meta.number(key));
}

fn float(meta: gguf.Model, key: []const u8) !f32 {
    const e = try meta.required(key);
    if (e.kind != .float32) return error.InvalidMetadata;
    return @bitCast(std.mem.readInt(u32, e.raw[0..4], .little));
}

fn u32Array(allocator: Allocator, meta: gguf.Model, key: []const u8) ![]u32 {
    const e = try meta.required(key);
    if (e.kind != .array) return error.InvalidMetadata;
    const elem = std.mem.readInt(u32, e.raw[0..4], .little);
    const n: usize = @intCast(std.mem.readInt(u64, e.raw[4..12], .little));
    const out = try allocator.alloc(u32, n);
    errdefer allocator.free(out);
    const width: usize = switch (elem) { 4, 5 => 4, 10, 11 => 8, else => return error.InvalidMetadata };
    if (e.raw.len < 12 + n * width) return error.InvalidMetadata;
    for (out, 0..) |*o, i| {
        o.* = if (width == 4) std.mem.readInt(u32, e.raw[12 + i * 4 ..][0..4], .little)
            else @intCast(std.mem.readInt(u64, e.raw[12 + i * 8 ..][0..8], .little));
    }
    return out;
}

pub fn open(io: Io, allocator: Allocator, path: []const u8) !*Model {
    const m = try allocator.create(Model);
    errdefer allocator.destroy(m);
    m.allocator = allocator;
    m.io = io;
    m.file = try Io.Dir.cwd().openFile(io, path, .{});
    errdefer m.file.close(io);
    const stat = try m.file.stat(io);
    m.map = try Io.File.MemoryMap.create(io, m.file, .{
        .len = @intCast(stat.size),
        .protection = .{ .read = true, .write = false },
        .populate = false,
    });
    errdefer m.map.destroy(io);
    m.bytes = m.map.memory;
    const prefix_len: usize = @intCast(@min(stat.size, gguf.max_header_bytes));
    m.meta = try gguf.parse(allocator, m.bytes[0..prefix_len], stat.size);
    errdefer m.meta.deinit();
    try m.meta.validateGyro();
    if (!std.mem.eql(u8, try m.meta.text("general.architecture"), "qwen4exp")) return error.UnsupportedArchitecture;
    m.sign_store = .empty;
    errdefer m.sign_store.deinit(allocator);
    try bind(m);
    return m;
}

fn bind(m: *Model) !void {
    const meta = m.meta;
    const a = "qwen4exp.";
    const ple_layers = try u32Array(m.allocator, meta, a ++ "ple.layers");
    defer m.allocator.free(ple_layers);
    if (ple_layers.len != 1) return error.UnsupportedConfiguration;
    const embd = meta.tensor("token_embd.weight") orelse return error.MissingTensor;
    var h = Hparams{
        .n_layer = try num(meta, a ++ "block_count"),
        .n_embd = try num(meta, a ++ "embedding_length"),
        .n_head = try num(meta, a ++ "attention.head_count"),
        .n_head_kv = try num(meta, a ++ "attention.head_count_kv"),
        .head_dim = try num(meta, a ++ "attention.key_length"),
        .n_rot = try num(meta, a ++ "rope.dimension_count"),
        .rope_base = try float(meta, a ++ "rope.freq_base"),
        .eps = try float(meta, a ++ "attention.layer_norm_rms_epsilon"),
        .n_expert = try num(meta, a ++ "expert_count"),
        .n_expert_used = try num(meta, a ++ "expert_used_count"),
        .n_ff_exp = try num(meta, a ++ "expert_feed_forward_length"),
        .n_ff_shexp = try num(meta, a ++ "expert_shared_feed_forward_length"),
        .ssm_conv = try num(meta, a ++ "ssm.conv_kernel"),
        .ssm_state = try num(meta, a ++ "ssm.state_size"),
        .ssm_k_heads = try num(meta, a ++ "ssm.group_count"),
        .ssm_v_heads = try num(meta, a ++ "ssm.time_step_rank"),
        .ssm_inner = try num(meta, a ++ "ssm.inner_size"),
        .hc = try num(meta, a ++ "hyper_connection.count"),
        .hc_rank = try num(meta, a ++ "hyper_connection.low_rank"),
        .idx_heads = try num(meta, a ++ "attention.indexer.head_count"),
        .idx_dim = try num(meta, a ++ "attention.indexer.key_length"),
        .idx_top_k = try num(meta, a ++ "attention.indexer.top_k"),
        .ple_layer = ple_layers[0],
        .ple_head_dim = try num(meta, a ++ "embedding_length_per_layer_input"),
        .ple_heads = (try num(meta, a ++ "ple.ngram_size") - 1) * try num(meta, a ++ "ple.heads_per_ngram"),
        .ple_conv = try num(meta, a ++ "ple.conv_kernel"),
        .ple_dilation = try num(meta, a ++ "ple.ngram_size"),
        .n_vocab = @intCast(embd.dims[1]),
        .context_length = try num(meta, a ++ "context_length"),
        .expert_weights_scale = 0,
    };
    if (meta.get(a ++ "expert_weights_scale") != null) h.expert_weights_scale = try float(meta, a ++ "expert_weights_scale");
    if (h.head_dim != try num(meta, a ++ "attention.value_length")) return error.UnsupportedConfiguration;
    if (h.ssm_inner != h.ssm_state * h.ssm_v_heads or h.ssm_v_heads % h.ssm_k_heads != 0) return error.UnsupportedConfiguration;
    if (h.n_head % h.n_head_kv != 0 or h.n_rot > h.head_dim or h.n_rot % 2 != 0) return error.UnsupportedConfiguration;
    if (h.ple_heads * h.ple_head_dim != h.n_embd) return error.UnsupportedConfiguration;
    if (h.hc < 2) return error.UnsupportedConfiguration;
    // Text positions repeat in every M-RoPE section, which reduces IMRoPE to NeoX ordering.
    const sections = try u32Array(m.allocator, meta, a ++ "rope.dimension_sections");
    defer m.allocator.free(sections);
    var section_sum: u32 = 0;
    for (sections) |s| section_sum += s;
    if (section_sum * 2 != h.n_rot) return error.UnsupportedConfiguration;
    m.h = h;
    m.ple_config = try ple.fromModel(meta);

    const ratios = try u32Array(m.allocator, meta, a ++ "attention.compress_ratios");
    defer m.allocator.free(ratios);
    if (ratios.len < h.n_layer) return error.UnsupportedConfiguration;
    const interval = if (meta.get(a ++ "full_attention_interval") != null) try num(meta, a ++ "full_attention_interval") else 4;

    // Rotation metadata: listed tensors store W D H.
    const names = try (try meta.required("prism.hadamard.weight_names")).strings(m.allocator);
    defer m.allocator.free(names);
    const widths = try (try meta.required("prism.hadamard.sign_widths")).ints32(m.allocator);
    defer m.allocator.free(widths);
    const sign_values = try (try meta.required("prism.hadamard.sign_values")).ints32(m.allocator);
    defer m.allocator.free(sign_values);
    if (try meta.number("prism.hadamard.block_size") != 128) return error.UnsupportedConfiguration;
    try m.sign_store.ensureTotalCapacity(m.allocator, sign_values.len);
    for (sign_values) |s| m.sign_store.appendAssumeCapacity(@floatFromInt(s));

    const Lookup = struct {
        names: [][]const u8, widths: []i32, store: []const f32,
        fn signs(self: @This(), name: []const u8, width: usize) !?[]const f32 {
            var listed = false;
            for (self.names) |n| if (std.mem.eql(u8, n, name)) { listed = true; };
            if (!listed) return null;
            var off: usize = 0;
            for (self.widths) |w| {
                if (@as(usize, @intCast(w)) == width) return self.store[off..][0..width];
                off += @intCast(w);
            }
            return error.UnsupportedConfiguration;
        }
    };
    const lookup = Lookup{ .names = names, .widths = widths, .store = m.sign_store.items };

    m.token_embd = try matOf(m, "token_embd.weight", h.n_embd, h.n_vocab);
    m.output = if (meta.tensor("output.weight") != null) try matOf(m, "output.weight", h.n_embd, h.n_vocab) else m.token_embd;
    m.hc_head = .{
        .norm = try f32Of(m, "output_hc_norm.weight", h.hcDim()),
        .down = try matOf(m, "output_hc_down.weight", h.hcDim(), h.hc_rank),
        .up = try matOf(m, "output_hc_up.weight", h.hc_rank, h.hcDim()),
        .inject = null,
    };
    const table = meta.tensor("per_layer_token_embd.weight") orelse return error.MissingTensor;
    m.ple_table = try matOf(m, "per_layer_token_embd.weight", h.ple_head_dim, @intCast(table.dims[1]));

    m.layers = try m.allocator.alloc(Layer, h.n_layer);
    errdefer m.allocator.free(m.layers);
    var buf: [128]u8 = undefined;
    const hd = h.hcDim();
    for (m.layers, 0..) |*l, i| {
        const il: u32 = @intCast(i);
        const n = struct {
            fn f(b: []u8, layer: u32, s: []const u8) []const u8 {
                return layerName(b, layer, s) catch unreachable;
            }
        }.f;
        l.index = il;
        l.hc_attn = .{
            .norm = try f32Of(m, n(&buf, il, "hc_attn_norm.weight"), hd),
            .down = try matOf(m, n(&buf, il, "hc_attn_down.weight"), hd, h.hc_rank),
            .up = try matOf(m, n(&buf, il, "hc_attn_up.weight"), h.hc_rank, hd),
            .inject = try matOf(m, n(&buf, il, "hc_attn_inject.weight"), hd, h.hc),
        };
        l.hc_ffn = .{
            .norm = try f32Of(m, n(&buf, il, "hc_ffn_norm.weight"), hd),
            .down = try matOf(m, n(&buf, il, "hc_ffn_down.weight"), hd, h.hc_rank),
            .up = try matOf(m, n(&buf, il, "hc_ffn_up.weight"), h.hc_rank, hd),
            .inject = try matOf(m, n(&buf, il, "hc_ffn_inject.weight"), hd, h.hc),
        };
        const recurrent = (il + 1) % interval != 0;
        if (recurrent) {
            const conv_ch = h.convChannels();
            l.mixer = .{ .linear = .{
                .qkv = try matOf(m, n(&buf, il, "attn_qkv.weight"), h.n_embd, conv_ch),
                .z = try matOf(m, n(&buf, il, "attn_gate.weight"), h.n_embd, h.ssm_inner),
                .beta = try matOf(m, n(&buf, il, "ssm_beta.weight"), h.n_embd, h.ssm_v_heads),
                .alpha = try matOf(m, n(&buf, il, "ssm_alpha.weight"), h.n_embd, h.ssm_v_heads),
                .out = try matOf(m, n(&buf, il, "ssm_out.weight"), h.ssm_inner, h.n_embd),
                .conv = try f32Of(m, n(&buf, il, "ssm_conv1d.weight"), h.ssm_conv * conv_ch),
                .dt_bias = try f32Of(m, n(&buf, il, "ssm_dt.bias"), h.ssm_v_heads),
                .a = try f32Of(m, n(&buf, il, "ssm_a"), h.ssm_v_heads),
                .norm = try f32Of(m, n(&buf, il, "ssm_norm.weight"), h.ssm_state),
            } };
        } else {
            if (ratios[i] == 0) return error.UnsupportedConfiguration;
            l.mixer = .{ .full = .{
                .wq = try matOf(m, n(&buf, il, "attn_q.weight"), h.n_embd, h.head_dim * h.n_head * 2),
                .wk = try matOf(m, n(&buf, il, "attn_k.weight"), h.n_embd, h.head_dim * h.n_head_kv),
                .wv = try matOf(m, n(&buf, il, "attn_v.weight"), h.n_embd, h.head_dim * h.n_head_kv),
                .wo = try matOf(m, n(&buf, il, "attn_output.weight"), h.head_dim * h.n_head, h.n_embd),
                .q_norm = try f32Of(m, n(&buf, il, "attn_q_norm.weight"), h.head_dim),
                .k_norm = try f32Of(m, n(&buf, il, "attn_k_norm.weight"), h.head_dim),
                .idx_q = try matOf(m, n(&buf, il, "indexer.q_proj.weight"), h.n_embd, h.idx_heads * h.idx_dim),
                .idx_k = try matOf(m, n(&buf, il, "indexer.k_proj.weight"), h.n_embd, h.idx_dim),
                .idx_q_norm = try f32Of(m, n(&buf, il, "indexer.q_norm.weight"), h.idx_dim),
                .idx_k_norm = try f32Of(m, n(&buf, il, "indexer.k_norm.weight"), h.idx_dim),
                .compress_ratio = ratios[i],
            } };
        }
        if (il == h.ple_layer) {
            if (!recurrent) return error.UnsupportedConfiguration;
            l.ple = .{
                .key = try matOf(m, n(&buf, il, "ple_key.weight"), h.n_embd, hd),
                .value = try matOf(m, n(&buf, il, "ple_value.weight"), h.n_embd, h.n_embd),
                .norm_key = try f32Of(m, n(&buf, il, "ple_norm_key.weight"), hd),
                .norm_query = try f32Of(m, n(&buf, il, "ple_norm_query.weight"), hd),
                .norm_conv = try f32Of(m, n(&buf, il, "ple_norm_conv.weight"), hd),
                .conv = try f32Of(m, n(&buf, il, "ple_conv1d.weight"), h.ple_conv * hd),
            };
        } else l.ple = null;
        l.router = try matOf(m, n(&buf, il, "ffn_gate_inp.weight"), h.n_embd, h.n_expert);
        const gate_name = n(&buf, il, "ffn_gate_exps.weight");
        l.gate_exps = try matOf(m, gate_name, h.n_embd, h.n_ff_exp * h.n_expert);
        l.gate_signs = try lookup.signs(gate_name, h.n_embd);
        const up_name = n(&buf, il, "ffn_up_exps.weight");
        l.up_exps = try matOf(m, up_name, h.n_embd, h.n_ff_exp * h.n_expert);
        l.up_signs = try lookup.signs(up_name, h.n_embd);
        const down_name = n(&buf, il, "ffn_down_exps.weight");
        l.down_exps = try matOf(m, down_name, h.n_ff_exp, h.n_embd * h.n_expert);
        l.down_signs = try lookup.signs(down_name, h.n_ff_exp);
        l.sh_gate = try matOf(m, n(&buf, il, "ffn_gate_shexp.weight"), h.n_embd, h.n_ff_shexp);
        l.sh_up = try matOf(m, n(&buf, il, "ffn_up_shexp.weight"), h.n_embd, h.n_ff_shexp);
        l.sh_down = try matOf(m, n(&buf, il, "ffn_down_shexp.weight"), h.n_ff_shexp, h.n_embd);
        l.sh_gate_inp = try f32Of(m, n(&buf, il, "ffn_gate_inp_shexp.weight"), h.n_embd);
    }
    // Every rotated weight must belong to an expert projection this graph rotates.
    for (names) |name| {
        if (!(std.mem.endsWith(u8, name, ".ffn_gate_exps.weight") or std.mem.endsWith(u8, name, ".ffn_up_exps.weight") or
            std.mem.endsWith(u8, name, ".ffn_down_exps.weight")))
        {
            std.log.err("prism.hadamard lists {s}, which this graph does not rotate", .{name});
            return error.UnsupportedConfiguration;
        }
    }
}
