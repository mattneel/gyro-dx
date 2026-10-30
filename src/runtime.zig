//! Model session runtime: tokenizer, graph execution backend, sampling, and prompt reuse.
const std = @import("std");
const model_mod = @import("model.zig");
const engine = @import("engine.zig");
const tokenizer_mod = @import("tokenizer.zig");
const sampler_mod = @import("sampler.zig");
const Pool = @import("pool.zig").Pool;
const Io = std.Io;
const Allocator = std.mem.Allocator;

pub const Options = struct {
    threads: usize = 0,
    max_ctx: u32 = 8192,
    prefill_chunk: u32 = 128,
    /// Creates an alternative graph backend. Null selects the CPU engine.
    backend: ?BackendFactory = null,
};

/// Graph execution backend for one sequence.
pub const Backend = struct {
    ctx: *anyopaque,
    vtable: *const VTable,
    name: []const u8,

    pub const VTable = struct {
        /// Processes tokens; writes next-token logits of the last token when logits is non-null.
        forward: *const fn (ctx: *anyopaque, tokens: []const u32, logits: ?[]f32) anyerror!void,
        reset: *const fn (ctx: *anyopaque) void,
        position: *const fn (ctx: *anyopaque) u32,
        capacity: *const fn (ctx: *anyopaque) u32,
        destroy: *const fn (ctx: *anyopaque) void,
    };

    pub fn forward(b: Backend, tokens: []const u32, logits: ?[]f32) !void {
        return b.vtable.forward(b.ctx, tokens, logits);
    }
    pub fn reset(b: Backend) void {
        b.vtable.reset(b.ctx);
    }
    pub fn position(b: Backend) u32 {
        return b.vtable.position(b.ctx);
    }
    pub fn capacity(b: Backend) u32 {
        return b.vtable.capacity(b.ctx);
    }
};

pub const BackendFactory = struct {
    ctx: ?*anyopaque = null,
    create: *const fn (ctx: ?*anyopaque, io: Io, allocator: Allocator, model: *model_mod.Model, max_ctx: u32) anyerror!Backend,
};

const CpuBackend = struct {
    session: engine.Session,

    const vtable = Backend.VTable{
        .forward = forward,
        .reset = reset,
        .position = position,
        .capacity = capacity,
        .destroy = destroy,
    };
    fn forward(ctx: *anyopaque, tokens: []const u32, logits: ?[]f32) anyerror!void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        return self.session.forward(tokens, logits, false);
    }
    fn reset(ctx: *anyopaque) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        self.session.reset();
    }
    fn position(ctx: *anyopaque) u32 {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        return self.session.pos;
    }
    fn capacity(ctx: *anyopaque) u32 {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        return self.session.max_ctx;
    }
    fn destroy(ctx: *anyopaque) void {
        const self: *CpuBackend = @ptrCast(@alignCast(ctx));
        const allocator = self.session.allocator;
        self.session.deinit();
        allocator.destroy(self);
    }
};

pub const Stats = struct {
    prompt_tokens: usize = 0,
    reused_tokens: usize = 0,
    generated_tokens: usize = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    stop_reason: StopReason = .length,

    pub fn prefillRate(s: Stats) f64 {
        const n = s.prompt_tokens - s.reused_tokens;
        return if (s.prefill_ns == 0) 0 else @as(f64, @floatFromInt(n)) * 1e9 / @as(f64, @floatFromInt(s.prefill_ns));
    }
    pub fn decodeRate(s: Stats) f64 {
        return if (s.decode_ns == 0) 0 else @as(f64, @floatFromInt(s.generated_tokens)) * 1e9 / @as(f64, @floatFromInt(s.decode_ns));
    }
};

pub const StopReason = enum { eos, length, stop_sequence, cancelled };

/// Receives generated text. Returning false stops generation.
pub const Sink = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, text: []const u8, token: u32) bool,
};

pub const Runtime = struct {
    allocator: Allocator,
    io: Io,
    model: *model_mod.Model,
    tok: tokenizer_mod.Tokenizer,
    pool: Pool,
    backend: Backend,
    history: std.ArrayList(u32),
    logits: []f32,
    eog: [4]u32,
    eog_len: usize,
    options: Options,

    pub fn create(io: Io, allocator: Allocator, path: []const u8, options: Options) !*Runtime {
        const rt = try allocator.create(Runtime);
        errdefer allocator.destroy(rt);
        rt.allocator = allocator;
        rt.io = io;
        rt.options = options;
        rt.model = try model_mod.open(io, allocator, path);
        errdefer rt.model.deinit();
        rt.tok = try tokenizer_mod.Tokenizer.init(allocator, rt.model.meta);
        errdefer rt.tok.deinit();
        const threads = if (options.threads == 0) try std.Thread.getCpuCount() else options.threads;
        try rt.pool.init(io, allocator, threads);
        errdefer rt.pool.deinit();
        const ctx = @min(options.max_ctx, rt.model.h.context_length);
        if (options.backend) |factory| {
            rt.backend = try factory.create(factory.ctx, io, allocator, rt.model, ctx);
        } else {
            const cpu = try allocator.create(CpuBackend);
            errdefer allocator.destroy(cpu);
            cpu.session = try engine.Session.init(allocator, rt.model, &rt.pool, ctx);
            rt.backend = .{ .ctx = cpu, .vtable = &CpuBackend.vtable, .name = "cpu" };
        }
        errdefer rt.backend.vtable.destroy(rt.backend.ctx);
        rt.history = .empty;
        rt.logits = try allocator.alloc(f32, rt.model.h.n_vocab);
        rt.eog_len = 0;
        rt.eog[rt.eog_len] = rt.tok.eos_id;
        rt.eog_len += 1;
        for ([_][]const u8{ "<|im_end|>", "<|endoftext|>" }) |text| {
            if (rt.tok.tokenId(text)) |id| {
                if (std.mem.indexOfScalar(u32, rt.eog[0..rt.eog_len], id) == null) {
                    rt.eog[rt.eog_len] = id;
                    rt.eog_len += 1;
                }
            }
        }
        return rt;
    }

    pub fn destroy(rt: *Runtime) void {
        const allocator = rt.allocator;
        allocator.free(rt.logits);
        rt.history.deinit(allocator);
        rt.backend.vtable.destroy(rt.backend.ctx);
        rt.pool.deinit();
        rt.tok.deinit();
        rt.model.deinit();
        allocator.destroy(rt.model);
        allocator.destroy(rt);
    }

    /// The CPU engine session, when the CPU backend is active.
    pub fn cpuSession(rt: *Runtime) ?*engine.Session {
        if (rt.backend.vtable != &CpuBackend.vtable) return null;
        const cpu: *CpuBackend = @ptrCast(@alignCast(rt.backend.ctx));
        return &cpu.session;
    }

    pub fn isEog(rt: *const Runtime, id: u32) bool {
        return std.mem.indexOfScalar(u32, rt.eog[0..rt.eog_len], id) != null;
    }

    pub fn reset(rt: *Runtime) void {
        rt.backend.reset();
        rt.history.clearRetainingCapacity();
    }

    /// Feeds prompt tokens, reusing the processed prefix when the new prompt extends it.
    /// Leaves next-token logits in rt.logits.
    pub fn prefill(rt: *Runtime, tokens: []const u32, stats: *Stats) !void {
        stats.prompt_tokens = tokens.len;
        const old = rt.history.items;
        var reuse: usize = 0;
        if (old.len > 0 and old.len < tokens.len and std.mem.eql(u32, old, tokens[0..old.len])) {
            reuse = old.len;
        } else {
            rt.reset();
        }
        stats.reused_tokens = reuse;
        const start = Io.Clock.awake.now(rt.io);
        var i = reuse;
        while (i < tokens.len) {
            const end = @min(tokens.len, i + rt.options.prefill_chunk);
            try rt.backend.forward(tokens[i..end], if (end == tokens.len) rt.logits else null);
            try rt.history.appendSlice(rt.allocator, tokens[i..end]);
            i = end;
        }
        stats.prefill_ns = @intCast(start.durationTo(Io.Clock.awake.now(rt.io)).nanoseconds);
    }

    /// Accepts one sampled token and computes the following logits.
    pub fn step(rt: *Runtime, token: u32) !void {
        try rt.backend.forward(&.{token}, rt.logits);
        try rt.history.append(rt.allocator, token);
    }

    /// Runs prefill and sampling until an end-of-generation token, max_tokens, or the sink stops it.
    pub fn generate(rt: *Runtime, prompt: []const u32, params: sampler_mod.Params, max_tokens: usize, sink: Sink) !Stats {
        var stats = Stats{};
        try rt.prefill(prompt, &stats);
        const start = Io.Clock.awake.now(rt.io);
        try rt.decodeLoop(params, max_tokens, sink, &stats);
        stats.decode_ns = @intCast(start.durationTo(Io.Clock.awake.now(rt.io)).nanoseconds);
        return stats;
    }

    fn decodeLoop(rt: *Runtime, params: sampler_mod.Params, max_tokens: usize, sink: Sink, stats: *Stats) !void {
        var sampler = try sampler_mod.Sampler.init(rt.allocator, rt.logits.len, params);
        defer sampler.deinit();
        var decoder = tokenizer_mod.StreamDecoder{};
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(rt.allocator);
        while (stats.generated_tokens < max_tokens) {
            const token = sampler.sample(rt.logits, rt.history.items);
            stats.generated_tokens += 1;
            if (rt.isEog(token)) {
                stats.stop_reason = .eos;
                // The end token belongs to the conversation so a following turn can reuse the prefix.
                if (rt.backend.position() < rt.backend.capacity()) {
                    try rt.backend.forward(&.{token}, null);
                    try rt.history.append(rt.allocator, token);
                }
                return;
            }
            text.clearRetainingCapacity();
            try decoder.push(rt.allocator, &rt.tok, token, &text);
            // history only holds tokens the graph has processed, so an unprocessed final token is not recorded.
            if (!sink.func(sink.ctx, text.items, token)) {
                stats.stop_reason = .cancelled;
                return;
            }
            if (stats.generated_tokens == max_tokens or rt.backend.position() + 1 >= rt.backend.capacity()) {
                stats.stop_reason = .length;
                break;
            }
            try rt.step(token);
        }
        text.clearRetainingCapacity();
        try decoder.flush(rt.allocator, &text);
        if (text.items.len > 0) _ = sink.func(sink.ctx, text.items, std.math.maxInt(u32));
    }
};
