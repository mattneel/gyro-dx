//! Inference commands: generate, chat, and parity dumps.
const std = @import("std");
const core = @import("gyro_dx");
const runtime = core.runtime;
const chat = core.chat;
const sampler = core.sampler;
const build_options = @import("build_options");
const gpu_engine = if (build_options.native) @import("gpu_engine.zig") else struct {};
const Io = std.Io;
const Allocator = std.mem.Allocator;

pub const Flags = struct {
    positional: std.ArrayList([]const u8) = .empty,
    values: std.StringHashMapUnmanaged([]const u8) = .empty,

    pub fn parse(allocator: Allocator, args: []const []const u8) !Flags {
        var f = Flags{};
        var i: usize = 0;
        while (i < args.len) : (i += 1) {
            const a = args[i];
            if (std.mem.startsWith(u8, a, "--")) {
                const name = a[2..];
                if (std.mem.indexOfScalar(u8, name, '=')) |eq| {
                    try f.values.put(allocator, name[0..eq], name[eq + 1 ..]);
                } else if (isSwitch(name)) {
                    try f.values.put(allocator, name, "true");
                } else {
                    if (i + 1 >= args.len) return error.MissingFlagValue;
                    i += 1;
                    try f.values.put(allocator, name, args[i]);
                }
            } else try f.positional.append(allocator, a);
        }
        return f;
    }

    fn isSwitch(name: []const u8) bool {
        for ([_][]const u8{ "chat", "no-think", "think", "raw", "step", "stats", "all-logits", "verbose", "debug-layer", "profile", "no-prefetch" }) |s|
            if (std.mem.eql(u8, s, name)) return true;
        return false;
    }

    pub fn str(f: *const Flags, name: []const u8) ?[]const u8 {
        return f.values.get(name);
    }
    pub fn has(f: *const Flags, name: []const u8) bool {
        return f.values.get(name) != null;
    }
    pub fn int(f: *const Flags, comptime T: type, name: []const u8, default: T) !T {
        const v = f.values.get(name) orelse return default;
        return std.fmt.parseInt(T, v, 10);
    }
    pub fn float(f: *const Flags, name: []const u8, default: f32) !f32 {
        const v = f.values.get(name) orelse return default;
        return std.fmt.parseFloat(f32, v);
    }
};

pub fn samplerParams(f: *const Flags) !sampler.Params {
    return .{
        .temperature = try f.float("temp", 0.6),
        .top_k = try f.int(u32, "top-k", 20),
        .top_p = try f.float("top-p", 0.95),
        .min_p = try f.float("min-p", 0.0),
        .repeat_penalty = try f.float("repeat-penalty", 1.0),
        .seed = try f.int(u64, "seed", 0),
    };
}

/// GPU backend options live for the process; the runtime keeps a pointer to them.
var gpu_options: if (build_options.native) gpu_engine.Options else void = undefined;

pub fn runtimeOptions(io: Io, allocator: Allocator, f: *const Flags) !runtime.Options {
    var o = runtime.Options{
        .threads = try f.int(usize, "threads", 0),
        .max_ctx = try f.int(u32, "ctx", 8192),
        .prefill_chunk = try f.int(u32, "batch", 128),
    };
    const backend = f.str("backend") orelse "cpu";
    if (std.mem.eql(u8, backend, "cpu")) return o;
    if (!std.mem.eql(u8, backend, "gpu")) return error.UnknownBackend;
    if (comptime !build_options.native) return error.NativeBackendNotBuilt;
    const exe_dir = try std.process.executableDirPathAlloc(io, allocator);
    const mode_text = f.str("experts") orelse "auto";
    gpu_options = .{
        .adapter = try f.int(u32, "adapter", 0),
        .debug = f.has("debug-layer"),
        .kernel_dir = f.str("kernels") orelse try std.fs.path.join(allocator, &.{ exe_dir, "kernels" }),
        .model_path = "",
        .expert_mode = std.meta.stringToEnum(gpu_engine.ExpertMode, mode_text) orelse return error.InvalidExpertMode,
        .cache_bytes = if (f.str("cache-gib")) |v| @intFromFloat(try std.fmt.parseFloat(f64, v) * 1073741824.0) else null,
        .reserve_bytes = @intFromFloat((try f.float("reserve-gib", 1.5)) * 1073741824.0),
        .profile = f.has("profile"),
        .prefetch = !f.has("no-prefetch"),
    };
    o.backend = gpu_engine.backendFactory(&gpu_options);
    return o;
}

pub fn openRuntime(io: Io, allocator: Allocator, f: *const Flags, path: []const u8) !*runtime.Runtime {
    const o = try runtimeOptions(io, allocator, f);
    if (comptime build_options.native) {
        if (o.backend != null) gpu_options.model_path = path;
    }
    return runtime.Runtime.create(io, allocator, path, o);
}

const StreamSink = struct {
    writer: *Io.Writer,
    failed: bool = false,
    fn emit(ctx: *anyopaque, text: []const u8, _: u32) bool {
        const self: *StreamSink = @ptrCast(@alignCast(ctx));
        self.writer.writeAll(text) catch {
            self.failed = true;
            return false;
        };
        self.writer.flush() catch {
            self.failed = true;
            return false;
        };
        return true;
    }
};

fn printStats(stats: runtime.Stats) void {
    std.debug.print("\n[prompt {d} tokens ({d} reused), {d:.2} tok/s prefill | generated {d} tokens, {d:.2} tok/s | stop: {s}]\n", .{
        stats.prompt_tokens, stats.reused_tokens, stats.prefillRate(), stats.generated_tokens, stats.decodeRate(),
        @tagName(stats.stop_reason),
    });
}

fn thinking(f: *const Flags) ?bool {
    if (f.has("no-think")) return false;
    if (f.has("think")) return true;
    return null;
}

/// gyro-dx generate MODEL --prompt TEXT [--raw] [--max-tokens N] [sampling flags]
pub fn generate(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer) !void {
    var f = try Flags.parse(allocator, args);
    if (f.positional.items.len != 1) return error.InvalidArguments;
    const prompt_text = f.str("prompt") orelse return error.MissingPrompt;
    const rt = try openRuntime(io, allocator, &f, f.positional.items[0]);
    defer rt.destroy();
    const text = if (f.has("raw")) prompt_text else try chat.render(allocator, &.{.{ .role = .user, .content = prompt_text }}, .{
        .enable_thinking = thinking(&f),
    });
    const tokens = try rt.tok.encode(allocator, text, true);
    var sink = StreamSink{ .writer = writer };
    const stats = try rt.generate(tokens, try samplerParams(&f), try f.int(usize, "max-tokens", 512), .{ .ctx = &sink, .func = StreamSink.emit });
    try writer.writeByte('\n');
    try writer.flush();
    if (f.has("stats")) printStats(stats);
}

/// gyro-dx tokenize MODEL --prompt TEXT [--raw] [--no-think]: prints comma-separated token IDs.
pub fn tokenize(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer) !void {
    var f = try Flags.parse(allocator, args);
    if (f.positional.items.len != 1) return error.InvalidArguments;
    const prompt_text = f.str("prompt") orelse return error.MissingPrompt;
    const model = try core.model.open(io, allocator, f.positional.items[0]);
    defer {
        model.deinit();
        allocator.destroy(model);
    }
    var tok = try core.tokenizer.Tokenizer.init(allocator, model.meta);
    defer tok.deinit();
    const text = if (f.has("raw")) prompt_text else try chat.render(allocator, &.{.{ .role = .user, .content = prompt_text }}, .{
        .enable_thinking = thinking(&f),
    });
    const tokens = try tok.encode(allocator, text, true);
    for (tokens, 0..) |t, i| try writer.print("{s}{d}", .{ if (i == 0) "" else ",", t });
    try writer.writeByte('\n');
}

/// gyro-dx chat MODEL [--system TEXT] [sampling flags]: interactive multi-turn session on stdin.
pub fn interactive(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer) !void {
    var f = try Flags.parse(allocator, args);
    if (f.positional.items.len != 1) return error.InvalidArguments;
    const rt = try openRuntime(io, allocator, &f, f.positional.items[0]);
    defer rt.destroy();
    var messages: std.ArrayList(chat.Message) = .empty;
    if (f.str("system")) |sys| try messages.append(allocator, .{ .role = .system, .content = sys });
    var in_buffer: [65536]u8 = undefined;
    var stdin = Io.File.Reader.init(.stdin(), io, &in_buffer);
    const params = try samplerParams(&f);
    const max_tokens = try f.int(usize, "max-tokens", 4096);
    while (true) {
        try writer.writeAll("\n> ");
        try writer.flush();
        const maybe_line = stdin.interface.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.LineTooLong,
            else => return err,
        };
        const line = maybe_line orelse break;
        const content = chat.trim(line);
        if (content.len == 0) continue;
        if (std.mem.eql(u8, content, "/reset")) {
            messages.shrinkRetainingCapacity(if (f.has("system")) 1 else 0);
            rt.reset();
            continue;
        }
        if (std.mem.eql(u8, content, "/quit")) break;
        try messages.append(allocator, .{ .role = .user, .content = try allocator.dupe(u8, content) });
        const prompt = try chat.render(allocator, messages.items, .{ .enable_thinking = thinking(&f) });
        const tokens = try rt.tok.encode(allocator, prompt, true);
        var reply: std.ArrayList(u8) = .empty;
        const Collect = struct {
            writer: *Io.Writer,
            reply: *std.ArrayList(u8),
            allocator: Allocator,
            fn emit(ctx: *anyopaque, text: []const u8, _: u32) bool {
                const self: *@This() = @ptrCast(@alignCast(ctx));
                self.reply.appendSlice(self.allocator, text) catch return false;
                self.writer.writeAll(text) catch return false;
                self.writer.flush() catch return false;
                return true;
            }
        };
        var collect = Collect{ .writer = writer, .reply = &reply, .allocator = allocator };
        const stats = try rt.generate(tokens, params, max_tokens, .{ .ctx = &collect, .func = Collect.emit });
        if (f.has("stats")) printStats(stats);
        // Keep the answer after the reasoning block as the assistant turn; the template drops old reasoning.
        const full = reply.items;
        var answer = full;
        var reasoning: ?[]const u8 = null;
        if (std.mem.indexOf(u8, full, "</think>")) |end| {
            const open_tag = std.mem.indexOf(u8, full, "<think>");
            const body_start = if (open_tag) |o| o + "<think>".len else 0;
            reasoning = chat.trim(full[body_start..end]);
            answer = full[end + "</think>".len ..];
        }
        try messages.append(allocator, .{ .role = .assistant, .content = chat.trim(answer), .reasoning_content = reasoning });
    }
}

const DumpObserver = struct {
    io: Io,
    dir: Io.Dir,
    step: usize,
    filter: ?[]const u8,
    allocator: Allocator,
    /// Repeated names within one step get a #N suffix, as in the reference dump tool.
    seen: std.StringHashMapUnmanaged(u32) = .empty,
    fn emit(ctx: *anyopaque, name: []const u8, layer: i32, values: []const f32, tokens: usize) void {
        const self: *DumpObserver = @ptrCast(@alignCast(ctx));
        var base_buf: [256]u8 = undefined;
        const base = if (layer >= 0)
            std.fmt.bufPrint(&base_buf, "{s}-{d}", .{ name, layer }) catch return
        else
            std.fmt.bufPrint(&base_buf, "{s}", .{name}) catch return;
        const entry = self.seen.getOrPut(self.allocator, base) catch return;
        if (!entry.found_existing) {
            entry.key_ptr.* = self.allocator.dupe(u8, base) catch return;
            entry.value_ptr.* = 0;
        }
        const occurrence = entry.value_ptr.*;
        entry.value_ptr.* += 1;
        var name_buf: [272]u8 = undefined;
        const full = if (occurrence == 0) base else std.fmt.bufPrint(&name_buf, "{s}#{d}", .{ base, occurrence }) catch return;
        if (self.filter) |flt| {
            var it = std.mem.splitScalar(u8, flt, ',');
            var ok = false;
            while (it.next()) |pat| {
                if (pat.len > 0 and pat[pat.len - 1] == '*') {
                    if (std.mem.startsWith(u8, full, pat[0 .. pat.len - 1])) ok = true;
                } else if (std.mem.eql(u8, pat, full)) ok = true;
            }
            if (!ok) return;
        }
        var path_buf: [512]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "s{d:0>3}.{s}.bin", .{ self.step, full }) catch return;
        const file = self.dir.createFile(self.io, path, .{}) catch return;
        defer file.close(self.io);
        const ne = [4]i64{ @intCast(values.len / tokens), @intCast(tokens), 1, 1 };
        file.writePositionalAll(self.io, std.mem.sliceAsBytes(&ne), 0) catch return;
        file.writePositionalAll(self.io, std.mem.sliceAsBytes(values), 32) catch return;
    }
};

/// gyro-dx dump MODEL OUT_DIR --tokens 1,2,3 [--step] [--filter a,b*]
/// Writes the same file format as the reference llama-gyro-dump tool.
pub fn dump(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer) !void {
    var f = try Flags.parse(allocator, args);
    if (f.positional.items.len != 2) return error.InvalidArguments;
    const rt = try openRuntime(io, allocator, &f, f.positional.items[0]);
    defer rt.destroy();
    try Io.Dir.cwd().createDirPath(io, f.positional.items[1]);
    var dir = try Io.Dir.cwd().openDir(io, f.positional.items[1], .{});
    defer dir.close(io);
    var tokens: std.ArrayList(u32) = .empty;
    if (f.str("tokens")) |list| {
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |t| if (t.len > 0) try tokens.append(allocator, try std.fmt.parseInt(u32, t, 10));
    } else if (f.str("prompt")) |p| {
        try tokens.appendSlice(allocator, try rt.tok.encode(allocator, p, true));
    } else return error.MissingPrompt;
    var obs = DumpObserver{ .io = io, .dir = dir, .step = 0, .filter = f.str("filter"), .allocator = allocator };
    if (rt.cpuSession()) |session| session.observer = .{ .ctx = &obs, .func = DumpObserver.emit };
    const logits = rt.logits;
    if (f.has("step")) {
        for (tokens.items, 0..) |t, i| {
            obs.step = i;
            obs.seen.clearRetainingCapacity();
            try rt.backend.forward(&.{t}, logits);
            writeLogits(io, dir, i, logits);
        }
    } else {
        try rt.backend.forward(tokens.items, logits);
        writeLogits(io, dir, 0, logits);
    }
    const gen = try f.int(usize, "gen", 0);
    if (gen > 0) {
        if (rt.cpuSession()) |session| session.observer = null;
        var list: std.ArrayList(u8) = .empty;
        for (0..gen) |_| {
            var best: u32 = 0;
            for (logits, 0..) |v, i| {
                if (v > logits[best]) best = @intCast(i);
            }
            try list.print(allocator, "{d}\n", .{best});
            if (rt.isEog(best)) break;
            try rt.backend.forward(&.{best}, logits);
        }
        try dir.writeFile(io, .{ .sub_path = "greedy.txt", .data = list.items });
    }
    try writer.print("dumped {d} tokens\n", .{tokens.items.len});
}

fn writeLogits(io: Io, dir: Io.Dir, step: usize, logits: []const f32) void {
    var path_buf: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "s{d:0>3}.logits.bin", .{step}) catch return;
    const file = dir.createFile(io, path, .{}) catch return;
    defer file.close(io);
    const ne = [4]i64{ @intCast(logits.len), 1, 1, 1 };
    file.writePositionalAll(io, std.mem.sliceAsBytes(&ne), 0) catch return;
    file.writePositionalAll(io, std.mem.sliceAsBytes(logits), 32) catch return;
}

/// gyro-dx bench MODEL [--prompt-tokens N] [--gen N] [--backend gpu] [--profile]
/// Prefills a fixed text prompt, then decodes greedily and reports timing as JSON.
pub fn bench(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer) !void {
    var f = try Flags.parse(allocator, args);
    if (f.positional.items.len != 1) return error.InvalidArguments;
    const rt = try openRuntime(io, allocator, &f, f.positional.items[0]);
    defer rt.destroy();
    const n_prompt = try f.int(usize, "prompt-tokens", 512);
    const n_gen = try f.int(usize, "gen", 128);
    const seed_text = "The quick brown fox jumps over the lazy dog while a compiler schedules kernels across many cores. ";
    var text: std.ArrayList(u8) = .empty;
    var tokens: []u32 = &.{};
    while (true) {
        try text.appendSlice(allocator, seed_text);
        tokens = try rt.tok.encode(allocator, text.items, false);
        if (tokens.len >= n_prompt) break;
    }
    tokens = tokens[0..n_prompt];
    var stats = runtime.Stats{};
    try rt.prefill(tokens, &stats);
    var latencies = try allocator.alloc(u64, n_gen);
    var token = argmax(rt.logits);
    const start = Io.Clock.awake.now(io);
    for (0..n_gen) |i| {
        const t0 = Io.Clock.awake.now(io);
        try rt.step(token);
        latencies[i] = @intCast(t0.durationTo(Io.Clock.awake.now(io)).nanoseconds);
        token = argmax(rt.logits);
    }
    const decode_ns: u64 = @intCast(start.durationTo(Io.Clock.awake.now(io)).nanoseconds);
    std.mem.sort(u64, latencies, {}, std.sort.asc(u64));
    const ms = struct {
        fn conv(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.conv;
    try writer.print("{f}\n", .{std.json.fmt(.{
        .backend = rt.backend.name,
        .prompt_tokens = n_prompt,
        .prefill_seconds = @as(f64, @floatFromInt(stats.prefill_ns)) / 1e9,
        .prefill_tokens_per_second = stats.prefillRate(),
        .generated_tokens = n_gen,
        .decode_tokens_per_second = @as(f64, @floatFromInt(n_gen)) * 1e9 / @as(f64, @floatFromInt(decode_ns)),
        .token_latency_ms = .{ .p50 = ms(latencies[n_gen / 2]), .p95 = ms(latencies[n_gen * 95 / 100]), .max = ms(latencies[n_gen - 1]) },
    }, .{ .whitespace = .indent_2 })});
    if (comptime build_options.native) {
        if (std.mem.eql(u8, rt.backend.name, "d3d12")) {
            const e: *gpu_engine.Engine = @ptrCast(@alignCast(rt.backend.ctx));
            try gpu_engine.report(e, writer);
        }
    }
}

fn argmax(logits: []const f32) u32 {
    var best: u32 = 0;
    for (logits, 0..) |v, i| {
        if (v > logits[best]) best = @intCast(i);
    }
    return best;
}
