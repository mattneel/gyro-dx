const std = @import("std");
const core = @import("gyro_dx");
const options = @import("build_options");
const Io = std.Io;
const Allocator = std.mem.Allocator;

// Agility exports must belong to the process executable.
pub export const D3D12SDKVersion: u32 = 722;
pub export const D3D12SDKPath: [*:0]const u8 = ".\\D3D12\\";

const usage =
    \\gyro-dx: Qwen Gyro engine foundation
    \\Pinned compiler: Zig 0.18.0-dev.738+ae9a8acc7
    \\
    \\Commands:
    \\  inspect MODEL.gguf
    \\  inspect-prefix HEADER.bin FULL_FILE_BYTES
    \\  manifest MODEL.gguf
    \\  plan MODEL.gguf AVAILABLE_GiB RESERVE_GiB
    \\  ranges MODEL.gguf LAYER EXPERT
    \\  replay MODEL.gguf TRACE.txt CACHE_SLOTS
    \\  ple-ranges MODEL.gguf TOKEN_ID [TOKEN_ID...]
    \\  probe [ADAPTER_INDEX]
    \\  stream MODEL.gguf LAYER EXPERT [ADAPTER_INDEX]
    \\  linalg SHADER.dxil [ADAPTER_INDEX]
    \\  gyro SHADER.dxil [ADAPTER_INDEX]
    \\
    \\inspect-prefix validates a header against a supplied size. It does not inspect payload bytes.
    \\plan uses a usable GPU budget, with an explicit reserve for state and scratch allocations.
    \\replay simulates completed demand reads from a routing trace. It does not measure speed.
    \\probe, stream, linalg, and gyro require the optional Windows native backend.
    \\This package does not yet generate text.
    \\
;

const Loaded = struct {
    prefix: []u8,
    model: core.gguf.Model,
    file: Io.File,
    fn deinit(self: *Loaded, io: Io, allocator: Allocator) void {
        self.model.deinit(); allocator.free(self.prefix); self.file.close(io);
    }
};

fn load(io: Io, allocator: Allocator, path: []const u8, expected_size: ?u64) !Loaded {
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);
    const stat = try file.stat(io);
    const full_size = expected_size orelse stat.size;
    if (expected_size != null and full_size < stat.size) return error.InvalidFileSize;
    const amount: usize = @intCast(@min(stat.size, core.gguf.max_header_bytes));
    const prefix = try allocator.alloc(u8, amount);
    errdefer allocator.free(prefix);
    if (try file.readPositionalAll(io, prefix, 0) != prefix.len) return error.UnexpectedEnd;
    var model = try core.gguf.parse(allocator, prefix, full_size);
    errdefer model.deinit();
    try model.validateGyro();
    return .{ .prefix = prefix, .model = model, .file = file };
}

fn outputJson(writer: *Io.Writer, value: anytype) !void {
    try std.json.Stringify.value(value, .{ .whitespace = .indent_2 }, writer);
    try writer.writeByte('\n');
}

fn inspect(writer: *Io.Writer, item: Loaded, prefix_only: bool) !void {
    const m = item.model;
    const summary = try core.plan.summarize(m);
    const ple = try core.ple.fromModel(m);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(item.prefix[0..@intCast(m.data_offset)], &hash, .{});
    const hash_text = std.fmt.bytesToHex(hash, .lower);
    try outputJson(writer, .{
        .architecture = try m.text("general.architecture"),
        .header_only = prefix_only,
        .file_bytes = m.file_size,
        .header_bytes = m.data_offset,
        .header_sha256 = &hash_text,
        .alignment = m.alignment,
        .tensor_count = m.tensors.len,
        .metadata_count = m.metadata.len,
        .summary = summary,
        .ple = .{ .layer = ple.layer, .heads = core.ple.head_count, .row_bytes = ple.row_bytes,
            .eos_token = ple.eos_token, .vocabulary_size = ple.vocab_size },
        .inference_available = false,
    });
}

fn parseU32(text: []const u8) !u32 { return std.fmt.parseInt(u32, text, 10); }
fn parseGiB(text: []const u8) !u64 {
    const value = try std.fmt.parseFloat(f64, text);
    if (!std.math.isFinite(value) or value <= 0 or value > 1048576) return error.InvalidBudget;
    return @intFromFloat(value * 1073741824.0);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    var buffer: [8192]u8 = undefined;
    var stdout = Io.File.Writer.init(.stdout(), init.io, &buffer);
    run(init.io, allocator, args, &stdout.interface) catch |err| {
        std.debug.print("gyro-dx: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    try stdout.interface.flush();
}

fn run(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer) !void {
    if (args.len < 2 or std.mem.eql(u8, args[1], "help") or std.mem.eql(u8, args[1], "--help")) {
        try writer.writeAll(usage); return;
    }
    const command = args[1];
    if (std.mem.eql(u8, command, "probe") or std.mem.eql(u8, command, "linalg") or std.mem.eql(u8, command, "gyro")) {
        if (comptime options.native) return @import("native.zig").runStandalone(io, allocator, args, writer);
        return error.NativeBackendNotBuilt;
    }
    if (args.len < 3) return error.InvalidArguments;
    const prefix_only = std.mem.eql(u8, command, "inspect-prefix");
    if (prefix_only and args.len != 4) return error.InvalidArguments;
    const expected_size = if (prefix_only) try std.fmt.parseInt(u64, args[3], 10) else null;
    var item = try load(io, allocator, args[2], expected_size);
    defer item.deinit(io, allocator);
    if (std.mem.eql(u8, command, "inspect") or prefix_only) {
        if (!prefix_only and args.len != 3) return error.InvalidArguments;
        return inspect(writer, item, prefix_only);
    }
    if (std.mem.eql(u8, command, "manifest")) {
        if (args.len != 3) return error.InvalidArguments;
        return outputJson(writer, .{ .file_bytes = item.model.file_size, .data_offset = item.model.data_offset,
            .tensors = item.model.tensors });
    }
    if (std.mem.eql(u8, command, "plan")) {
        if (args.len != 5) return error.InvalidArguments;
        const summary = try core.plan.summarize(item.model);
        return outputJson(writer, try core.plan.budget(summary, try parseGiB(args[3]), try parseGiB(args[4])));
    }
    if (std.mem.eql(u8, command, "ranges")) {
        if (args.len != 5) return error.InvalidArguments;
        return outputJson(writer, try core.plan.expertBundle(item.model, try parseU32(args[3]), try parseU32(args[4])));
    }
    if (std.mem.eql(u8, command, "replay")) {
        if (args.len != 5) return error.InvalidArguments;
        const summary = try core.plan.summarize(item.model);
        if (summary.min_expert_bundle_bytes != summary.max_expert_bundle_bytes) return error.MixedExpertBundleSizes;
        const trace_bytes = try Io.Dir.cwd().readFileAlloc(io, args[3], allocator, .limited(64 * 1024 * 1024));
        defer allocator.free(trace_bytes);
        const report = try core.trace.replay(allocator, trace_bytes, .{
            .layer_count = summary.layers,
            .expert_count = summary.experts_per_layer,
            .bundle_bytes = summary.max_expert_bundle_bytes,
            .slot_count = try parseU32(args[4]),
        });
        return outputJson(writer, report);
    }
    if (std.mem.eql(u8, command, "ple-ranges")) {
        if (args.len < 4) return error.InvalidArguments;
        const config = try core.ple.fromModel(item.model);
        var history = core.ple.History{};
        for (args[3..]) |token_text| {
            const token = try parseU32(token_text);
            const rows = try history.accept(config, token);
            var useful: [16]core.gguf.Range = undefined;
            for (rows, 0..) |row, i| useful[i] = row.range;
            const pages = try core.ple.coalescePages(allocator, &useful, item.model.file_size, 4096);
            defer allocator.free(pages);
            try outputJson(writer, .{ .token = token, .rows = rows, .pages = pages });
        }
        return;
    }
    if (std.mem.eql(u8, command, "stream")) {
        if (comptime options.native) return @import("native.zig").stream(io, allocator, args, writer, item.model, item.file);
        return error.NativeBackendNotBuilt;
    }
    return error.InvalidArguments;
}
