const std = @import("std");
const core = @import("gyro_dx");
const c = @import("native_abi.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const timeout_ms: u32 = 30000;

fn check(result: c.gd_result) !void {
    if (result < 0) {
        std.debug.print("Native HRESULT: 0x{x:0>8}\n", .{@as(u32, @bitCast(result))});
        return error.NativeBackendFailed;
    }
}

fn create(adapter: u32, require_linalg: bool) !*c.gd_context {
    var context: ?*c.gd_context = null;
    const opts = c.gd_options{ .adapter_index = adapter, .enable_debug = 0,
        .require_linalg = @intFromBool(require_linalg), .staging_bytes = 32 * 1024 * 1024, .queue_capacity = 1024 };
    try check(c.gd_create(&opts, &context));
    return context orelse error.NativeBackendFailed;
}

fn destroy(context: *c.gd_context) void {
    check(c.gd_destroy(context, timeout_ms)) catch {
        std.debug.print("The backend did not drain before process exit.\n", .{});
    };
}

fn adapterArg(args: []const []const u8, index: usize) !u32 {
    return if (args.len > index) try std.fmt.parseInt(u32, args[index], 10) else 0;
}

pub fn runStandalone(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer) !void {
    const is_probe = std.mem.eql(u8, args[1], "probe");
    if (std.mem.eql(u8, args[1], "gyro")) return validateGyro(io, allocator, args, writer);
    if (is_probe) {
        if (args.len < 2 or args.len > 3) return error.InvalidArguments;
        const context = try create(try adapterArg(args, 2), false);
        defer destroy(context);
        var caps: c.gd_caps = undefined;
        try check(c.gd_get_caps(context, &caps));
        const adapter_name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&caps.adapter_name)), 0);
        try std.json.Stringify.value(.{
            .adapter = adapter_name,
            .abi_version = caps.abi_version,
            .vendor_id = caps.vendor_id,
            .device_id = caps.device_id,
            .shader_model = caps.shader_model,
            .linalg_tier = caps.linalg_tier,
            .wave16_fp16_fp32_flags = caps.wave16_fp16_fp32_flags,
            .vector_fp16_fp32_flags = caps.vector_fp16_fp32_flags,
            .experimental_features_enabled = caps.experimental_features_enabled != 0,
            .dedicated_video_bytes = caps.dedicated_video_bytes,
            .local_budget_bytes = caps.local_budget_bytes,
            .local_usage_bytes = caps.local_usage_bytes,
        }, .{ .whitespace = .indent_2 }, writer);
    } else {
        if (args.len < 3 or args.len > 4) return error.InvalidArguments;
        const shader = try Io.Dir.cwd().readFileAlloc(io, args[2], allocator, .limited(16 * 1024 * 1024));
        defer allocator.free(shader);
        const context = try create(try adapterArg(args, 3), true);
        defer destroy(context);
        var max_abs_error: f32 = undefined;
        try check(c.gd_validate_linalg(context, shader.ptr, shader.len, &max_abs_error, timeout_ms));
        try std.json.Stringify.value(.{ .matrix_m = 16, .matrix_n = 16, .matrix_k = 16,
            .max_absolute_error = max_abs_error, .validation_passed = true }, .{ .whitespace = .indent_2 }, writer);
    }
    try writer.writeByte('\n');
}

/// This command validates byte transport. It does not execute expert matmuls.
pub fn stream(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer,
    model: core.gguf.Model, source: Io.File) !void
{
    if (args.len < 5 or args.len > 6) return error.InvalidArguments;
    const layer = try std.fmt.parseInt(u32, args[3], 10);
    const expert = try std.fmt.parseInt(u32, args[4], 10);
    const projections = try core.plan.expertBundle(model, layer, expert);
    var ranges: [3]c.gd_range = undefined;
    var bytes: u64 = 0;
    for (projections, 0..) |p, i| {
        if (p.bytes > 32 * 1024 * 1024) return error.ExpertProjectionTooLarge;
        ranges[i] = .{ .file_offset = p.file_offset, .buffer_offset = bytes,
            .bytes = @intCast(p.bytes), .reserved = 0 };
        bytes = try std.math.add(u64, bytes, p.bytes);
    }
    if (bytes > std.math.maxInt(u32)) return error.ExpertBundleTooLarge;
    const context = try create(try adapterArg(args, 5), false);
    defer destroy(context);
    const filename = try allocator.dupeSentinel(u8, args[2], 0);
    defer allocator.free(filename);
    var file: ?*c.gd_file = null;
    try check(c.gd_open_file(context, filename.ptr, &file));
    var source_size: u64 = 0;
    try check(c.gd_file_size(file, &source_size));
    if (source_size != model.file_size) return error.ModelChanged;
    var buffer: ?*c.gd_buffer = null;
    try check(c.gd_create_buffer(context, bytes, &buffer));
    var ticket: u64 = 0;
    try check(c.gd_read_ranges(context, file, buffer, &ranges, ranges.len, &ticket));
    try check(c.gd_wait_storage(context, ticket, timeout_ms));
    const actual = try allocator.alloc(u8, @intCast(bytes));
    defer allocator.free(actual);
    const expected = try allocator.alloc(u8, @intCast(bytes));
    defer allocator.free(expected);
    try check(c.gd_readback(context, buffer, 0, @intCast(bytes), actual.ptr, timeout_ms));
    for (ranges) |range| {
        const offset: usize = @intCast(range.buffer_offset);
        const view = expected[offset..][0..range.bytes];
        if (try source.readPositionalAll(io, view, range.file_offset) != view.len) return error.UnexpectedEnd;
    }
    if (!std.mem.eql(u8, actual, expected)) return error.TransferMismatch;
    try std.json.Stringify.value(.{ .layer = layer, .expert = expert, .bytes = bytes,
        .storage_ticket = ticket, .readback_equal = true, .expert_compute_executed = false },
        .{ .whitespace = .indent_2 }, writer);
    try writer.writeByte('\n');
}

fn validateGyro(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer) !void {
    if (args.len < 3 or args.len > 4) return error.InvalidArguments;
    const shader = try Io.Dir.cwd().readFileAlloc(io, args[2], allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(shader);
    const context = try create(try adapterArg(args, 3), false);
    defer destroy(context);
    const codebook = @embedFile("gyro_codebook.f16");
    const scales = [_]u16{ 0, 0x8000, 1, 0x8001, 0x3ff, 0x83ff, 0x400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff };
    var state: u32 = 0x4759524f;
    for (144..152) |type_value| {
        const type_id: u32 = @intCast(type_value);
        const info = core.gyro.typeInfo(type_id).?;
        const encoded = try allocator.alloc(u8, @as(usize, info.block_bytes) * 64);
        defer allocator.free(encoded);
        const expected = try allocator.alloc(f32, 128 * 64);
        defer allocator.free(expected);
        for (encoded) |*byte| {
            state = state *% 1664525 +% 1013904223;
            byte.* = @truncate(state >> 24);
        }
        for (0..64) |block| {
            const offset = block * info.block_bytes;
            std.mem.writeInt(u16, encoded[offset..][0..2], scales[block % scales.len], .little);
        }
        try core.gyro.decodeRow(type_id, 15, encoded, expected);
        var max_abs_error: f32 = undefined;
        try check(c.gd_validate_gyro(context, shader.ptr, shader.len, codebook.ptr, codebook.len,
            encoded.ptr, encoded.len, type_id, expected.ptr, @intCast(expected.len), &max_abs_error, timeout_ms));
        try std.json.Stringify.value(.{ .type_id = type_id, .type_name = info.name,
            .blocks = 64, .values = expected.len, .max_absolute_error = max_abs_error,
            .validation_passed = true }, .{ .whitespace = .indent_2 }, writer);
        try writer.writeByte('\n');
    }
}
