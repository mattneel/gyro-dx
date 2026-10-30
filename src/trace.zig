//! Offline replay for compressed expert cache policy.
//! Each line has one token ID, one layer ID, and one or more expert IDs.
//! Replay resolves demand I/O and compute before the next line.
//! The result contains byte counts. It contains no timing or throughput estimate.
const std = @import("std");
const cache_mod = @import("cache.zig");

pub const Options = struct {
    slot_count: u32,
    bundle_bytes: u64,
    layer_count: u32,
    expert_count: u32,
};

pub const Summary = struct {
    requests: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    loaded_bytes: u64 = 0,
    steps: u64 = 0,
};

pub const Error = error{
    InvalidOptions,
    InvalidInteger,
    MissingField,
    EmptyRoute,
    RouteOutOfRange,
    DuplicateRoute,
    ArithmeticOverflow,
} || cache_mod.Error || std.mem.Allocator.Error;

fn decimal(comptime T: type, text: []const u8) Error!T {
    if (text.len == 0) return error.InvalidInteger;
    for (text) |c| if (c < '0' or c > '9') return error.InvalidInteger;
    return std.fmt.parseInt(T, text, 10) catch error.InvalidInteger;
}

fn add(target: *u64, value: u64) Error!void {
    const result = @addWithOverflow(target.*, value);
    if (result[1] != 0) return error.ArithmeticOverflow;
    target.* = result[0];
}

fn isResident(cache: *const cache_mod.Cache, key: cache_mod.Key) bool {
    for (cache.slots) |slot| {
        if (slot.state == .ready and slot.key.layer == key.layer and slot.key.expert == key.expert and slot.key.variant == key.variant) return true;
    }
    return false;
}

/// Simulate the cache for every trace line in input order.
/// Token IDs label trace rows. They do not change cache keys or enforce token order.
/// The caller supplies the exact compressed bundle size from the model manifest.
pub fn replay(allocator: std.mem.Allocator, text: []const u8, options: Options) Error!Summary {
    if (options.layer_count == 0 or options.expert_count == 0 or options.bundle_bytes == 0 or options.slot_count == 0) return error.InvalidOptions;
    var cache = try cache_mod.Cache.init(allocator, options.slot_count, options.bundle_bytes);
    defer cache.deinit();
    var requests: std.ArrayList(cache_mod.Request) = .empty;
    defer requests.deinit(allocator);
    var leases: std.ArrayList(cache_mod.Lease) = .empty;
    defer leases.deinit(allocator);
    var summary: Summary = .{};

    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const comment = std.mem.indexOfScalar(u8, raw, '#') orelse raw.len;
        const line = std.mem.trim(u8, raw[0..comment], " \t\r");
        if (line.len == 0) continue;
        var fields = std.mem.tokenizeAny(u8, line, " \t\r");
        _ = try decimal(u64, fields.next() orelse return error.MissingField);
        const layer = try decimal(u32, fields.next() orelse return error.MissingField);
        if (layer >= options.layer_count) return error.RouteOutOfRange;
        requests.clearRetainingCapacity();
        while (fields.next()) |field| {
            const expert = try decimal(u32, field);
            if (expert >= options.expert_count) return error.RouteOutOfRange;
            for (requests.items) |earlier| if (earlier.key.expert == expert) return error.DuplicateRoute;
            try requests.append(allocator, .{
                .key = .{ .layer = layer, .expert = expert },
                .bytes = options.bundle_bytes,
            });
        }
        if (requests.items.len == 0) return error.EmptyRoute;
        try leases.resize(allocator, requests.items.len);
        const step = @addWithOverflow(summary.steps, 1);
        if (step[1] != 0) return error.ArithmeticOverflow;
        const load: cache_mod.LoadFence = .{ .value = step[0] };
        const compute: cache_mod.ComputeFence = .{ .value = step[0] };
        var hits: u64 = 0;
        for (requests.items) |request| {
            if (isResident(&cache, request.key)) hits += 1;
        }
        try cache.acquireBatch(requests.items, load, leases.items);
        const count: u64 = @intCast(requests.items.len);
        const misses = count - hits;
        if (misses != 0) try cache.completeLoad(load, load, true);
        for (leases.items) |lease| try cache.retire(lease, compute);
        try cache.observeCompute(compute);

        const loaded = @mulWithOverflow(misses, options.bundle_bytes);
        if (loaded[1] != 0) return error.ArithmeticOverflow;
        try add(&summary.requests, count);
        try add(&summary.hits, hits);
        try add(&summary.misses, misses);
        try add(&summary.loaded_bytes, loaded[0]);
        summary.steps = step[0];
    }
    return summary;
}

test "replay counts exact misses after deterministic eviction" {
    const trace =
        \\# token layer experts
        \\0 0 0 1
        \\0 1 0
        \\1 0 0 1
        \\1 1 0
    ;
    const summary = try replay(std.testing.allocator, trace, .{
        .slot_count = 2,
        .bundle_bytes = 128,
        .layer_count = 2,
        .expert_count = 2,
    });
    try std.testing.expectEqual(Summary{
        .requests = 6,
        .hits = 1,
        .misses = 5,
        .loaded_bytes = 640,
        .steps = 4,
    }, summary);
}

test "enough slots retain experts across token IDs" {
    const summary = try replay(std.testing.allocator, "0 0 0 1\n0 1 0\n1 0 0 1\n1 1 0\n", .{
        .slot_count = 3,
        .bundle_bytes = 1_305_600,
        .layer_count = 2,
        .expert_count = 2,
    });
    try std.testing.expectEqual(@as(u64, 3), summary.hits);
    try std.testing.expectEqual(@as(u64, 3), summary.misses);
    try std.testing.expectEqual(@as(u64, 3_916_800), summary.loaded_bytes);
}

test "replay accepts tabs comments blank lines and variable route counts" {
    const summary = try replay(std.testing.allocator, "\r\n# trace\n9\t0\t1\t2 # route\r\n10 0 1\n", .{
        .slot_count = 2,
        .bundle_bytes = 16,
        .layer_count = 1,
        .expert_count = 3,
    });
    try std.testing.expectEqual(@as(u64, 2), summary.steps);
    try std.testing.expectEqual(@as(u64, 3), summary.requests);
    try std.testing.expectEqual(@as(u64, 1), summary.hits);
    try std.testing.expectEqual(@as(u64, 32), summary.loaded_bytes);
}

test "replay rejects duplicate and out of range experts" {
    const options: Options = .{ .slot_count = 2, .bundle_bytes = 128, .layer_count = 2, .expert_count = 3 };
    try std.testing.expectError(error.DuplicateRoute, replay(std.testing.allocator, "0 0 1 1", options));
    try std.testing.expectError(error.RouteOutOfRange, replay(std.testing.allocator, "0 2 1", options));
    try std.testing.expectError(error.RouteOutOfRange, replay(std.testing.allocator, "0 1 3", options));
}

test "replay rejects incomplete and malformed routes" {
    const options: Options = .{ .slot_count = 2, .bundle_bytes = 128, .layer_count = 2, .expert_count = 3 };
    try std.testing.expectError(error.MissingField, replay(std.testing.allocator, "0", options));
    try std.testing.expectError(error.EmptyRoute, replay(std.testing.allocator, "0 1", options));
    try std.testing.expectError(error.InvalidInteger, replay(std.testing.allocator, "-1 0 1", options));
    try std.testing.expectError(error.InvalidInteger, replay(std.testing.allocator, "0 0 nope", options));
    try std.testing.expectError(error.InvalidInteger, replay(std.testing.allocator, "0 4294967296 1", options));
}

test "replay requires enough slots for the complete route" {
    const options: Options = .{ .slot_count = 1, .bundle_bytes = 128, .layer_count = 1, .expert_count = 2 };
    try std.testing.expectError(error.CacheFull, replay(std.testing.allocator, "0 0 0 1", options));
}

test "empty trace yields zero statistics" {
    const options: Options = .{ .slot_count = 1, .bundle_bytes = 128, .layer_count = 1, .expert_count = 1 };
    try std.testing.expectEqual(Summary{}, try replay(std.testing.allocator, "\n# empty\n", options));
}
