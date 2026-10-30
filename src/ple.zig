//! PLE address calculation for the released Qwen4Exp 3-gram configuration.
//! The hash follows agentionai/llama.cpp at 9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f.
//! Previous tokens use newest-first order. A missing predecessor acts as EOS.
const std = @import("std");
const gguf = @import("gguf.zig");

pub const head_count = 16;
pub const Error = gguf.ParseError || error{
    UnsupportedArchitecture,
    UnsupportedPleConfiguration,
    MissingPleTable,
    InvalidToken,
    InvalidHeadRange,
    InvalidPageSize,
    InvalidReadRange,
    TooManyRanges,
};

pub const Row = struct {
    head: u32,
    row: u64,
    range: gguf.Range,
};

pub const Config = struct {
    layer: u32,
    eos_token: u32,
    vocab_size: u32,
    multipliers: [3]u64,
    head_offsets: [head_count]u64,
    head_vocab_sizes: [head_count]u64,
    table_offset: u64,
    table_rows: u64,
    row_bytes: u64,
    table_type: u32,

    pub fn rowRanges(config: Config, current: u32, previous: [2]?u32) Error![head_count]Row {
        if (current >= config.vocab_size) return error.InvalidToken;
        var context = [_]u64{ current, config.eos_token, config.eos_token };
        var cut = false;
        for (previous, 0..) |token, index| {
            if (!cut) {
                if (token) |value| {
                    if (value >= config.vocab_size) return error.InvalidToken;
                    if (value == config.eos_token) {
                        cut = true;
                    } else {
                        context[index + 1] = value;
                    }
                } else {
                    cut = true;
                }
            }
        }

        var result: [head_count]Row = undefined;
        var mixed = context[0] *% config.multipliers[0];
        for (1..3) |gram_index| {
            mixed ^= context[gram_index] *% config.multipliers[gram_index];
            for (0..8) |within_gram| {
                const head = (gram_index - 1) * 8 + within_gram;
                const vocab = config.head_vocab_sizes[head];
                if (vocab == 0) return error.InvalidHeadRange;
                const row = try checkedAdd(mixed % vocab, config.head_offsets[head]);
                if (row >= config.table_rows) return error.InvalidHeadRange;
                result[head] = .{
                    .head = @intCast(head),
                    .row = row,
                    .range = .{
                        .file_offset = try checkedAdd(config.table_offset, try checkedMul(row, config.row_bytes)),
                        .bytes = config.row_bytes,
                    },
                };
            }
        }
        return result;
    }
};

pub const History = struct {
    previous: [2]?u32 = .{ null, null },

    /// Compute addresses for one accepted token, then advance the token history.
    /// Speculative tokens need a separate History copy until acceptance.
    pub fn accept(history: *History, config: Config, token: u32) Error![head_count]Row {
        const rows = try config.rowRanges(token, history.previous);
        const newest = history.previous[0];
        history.previous[1] = newest;
        history.previous[0] = token;
        return rows;
    }

    pub fn reset(history: *History) void {
        history.* = .{};
    }
};

pub fn fromModel(model: gguf.Model) Error!Config {
    if (!std.mem.eql(u8, try model.text("general.architecture"), "qwen4exp"))
        return error.UnsupportedArchitecture;
    if (try model.number("qwen4exp.ple.ngram_size") != 3 or
        try model.number("qwen4exp.ple.heads_per_ngram") != 8)
        return error.UnsupportedPleConfiguration;

    const table = model.tensor("per_layer_token_embd.weight") orelse return error.MissingPleTable;
    if (table.rank != 2 or table.dims[0] == 0 or table.dims[1] == 0)
        return error.InvalidDimensions;
    const row_layout = gguf.layout(table.type_id) orelse return error.UnsupportedTensorType;
    if (table.dims[0] % row_layout.block_weights != 0) return error.InvalidDimensions;
    const row_bytes = try checkedMul(table.dims[0] / row_layout.block_weights, row_layout.block_bytes);
    const expected_bytes = try checkedMul(row_bytes, table.dims[1]);
    if (table.byte_length == null or table.byte_length.? != expected_bytes)
        return error.InvalidTensorRange;
    if (try checkedAdd(table.file_offset, expected_bytes) > model.file_size)
        return error.InvalidTensorRange;
    if (try model.number("qwen4exp.embedding_length_per_layer_input") != table.dims[0])
        return error.InvalidDimensions;

    const embedding = model.tensor("token_embd.weight") orelse return error.InvalidMetadata;
    if (embedding.rank != 2 or embedding.dims[1] == 0 or embedding.dims[1] > std.math.maxInt(u32))
        return error.InvalidDimensions;
    const eos = try model.number("qwen4exp.ple.eos_token_id");
    if (eos >= embedding.dims[1]) return error.InvalidToken;

    var config: Config = .{
        .layer = undefined,
        .eos_token = @intCast(eos),
        .vocab_size = @intCast(embedding.dims[1]),
        .multipliers = undefined,
        .head_offsets = undefined,
        .head_vocab_sizes = undefined,
        .table_offset = table.file_offset,
        .table_rows = table.dims[1],
        .row_bytes = row_bytes,
        .table_type = table.type_id,
    };
    try unsignedArray(try model.required("qwen4exp.ple.layer_multipliers"), &config.multipliers);
    try unsignedArray(try model.required("qwen4exp.ple.head_offsets"), &config.head_offsets);
    try unsignedArray(try model.required("qwen4exp.ple.head_vocab_sizes"), &config.head_vocab_sizes);
    const layer_entry = try model.required("qwen4exp.ple.layers");
    if (layer_entry.kind != .array or layer_entry.raw.len != 16 or
        std.mem.readInt(u32, layer_entry.raw[0..4], .little) != @intFromEnum(gguf.ValueType.int32) or
        std.mem.readInt(u64, layer_entry.raw[4..12], .little) != 1)
        return error.UnsupportedPleConfiguration;
    const layer = std.mem.readInt(i32, layer_entry.raw[12..16], .little);
    if (layer < 0 or @as(u64, @intCast(layer)) >= try model.number("qwen4exp.block_count"))
        return error.InvalidMetadata;
    config.layer = @intCast(layer);
    for (config.head_offsets, config.head_vocab_sizes) |offset, vocab| {
        if (vocab == 0 or offset >= config.table_rows or vocab > config.table_rows - offset)
            return error.InvalidHeadRange;
        if (offset + vocab > std.math.maxInt(i32)) return error.InvalidHeadRange;
    }
    return config;
}

fn unsignedArray(entry: gguf.Entry, destination: []u64) Error!void {
    if (entry.kind != .array or entry.raw.len < 12) return error.InvalidMetadata;
    const kind = std.mem.readInt(u32, entry.raw[0..4], .little);
    const count = std.mem.readInt(u64, entry.raw[4..12], .little);
    if (kind != @intFromEnum(gguf.ValueType.uint64) or count != destination.len)
        return error.UnsupportedPleConfiguration;
    if (entry.raw.len != 12 + destination.len * 8) return error.InvalidMetadata;
    for (destination, 0..) |*value, index| {
        value.* = std.mem.readInt(u64, entry.raw[12 + index * 8 ..][0..8], .little);
    }
}

fn checkedAdd(a: u64, b: u64) Error!u64 {
    return std.math.add(u64, a, b) catch error.IntegerOverflow;
}

fn checkedMul(a: u64, b: u64) Error!u64 {
    return std.math.mul(u64, a, b) catch error.IntegerOverflow;
}

fn offsetLess(_: void, a: gguf.Range, b: gguf.Range) bool {
    return a.file_offset < b.file_offset;
}

/// Expand reads to page boundaries and merge adjacent or overlapping pages.
/// The last range ends at the file size when the final page is incomplete.
/// The caller owns the returned slice.
pub fn coalescePages(allocator: std.mem.Allocator, ranges: []const gguf.Range, file_size: u64, page_size: u64) Error![]gguf.Range {
    if (page_size == 0 or !std.math.isPowerOfTwo(page_size)) return error.InvalidPageSize;
    if (ranges.len > 65536) return error.TooManyRanges;
    const pages = try allocator.alloc(gguf.Range, ranges.len);
    errdefer allocator.free(pages);
    for (ranges, pages) |range, *page| {
        if (range.bytes == 0 or range.file_offset >= file_size or range.bytes > file_size - range.file_offset)
            return error.InvalidReadRange;
        const start = range.file_offset - range.file_offset % page_size;
        const end = range.file_offset + range.bytes;
        const padding = (page_size - end % page_size) % page_size;
        const last = end + @min(padding, file_size - end);
        page.* = .{ .file_offset = start, .bytes = last - start };
    }
    std.mem.sort(gguf.Range, pages, {}, offsetLess);
    var count: usize = 0;
    for (pages) |page| {
        if (count != 0) {
            const previous = &pages[count - 1];
            const previous_end = previous.file_offset + previous.bytes;
            if (page.file_offset <= previous_end) {
                previous.bytes = @max(previous_end, page.file_offset + page.bytes) - previous.file_offset;
                continue;
            }
        }
        pages[count] = page;
        count += 1;
    }
    return allocator.realloc(pages, count);
}

test "PLE page requests merge, cover row tails, and stop at EOF" {
    const input = [_]gguf.Range{
        .{ .file_offset = 4090, .bytes = 170 },
        .{ .file_offset = 160, .bytes = 170 },
        .{ .file_offset = 9000, .bytes = 170 },
        .{ .file_offset = 12200, .bytes = 170 },
        .{ .file_offset = 12400, .bytes = 170 },
    };
    const pages = try coalescePages(std.testing.allocator, &input, 12570, 4096);
    defer std.testing.allocator.free(pages);
    try std.testing.expectEqual(@as(usize, 1), pages.len);
    try std.testing.expectEqual(@as(u64, 0), pages[0].file_offset);
    try std.testing.expectEqual(@as(u64, 12570), pages[0].bytes);
    try std.testing.expectError(error.InvalidPageSize, coalescePages(std.testing.allocator, &input, 12570, 3));
    try std.testing.expectError(error.InvalidReadRange, coalescePages(std.testing.allocator, &input, 12450, 4096));
    const distant = [_]gguf.Range{ .{ .file_offset = 4100, .bytes = 170 }, .{ .file_offset = 20000, .bytes = 170 } };
    const separate = try coalescePages(std.testing.allocator, &distant, 24000, 4096);
    defer std.testing.allocator.free(separate);
    try std.testing.expectEqual(@as(usize, 2), separate.len);
    try std.testing.expectEqual(@as(u64, 4096), separate[0].file_offset);
    try std.testing.expectEqual(@as(u64, 4096), separate[0].bytes);
}

fn fixtureConfig() Config {
    return .{
        .layer = 1,
        .eos_token = 248044,
        .vocab_size = 248320,
        .multipliers = .{ 23703573157769, 20109073645365, 8052911324071 },
        .head_offsets = .{ 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275 },
        .head_vocab_sizes = .{ 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 },
        .table_offset = 693477216,
        .table_rows = 320001536,
        .row_bytes = 170,
        .table_type = 8,
    };
}

test "PLE rows match an independent upstream formula oracle" {
    const vectors = @import("ple_vectors.zig");
    const Case = struct { token: u32, previous: [2]?u32 };
    const cases = [_]Case{
        .{ .token = 1, .previous = .{ null, null } },
        .{ .token = 12345, .previous = .{ 456, 789 } },
        .{ .token = 248044, .previous = .{ 456, 789 } },
        .{ .token = 123, .previous = .{ 248044, 555 } },
        .{ .token = 123, .previous = .{ null, 555 } },
        .{ .token = 123, .previous = .{ 555, 248044 } },
        .{ .token = 248319, .previous = .{ 248318, 248317 } },
        .{ .token = 0, .previous = .{ 0, 0 } },
    };
    const config = fixtureConfig();
    for (cases, vectors.expected) |case, expected| {
        const actual = try config.rowRanges(case.token, case.previous);
        for (actual, expected, 0..) |row, row_id, head| {
            try std.testing.expectEqual(row_id, row.row);
            try std.testing.expectEqual(@as(u32, @intCast(head)), row.head);
            try std.testing.expectEqual(config.table_offset + row_id * 170, row.range.file_offset);
            try std.testing.expectEqual(@as(u64, 170), row.range.bytes);
        }
    }
    var wrapping = config;
    wrapping.multipliers = .{ std.math.maxInt(u64), std.math.maxInt(u64) - 16, std.math.maxInt(u64) - 32 };
    const actual = try wrapping.rowRanges(248319, .{ 248318, 248317 });
    for (actual, vectors.wrapping) |row, expected| try std.testing.expectEqual(expected, row.row);
}

test "PLE history commits only valid tokens and preserves EOS token context" {
    const config = fixtureConfig();
    var history: History = .{};
    _ = try history.accept(config, 789);
    _ = try history.accept(config, 456);
    const eos_rows = try history.accept(config, 248044);
    const expected = try config.rowRanges(248044, .{ 456, 789 });
    for (eos_rows, expected) |row, expected_row| try std.testing.expectEqual(expected_row.row, row.row);
    try std.testing.expectEqual(@as(?u32, 248044), history.previous[0]);
    try std.testing.expectEqual(@as(?u32, 456), history.previous[1]);
    try std.testing.expectError(error.InvalidToken, history.accept(config, 248320));
    try std.testing.expectEqual(@as(?u32, 248044), history.previous[0]);
    const next_rows = try history.accept(config, 123);
    const fresh_rows = try config.rowRanges(123, .{ null, null });
    for (next_rows, fresh_rows) |row, expected_row| try std.testing.expectEqual(expected_row.row, row.row);
    history.reset();
    try std.testing.expectEqual(@as(?u32, null), history.previous[0]);
    try std.testing.expectEqual(@as(?u32, null), history.previous[1]);
}
