const std = @import("std");
const gyro = @import("gyro.zig");
const Allocator = std.mem.Allocator;

pub const max_header_bytes = 64 * 1024 * 1024;
pub const ParseError = error{
    UnexpectedEnd, BadMagic, UnsupportedVersion, InvalidType, InvalidBoolean,
    LimitExceeded, IntegerOverflow, InvalidAlignment, InvalidDimensions,
    DuplicateKey, DuplicateTensor, InvalidTensorRange, OverlappingTensors,
    UnsupportedTensorType, InvalidExpert, InvalidMetadata, UnsupportedGyroMetadata,
} || Allocator.Error;

pub const ValueType = enum(u32) {
    uint8 = 0, int8 = 1, uint16 = 2, int16 = 3, uint32 = 4, int32 = 5,
    float32 = 6, boolean = 7, string = 8, array = 9, uint64 = 10,
    int64 = 11, float64 = 12,
};

const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(c: *Cursor, n: usize) ParseError![]const u8 {
        if (n > c.bytes.len - c.pos) return error.UnexpectedEnd;
        const result = c.bytes[c.pos..][0..n];
        c.pos += n;
        return result;
    }

    fn int(c: *Cursor, comptime T: type) ParseError!T {
        const bytes = try c.take(@sizeOf(T));
        return std.mem.readInt(T, bytes[0..@sizeOf(T)], .little);
    }

    fn string(c: *Cursor) ParseError![]const u8 {
        const len = try c.int(u64);
        if (len > max_header_bytes) return error.LimitExceeded;
        return c.take(@intCast(len));
    }

    fn tag(c: *Cursor) ParseError!ValueType {
        return std.enums.fromInt(ValueType, try c.int(u32)) orelse error.InvalidType;
    }

    fn skip(c: *Cursor, kind: ValueType, depth: u8) ParseError!void {
        if (depth > 8) return error.LimitExceeded;
        switch (kind) {
            .uint8, .int8 => _ = try c.take(1),
            .boolean => if (try c.int(u8) > 1) { return error.InvalidBoolean; },
            .uint16, .int16 => _ = try c.take(2),
            .uint32, .int32, .float32 => _ = try c.take(4),
            .uint64, .int64, .float64 => _ = try c.take(8),
            .string => _ = try c.string(),
            .array => {
                const elem_type = try c.tag();
                const n = try c.int(u64);
                if (n > 16 * 1024 * 1024) return error.LimitExceeded;
                // This also validates booleans and nested array lengths.
                for (0..@intCast(n)) |_| try c.skip(elem_type, depth + 1);
            },
        }
    }
};

pub const Entry = struct {
    key: []const u8,
    kind: ValueType,
    raw: []const u8,

    pub fn unsigned(e: Entry) ParseError!u64 {
        var c = Cursor{ .bytes = e.raw };
        return switch (e.kind) {
            .uint8 => try c.int(u8), .uint16 => try c.int(u16),
            .uint32 => try c.int(u32), .uint64 => try c.int(u64),
            else => error.InvalidMetadata,
        };
    }

    pub fn string(e: Entry) ParseError![]const u8 {
        if (e.kind != .string) return error.InvalidMetadata;
        var c = Cursor{ .bytes = e.raw };
        return c.string();
    }

    pub fn boolean(e: Entry) ParseError!bool {
        if (e.kind != .boolean or e.raw.len != 1) return error.InvalidMetadata;
        return e.raw[0] == 1;
    }

    pub fn strings(e: Entry, allocator: Allocator) ParseError![][]const u8 {
        if (e.kind != .array) return error.InvalidMetadata;
        var c = Cursor{ .bytes = e.raw };
        if (try c.tag() != .string) return error.InvalidMetadata;
        const n = try c.int(u64);
        const result = try allocator.alloc([]const u8, @intCast(n));
        errdefer allocator.free(result);
        for (result) |*value| value.* = try c.string();
        return result;
    }

    pub fn ints32(e: Entry, allocator: Allocator) ParseError![]i32 {
        if (e.kind != .array) return error.InvalidMetadata;
        var c = Cursor{ .bytes = e.raw };
        if (try c.tag() != .int32) return error.InvalidMetadata;
        const n = try c.int(u64);
        const result = try allocator.alloc(i32, @intCast(n));
        errdefer allocator.free(result);
        for (result) |*value| value.* = try c.int(i32);
        return result;
    }
};

pub const Layout = struct { name: []const u8, block_weights: u64, block_bytes: u64 };

pub fn layout(type_id: u32) ?Layout {
    if (gyro.typeInfo(type_id)) |info| return .{
        .name = info.name, .block_weights = info.block_weights, .block_bytes = info.block_bytes,
    };
    return switch (type_id) {
        0 => .{ .name = "F32", .block_weights = 1, .block_bytes = 4 },
        1 => .{ .name = "F16", .block_weights = 1, .block_bytes = 2 },
        8 => .{ .name = "Q8_0", .block_weights = 32, .block_bytes = 34 },
        30 => .{ .name = "BF16", .block_weights = 1, .block_bytes = 2 },
        else => null,
    };
}

pub const Range = struct { file_offset: u64, bytes: u64 };
pub const Tensor = struct {
    name: []const u8,
    rank: u8,
    dims: [4]u64,
    type_id: u32,
    relative_offset: u64,
    file_offset: u64 = 0,
    byte_length: ?u64 = null,

    pub fn expertRange(t: Tensor, expert: u32) ParseError!Range {
        if (t.rank != 3 or expert >= t.dims[2]) return error.InvalidExpert;
        const n = t.byte_length orelse return error.UnsupportedTensorType;
        const stride = n / t.dims[2];
        return .{ .file_offset = try add(t.file_offset, try mul(stride, expert)), .bytes = stride };
    }

    pub fn isExpert(t: Tensor) bool {
        const suffixes = [_][]const u8{
            ".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight",
        };
        for (suffixes) |suffix| if (std.mem.endsWith(u8, t.name, suffix)) return true;
        return false;
    }
};

pub const Model = struct {
    allocator: Allocator,
    version: u32,
    alignment: u32,
    data_offset: u64,
    file_size: u64,
    metadata: []Entry,
    tensors: []Tensor,

    pub fn deinit(m: *Model) void {
        m.allocator.free(m.metadata);
        m.allocator.free(m.tensors);
        m.* = undefined;
    }

    pub fn get(m: Model, key: []const u8) ?Entry {
        for (m.metadata) |entry| if (std.mem.eql(u8, entry.key, key)) return entry;
        return null;
    }

    pub fn required(m: Model, key: []const u8) ParseError!Entry {
        return m.get(key) orelse error.InvalidMetadata;
    }

    pub fn number(m: Model, key: []const u8) ParseError!u64 {
        return (try m.required(key)).unsigned();
    }

    pub fn text(m: Model, key: []const u8) ParseError![]const u8 {
        return (try m.required(key)).string();
    }

    pub fn tensor(m: Model, name: []const u8) ?Tensor {
        for (m.tensors) |t| if (std.mem.eql(u8, t.name, name)) return t;
        return null;
    }

    /// Validates the released version-1 rotation metadata. No transform is implicit.
    pub fn validateGyro(m: Model) ParseError!void {
        var has_gyro = false;
        for (m.tensors) |t| if (gyro.typeInfo(t.type_id) != null) { has_gyro = true; };
        if (!has_gyro) return;
        if (try m.number("agention.trellis.state_bits") != 15) return error.UnsupportedGyroMetadata;
        if (try m.number("prism.hadamard.version") != 1) return error.UnsupportedGyroMetadata;
        const b = try m.number("prism.hadamard.block_size");
        if (b == 0 or b > 65536 or !std.math.isPowerOfTwo(b)) return error.InvalidMetadata;
        if (!std.mem.eql(u8, try m.text("prism.hadamard.transform"), "normalized-sylvester-walsh-hadamard"))
            return error.UnsupportedGyroMetadata;
        if (!std.mem.eql(u8, try m.text("prism.hadamard.axis"), "input-last-dimension"))
            return error.UnsupportedGyroMetadata;
        const mode = try m.text("prism.hadamard.sign_mode");
        const explicit = std.mem.eql(u8, mode, "explicit");
        if (!explicit and !std.mem.eql(u8, mode, "identity")) return error.UnsupportedGyroMetadata;

        const names = try (try m.required("prism.hadamard.weight_names")).strings(m.allocator);
        defer m.allocator.free(names);
        if (names.len == 0) return error.InvalidMetadata;
        var name_set = std.StringHashMap(void).init(m.allocator);
        defer name_set.deinit();
        const widths = if (explicit)
            try (try m.required("prism.hadamard.sign_widths")).ints32(m.allocator)
        else try m.allocator.alloc(i32, 0);
        defer m.allocator.free(widths);
        var sum: u64 = 0;
        for (widths, 0..) |w, i| {
            if (w <= 0 or @as(u64, @intCast(w)) % b != 0) return error.InvalidMetadata;
            for (widths[0..i]) |earlier| if (w == earlier) return error.InvalidMetadata;
            sum = try add(sum, @intCast(w));
        }
        if (explicit) {
            const signs = try (try m.required("prism.hadamard.sign_values")).ints32(m.allocator);
            defer m.allocator.free(signs);
            if (signs.len != sum) return error.InvalidMetadata;
            for (signs) |s| if (s != 1 and s != -1) return error.InvalidMetadata;
        } else if (m.get("prism.hadamard.sign_widths") != null or m.get("prism.hadamard.sign_values") != null) {
            return error.InvalidMetadata;
        }
        for (names) |name| {
            if ((try name_set.getOrPut(name)).found_existing) return error.InvalidMetadata;
            const t = m.tensor(name) orelse return error.InvalidMetadata;
            if (t.dims[0] % b != 0) return error.InvalidMetadata;
            if (explicit) {
                var found = false;
                for (widths) |w| if (@as(u64, @intCast(w)) == t.dims[0]) { found = true; };
                if (!found) return error.InvalidMetadata;
            }
        }
        for ([_][]const u8{ "prism.hadamard.gdn_v_grouped", "prism.hadamard.tied_output" }) |key| {
            if (m.get(key)) |entry| if (try entry.boolean()) return error.UnsupportedGyroMetadata;
        }
        if (m.get("prism.hadamard.inverse_weight_names")) |entry| {
            const inverse_names = try entry.strings(m.allocator);
            defer m.allocator.free(inverse_names);
            if (inverse_names.len != 0) return error.UnsupportedGyroMetadata;
        }
    }
};

fn add(a: u64, b: u64) ParseError!u64 {
    return std.math.add(u64, a, b) catch error.IntegerOverflow;
}

fn mul(a: u64, b: u64) ParseError!u64 {
    return std.math.mul(u64, a, b) catch error.IntegerOverflow;
}

fn lessOffset(_: void, a: Tensor, b: Tensor) bool { return a.file_offset < b.file_offset; }

/// The returned strings borrow prefix. file_size is the full file size, not the prefix size.
pub fn parse(allocator: Allocator, prefix: []const u8, file_size: u64) ParseError!Model {
    var c = Cursor{ .bytes = prefix };
    if (!std.mem.eql(u8, try c.take(4), "GGUF")) return error.BadMagic;
    const version = try c.int(u32);
    if (version != 3) return error.UnsupportedVersion;
    const tensor_count = try c.int(u64);
    const metadata_count = try c.int(u64);
    if (tensor_count > 1000000 or metadata_count > 1000000) return error.LimitExceeded;
    // A metadata entry needs a one-byte key, its type, and at least one value byte.
    if (metadata_count > (prefix.len - c.pos) / 14) return error.UnexpectedEnd;
    const metadata = try allocator.alloc(Entry, @intCast(metadata_count));
    errdefer allocator.free(metadata);
    var keys = std.StringHashMap(void).init(allocator);
    defer keys.deinit();
    var alignment: u32 = 32;
    for (metadata) |*entry| {
        const key = try c.string();
        if (key.len == 0 or key.len > 65535) return error.InvalidMetadata;
        if ((try keys.getOrPut(key)).found_existing) return error.DuplicateKey;
        const kind = try c.tag();
        const start = c.pos;
        try c.skip(kind, 0);
        entry.* = .{ .key = key, .kind = kind, .raw = prefix[start..c.pos] };
        if (std.mem.eql(u8, key, "general.alignment")) {
            if (kind != .uint32) return error.InvalidAlignment;
            alignment = @intCast(try entry.unsigned());
        }
    }
    if (alignment == 0 or alignment > 65536 or !std.math.isPowerOfTwo(alignment)) return error.InvalidAlignment;
    if (tensor_count > (prefix.len - c.pos) / 33) return error.UnexpectedEnd;
    const tensors = try allocator.alloc(Tensor, @intCast(tensor_count));
    errdefer allocator.free(tensors);
    keys.clearRetainingCapacity();
    for (tensors) |*t| {
        const name = try c.string();
        if (name.len == 0 or name.len > 4096) return error.InvalidMetadata;
        if ((try keys.getOrPut(name)).found_existing) return error.DuplicateTensor;
        const rank = try c.int(u32);
        if (rank == 0 or rank > 4) return error.InvalidDimensions;
        var dims = [4]u64{ 1, 1, 1, 1 };
        for (dims[0..rank]) |*d| {
            d.* = try c.int(u64);
            if (d.* == 0) return error.InvalidDimensions;
        }
        const type_id = try c.int(u32);
        const offset = try c.int(u64);
        if (offset % alignment != 0) return error.InvalidAlignment;
        var byte_length: ?u64 = null;
        if (layout(type_id)) |l| {
            if (dims[0] % l.block_weights != 0) return error.InvalidDimensions;
            var n = try mul(dims[0] / l.block_weights, l.block_bytes);
            for (dims[1..rank]) |d| n = try mul(n, d);
            byte_length = n;
        }
        t.* = .{ .name = name, .rank = @intCast(rank), .dims = dims, .type_id = type_id,
            .relative_offset = offset, .byte_length = byte_length };
    }
    const mask: u64 = alignment - 1;
    const data_offset = (try add(c.pos, mask)) & ~mask;
    if (data_offset > max_header_bytes) return error.LimitExceeded;
    if (data_offset > file_size) return error.InvalidTensorRange;
    if (data_offset > prefix.len) return error.UnexpectedEnd;
    for (tensors) |*t| {
        t.file_offset = try add(data_offset, t.relative_offset);
        if (t.file_offset >= file_size) return error.InvalidTensorRange;
        if (t.byte_length) |n| if (try add(t.file_offset, n) > file_size) return error.InvalidTensorRange;
    }
    std.sort.heap(Tensor, tensors, {}, lessOffset);
    for (tensors, 0..) |t, i| {
        if (i + 1 == tensors.len) break;
        if (tensors[i + 1].file_offset == t.file_offset) return error.OverlappingTensors;
        if (t.byte_length) |n| if (try add(t.file_offset, n) > tensors[i + 1].file_offset)
            return error.OverlappingTensors;
    }
    return .{ .allocator = allocator, .version = version, .alignment = alignment,
        .data_offset = data_offset, .file_size = file_size, .metadata = metadata, .tensors = tensors };
}

const Fixture = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn int(f: *Fixture, comptime T: type, value: T) !void {
        var bytes: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &bytes, value, .little);
        try f.bytes.appendSlice(std.testing.allocator, &bytes);
    }
    fn str(f: *Fixture, value: []const u8) !void {
        try f.int(u64, value.len);
        try f.bytes.appendSlice(std.testing.allocator, value);
    }
    fn header(f: *Fixture, nt: u64, nk: u64) !void {
        try f.bytes.appendSlice(std.testing.allocator, "GGUF");
        try f.int(u32, 3); try f.int(u64, nt); try f.int(u64, nk);
    }
    fn tensor(f: *Fixture, name: []const u8, dims: []const u64, typ: u32, offset: u64) !void {
        try f.str(name); try f.int(u32, @intCast(dims.len));
        for (dims) |d| try f.int(u64, d);
        try f.int(u32, typ); try f.int(u64, offset);
    }
    fn pad(f: *Fixture) !void {
        while (f.bytes.items.len % 32 != 0) try f.bytes.append(std.testing.allocator, 0);
    }
    fn deinit(f: *Fixture) void { f.bytes.deinit(std.testing.allocator); }
};

test "GGUF expert slices use block bytes and absolute file offsets" {
    var f = Fixture{}; defer f.deinit();
    try f.header(1, 0);
    try f.tensor("blk.0.ffn_down_exps.weight", &.{ 640, 2560, 256 }, 144, 0);
    try f.pad();
    const payload: u64 = 435200 * 256;
    var m = try parse(std.testing.allocator, f.bytes.items, f.bytes.items.len + payload);
    defer m.deinit();
    const r = try m.tensors[0].expertRange(255);
    try std.testing.expectEqual(@as(u64, 435200), r.bytes);
    try std.testing.expectEqual(m.data_offset + 435200 * 255, r.file_offset);
    try std.testing.expectError(error.InvalidExpert, m.tensors[0].expertRange(256));
    for (0..f.bytes.items.len) |end| {
        try std.testing.expectError(error.UnexpectedEnd, parse(std.testing.allocator, f.bytes.items[0..end], m.file_size));
    }
}

test "GGUF detects overlapping ranges and arithmetic overflow" {
    var f = Fixture{}; defer f.deinit();
    try f.header(2, 0);
    try f.tensor("a", &.{32}, 0, 0);
    try f.tensor("b", &.{32}, 0, 32);
    try f.pad();
    try std.testing.expectError(error.OverlappingTensors, parse(std.testing.allocator, f.bytes.items, 4096));
    f.bytes.clearRetainingCapacity();
    try f.header(1, 0);
    try f.tensor("a", &.{ std.math.maxInt(u64), 2 }, 0, 0);
    try f.pad();
    try std.testing.expectError(error.IntegerOverflow, parse(std.testing.allocator, f.bytes.items, std.math.maxInt(u64)));
}

test "GGUF rejects duplicate keys and invalid boolean values" {
    var f = Fixture{}; defer f.deinit();
    try f.header(0, 2);
    for (0..2) |_| { try f.str("a"); try f.int(u32, 4); try f.int(u32, 1); }
    try f.pad();
    try std.testing.expectError(error.DuplicateKey, parse(std.testing.allocator, f.bytes.items, 4096));
    f.bytes.clearRetainingCapacity();
    try f.header(0, 1); try f.str("a"); try f.int(u32, 7); try f.int(u8, 2); try f.pad();
    try std.testing.expectError(error.InvalidBoolean, parse(std.testing.allocator, f.bytes.items, 4096));
}

test "GGUF preserves unsupported tensor types without inventing sizes" {
    var f = Fixture{}; defer f.deinit();
    try f.header(1, 0); try f.tensor("unknown", &.{ 128, 2, 4 }, 999, 0); try f.pad();
    var m = try parse(std.testing.allocator, f.bytes.items, 4096); defer m.deinit();
    try std.testing.expectEqual(@as(?u64, null), m.tensors[0].byte_length);
    try std.testing.expectError(error.UnsupportedTensorType, m.tensors[0].expertRange(0));
}
