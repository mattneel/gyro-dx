//! Gyro trellis decoder and block Hadamard transform.
//! The normative format is AgentionAI SPEC.md version 1.1.
//! Upstream revision: 587a5269848eb07996c9c1433f9ad638604eb4d8.
//! The codebook and decoder formulas derive from the Apache-2.0 reference.
//! Copyright 2026 AgentionAI (Laurent Zuijdwijk). See third_party/agention-gyro/NOTICE.
const std = @import("std");

pub const block_weights: u16 = 128;
pub const state_bits: u32 = 15;
pub const codebook_bytes = @embedFile("gyro_codebook.f16");

pub const Type = enum(u32) {
    tq2_t = 144,
    tqk6 = 145,
    tqk7 = 146,
    tqp5 = 147,
    tqp6 = 148,
    tqp7 = 149,
    tqk5 = 150,
    tqk4 = 151,
};

pub const TypeInfo = struct {
    id: u32,
    name: []const u8,
    k: u8,
    steps: u8,
    block_bytes: u16,
    block_weights: u16 = 128,
};

pub const types = [_]TypeInfo{
    .{ .id = 144, .name = "TQ2_T", .k = 8, .steps = 32, .block_bytes = 34 },
    .{ .id = 145, .name = "TQK6", .k = 6, .steps = 32, .block_bytes = 26 },
    .{ .id = 146, .name = "TQK7", .k = 7, .steps = 32, .block_bytes = 30 },
    .{ .id = 147, .name = "TQP5", .k = 5, .steps = 64, .block_bytes = 42 },
    .{ .id = 148, .name = "TQP6", .k = 6, .steps = 64, .block_bytes = 50 },
    .{ .id = 149, .name = "TQP7", .k = 7, .steps = 64, .block_bytes = 58 },
    .{ .id = 150, .name = "TQK5", .k = 5, .steps = 32, .block_bytes = 22 },
    .{ .id = 151, .name = "TQK4", .k = 4, .steps = 32, .block_bytes = 18 },
};

pub const Error = error{
    UnsupportedType,
    MissingStateBits,
    UnsupportedStateBits,
    InvalidRowLength,
    InvalidSourceLength,
    IntegerOverflow,
    InvalidBlockSize,
    InvalidSignsLength,
    InvalidSign,
};

pub fn typeInfo(id: u32) ?TypeInfo {
    for (types) |info| if (info.id == id) return info;
    return null;
}

pub fn validateStateBits(value: ?u32) Error!void {
    const value_present = value orelse return error.MissingStateBits;
    if (value_present != state_bits) return error.UnsupportedStateBits;
}

pub fn blockBytes(id: u32) Error!u16 {
    const info = typeInfo(id) orelse return error.UnsupportedType;
    return info.block_bytes;
}

pub fn rowBytes(id: u32, weight_count: u64) Error!u64 {
    const info = typeInfo(id) orelse return error.UnsupportedType;
    if (weight_count % block_weights != 0) return error.InvalidRowLength;
    const result = @mulWithOverflow(weight_count / block_weights, @as(u64, info.block_bytes));
    if (result[1] != 0) return error.IntegerOverflow;
    return result[0];
}

/// Decode packed blocks into their stored basis.
/// The source length must equal the encoded length of the destination.
/// This function does not apply the per-tensor Hadamard transform.
pub fn decodeRow(id: u32, declared_state_bits: ?u32, source: []const u8, destination: []f32) Error!void {
    @setFloatMode(.strict);
    try validateStateBits(declared_state_bits);
    const info = typeInfo(id) orelse return error.UnsupportedType;
    const expected = try rowBytes(id, destination.len);
    if (source.len != expected) return error.InvalidSourceLength;
    var output_index: usize = 0;
    var offset: usize = 0;
    while (offset < source.len) : (offset += info.block_bytes) {
        const block = source[offset..][0..info.block_bytes];
        const scale = fp16ToFp32(readU16(block[0..2]));
        const stream = block[2..];
        for (0..info.steps) |step| {
            const state = trellisState(info, stream, step);
            const hash = (state & 0x7fff) *% @as(u32, 0x9e3779b1);
            const pair_count: usize = if (info.steps == 64) 1 else 2;
            for (0..pair_count) |pair| {
                const index: usize = if (pair == 0) hash >> 21 else (hash >> 10) & 2047;
                destination[output_index] = scale * codebook(index * 2);
                destination[output_index + 1] = scale * codebook(index * 2 + 1);
                output_index += 2;
            }
        }
    }
}

fn trellisState(info: TypeInfo, stream: []const u8, step: usize) u32 {
    if (info.id == @intFromEnum(Type.tq2_t)) {
        return (@as(u32, stream[(step + 31) % 32]) << 8) | stream[step];
    }
    const offset: usize = (@as(usize, info.steps) - 1 - step) * info.k;
    const byte = offset >> 3;
    const word = @as(u32, stream[byte]) |
        (@as(u32, stream[(byte + 1) % stream.len]) << 8) |
        (@as(u32, stream[(byte + 2) % stream.len]) << 16);
    return (word >> @as(u5, @intCast(offset & 7))) & 0xffff;
}

fn readU16(bytes: []const u8) u16 {
    return @as(u16, bytes[0]) | (@as(u16, bytes[1]) << 8);
}

fn codebook(index: usize) f32 {
    return fp16ToFp32(readU16(codebook_bytes[index * 2 ..][0..2]));
}

/// Convert every binary16 bit pattern without hardware subnormal assumptions.
pub fn fp16ToFp32(half: u16) f32 {
    const sign = @as(u32, half & 0x8000) << 16;
    const exponent = (half >> 10) & 31;
    var mantissa: u32 = half & 1023;
    if (exponent == 31) return @bitCast(sign | 0x7f800000 | (mantissa << 13));
    if (exponent != 0) return @bitCast(sign | ((@as(u32, exponent) + 112) << 23) | (mantissa << 13));
    if (mantissa == 0) return @bitCast(sign);
    var output_exponent: u32 = 113;
    while ((mantissa & 1024) == 0) {
        mantissa <<= 1;
        output_exponent -= 1;
    }
    return @bitCast(sign | (output_exponent << 23) | ((mantissa & 1023) << 13));
}

fn validateTransform(values: []f32, block_size: usize, signs: ?[]const i8) Error!void {
    if (block_size == 0 or !std.math.isPowerOfTwo(block_size)) return error.InvalidBlockSize;
    if (values.len % block_size != 0) return error.InvalidRowLength;
    if (signs) |sign_values| {
        if (sign_values.len != values.len) return error.InvalidSignsLength;
        for (sign_values) |sign| if (sign != 1 and sign != -1) return error.InvalidSign;
    }
}

fn hadamard(values: []f32, block_size: usize) void {
    @setFloatMode(.strict);
    const normalization: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(block_size)));
    var block_start: usize = 0;
    while (block_start < values.len) : (block_start += block_size) {
        const block = values[block_start..][0..block_size];
        var width: usize = 1;
        while (width < block_size) : (width *= 2) {
            var start: usize = 0;
            while (start < block_size) : (start += 2 * width) {
                for (0..width) |j| {
                    const a = block[start + j];
                    const b = block[start + j + width];
                    block[start + j] = a + b;
                    block[start + j + width] = a - b;
                }
            }
        }
        for (block) |*value| value.* *= normalization;
    }
}

/// Apply T(x) = H_b (signs * x) along one input-feature row.
/// The same transform applies to activations and stored weight rows.
/// A null sign vector represents identity signs.
pub fn rotate(values: []f32, block_size: usize, signs: ?[]const i8) Error!void {
    @setFloatMode(.strict);
    try validateTransform(values, block_size, signs);
    if (signs) |sign_values| {
        for (values, sign_values) |*value, sign| value.* *= @floatFromInt(sign);
    }
    hadamard(values, block_size);
}

/// Apply T^-1(x) = signs * H_b(x) along one input-feature row.
/// Use this function after decodeRow for weights in the original basis.
pub fn unrotate(values: []f32, block_size: usize, signs: ?[]const i8) Error!void {
    @setFloatMode(.strict);
    try validateTransform(values, block_size, signs);
    hadamard(values, block_size);
    if (signs) |sign_values| {
        for (values, sign_values) |*value, sign| value.* *= @floatFromInt(sign);
    }
}

test "normative codebook hash" {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(codebook_bytes, &digest, .{});
    const expected = [_]u8{ 0xee, 0x48, 0x1f, 0x15, 0x1d, 0x89, 0x47, 0xce, 0x8b, 0x08, 0x81, 0x7f, 0xdf, 0x89, 0xd0, 0xa5, 0x6d, 0x64, 0x7b, 0x64, 0x35, 0xce, 0xb3, 0xdf, 0x0f, 0x07, 0x9e, 0xa7, 0x48, 0xe0, 0xe5, 0x88 };
    try std.testing.expectEqualSlices(u8, &expected, &digest);
}

test "type ids and source lengths fail closed" {
    try std.testing.expect(typeInfo(35) == null);
    try std.testing.expectError(error.UnsupportedType, rowBytes(35, 128));
    try std.testing.expectError(error.InvalidRowLength, rowBytes(144, 127));
    try std.testing.expectEqual(@as(u64, 435200), try rowBytes(144, 640 * 2560));
    var destination: [128]f32 = undefined;
    const source: [34]u8 = @splat(0);
    try std.testing.expectError(error.MissingStateBits, decodeRow(144, null, &source, &destination));
    try std.testing.expectError(error.UnsupportedStateBits, decodeRow(144, 16, &source, &destination));
    try std.testing.expectError(error.InvalidSourceLength, decodeRow(144, 15, source[0..33], &destination));
}

test "binary16 edge values" {
    try std.testing.expectEqual(@as(u32, 0x00000000), @as(u32, @bitCast(fp16ToFp32(0))));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(fp16ToFp32(0x8000))));
    try std.testing.expectEqual(@as(u32, 0x33800000), @as(u32, @bitCast(fp16ToFp32(1))));
    try std.testing.expectEqual(@as(u32, 0x38800000), @as(u32, @bitCast(fp16ToFp32(0x0400))));
    try std.testing.expectEqual(@as(f32, 65504), fp16ToFp32(0x7bff));
    try std.testing.expect(std.math.isInf(fp16ToFp32(0x7c00)));
    try std.testing.expect(std.math.isNan(fp16ToFp32(0x7e00)));
}

test "rotation order and invalid signs" {
    const signs = [_]i8{ 1, -1, 1, 1 };
    var values = [_]f32{ 1, 2, 3, 4 };
    try rotate(&values, 4, &signs);
    try std.testing.expectEqualSlices(f32, &.{ 3, 1, -4, 2 }, &values);
    try unrotate(&values, 4, &signs);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &values);
    try std.testing.expectError(error.InvalidBlockSize, rotate(&values, 3, null));
    try std.testing.expectError(error.InvalidSignsLength, rotate(&values, 4, signs[0..3]));
    const invalid = [_]i8{ 1, -1, 0, 1 };
    try std.testing.expectError(error.InvalidSign, rotate(&values, 4, &invalid));
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &values);
}
