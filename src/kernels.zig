//! CPU reference kernels for the Qwen4Exp graph.
//! Weights stay in their GGUF encoding. Activations are float32.
const std = @import("std");
const gyro = @import("gyro.zig");
const Pool = @import("pool.zig").Pool;

pub const type_f32: u32 = 0;
pub const type_f16: u32 = 1;
pub const type_q8_0: u32 = 8;
pub const type_bf16: u32 = 30;

pub const max_cols = 16384;

/// Normative codebook expanded to float32 at compile time: 2,048 pairs.
pub const codebook: [4096]f32 = blk: {
    @setEvalBranchQuota(100000);
    var table: [4096]f32 = undefined;
    for (0..4096) |i| {
        const half = @as(u16, gyro.codebook_bytes[i * 2]) | (@as(u16, gyro.codebook_bytes[i * 2 + 1]) << 8);
        table[i] = gyro.fp16ToFp32(half);
    }
    break :blk table;
};

pub inline fn f16ToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

pub inline fn f32ToF16(value: f32) u16 {
    return @bitCast(@as(f16, @floatCast(value)));
}

pub inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// A row-major matrix with `rows` outputs of `cols` inputs each, in its stored encoding.
pub const Matrix = struct {
    type_id: u32,
    cols: usize,
    rows: usize,
    row_bytes: usize,
    data: []const u8,

    pub fn row(m: Matrix, r: usize) []const u8 {
        return m.data[r * m.row_bytes ..][0..m.row_bytes];
    }

    /// Rows [start, start + count) as a matrix.
    pub fn slice(m: Matrix, start: usize, count: usize) Matrix {
        return .{ .type_id = m.type_id, .cols = m.cols, .rows = count, .row_bytes = m.row_bytes,
            .data = m.data[start * m.row_bytes ..][0 .. count * m.row_bytes] };
    }
};

pub fn rowBytes(type_id: u32, cols: usize) !usize {
    return switch (type_id) {
        type_f32 => cols * 4,
        type_f16, type_bf16 => cols * 2,
        type_q8_0 => if (cols % 32 != 0) error.InvalidDimensions else cols / 32 * 34,
        else => blk: {
            const info = gyro.typeInfo(type_id) orelse return error.UnsupportedTensorType;
            if (cols % 128 != 0) return error.InvalidDimensions;
            break :blk cols / 128 * info.block_bytes;
        },
    };
}

const V8 = @Vector(8, f32);
const V32 = @Vector(32, f32);

fn dotF32(row: []const u8, x: []const f32) f32 {
    @setFloatMode(.optimized);
    var acc: V8 = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const w: V8 = @bitCast(row[i * 4 ..][0..32].*);
        const v: V8 = x[i..][0..8].*;
        acc += w * v;
    }
    var sum = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) sum += @as(f32, @bitCast(row[i * 4 ..][0..4].*)) * x[i];
    return sum;
}

fn dotHalf(comptime bf: bool, row: []const u8, x: []const f32) f32 {
    @setFloatMode(.optimized);
    var acc: V8 = @splat(0);
    var i: usize = 0;
    while (i + 8 <= x.len) : (i += 8) {
        const bits: @Vector(8, u16) = @bitCast(row[i * 2 ..][0..16].*);
        const w: V8 = if (bf)
            @bitCast(@as(@Vector(8, u32), bits) << @splat(16))
        else
            @floatCast(@as(@Vector(8, f16), @bitCast(bits)));
        const v: V8 = x[i..][0..8].*;
        acc += w * v;
    }
    var sum = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) {
        const bits = std.mem.readInt(u16, row[i * 2 ..][0..2], .little);
        sum += (if (bf) bf16ToF32(bits) else f16ToF32(bits)) * x[i];
    }
    return sum;
}

fn dotQ8(row: []const u8, x: []const f32) f32 {
    @setFloatMode(.optimized);
    var total: f32 = 0;
    const blocks = x.len / 32;
    for (0..blocks) |b| {
        const block = row[b * 34 ..][0..34];
        const d = f16ToF32(std.mem.readInt(u16, block[0..2], .little));
        const q: @Vector(32, i8) = @bitCast(block[2..34].*);
        const w: V32 = @floatFromInt(q);
        const v: V32 = x[b * 32 ..][0..32].*;
        total += d * @reduce(.Add, w * v);
    }
    return total;
}

inline fn tq2State(stream: *const [32]u8, step: usize) u32 {
    return (@as(u32, stream[(step + 31) & 31]) << 8) | stream[step];
}

fn genericState(info: gyro.TypeInfo, stream: []const u8, step: usize) u32 {
    const offset: usize = (@as(usize, info.steps) - 1 - step) * info.k;
    const byte = offset >> 3;
    const word = @as(u32, stream[byte]) |
        (@as(u32, stream[(byte + 1) % stream.len]) << 8) |
        (@as(u32, stream[(byte + 2) % stream.len]) << 16);
    return (word >> @as(u5, @intCast(offset & 7))) & 0xffff;
}

/// Decodes one Gyro block of 128 values in the stored basis.
pub fn decodeGyroBlock(info: gyro.TypeInfo, block: []const u8, out: *[128]f32) void {
    const scale = f16ToF32(std.mem.readInt(u16, block[0..2], .little));
    const stream = block[2..info.block_bytes];
    if (info.steps == 32) {
        for (0..32) |step| {
            const state = if (info.id == 144) tq2State(stream[0..32], step) else genericState(info, stream, step);
            const h = (state & 0x7fff) *% @as(u32, 0x9e3779b1);
            const a = (h >> 21) * 2;
            const b = ((h >> 10) & 2047) * 2;
            out[step * 4 + 0] = scale * codebook[a];
            out[step * 4 + 1] = scale * codebook[a + 1];
            out[step * 4 + 2] = scale * codebook[b];
            out[step * 4 + 3] = scale * codebook[b + 1];
        }
    } else {
        for (0..64) |step| {
            const state = genericState(info, stream, step);
            const h = (state & 0x7fff) *% @as(u32, 0x9e3779b1);
            const a = (h >> 21) * 2;
            out[step * 2 + 0] = scale * codebook[a];
            out[step * 2 + 1] = scale * codebook[a + 1];
        }
    }
}

fn dotTq2(row: []const u8, x: []const f32) f32 {
    @setFloatMode(.optimized);
    var total: f32 = 0;
    const blocks = x.len / 128;
    for (0..blocks) |bi| {
        const block = row[bi * 34 ..][0..34];
        const scale = f16ToF32(std.mem.readInt(u16, block[0..2], .little));
        const stream = block[2..34];
        const xb = x[bi * 128 ..][0..128];
        var acc: @Vector(4, f32) = @splat(0);
        inline for (0..32) |step| {
            const state = tq2State(stream, step);
            const h = (state & 0x7fff) *% @as(u32, 0x9e3779b1);
            const a = (h >> 21) * 2;
            const b = ((h >> 10) & 2047) * 2;
            const w: @Vector(4, f32) = .{ codebook[a], codebook[a + 1], codebook[b], codebook[b + 1] };
            const v: @Vector(4, f32) = xb[step * 4 ..][0..4].*;
            acc += w * v;
        }
        total += scale * @reduce(.Add, acc);
    }
    return total;
}

fn dotGyroGeneric(info: gyro.TypeInfo, row: []const u8, x: []const f32) f32 {
    @setFloatMode(.optimized);
    var total: f32 = 0;
    var tmp: [128]f32 = undefined;
    const blocks = x.len / 128;
    for (0..blocks) |bi| {
        decodeGyroBlock(info, row[bi * info.block_bytes ..][0..info.block_bytes], &tmp);
        const xb = x[bi * 128 ..][0..128];
        var acc: V8 = @splat(0);
        var i: usize = 0;
        while (i < 128) : (i += 8) acc += @as(V8, tmp[i..][0..8].*) * @as(V8, xb[i..][0..8].*);
        total += @reduce(.Add, acc);
    }
    return total;
}

pub fn dotRow(type_id: u32, row: []const u8, x: []const f32) f32 {
    return switch (type_id) {
        type_f32 => dotF32(row, x),
        type_f16 => dotHalf(false, row, x),
        type_bf16 => dotHalf(true, row, x),
        type_q8_0 => dotQ8(row, x),
        144 => dotTq2(row, x),
        else => dotGyroGeneric(gyro.typeInfo(type_id).?, row, x),
    };
}

/// Decodes one row into float32 values in the stored basis.
pub fn decodeRow(type_id: u32, row: []const u8, out: []f32) void {
    switch (type_id) {
        type_f32 => for (out, 0..) |*o, i| {
            o.* = @bitCast(row[i * 4 ..][0..4].*);
        },
        type_f16 => for (out, 0..) |*o, i| {
            o.* = f16ToF32(std.mem.readInt(u16, row[i * 2 ..][0..2], .little));
        },
        type_bf16 => for (out, 0..) |*o, i| {
            o.* = bf16ToF32(std.mem.readInt(u16, row[i * 2 ..][0..2], .little));
        },
        type_q8_0 => for (0..out.len / 32) |b| {
            const block = row[b * 34 ..][0..34];
            const d = f16ToF32(std.mem.readInt(u16, block[0..2], .little));
            for (0..32) |j| out[b * 32 + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + j]))));
        },
        else => {
            const info = gyro.typeInfo(type_id).?;
            for (0..out.len / 128) |b|
                decodeGyroBlock(info, row[b * info.block_bytes ..][0..info.block_bytes], out[b * 128 ..][0..128]);
        },
    }
}

const MatvecCtx = struct { m: Matrix, x: []const f32, y: []f32, n: usize };

fn matvecRange(c: *const MatvecCtx, start: usize, end: usize) void {
    if (c.n == 1) {
        for (start..end) |r| c.y[r] = dotRow(c.m.type_id, c.m.row(r), c.x);
        return;
    }
    // Several activation rows: decode a weight row once, then reuse it.
    var buffer: [max_cols]f32 align(64) = undefined;
    const w = buffer[0..c.m.cols];
    for (start..end) |r| {
        decodeRow(c.m.type_id, c.m.row(r), w);
        for (0..c.n) |t| c.y[t * c.m.rows + r] = dotDense(w, c.x[t * c.m.cols ..][0..c.m.cols]);
    }
}

pub fn dotDense(a: []const f32, b: []const f32) f32 {
    @setFloatMode(.optimized);
    var acc: V8 = @splat(0);
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) acc += @as(V8, a[i..][0..8].*) * @as(V8, b[i..][0..8].*);
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// y[t][r] = sum_i W[r][i] x[t][i] for n activation rows. x is n*cols, y is n*rows.
pub fn matmul(pool: *Pool, m: Matrix, x: []const f32, y: []f32, n: usize) void {
    std.debug.assert(x.len >= n * m.cols and y.len >= n * m.rows and m.cols <= max_cols);
    const ctx = MatvecCtx{ .m = m, .x = x, .y = y, .n = n };
    pool.run(m.rows, pool.chunkFor(m.rows, 8), &ctx, matvecRange);
}

pub fn matvec(pool: *Pool, m: Matrix, x: []const f32, y: []f32) void {
    matmul(pool, m, x, y, 1);
}

/// Serial single-row product for small matrices inside an already parallel task.
pub fn matvecSerial(m: Matrix, x: []const f32, y: []f32) void {
    for (0..m.rows) |r| y[r] = dotRow(m.type_id, m.row(r), x);
}

// ---- elementwise and normalization helpers ----

pub fn rmsNorm(out: []f32, x: []const f32, weight: ?[]const f32, eps: f32) void {
    @setFloatMode(.optimized);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    if (weight) |w| {
        for (out, x, w) |*o, v, g| o.* = v * scale * g;
    } else {
        for (out, x) |*o, v| o.* = v * scale;
    }
}

/// Matches build_gdn_l2_norm: rms_norm(x, eps / n) / sqrt(n).
pub fn l2Norm(x: []f32, eps: f32) void {
    @setFloatMode(.optimized);
    const n: f32 = @floatFromInt(x.len);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const scale = (1.0 / @sqrt(ss / n + eps / n)) / @sqrt(n);
    for (x) |*v| v.* *= scale;
}

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Matches ggml_softplus: log(1 + exp(x)), linear above 20.
pub inline fn softplus(x: f32) f32 {
    return if (x > 20.0) x else @log(1.0 + @exp(x));
}

/// In-place normalized block Hadamard transform after a sign multiply: H (signs * x).
pub fn rotate(x: []f32, block: usize, signs: []const f32) void {
    @setFloatMode(.optimized);
    for (x, signs) |*v, s| v.* *= s;
    const norm: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(block)));
    var start: usize = 0;
    while (start < x.len) : (start += block) {
        const b = x[start..][0..block];
        var width: usize = 1;
        while (width < block) : (width *= 2) {
            var s: usize = 0;
            while (s < block) : (s += 2 * width) {
                for (0..width) |j| {
                    const p = b[s + j];
                    const q = b[s + j + width];
                    b[s + j] = p + q;
                    b[s + j + width] = p - q;
                }
            }
        }
        for (b) |*v| v.* *= norm;
    }
}

pub fn softmaxInPlace(x: []f32) void {
    var m: f32 = -std.math.inf(f32);
    for (x) |v| m = @max(m, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - m);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

/// NeoX rotary embedding on the first n_rot dims of one head: pairs (i, i + n_rot/2).
pub fn ropeNeox(head: []f32, n_rot: usize, pos: f32, freq_base: f32) void {
    const half = n_rot / 2;
    const theta_scale = std.math.pow(f32, freq_base, -2.0 / @as(f32, @floatFromInt(n_rot)));
    var theta = pos;
    for (0..half) |i| {
        const c = @cos(theta);
        const s = @sin(theta);
        const a = head[i];
        const b = head[i + half];
        head[i] = a * c - b * s;
        head[i + half] = a * s + b * c;
        theta *= theta_scale;
    }
}

test "q8_0 dot and decode agree" {
    var row: [68]u8 = undefined;
    const scales = [_]f32{ 0.5, -0.25 };
    for (0..2) |b| {
        std.mem.writeInt(u16, row[b * 34 ..][0..2], f32ToF16(scales[b]), .little);
        for (0..32) |j| row[b * 34 + 2 + j] = @bitCast(@as(i8, @intCast(@as(i32, @intCast(j)) - 16 + @as(i32, @intCast(b)))));
    }
    var x: [64]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) - 3.0;
    var dense: [64]f32 = undefined;
    decodeRow(type_q8_0, &row, &dense);
    var expected: f32 = 0;
    for (dense, x) |w, v| expected += w * v;
    try std.testing.expectApproxEqAbs(expected, dotRow(type_q8_0, &row, &x), 1e-4);
    try std.testing.expectEqual(@as(f32, 0.5 * -16.0), dense[0]);
}

test "fused gyro dot matches the strict reference decoder" {
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    inline for (gyro.types) |info| {
        var row: [2 * 58]u8 = undefined;
        const bytes = row[0 .. 2 * @as(usize, info.block_bytes)];
        random.bytes(bytes);
        for (0..2) |b| std.mem.writeInt(u16, bytes[b * info.block_bytes ..][0..2], f32ToF16(0.03125), .little);
        var reference: [256]f32 = undefined;
        try gyro.decodeRow(info.id, 15, bytes, &reference);
        var fast: [256]f32 = undefined;
        decodeRow(info.id, bytes, &fast);
        try std.testing.expectEqualSlices(f32, &reference, &fast);
        var x: [256]f32 = undefined;
        for (&x) |*v| v.* = random.float(f32) - 0.5;
        var expected: f32 = 0;
        for (reference, x) |w, v| expected += w * v;
        try std.testing.expectApproxEqAbs(expected, dotRow(info.id, bytes, &x), 1e-4);
    }
}

test "rotation matches the strict module transform" {
    var a: [256]f32 = undefined;
    var b: [256]f32 = undefined;
    var signs_i8: [256]i8 = undefined;
    var signs: [256]f32 = undefined;
    for (0..256) |i| {
        a[i] = @floatFromInt(@as(i32, @intCast(i % 13)) - 6);
        b[i] = a[i];
        signs_i8[i] = if (i % 3 == 0) -1 else 1;
        signs[i] = @floatFromInt(signs_i8[i]);
    }
    try gyro.rotate(&a, 128, &signs_i8);
    rotate(&b, 128, &signs);
    for (a, b) |p, q| try std.testing.expectApproxEqAbs(p, q, 1e-4);
}
