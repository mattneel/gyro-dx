const std = @import("std");
const gyro = @import("gyro");
const fixture = @embedFile("gyro_vectors.bin");

const Cursor = struct {
    pos: usize = 0,

    fn take(self: *Cursor, count: usize) ![]const u8 {
        if (count > fixture.len - self.pos) return error.TruncatedFixture;
        const result = fixture[self.pos..][0..count];
        self.pos += count;
        return result;
    }

    fn word(self: *Cursor) !u32 {
        const bytes = try self.take(4);
        return std.mem.readInt(u32, bytes[0..4], .little);
    }

    fn expectFloatBits(self: *Cursor, actual: []const f32) !void {
        for (actual) |value| {
            const expected = try self.word();
            try std.testing.expectEqual(expected, @as(u32, @bitCast(value)));
        }
    }
};

test "512 independent C oracle blocks match all 65536 float bits" {
    var cursor: Cursor = .{};
    try std.testing.expectEqualStrings("GYROV1\x00\x00", try cursor.take(8));
    const cases = try cursor.word();
    try std.testing.expectEqual(@as(u32, 512), cases);
    for (0..cases) |_| {
        const type_id = try cursor.word();
        const size = try cursor.word();
        const encoded = try cursor.take(size);
        var decoded: [128]f32 = undefined;
        try gyro.decodeRow(type_id, 15, encoded, &decoded);
        try cursor.expectFloatBits(&decoded);
    }

    const rotation_count = try cursor.word();
    try std.testing.expectEqual(@as(u32, 2), rotation_count);
    for (0..rotation_count) |_| {
        const width = try cursor.word();
        const block = try cursor.word();
        const values = try std.testing.allocator.alloc(f32, width);
        defer std.testing.allocator.free(values);
        const signs = try std.testing.allocator.alloc(i8, width);
        defer std.testing.allocator.free(signs);
        for (values) |*value| value.* = @bitCast(try cursor.word());
        const sign_bytes = try cursor.take(width);
        for (signs, sign_bytes) |*sign, byte| sign.* = @bitCast(byte);
        try gyro.rotate(values, block, signs);
        try cursor.expectFloatBits(values);
        try gyro.unrotate(values, block, signs);
        try cursor.expectFloatBits(values);
    }
    try std.testing.expectEqual(fixture.len, cursor.pos);
}
