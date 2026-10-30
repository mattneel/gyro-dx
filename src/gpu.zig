//! Thin Zig wrapper over the native bindless compute and DirectStorage bridge.
const std = @import("std");
const c = @import("native_abi.zig");

pub const timeout_ms: u32 = 60000;

pub fn check(result: c.gd_result) !void {
    if (result < 0) {
        std.log.err("native HRESULT 0x{x:0>8}", .{@as(u32, @bitCast(result))});
        std.debug.dumpCurrentStackTrace(.{});
        return error.NativeBackendFailed;
    }
}

pub const Buffer = struct {
    handle: *c.gd_buffer,
    srv: u32,
    uav: u32,
    bytes: u64,
    mapped: ?[*]u8,

    pub fn floats(b: Buffer) []f32 {
        const p: [*]f32 = @ptrCast(@alignCast(b.mapped.?));
        return p[0..@intCast(b.bytes / 4)];
    }
    pub fn words(b: Buffer) []u32 {
        const p: [*]u32 = @ptrCast(@alignCast(b.mapped.?));
        return p[0..@intCast(b.bytes / 4)];
    }
};

pub const Kind = enum(u32) { device = c.resource_device, host = c.resource_host };

pub const Device = struct {
    ctx: *c.gd_context,
    caps: c.gd_caps,
    allocated_device_bytes: u64 = 0,
    allocated_host_bytes: u64 = 0,
    next_storage: u64 = 0,

    pub fn init(adapter: u32, debug: bool) !Device {
        var context: ?*c.gd_context = null;
        const opts = c.gd_options{ .adapter_index = adapter, .enable_debug = @intFromBool(debug), .require_linalg = 0,
            .staging_bytes = 32 * 1024 * 1024, .queue_capacity = 1024 };
        try check(c.gd_create(&opts, &context));
        var d = Device{ .ctx = context.?, .caps = undefined };
        try check(c.gd_get_caps(d.ctx, &d.caps));
        if (d.caps.shader_model < 0x66) {
            _ = c.gd_destroy(d.ctx, timeout_ms);
            return error.ShaderModel66Required;
        }
        return d;
    }

    pub fn deinit(d: *Device) void {
        check(c.gd_destroy(d.ctx, timeout_ms)) catch {};
    }

    pub fn adapterName(d: *const Device) []const u8 {
        return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&d.caps.adapter_name)), 0);
    }

    pub fn budget(d: *Device) !struct { budget: u64, usage: u64 } {
        try check(c.gd_get_caps(d.ctx, &d.caps));
        return .{ .budget = d.caps.local_budget_bytes, .usage = d.caps.local_usage_bytes };
    }

    pub fn buffer(d: *Device, bytes: u64, kind: Kind) !Buffer {
        var handle: ?*c.gd_buffer = null;
        var info: c.gd_buffer_info = undefined;
        try check(c.gd_create_resource(d.ctx, @max(bytes, 256), @intFromEnum(kind), &handle, &info));
        if (kind == .device) d.allocated_device_bytes += info.allocated_bytes else d.allocated_host_bytes += info.allocated_bytes;
        return .{ .handle = handle.?, .srv = info.srv_index, .uav = info.uav_index, .bytes = @max(bytes, 256),
            .mapped = if (info.mapped) |m| @ptrCast(m) else null };
    }

    pub fn release(d: *Device, b: Buffer) void {
        check(c.gd_release_buffer(d.ctx, b.handle, timeout_ms)) catch {};
    }

    pub fn pipeline(d: *Device, dxil: []const u8) !*c.gd_pipeline {
        var p: ?*c.gd_pipeline = null;
        try check(c.gd_create_pipeline(d.ctx, dxil.ptr, dxil.len, &p));
        return p.?;
    }

    pub fn begin(d: *Device) !void {
        try check(c.gd_begin(d.ctx, timeout_ms));
    }

    pub fn dispatch(d: *Device, p: *c.gd_pipeline, constants: []const u32, x: u32, y: u32, z: u32) !void {
        try check(c.gd_dispatch(d.ctx, p, constants.ptr, @intCast(constants.len), x, y, z));
    }

    pub fn barrier(d: *Device) !void {
        try check(c.gd_barrier(d.ctx));
    }

    pub fn timestamp(d: *Device) !u32 {
        var index: u32 = undefined;
        try check(c.gd_timestamp(d.ctx, &index));
        return index;
    }

    pub fn submit(d: *Device) !u64 {
        var ticket: u64 = undefined;
        try check(c.gd_submit(d.ctx, &ticket));
        return ticket;
    }

    pub fn wait(d: *Device, ticket: u64) !void {
        try check(c.gd_wait_compute(d.ctx, ticket, timeout_ms));
    }

    pub fn timestamps(d: *Device, out: []u64) !u64 {
        var frequency: u64 = undefined;
        try check(c.gd_read_timestamps(d.ctx, @intCast(out.len), out.ptr, &frequency));
        return frequency;
    }

    pub fn openFile(d: *Device, path: [:0]const u8) !*c.gd_file {
        var f: ?*c.gd_file = null;
        try check(c.gd_open_file(d.ctx, path.ptr, &f));
        return f.?;
    }

    /// Enqueues raw ranges; returns the storage ticket. compute_ticket orders the writes after
    /// that compute submission.
    pub fn readRanges(d: *Device, file: *c.gd_file, b: Buffer, ranges: []const c.gd_range, compute_ticket: u64) !u64 {
        var ticket: u64 = undefined;
        try check(c.gd_read_ranges_after(d.ctx, file, b.handle, ranges.ptr, @intCast(ranges.len), compute_ticket, &ticket));
        d.next_storage = ticket;
        return ticket;
    }

    /// Waits on the CPU for a storage ticket and verifies its status.
    pub fn waitStorage(d: *Device, ticket: u64) !void {
        // A completed fence can precede its status result; the bridge then reports E_PENDING.
        const e_pending: c.gd_result = @bitCast(@as(u32, 0x8000000A));
        var attempts: u32 = 0;
        while (true) : (attempts += 1) {
            const result = c.gd_wait_storage(d.ctx, ticket, timeout_ms);
            if (result != e_pending or attempts >= 100_000) return check(result);
            std.atomic.spinLoopHint();
        }
    }

    /// Last storage ticket whose fence completed.
    pub fn completedStorage(d: *Device) u64 {
        var storage: u64 = 0;
        var compute: u64 = 0;
        _ = c.gd_poll(d.ctx, &storage, &compute);
        return storage;
    }

    pub fn queueWaitStorage(d: *Device, ticket: u64) !void {
        try check(c.gd_queue_wait_storage(d.ctx, ticket));
    }
};
