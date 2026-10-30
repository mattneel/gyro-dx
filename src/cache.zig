//! A host cache policy for compressed expert payloads.
//! The caller owns the storage queue, resources, and both fence timelines.
//! A completed storage fence proves completion. The status result proves success.
const std = @import("std");

pub const Key = struct {
    layer: u32,
    expert: u32,
    variant: u32 = 0,

    fn eql(a: Key, b: Key) bool {
        return a.layer == b.layer and a.expert == b.expert and a.variant == b.variant;
    }
};

pub const Request = struct { key: Key, bytes: u64 };
pub const LoadFence = struct { value: u64 };
pub const ComputeFence = struct { value: u64 };
pub const Lease = struct { slot: u32, generation: u64 };
pub const State = enum { empty, loading, ready, failed };

pub const Slot = struct {
    state: State = .empty,
    key: Key = .{ .layer = 0, .expert = 0 },
    bytes: u64 = 0,
    generation: u64 = 0,
    pins: u32 = 0,
    last_access: u64 = 0,
    load_fence: LoadFence = .{ .value = 0 },
    compute_fence: ComputeFence = .{ .value = 0 },
};

pub const Error = error{
    InvalidCapacity,
    InvalidRequest,
    OutputLengthMismatch,
    OversizedRequest,
    SizeMismatch,
    CacheFull,
    CounterOverflow,
    FenceNotMonotonic,
    FenceNotComplete,
    UnknownLoadFence,
    StaleLease,
    NotPinned,
    NotReady,
};

pub const Cache = struct {
    allocator: std.mem.Allocator,
    slots: []Slot,
    slot_bytes: u64,
    access_clock: u64 = 0,
    last_load: LoadFence = .{ .value = 0 },
    last_compute: ComputeFence = .{ .value = 0 },
    completed_compute: ComputeFence = .{ .value = 0 },

    pub fn init(allocator: std.mem.Allocator, slot_count: u32, slot_bytes: u64) (Error || std.mem.Allocator.Error)!Cache {
        if (slot_count == 0 or slot_bytes == 0) return error.InvalidCapacity;
        if (slot_bytes > std.math.maxInt(u64) / @as(u64, slot_count)) return error.InvalidCapacity;
        const slots = try allocator.alloc(Slot, slot_count);
        @memset(slots, .{});
        return .{ .allocator = allocator, .slots = slots, .slot_bytes = slot_bytes };
    }

    pub fn deinit(self: *Cache) void {
        self.allocator.free(self.slots);
        self.* = undefined;
    }

    pub fn capacityBytes(self: *const Cache) u64 {
        return self.slot_bytes * @as(u64, @intCast(self.slots.len));
    }

    pub fn payloadBytes(self: *const Cache) u64 {
        var bytes: u64 = 0;
        for (self.slots) |slot| if (slot.state != .empty) {
            bytes += slot.bytes;
        };
        return bytes;
    }

    fn existing(self: *const Cache, key: Key) ?usize {
        for (self.slots, 0..) |slot, i| {
            if ((slot.state == .ready or slot.state == .loading) and slot.key.eql(key)) return i;
        }
        return null;
    }

    fn evictable(self: *const Cache, slot: Slot) bool {
        return slot.state != .loading and slot.pins == 0 and
            slot.compute_fence.value <= self.completed_compute.value and
            slot.generation != std.math.maxInt(u64);
    }

    fn requested(requests: []const Request, key: Key) bool {
        for (requests) |request| if (request.key.eql(key)) return true;
        return false;
    }

    fn first(requests: []const Request, index: usize) bool {
        for (requests[0..index]) |request| if (request.key.eql(requests[index].key)) return false;
        return true;
    }

    fn references(requests: []const Request, key: Key) usize {
        var count: usize = 0;
        for (requests) |request| if (request.key.eql(key)) {
            count += 1;
        };
        return count;
    }

    fn victim(self: *const Cache) usize {
        var selected: ?usize = null;
        for (self.slots, 0..) |slot, i| {
            if (!self.evictable(slot)) continue;
            if (slot.state == .empty) return i;
            if (selected == null or slot.last_access < self.slots[selected.?].last_access) selected = i;
        }
        // Admission preflight proves that a victim exists.
        return selected.?;
    }

    /// Reserve all requests, or leave the cache and output unchanged.
    /// Each output lease owns one pin, including duplicate keys.
    /// A cache miss receives load_fence. The caller must submit that storage batch.
    /// If submission fails, the caller must resolve or abandon the whole backend.
    pub fn acquireBatch(self: *Cache, requests: []const Request, load_fence: LoadFence, output: []Lease) Error!void {
        if (output.len != requests.len) return error.OutputLengthMismatch;
        if (requests.len > std.math.maxInt(u32)) return error.CounterOverflow;
        if (requests.len > std.math.maxInt(u64) - self.access_clock) return error.CounterOverflow;

        var misses: usize = 0;
        for (requests, 0..) |request, i| {
            if (request.bytes == 0) return error.InvalidRequest;
            if (request.bytes > self.slot_bytes) return error.OversizedRequest;
            for (requests[0..i]) |earlier| {
                if (earlier.key.eql(request.key) and earlier.bytes != request.bytes) return error.SizeMismatch;
            }
            if (!first(requests, i)) continue;
            if (self.existing(request.key)) |index| {
                const slot = self.slots[index];
                if (slot.bytes != request.bytes) return error.SizeMismatch;
                const refs = references(requests, request.key);
                if (refs > std.math.maxInt(u32) - slot.pins) return error.CounterOverflow;
            } else {
                misses += 1;
            }
        }

        if (misses > 0) {
            if (load_fence.value == 0 or load_fence.value <= self.last_load.value) return error.FenceNotMonotonic;
            var available: usize = 0;
            for (self.slots) |slot| {
                if (!self.evictable(slot)) continue;
                if (slot.state == .ready and requested(requests, slot.key)) continue;
                available += 1;
            }
            if (misses > available) return error.CacheFull;
        }

        // Protect existing hits before any victim selection.
        for (requests, 0..) |request, i| {
            if (self.existing(request.key)) |index| {
                var slot = &self.slots[index];
                slot.pins += 1;
                self.access_clock += 1;
                slot.last_access = self.access_clock;
                output[i] = .{ .slot = @intCast(index), .generation = slot.generation };
            } else output[i] = .{ .slot = 0, .generation = 0 };
        }

        for (requests, 0..) |request, i| {
            if (output[i].generation != 0) continue;
            const index = self.existing(request.key) orelse index: {
                const index = self.victim();
                const generation = self.slots[index].generation + 1;
                self.slots[index] = .{
                    .state = .loading,
                    .key = request.key,
                    .bytes = request.bytes,
                    .generation = generation,
                    .load_fence = load_fence,
                };
                break :index index;
            };
            var slot = &self.slots[index];
            slot.pins += 1;
            self.access_clock += 1;
            slot.last_access = self.access_clock;
            output[i] = .{ .slot = @intCast(index), .generation = slot.generation };
        }
        if (misses > 0) self.last_load = load_fence;
    }

    pub fn get(self: *const Cache, lease: Lease) Error!*const Slot {
        if (lease.slot >= self.slots.len) return error.StaleLease;
        const slot = &self.slots[lease.slot];
        if (slot.state == .empty or slot.generation != lease.generation) return error.StaleLease;
        return slot;
    }

    /// Resolve the status for one storage batch after its fence completes.
    /// A failed batch remains unreadable. Its pins still require release.
    pub fn completeLoad(self: *Cache, batch: LoadFence, observed: LoadFence, success: bool) Error!void {
        if (batch.value == 0 or observed.value < batch.value) return error.FenceNotComplete;
        var found = false;
        for (self.slots) |*slot| {
            if (slot.state == .loading and slot.load_fence.value == batch.value) {
                slot.state = if (success) .ready else .failed;
                found = true;
            }
        }
        if (!found) return error.UnknownLoadFence;
    }

    /// Release a pin that never entered a compute submission.
    pub fn release(self: *Cache, lease: Lease) Error!void {
        const slot = try self.get(lease);
        if (slot.pins == 0) return error.NotPinned;
        self.slots[lease.slot].pins -= 1;
    }

    /// Retire one pin after the caller submits the expert's last GPU use.
    /// Use one monotonic compute fence timeline for this cache.
    pub fn retire(self: *Cache, lease: Lease, fence: ComputeFence) Error!void {
        const slot = try self.get(lease);
        if (slot.pins == 0) return error.NotPinned;
        if (slot.state != .ready) return error.NotReady;
        if (fence.value == 0 or fence.value <= self.completed_compute.value or fence.value < self.last_compute.value) return error.FenceNotMonotonic;
        self.slots[lease.slot].pins -= 1;
        self.slots[lease.slot].compute_fence = fence;
        self.last_compute = fence;
    }

    /// Observe the compute fence after the backend reads its completed value.
    pub fn observeCompute(self: *Cache, fence: ComputeFence) Error!void {
        if (fence.value < self.completed_compute.value) return error.FenceNotMonotonic;
        self.completed_compute = fence;
    }
};

fn req(expert: u32) Request {
    return .{ .key = .{ .layer = 0, .expert = expert }, .bytes = 128 };
}

test "misses stay pinned and unreadable until storage completes successfully" {
    var cache = try Cache.init(std.testing.allocator, 1, 256);
    defer cache.deinit();
    var leases: [1]Lease = undefined;
    try cache.acquireBatch(&.{req(1)}, .{ .value = 7 }, &leases);
    try std.testing.expectEqual(State.loading, (try cache.get(leases[0])).state);
    try std.testing.expectError(error.NotReady, cache.retire(leases[0], .{ .value = 1 }));
    try std.testing.expectError(error.FenceNotComplete, cache.completeLoad(.{ .value = 7 }, .{ .value = 6 }, true));
    try cache.completeLoad(.{ .value = 7 }, .{ .value = 7 }, true);
    try std.testing.expectEqual(State.ready, (try cache.get(leases[0])).state);
    try cache.release(leases[0]);
}

test "storage completion cannot satisfy a compute retirement fence" {
    var cache = try Cache.init(std.testing.allocator, 1, 256);
    defer cache.deinit();
    var lease: [1]Lease = undefined;
    try cache.acquireBatch(&.{req(1)}, .{ .value = 100 }, &lease);
    try cache.completeLoad(.{ .value = 100 }, .{ .value = 100 }, true);
    try cache.retire(lease[0], .{ .value = 2 });
    try std.testing.expectError(error.CacheFull, cache.acquireBatch(&.{req(2)}, .{ .value = 101 }, &lease));
    try cache.observeCompute(.{ .value = 1 });
    try std.testing.expectError(error.CacheFull, cache.acquireBatch(&.{req(2)}, .{ .value = 101 }, &lease));
    try cache.observeCompute(.{ .value = 2 });
    try cache.acquireBatch(&.{req(2)}, .{ .value = 101 }, &lease);
}

test "failed batch admission leaves state and output unchanged" {
    var cache = try Cache.init(std.testing.allocator, 2, 256);
    defer cache.deinit();
    var lease: [1]Lease = undefined;
    try cache.acquireBatch(&.{req(1)}, .{ .value = 1 }, &lease);
    const saved = cache.slots[0];
    var output = [_]Lease{ .{ .slot = 91, .generation = 8 }, .{ .slot = 92, .generation = 9 } };
    try std.testing.expectError(error.CacheFull, cache.acquireBatch(&.{ req(2), req(3) }, .{ .value = 2 }, &output));
    try std.testing.expectEqual(saved, cache.slots[0]);
    try std.testing.expectEqual(State.empty, cache.slots[1].state);
    try std.testing.expectEqual(@as(u32, 91), output[0].slot);
    try std.testing.expectEqual(@as(u64, 1), cache.last_load.value);
}

test "duplicates share one slot and own independent pins" {
    var cache = try Cache.init(std.testing.allocator, 1, 256);
    defer cache.deinit();
    var leases: [3]Lease = undefined;
    try cache.acquireBatch(&.{ req(2), req(2), req(2) }, .{ .value = 1 }, &leases);
    try std.testing.expectEqual(leases[0], leases[2]);
    try std.testing.expectEqual(@as(u32, 3), (try cache.get(leases[0])).pins);
    for (leases) |lease| try cache.release(lease);
    try std.testing.expectError(error.NotPinned, cache.release(leases[0]));
    // No pins does not permit eviction of a pending storage destination.
    var output: [1]Lease = undefined;
    try std.testing.expectError(error.CacheFull, cache.acquireBatch(&.{req(3)}, .{ .value = 2 }, &output));
}

test "batch hits survive victim selection and leases detect stale generations" {
    var cache = try Cache.init(std.testing.allocator, 2, 256);
    defer cache.deinit();
    var initial: [2]Lease = undefined;
    try cache.acquireBatch(&.{ req(1), req(2) }, .{ .value = 1 }, &initial);
    try cache.completeLoad(.{ .value = 1 }, .{ .value = 1 }, true);
    for (initial) |lease| try cache.release(lease);
    var next: [2]Lease = undefined;
    // Key 1 is the oldest. The batch still protects it before it selects a victim.
    try cache.acquireBatch(&.{ req(3), req(1) }, .{ .value = 2 }, &next);
    try std.testing.expectEqual(initial[0], next[1]);
    try std.testing.expectError(error.StaleLease, cache.get(initial[1]));
    try std.testing.expectEqual(@as(u32, 3), (try cache.get(next[0])).key.expert);
}

test "failed I/O cannot become an expert hit" {
    var cache = try Cache.init(std.testing.allocator, 1, 256);
    defer cache.deinit();
    var lease: [1]Lease = undefined;
    try cache.acquireBatch(&.{req(1)}, .{ .value = 1 }, &lease);
    try cache.completeLoad(.{ .value = 1 }, .{ .value = 1 }, false);
    try std.testing.expectEqual(State.failed, (try cache.get(lease[0])).state);
    try std.testing.expectError(error.NotReady, cache.retire(lease[0], .{ .value = 1 }));
    var retry: [1]Lease = undefined;
    try std.testing.expectError(error.CacheFull, cache.acquireBatch(&.{req(1)}, .{ .value = 2 }, &retry));
    try cache.release(lease[0]);
    try cache.acquireBatch(&.{req(1)}, .{ .value = 2 }, &retry);
    try std.testing.expectError(error.StaleLease, cache.get(lease[0]));
}

test "capacity and payload validation reject malformed requests" {
    try std.testing.expectError(error.InvalidCapacity, Cache.init(std.testing.allocator, 0, 128));
    try std.testing.expectError(error.InvalidCapacity, Cache.init(std.testing.allocator, 2, std.math.maxInt(u64)));
    var cache = try Cache.init(std.testing.allocator, 2, 256);
    defer cache.deinit();
    var leases: [2]Lease = undefined;
    const oversized: Request = .{ .key = .{ .layer = 1, .expert = 1 }, .bytes = 257 };
    try std.testing.expectError(error.OversizedRequest, cache.acquireBatch(&.{req(1), oversized}, .{ .value = 1 }, &leases));
    try std.testing.expectEqual(@as(u64, 0), cache.payloadBytes());
    const mismatch: Request = .{ .key = req(1).key, .bytes = 129 };
    try std.testing.expectError(error.SizeMismatch, cache.acquireBatch(&.{req(1), mismatch}, .{ .value = 1 }, &leases));
    try std.testing.expectEqual(@as(u64, 512), cache.capacityBytes());
}

test "deterministic LRU evicts the oldest eligible expert" {
    var cache = try Cache.init(std.testing.allocator, 2, 256);
    defer cache.deinit();
    var initial: [2]Lease = undefined;
    try cache.acquireBatch(&.{ req(1), req(2) }, .{ .value = 1 }, &initial);
    try cache.completeLoad(.{ .value = 1 }, .{ .value = 1 }, true);
    for (initial) |lease| try cache.release(lease);
    var touch: [1]Lease = undefined;
    try cache.acquireBatch(&.{req(1)}, .{ .value = 0 }, &touch);
    try cache.release(touch[0]);
    try cache.acquireBatch(&.{req(3)}, .{ .value = 2 }, &touch);
    try std.testing.expectEqual(initial[1].slot, touch[0].slot);
    try std.testing.expectEqual(@as(u64, 256), cache.payloadBytes());
}

test "compute observations and submissions require a monotonic timeline" {
    var cache = try Cache.init(std.testing.allocator, 2, 256);
    defer cache.deinit();
    var leases: [2]Lease = undefined;
    try cache.acquireBatch(&.{ req(1), req(2) }, .{ .value = 1 }, &leases);
    try cache.completeLoad(.{ .value = 1 }, .{ .value = 1 }, true);
    try cache.retire(leases[0], .{ .value = 9 });
    try std.testing.expectError(error.FenceNotMonotonic, cache.retire(leases[1], .{ .value = 8 }));
    try cache.retire(leases[1], .{ .value = 9 });
    try cache.observeCompute(.{ .value = 9 });
    try std.testing.expectError(error.FenceNotMonotonic, cache.observeCompute(.{ .value = 8 }));
}
