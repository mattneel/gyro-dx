//! Fixed worker pool for data-parallel CPU kernels.
//! The calling thread participates in every job. Workers spin briefly and then park on a futex.
const std = @import("std");
const Io = std.Io;

const Job = struct {
    func: *const fn (*anyopaque, usize, usize) void,
    ctx: *anyopaque,
    count: usize,
    chunk: usize,
};

pub const Pool = struct {
    io: Io,
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    generation: std.atomic.Value(u32) = .init(0),
    pending: std.atomic.Value(u32) = .init(0),
    next: std.atomic.Value(usize) = .init(0),
    quit: std.atomic.Value(bool) = .init(false),
    job: Job = undefined,

    /// thread_count includes the calling thread. A value of 1 runs every job inline.
    pub fn init(pool: *Pool, io: Io, allocator: std.mem.Allocator, thread_count: usize) !void {
        pool.* = .{ .io = io, .allocator = allocator, .threads = &.{} };
        const workers = @max(thread_count, 1) - 1;
        pool.threads = try allocator.alloc(std.Thread, workers);
        var started: usize = 0;
        errdefer {
            pool.quit.store(true, .release);
            _ = pool.generation.fetchAdd(1, .acq_rel);
            io.futexWake(u32, &pool.generation.raw, std.math.maxInt(u32));
            for (pool.threads[0..started]) |t| t.join();
            allocator.free(pool.threads);
        }
        for (pool.threads) |*t| {
            t.* = try std.Thread.spawn(.{}, worker, .{pool});
            started += 1;
        }
    }

    pub fn deinit(pool: *Pool) void {
        pool.quit.store(true, .release);
        _ = pool.generation.fetchAdd(1, .acq_rel);
        pool.io.futexWake(u32, &pool.generation.raw, std.math.maxInt(u32));
        for (pool.threads) |t| t.join();
        pool.allocator.free(pool.threads);
    }

    pub fn threadCount(pool: *const Pool) usize {
        return pool.threads.len + 1;
    }

    fn drain(pool: *Pool) void {
        const job = pool.job;
        while (true) {
            const start = pool.next.fetchAdd(job.chunk, .monotonic);
            if (start >= job.count) break;
            job.func(job.ctx, start, @min(start + job.chunk, job.count));
        }
    }

    fn worker(pool: *Pool) void {
        var seen: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            while (pool.generation.load(.acquire) == seen) {
                if (spins < 4096) {
                    std.atomic.spinLoopHint();
                    spins += 1;
                } else {
                    pool.io.futexWaitUncancelable(u32, &pool.generation.raw, seen);
                }
            }
            seen = pool.generation.load(.acquire);
            if (pool.quit.load(.acquire)) return;
            pool.drain();
            if (pool.pending.fetchSub(1, .acq_rel) == 1)
                pool.io.futexWake(u32, &pool.pending.raw, 1);
        }
    }

    /// Runs func(ctx, start, end) over [0, count) in chunks. Blocks until every chunk finishes.
    pub fn run(pool: *Pool, count: usize, chunk: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize, usize) void) void {
        if (count == 0) return;
        const Ctx = @TypeOf(ctx);
        const Wrap = struct {
            fn call(p: *anyopaque, start: usize, end: usize) void {
                func(@as(Ctx, @ptrCast(@alignCast(p))), start, end);
            }
        };
        const c = @max(chunk, 1);
        if (pool.threads.len == 0 or count <= c) {
            func(ctx, 0, count);
            return;
        }
        pool.job = .{ .func = Wrap.call, .ctx = @ptrCast(@constCast(ctx)), .count = count, .chunk = c };
        pool.next.store(0, .release);
        pool.pending.store(@intCast(pool.threads.len), .release);
        _ = pool.generation.fetchAdd(1, .acq_rel);
        pool.io.futexWake(u32, &pool.generation.raw, std.math.maxInt(u32));
        pool.drain();
        var spins: u32 = 0;
        while (true) {
            const left = pool.pending.load(.acquire);
            if (left == 0) break;
            if (spins < 1 << 16) {
                std.atomic.spinLoopHint();
                spins += 1;
            } else {
                pool.io.futexWaitUncancelable(u32, &pool.pending.raw, left);
            }
        }
    }

    /// Chooses a chunk so that each thread receives several chunks.
    pub fn chunkFor(pool: *const Pool, count: usize, min_chunk: usize) usize {
        const target = count / (pool.threadCount() * 4);
        return @max(min_chunk, target);
    }
};

test "pool covers every index exactly once" {
    var pool: Pool = undefined;
    try pool.init(std.testing.io, std.testing.allocator, 4);
    defer pool.deinit();
    var hits: [1000]std.atomic.Value(u32) = @splat(.init(0));
    const Ctx = struct { hits: []std.atomic.Value(u32) };
    const ctx = Ctx{ .hits = &hits };
    for (0..50) |_| {
        pool.run(hits.len, 7, &ctx, struct {
            fn f(c: *const Ctx, start: usize, end: usize) void {
                for (start..end) |i| _ = c.hits[i].fetchAdd(1, .monotonic);
            }
        }.f);
    }
    for (hits) |h| try std.testing.expectEqual(@as(u32, 50), h.load(.monotonic));
}
