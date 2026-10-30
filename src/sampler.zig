//! Token sampling: greedy, temperature, top-k, top-p, min-p, and a repetition penalty.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Params = struct {
    temperature: f32 = 1.0,
    top_k: u32 = 20,
    top_p: f32 = 0.95,
    min_p: f32 = 0.0,
    repeat_penalty: f32 = 1.0,
    repeat_window: u32 = 64,
    seed: u64 = 0,
};

const Candidate = struct { id: u32, logit: f32 };

pub const Sampler = struct {
    allocator: Allocator,
    params: Params,
    rng: std.Random.DefaultPrng,
    candidates: []Candidate,

    pub fn init(allocator: Allocator, n_vocab: usize, params: Params) !Sampler {
        return .{
            .allocator = allocator,
            .params = params,
            .rng = .init(params.seed),
            .candidates = try allocator.alloc(Candidate, n_vocab),
        };
    }

    pub fn deinit(s: *Sampler) void {
        s.allocator.free(s.candidates);
    }

    /// Samples one token. `history` is the token sequence so far for the repetition penalty.
    pub fn sample(s: *Sampler, logits: []const f32, history: []const u32) u32 {
        const p = s.params;
        if (p.temperature <= 0) return argmax(logits, history, p);
        const c = s.candidates[0..logits.len];
        for (c, logits, 0..) |*cand, l, i| cand.* = .{ .id = @intCast(i), .logit = l };
        applyPenalty(c, history, p);
        // Top-k keeps the k largest logits; a zero k keeps every token.
        var n: usize = c.len;
        const k: usize = if (p.top_k == 0) c.len else @min(p.top_k, c.len);
        if (k < c.len) {
            selectTop(c, k);
            n = k;
        }
        const live = c[0..n];
        std.mem.sort(Candidate, live, {}, struct {
            fn f(_: void, a: Candidate, b: Candidate) bool {
                return if (a.logit != b.logit) a.logit > b.logit else a.id < b.id;
            }
        }.f);
        // Softmax with temperature.
        const max = live[0].logit;
        var sum: f64 = 0;
        for (live) |*cand| {
            const e = @exp(@as(f64, (cand.logit - max) / p.temperature));
            cand.logit = @floatCast(e);
            sum += e;
        }
        for (live) |*cand| cand.logit = @floatCast(cand.logit / sum);
        // Min-p relative to the top probability.
        if (p.min_p > 0) {
            const threshold = live[0].logit * p.min_p;
            var keep: usize = 1;
            while (keep < n and live[keep].logit >= threshold) keep += 1;
            n = keep;
        }
        // Top-p: smallest prefix whose mass reaches top_p.
        if (p.top_p < 1.0) {
            var mass: f32 = 0;
            var keep: usize = 0;
            while (keep < n) {
                mass += live[keep].logit;
                keep += 1;
                if (mass >= p.top_p) break;
            }
            n = keep;
        }
        var total: f32 = 0;
        for (live[0..n]) |cand| total += cand.logit;
        var r = s.rng.random().float(f32) * total;
        for (live[0..n]) |cand| {
            r -= cand.logit;
            if (r <= 0) return cand.id;
        }
        return live[n - 1].id;
    }
};

fn penalized(logit: f32, penalty: f32) f32 {
    return if (logit > 0) logit / penalty else logit * penalty;
}

fn applyPenalty(c: []Candidate, history: []const u32, p: Params) void {
    if (p.repeat_penalty == 1.0) return;
    const start = history.len -| p.repeat_window;
    for (history[start..], start..) |id, i| {
        // Penalize each distinct token once.
        if (std.mem.indexOfScalar(u32, history[start..i], id) != null) continue;
        if (id < c.len) c[id].logit = penalized(c[id].logit, p.repeat_penalty);
    }
}

fn argmax(logits: []const f32, history: []const u32, p: Params) u32 {
    var best: u32 = 0;
    var best_value = -std.math.inf(f32);
    const start = history.len -| p.repeat_window;
    for (logits, 0..) |l, i| {
        var v = l;
        if (p.repeat_penalty != 1.0 and std.mem.indexOfScalar(u32, history[start..], @intCast(i)) != null)
            v = penalized(v, p.repeat_penalty);
        if (v > best_value) {
            best_value = v;
            best = @intCast(i);
        }
    }
    return best;
}

/// Partitions c so its first k entries hold the k largest logits (quickselect).
fn selectTop(c: []Candidate, k: usize) void {
    var lo: usize = 0;
    var hi: usize = c.len;
    while (hi - lo > 1) {
        const pivot = c[lo + (hi - lo) / 2].logit;
        var i = lo;
        var j = hi - 1;
        while (i <= j) {
            while (c[i].logit > pivot) i += 1;
            while (c[j].logit < pivot) j -= 1;
            if (i <= j) {
                std.mem.swap(Candidate, &c[i], &c[j]);
                i += 1;
                if (j == 0) break;
                j -= 1;
            }
        }
        if (k <= j) {
            hi = j + 1;
        } else if (k >= i) {
            lo = i;
        } else return;
    }
}

test "greedy and top-k sampling" {
    var s = try Sampler.init(std.testing.allocator, 6, .{ .temperature = 0 });
    defer s.deinit();
    const logits = [_]f32{ 0.1, 3.0, -1.0, 2.9, 0.0, 1.0 };
    try std.testing.expectEqual(@as(u32, 1), s.sample(&logits, &.{}));
    s.params = .{ .temperature = 1.0, .top_k = 2, .top_p = 1.0, .seed = 3 };
    var seen: [6]bool = @splat(false);
    for (0..200) |_| seen[s.sample(&logits, &.{})] = true;
    try std.testing.expect(seen[1] and seen[3]);
    try std.testing.expect(!seen[0] and !seen[2] and !seen[4] and !seen[5]);
    // Repetition penalty demotes the leader below the runner-up under greedy decoding.
    s.params = .{ .temperature = 0, .repeat_penalty = 1.5 };
    try std.testing.expectEqual(@as(u32, 3), s.sample(&logits, &.{1}));
}

test "top-p keeps the smallest prefix reaching the mass" {
    var s = try Sampler.init(std.testing.allocator, 3, .{ .temperature = 1.0, .top_k = 0, .top_p = 0.5, .seed = 1 });
    defer s.deinit();
    const logits = [_]f32{ 5.0, 0.0, 0.0 };
    for (0..100) |_| try std.testing.expectEqual(@as(u32, 0), s.sample(&logits, &.{}));
}
