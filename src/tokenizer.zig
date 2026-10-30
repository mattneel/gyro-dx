//! Byte-level GPT-2 BPE tokenizer with the `qwen35` pre-tokenizer, built from GGUF metadata.
//!
//! Pre-tokenizer regex (exactly as llama.cpp / HF tokenizer.json):
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
//!   ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! No Unicode normalization is applied (matching llama.cpp); inputs are expected to be NFC.
const std = @import("std");
const gguf = @import("gguf.zig");
const tables = @import("unicode_tables.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ UnsupportedTokenizer, InvalidVocab } || gguf.ParseError;

pub const token_type_control: i32 = 3;
pub const token_type_user_defined: i32 = 4;

const Merge = struct { rank: u32, id: u32 };
const dead: u32 = std.math.maxInt(u32);

pub const Tokenizer = struct {
    allocator: Allocator,
    /// Decoded (raw byte) piece of every token, concatenated.
    blob: []u8,
    /// offsets[i]..offsets[i+1] is token i inside blob; len = vocab_size + 1.
    offsets: []u32,
    types: []u8,
    lookup: std.StringHashMapUnmanaged(u32),
    merges: std.AutoHashMapUnmanaged(u64, Merge),
    byte_token: [256]u32,
    /// Control/user-defined token ids, longest text first.
    specials: []u32,
    special_first: [256]bool,
    vocab_size: u32,
    eos_id: u32,
    bos_id: ?u32,
    pad_id: ?u32,
    add_bos: bool,

    /// Copies everything it needs out of `model`; `model`'s backing bytes may be freed afterwards.
    pub fn init(allocator: Allocator, model: gguf.Model) Error!Tokenizer {
        if (!std.mem.eql(u8, try model.text("tokenizer.ggml.model"), "gpt2")) return error.UnsupportedTokenizer;
        if (!std.mem.eql(u8, try model.text("tokenizer.ggml.pre"), "qwen35")) return error.UnsupportedTokenizer;

        const tokens = try (try model.required("tokenizer.ggml.tokens")).strings(allocator);
        defer allocator.free(tokens);
        const raw_types = try (try model.required("tokenizer.ggml.token_type")).ints32(allocator);
        defer allocator.free(raw_types);
        if (raw_types.len != tokens.len or tokens.len >= dead) return error.InvalidVocab;
        const merge_strings = try (try model.required("tokenizer.ggml.merges")).strings(allocator);
        defer allocator.free(merge_strings);

        const decode_map = byteDecodeMap();
        var blob: std.ArrayList(u8) = .empty;
        errdefer blob.deinit(allocator);
        const offsets = try allocator.alloc(u32, tokens.len + 1);
        errdefer allocator.free(offsets);
        const types = try allocator.alloc(u8, tokens.len);
        errdefer allocator.free(types);
        for (tokens, 0..) |tok, i| {
            offsets[i] = std.math.cast(u32, blob.items.len) orelse return error.InvalidVocab;
            const kind = raw_types[i];
            types[i] = std.math.cast(u8, kind) orelse 0;
            if (kind == token_type_control or kind == token_type_user_defined) {
                try blob.appendSlice(allocator, tok);
            } else try decodeMapped(allocator, &decode_map, tok, &blob);
        }
        offsets[tokens.len] = std.math.cast(u32, blob.items.len) orelse return error.InvalidVocab;
        const blob_bytes = try blob.toOwnedSlice(allocator);
        errdefer allocator.free(blob_bytes);

        var self: Tokenizer = .{
            .allocator = allocator,
            .blob = blob_bytes,
            .offsets = offsets,
            .types = types,
            .lookup = .empty,
            .merges = .empty,
            .byte_token = undefined,
            .specials = &.{},
            .special_first = @splat(false),
            .vocab_size = @intCast(tokens.len),
            .eos_id = @intCast(try model.number("tokenizer.ggml.eos_token_id")),
            .bos_id = optionalId(model, "tokenizer.ggml.bos_token_id"),
            .pad_id = optionalId(model, "tokenizer.ggml.padding_token_id"),
            .add_bos = if (model.get("tokenizer.ggml.add_bos_token")) |e| try e.boolean() else false,
        };
        errdefer self.lookup.deinit(allocator);
        errdefer self.merges.deinit(allocator);
        if (self.eos_id >= self.vocab_size) return error.InvalidVocab;

        try self.lookup.ensureTotalCapacity(allocator, @intCast(tokens.len));
        for (0..tokens.len) |i| {
            const gop = self.lookup.getOrPutAssumeCapacity(self.pieceBytes(@intCast(i)));
            if (!gop.found_existing) gop.value_ptr.* = @intCast(i);
        }
        for (0..256) |b| {
            const byte: u8 = @intCast(b);
            self.byte_token[b] = self.lookup.get(&.{byte}) orelse return error.InvalidVocab;
        }

        try self.merges.ensureTotalCapacity(allocator, @intCast(merge_strings.len));
        var scratch: std.ArrayList(u8) = .empty;
        defer scratch.deinit(allocator);
        for (merge_strings, 0..) |line, rank| {
            const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidVocab;
            scratch.clearRetainingCapacity();
            try decodeMapped(allocator, &decode_map, line[0..sp], &scratch);
            const left_len = scratch.items.len;
            try decodeMapped(allocator, &decode_map, line[sp + 1 ..], &scratch);
            const left = self.lookup.get(scratch.items[0..left_len]) orelse return error.InvalidVocab;
            const right = self.lookup.get(scratch.items[left_len..]) orelse return error.InvalidVocab;
            const merged = self.lookup.get(scratch.items) orelse return error.InvalidVocab;
            const gop = self.merges.getOrPutAssumeCapacity(pairKey(left, right));
            if (!gop.found_existing) gop.value_ptr.* = .{ .rank = @intCast(rank), .id = merged };
        }

        var specials: std.ArrayList(u32) = .empty;
        errdefer specials.deinit(allocator);
        for (0..tokens.len) |i| {
            if (raw_types[i] != token_type_control and raw_types[i] != token_type_user_defined) continue;
            const text = self.pieceBytes(@intCast(i));
            if (text.len == 0) continue;
            try specials.append(allocator, @intCast(i));
            self.special_first[text[0]] = true;
        }
        std.mem.sort(u32, specials.items, &self, longerFirst);
        self.specials = try specials.toOwnedSlice(allocator);
        return self;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.allocator.free(self.blob);
        self.allocator.free(self.offsets);
        self.allocator.free(self.types);
        self.allocator.free(self.specials);
        self.lookup.deinit(self.allocator);
        self.merges.deinit(self.allocator);
        self.* = undefined;
    }

    /// Raw bytes of a token, borrowed from the tokenizer (empty for an out-of-range id).
    pub fn pieceBytes(self: *const Tokenizer, id: u32) []const u8 {
        if (id >= self.vocab_size) return &.{};
        return self.blob[self.offsets[id]..self.offsets[id + 1]];
    }

    /// Owned copy of `pieceBytes`.
    pub fn piece(self: *const Tokenizer, allocator: Allocator, id: u32) Allocator.Error![]u8 {
        return allocator.dupe(u8, self.pieceBytes(id));
    }

    /// Token whose raw byte text equals `text` (for control tokens this is their literal text).
    pub fn tokenId(self: *const Tokenizer, text: []const u8) ?u32 {
        return self.lookup.get(text);
    }

    /// True for control tokens (token_type 3), e.g. <|im_start|>, <|endoftext|>.
    pub fn isControl(self: *const Tokenizer, id: u32) bool {
        return id < self.vocab_size and self.types[id] == token_type_control;
    }

    /// Tokens recognized verbatim in text when `parse_special` is set (token_type 3 or 4).
    pub fn isSpecial(self: *const Tokenizer, id: u32) bool {
        return id < self.vocab_size and (self.types[id] == token_type_control or self.types[id] == token_type_user_defined);
    }

    /// Concatenated raw bytes of `ids` (invalid ids contribute nothing).
    pub fn decode(self: *const Tokenizer, allocator: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| try out.appendSlice(allocator, self.pieceBytes(id));
        return out.toOwnedSlice(allocator);
    }

    /// Tokenizes `text`. With `parse_special`, control/user-defined token texts are emitted as their
    /// single token. A BOS token is prepended only if `tokenizer.ggml.add_bos_token` is true.
    pub fn encode(self: *const Tokenizer, allocator: Allocator, text: []const u8, parse_special: bool) Allocator.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);
        var work: Work = .{};
        defer work.deinit(allocator);
        if (self.add_bos) if (self.bos_id) |id| try out.append(allocator, id);
        var seg_start: usize = 0;
        if (parse_special and self.specials.len > 0) {
            var i: usize = 0;
            while (i < text.len) {
                if (self.special_first[text[i]]) {
                    if (self.matchSpecial(text[i..])) |id| {
                        try self.encodePlain(allocator, text[seg_start..i], &out, &work);
                        try out.append(allocator, id);
                        i += self.pieceBytes(id).len;
                        seg_start = i;
                        continue;
                    }
                }
                i += 1;
            }
        }
        try self.encodePlain(allocator, text[seg_start..], &out, &work);
        return out.toOwnedSlice(allocator);
    }

    fn matchSpecial(self: *const Tokenizer, rest: []const u8) ?u32 {
        for (self.specials) |id| {
            if (std.mem.startsWith(u8, rest, self.pieceBytes(id))) return id;
        }
        return null;
    }

    fn encodePlain(self: *const Tokenizer, allocator: Allocator, text: []const u8, out: *std.ArrayList(u32), work: *Work) Allocator.Error!void {
        var pos: usize = 0;
        while (pos < text.len) {
            const end = splitEnd(text, pos);
            try self.bpeWord(allocator, text[pos..end], out, work);
            pos = end;
        }
    }

    fn bpeWord(self: *const Tokenizer, allocator: Allocator, word: []const u8, out: *std.ArrayList(u32), work: *Work) Allocator.Error!void {
        const n = word.len;
        if (n == 1) return out.append(allocator, self.byte_token[word[0]]);
        try work.reset(allocator, n);
        const ids = work.ids.items;
        const next = work.next.items;
        const prev = work.prev.items;
        for (word, 0..) |b, i| {
            ids[i] = self.byte_token[b];
            next[i] = @intCast(i + 1);
            prev[i] = if (i == 0) dead else @intCast(i - 1);
        }
        for (0..n - 1) |i| try self.pushPair(allocator, &work.heap, ids, @intCast(i), @intCast(i + 1));
        while (work.heap.pop()) |top| {
            const left = top.left;
            const right = next[left];
            if (ids[left] != top.left_id or right >= n or ids[right] != top.right_id) continue;
            ids[left] = top.merged;
            ids[right] = dead;
            const after = next[right];
            next[left] = after;
            if (after < n) prev[after] = left;
            if (prev[left] != dead) try self.pushPair(allocator, &work.heap, ids, prev[left], left);
            if (after < n) try self.pushPair(allocator, &work.heap, ids, left, after);
        }
        var i: u32 = 0;
        while (i < n) : (i = next[i]) try out.append(allocator, ids[i]);
    }

    fn pushPair(self: *const Tokenizer, allocator: Allocator, heap: *Heap, ids: []const u32, left: u32, right: u32) Allocator.Error!void {
        const m = self.merges.get(pairKey(ids[left], ids[right])) orelse return;
        try heap.push(allocator, .{
            .rank = m.rank, .left = left, .left_id = ids[left], .right_id = ids[right], .merged = m.id,
        });
    }

    fn longerFirst(self: *Tokenizer, a: u32, b: u32) bool {
        const la = self.pieceBytes(a).len;
        const lb = self.pieceBytes(b).len;
        return if (la != lb) la > lb else a < b;
    }
};

fn optionalId(model: gguf.Model, key: []const u8) ?u32 {
    const entry = model.get(key) orelse return null;
    return std.math.cast(u32, entry.unsigned() catch return null);
}

fn pairKey(a: u32, b: u32) u64 {
    return (@as(u64, a) << 32) | b;
}

// ---------------------------------------------------------------------------------------------
// BPE work buffers and the merge priority queue.

const Bigram = struct { rank: u32, left: u32, left_id: u32, right_id: u32, merged: u32 };

const Heap = struct {
    items: std.ArrayList(Bigram) = .empty,

    fn less(a: Bigram, b: Bigram) bool {
        return if (a.rank != b.rank) a.rank < b.rank else a.left < b.left;
    }

    fn push(h: *Heap, allocator: Allocator, item: Bigram) Allocator.Error!void {
        try h.items.append(allocator, item);
        const a = h.items.items;
        var i = a.len - 1;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (!less(a[i], a[parent])) break;
            std.mem.swap(Bigram, &a[i], &a[parent]);
            i = parent;
        }
    }

    fn pop(h: *Heap) ?Bigram {
        const a = h.items.items;
        if (a.len == 0) return null;
        const top = a[0];
        a[0] = a[a.len - 1];
        h.items.items.len -= 1;
        const s = h.items.items;
        var i: usize = 0;
        while (true) {
            const l = 2 * i + 1;
            if (l >= s.len) break;
            const r = l + 1;
            const child = if (r < s.len and less(s[r], s[l])) r else l;
            if (!less(s[child], s[i])) break;
            std.mem.swap(Bigram, &s[child], &s[i]);
            i = child;
        }
        return top;
    }
};

const Work = struct {
    ids: std.ArrayList(u32) = .empty,
    next: std.ArrayList(u32) = .empty,
    prev: std.ArrayList(u32) = .empty,
    heap: Heap = .{},

    fn reset(w: *Work, allocator: Allocator, n: usize) Allocator.Error!void {
        try w.ids.resize(allocator, n);
        try w.next.resize(allocator, n);
        try w.prev.resize(allocator, n);
        w.heap.items.clearRetainingCapacity();
    }

    fn deinit(w: *Work, allocator: Allocator) void {
        w.ids.deinit(allocator);
        w.next.deinit(allocator);
        w.prev.deinit(allocator);
        w.heap.items.deinit(allocator);
    }
};

// ---------------------------------------------------------------------------------------------
// GPT-2 byte <-> unicode mapping.

/// codepoint -> byte for the GPT-2 byte-level alphabet; -1 where the codepoint is not in it.
fn byteDecodeMap() [324]i16 {
    var map: [324]i16 = @splat(-1);
    var extra: u16 = 0;
    for (0..256) |b| {
        const printable = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255);
        if (printable) {
            map[b] = @intCast(b);
        } else {
            map[256 + extra] = @intCast(b);
            extra += 1;
        }
    }
    return map;
}

/// Appends the bytes `mapped` (GPT-2 unicode alphabet, UTF-8) stands for. Characters outside the
/// alphabet are copied verbatim.
fn decodeMapped(allocator: Allocator, map: *const [324]i16, mapped: []const u8, out: *std.ArrayList(u8)) Allocator.Error!void {
    var i: usize = 0;
    while (i < mapped.len) {
        const c = decodeCp(mapped, i);
        if (c.v < map.len and map[c.v] >= 0) {
            try out.append(allocator, @intCast(map[c.v]));
        } else try out.appendSlice(allocator, mapped[i..][0..c.len]);
        i += c.len;
    }
}

// ---------------------------------------------------------------------------------------------
// Pre-tokenizer.

const Cp = struct { v: u21, len: u3 };
const invalid_cp: u21 = 0x1FFFFF;

fn decodeCp(text: []const u8, i: usize) Cp {
    const b = text[i];
    if (b < 0x80) return .{ .v = b, .len = 1 };
    const n = std.unicode.utf8ByteSequenceLength(b) catch return .{ .v = invalid_cp, .len = 1 };
    if (i + n > text.len) return .{ .v = invalid_cp, .len = 1 };
    const v = std.unicode.utf8Decode(text[i..][0..n]) catch return .{ .v = invalid_cp, .len = 1 };
    return .{ .v = v, .len = @intCast(n) };
}

const Class = enum { letter, mark, number, space, other };

fn isSpace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

fn classify(cp: u21) Class {
    if (cp < 0x80) {
        return switch (cp) {
            'a'...'z', 'A'...'Z' => .letter,
            '0'...'9' => .number,
            0x09...0x0D, 0x20 => .space,
            else => .other,
        };
    }
    if (isSpace(cp)) return .space;
    var lo: usize = 0;
    var hi: usize = tables.ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = tables.ranges[mid];
        if (cp < r.lo) {
            hi = mid;
        } else if (cp > r.hi) {
            lo = mid + 1;
        } else return switch (r.class) {
            1 => .letter,
            2 => .mark,
            else => .number,
        };
    }
    return .other;
}

fn letterRun(text: []const u8, start: usize) usize {
    var q = start;
    while (q < text.len) {
        const c = decodeCp(text, q);
        const k = classify(c.v);
        if (k != .letter and k != .mark) break;
        q += c.len;
    }
    return q;
}

fn lowerAscii(b: u8) u8 {
    return if (b >= 'A' and b <= 'Z') b + 32 else b;
}

/// End offset of the pre-token starting at `start` (always > start).
pub fn splitEnd(text: []const u8, start: usize) usize {
    const len = text.len;
    const c0 = decodeCp(text, start);
    const k0 = classify(c0.v);

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c0.v == '\'' and start + 1 < len) {
        const a = lowerAscii(text[start + 1]);
        if (a == 's' or a == 't' or a == 'm' or a == 'd') return start + 2;
        if (start + 2 < len) {
            const b = lowerAscii(text[start + 2]);
            if ((a == 'r' and b == 'e') or (a == 'v' and b == 'e') or (a == 'l' and b == 'l')) return start + 3;
        }
    }

    // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if (c0.v != '\r' and c0.v != '\n' and k0 != .letter and k0 != .number) {
        const p1 = start + c0.len;
        const e = letterRun(text, p1);
        if (e > p1) return e;
    }
    if (k0 == .letter or k0 == .mark) return letterRun(text, start + c0.len);

    // \p{N}
    if (k0 == .number) return start + c0.len;

    // ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    {
        const run_start = start + @intFromBool(c0.v == ' ');
        var q = run_start;
        while (q < len) {
            const c = decodeCp(text, q);
            if (classify(c.v) != .other) break;
            q += c.len;
        }
        if (q > run_start) {
            while (q < len and (text[q] == '\r' or text[q] == '\n')) q += 1;
            return q;
        }
    }

    if (k0 != .space) return start + c0.len;

    // \s*[\r\n]+ | \s+(?!\S) | \s+
    var q = start;
    var last_newline_end: usize = 0;
    var last_start = start;
    var count: usize = 0;
    while (q < len) {
        const c = decodeCp(text, q);
        if (classify(c.v) != .space) break;
        if (c.v == '\r' or c.v == '\n') last_newline_end = q + 1;
        last_start = q;
        q += c.len;
        count += 1;
    }
    if (last_newline_end != 0) return last_newline_end;
    if (q >= len) return q;
    if (count > 1) return last_start;
    return q;
}

// ---------------------------------------------------------------------------------------------
// Streaming detokenizer.

/// Turns a token stream into UTF-8 text without ever splitting a multi-byte sequence across calls.
pub const StreamDecoder = struct {
    pending: [3]u8 = undefined,
    pending_len: u2 = 0,

    /// Appends the bytes of `id` to `out`, holding back a trailing incomplete UTF-8 sequence.
    pub fn push(self: *StreamDecoder, allocator: Allocator, tok: *const Tokenizer, id: u32, out: *std.ArrayList(u8)) Allocator.Error!void {
        const start = out.items.len;
        try out.appendSlice(allocator, self.pending[0..self.pending_len]);
        try out.appendSlice(allocator, tok.pieceBytes(id));
        const hold = incompleteTail(out.items[start..]);
        self.pending_len = @intCast(hold);
        if (hold > 0) {
            @memcpy(self.pending[0..hold], out.items[out.items.len - hold ..]);
            out.items.len -= hold;
        }
    }

    /// Emits whatever is still held back (an incomplete sequence at end of generation).
    pub fn flush(self: *StreamDecoder, allocator: Allocator, out: *std.ArrayList(u8)) Allocator.Error!void {
        try out.appendSlice(allocator, self.pending[0..self.pending_len]);
        self.pending_len = 0;
    }

    pub fn reset(self: *StreamDecoder) void {
        self.pending_len = 0;
    }

    /// Length of a trailing byte run that is a proper prefix of a valid UTF-8 sequence (0..3).
    fn incompleteTail(bytes: []const u8) usize {
        var back: usize = 1;
        while (back <= 3 and back <= bytes.len) : (back += 1) {
            const b = bytes[bytes.len - back];
            if (b & 0xC0 == 0x80) continue; // continuation: keep looking for the lead byte
            const need = std.unicode.utf8ByteSequenceLength(b) catch return 0;
            return if (need > back) back else 0;
        }
        return 0;
    }
};

// ---------------------------------------------------------------------------------------------
// Tests. Vocabulary-dependent tests need the GGUF header named by $GYRO_HEADER or the default path;
// they skip when it is absent.

const vectors = @import("tokenizer_vectors.zig");

const default_header_path = "C:/src/gyro-work/coder-header.bin";
const default_full_size: u64 = 75796732256;

pub const TestHeader = struct {
    prefix: []u8,
    model: gguf.Model,

    pub fn load(allocator: Allocator) !TestHeader {
        const io = std.testing.io;
        const env_path = std.testing.environ.getAlloc(allocator, "GYRO_HEADER") catch null;
        defer if (env_path) |p| allocator.free(p);
        const path = env_path orelse default_header_path;
        const prefix = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(gguf.max_header_bytes)) catch
            return error.SkipZigTest;
        errdefer allocator.free(prefix);
        const model = try gguf.parse(allocator, prefix, default_full_size);
        return .{ .prefix = prefix, .model = model };
    }

    pub fn deinit(h: *TestHeader, allocator: Allocator) void {
        h.model.deinit();
        allocator.free(h.prefix);
    }
};

test "pre-tokenizer splits follow the qwen35 regex" {
    const Case = struct { text: []const u8, parts: []const []const u8 };
    const cases = [_]Case{
        .{ .text = "Hello, world!", .parts = &.{ "Hello", ",", " world", "!" } },
        .{ .text = "it's 12345", .parts = &.{ "it", "'s", " ", "1", "2", "3", "4", "5" } },
        .{ .text = "I'M WE'RE they'LL", .parts = &.{ "I", "'M", " WE", "'RE", " they", "'LL" } },
        .{ .text = "a  \n\n  b", .parts = &.{ "a", "  \n\n", " ", " b" } },
        .{ .text = "x   y", .parts = &.{ "x", "  ", " y" } },
        .{ .text = "end   ", .parts = &.{ "end", "   " } },
        .{ .text = "a.\r\nb", .parts = &.{ "a", ".\r\n", "b" } },
        .{ .text = "\tfoo(\"bar\");\n", .parts = &.{ "\tfoo", "(\"", "bar", "\");\n" } },
        .{ .text = "日本語です", .parts = &.{"日本語です"} },
        .{ .text = "e\u{301}x", .parts = &.{"e\u{301}x"} },
    };
    for (cases) |c| {
        var pos: usize = 0;
        for (c.parts) |expected| {
            const end = splitEnd(c.text, pos);
            try std.testing.expectEqualStrings(expected, c.text[pos..end]);
            pos = end;
        }
        try std.testing.expectEqual(c.text.len, pos);
    }
}

test "stream decoder never splits UTF-8 sequences" {
    const allocator = std.testing.allocator;
    // Euro sign E2 82 AC then 'x', delivered one byte at a time via a fake tokenizer view.
    var blob = [_]u8{ 0xE2, 0x82, 0xAC, 'x', 0xFF };
    var offsets = [_]u32{ 0, 1, 2, 3, 4, 5 };
    var tok: Tokenizer = undefined;
    tok.blob = &blob;
    tok.offsets = &offsets;
    tok.vocab_size = 5;
    var dec: StreamDecoder = .{};
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try dec.push(allocator, &tok, 0, &out);
    try dec.push(allocator, &tok, 1, &out);
    try std.testing.expectEqual(@as(usize, 0), out.items.len);
    try dec.push(allocator, &tok, 2, &out);
    try std.testing.expectEqualStrings("\u{20AC}", out.items);
    try dec.push(allocator, &tok, 3, &out);
    try dec.push(allocator, &tok, 0, &out);
    try dec.push(allocator, &tok, 4, &out); // lone E2 followed by FF: E2 is flushed as-is
    try dec.flush(allocator, &out);
    try std.testing.expectEqualStrings("\u{20AC}x\xE2\xFF", out.items);
}

test "encode matches HF tokenizers reference ids and round-trips" {
    const allocator = std.testing.allocator;
    var header = try TestHeader.load(allocator);
    defer header.deinit(allocator);
    var tok = try Tokenizer.init(allocator, header.model);
    defer tok.deinit();
    try std.testing.expectEqual(@as(u32, 248320), tok.vocab_size);
    try std.testing.expectEqual(@as(u32, 248046), tok.eos_id);
    for (vectors.cases) |c| {
        const ids = try tok.encode(allocator, c.text, true);
        defer allocator.free(ids);
        std.testing.expectEqualSlices(u32, c.ids, ids) catch |err| {
            std.debug.print("case: {any}\n", .{c.text});
            return err;
        };
        const back = try tok.decode(allocator, ids);
        defer allocator.free(back);
        try std.testing.expectEqualStrings(c.text, back);
    }
}

test "special tokens only parse when requested and large inputs encode quickly" {
    const allocator = std.testing.allocator;
    var header = try TestHeader.load(allocator);
    defer header.deinit(allocator);
    var tok = try Tokenizer.init(allocator, header.model);
    defer tok.deinit();
    const im_start = tok.tokenId("<|im_start|>").?;
    try std.testing.expect(tok.isControl(im_start));
    const plain = try tok.encode(allocator, "<|im_start|>", false);
    defer allocator.free(plain);
    try std.testing.expect(plain.len > 1);
    const parsed = try tok.encode(allocator, "<|im_start|>", true);
    defer allocator.free(parsed);
    try std.testing.expectEqualSlices(u32, &.{im_start}, parsed);

    var big: std.ArrayList(u8) = .empty;
    defer big.deinit(allocator);
    while (big.items.len < 200 * 1024) try big.appendSlice(allocator, "fn main() { let x = 42; // héllo wörld 日本\n    println!(\"{}\", x); }\n");
    const ids = try tok.encode(allocator, big.items, true);
    defer allocator.free(ids);
    const back = try tok.decode(allocator, ids);
    defer allocator.free(back);
    try std.testing.expectEqualStrings(big.items, back);
}
