//! Hand port of the GGUF `tokenizer.chat_template` (Qwen3.8-Flash-Next-Gyro-Coder, Unsloth variant)
//! for text-only conversations. `verifyTemplate` checks that a model's template is byte-identical to
//! the one this port was written against.
//!
//! Supported: system/developer/user/assistant/tool messages with string content, assistant
//! `reasoning_content`, assistant `tool_calls` whose argument values are strings, `enable_thinking`,
//! `preserve_thinking`, `reasoning_effort`, `add_generation_prompt`.
//! Not representable (and therefore impossible to misrender): the `tools` definition block (needs
//! Jinja `tojson`), non-string tool-call argument values, and image/video/list message content.
const std = @import("std");
const gguf = @import("gguf.zig");
const Allocator = std.mem.Allocator;

pub const template_sha256_hex = "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce";

pub const Role = enum { system, developer, user, assistant, tool };

pub const Argument = struct { name: []const u8, value: []const u8 };

pub const ToolCall = struct { name: []const u8, arguments: []const Argument = &.{} };

pub const Message = struct {
    role: Role,
    content: []const u8 = "",
    /// Assistant only; rendered inside <think> blocks.
    reasoning_content: ?[]const u8 = null,
    /// Assistant only.
    tool_calls: []const ToolCall = &.{},
};

pub const Options = struct {
    add_generation_prompt: bool = true,
    /// null = variable undefined in the template (thinking on, with reasoning instructions).
    enable_thinking: ?bool = null,
    /// null = undefined (reasoning of every assistant turn is kept).
    preserve_thinking: ?bool = null,
    /// null = undefined (default "xhigh"). One of xhigh, high, medium, low.
    reasoning_effort: ?[]const u8 = null,
};

pub const RenderError = error{
    /// 'No messages provided.'
    NoMessages,
    /// 'System message must be at the beginning.'
    SystemMessageNotFirst,
    /// 'Unexpected reasoning effort ...'
    UnexpectedReasoningEffort,
    TemplateMismatch,
} || Allocator.Error;

/// Fails with `error.TemplateMismatch` unless `template` is the exact template this port implements.
pub fn verifyTemplate(template: []const u8) error{TemplateMismatch}!void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(template, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &hex, template_sha256_hex)) return error.TemplateMismatch;
}

/// `verifyTemplate` applied to the model's `tokenizer.chat_template`.
pub fn verifyModelTemplate(model: gguf.Model) (RenderError || gguf.ParseError)!void {
    try verifyTemplate(try model.text("tokenizer.chat_template"));
}

const xhigh_instructions = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const low_instructions = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

/// Renders the prompt text (with literal `<|im_start|>` etc.; tokenize with parse_special=true).
pub fn render(allocator: Allocator, messages: []const Message, options: Options) RenderError![]u8 {
    if (messages.len == 0) return error.NoMessages;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    // Leading system/developer messages are merged into one system block.
    var merged: std.ArrayList(u8) = .empty;
    defer merged.deinit(allocator);
    var num_sys: usize = 0;
    for (messages, 0..) |m, i| {
        if (num_sys == i and (m.role == .system or m.role == .developer)) {
            const text = trim(m.content);
            if (text.len > 0) {
                if (merged.items.len > 0) try merged.append(allocator, '\n');
                try merged.appendSlice(allocator, text);
            }
            num_sys += 1;
        }
    }

    var instructions: []const u8 = "";
    if (options.enable_thinking == null or options.enable_thinking.?) {
        var effort: []const u8 = options.reasoning_effort orelse "xhigh";
        if (std.mem.eql(u8, effort, "high")) effort = "xhigh";
        if (std.mem.eql(u8, effort, "xhigh")) {
            instructions = xhigh_instructions;
        } else if (std.mem.eql(u8, effort, "low")) {
            instructions = low_instructions;
        } else if (!std.mem.eql(u8, effort, "medium")) return error.UnexpectedReasoningEffort;
    }

    if (merged.items.len > 0) {
        try out.appendSlice(allocator, "<|im_start|>system\n");
        if (instructions.len > 0) {
            try out.appendSlice(allocator, instructions);
            try out.appendSlice(allocator, "\n\n");
        }
        try out.appendSlice(allocator, merged.items);
        try out.appendSlice(allocator, "<|im_end|>\n");
    } else if (instructions.len > 0) {
        try out.appendSlice(allocator, "<|im_start|>system\n");
        try out.appendSlice(allocator, instructions);
        try out.appendSlice(allocator, "<|im_end|>\n");
    }

    // Index of the last real user query (a user turn that is not a bare tool response).
    var last_query_index: usize = messages.len - 1;
    var j = messages.len;
    while (j > 0) {
        j -= 1;
        if (messages[j].role != .user) continue;
        const c = trim(messages[j].content);
        if (!(std.mem.startsWith(u8, c, "<tool_response>") and std.mem.endsWith(u8, c, "</tool_response>"))) {
            last_query_index = j;
            break;
        }
    }

    for (messages[num_sys..], num_sys..) |m, i| {
        const content = trim(m.content);
        switch (m.role) {
            .system, .developer => return error.SystemMessageNotFirst,
            .user => {
                try out.appendSlice(allocator, "<|im_start|>user\n");
                try out.appendSlice(allocator, content);
                try out.appendSlice(allocator, "<|im_end|>\n");
            },
            .assistant => {
                const reasoning = trim(m.reasoning_content orelse "");
                try out.appendSlice(allocator, "<|im_start|>assistant\n");
                if ((options.preserve_thinking orelse true) or i > last_query_index) {
                    try out.appendSlice(allocator, "<think>\n");
                    try out.appendSlice(allocator, reasoning);
                    try out.appendSlice(allocator, "\n</think>\n\n");
                }
                try out.appendSlice(allocator, content);
                for (m.tool_calls, 0..) |call, k| {
                    if (k == 0) {
                        if (content.len > 0) try out.appendSlice(allocator, "\n\n");
                    } else try out.append(allocator, '\n');
                    try out.appendSlice(allocator, "<tool_call>\n<function=");
                    try out.appendSlice(allocator, call.name);
                    try out.appendSlice(allocator, ">\n");
                    for (call.arguments) |arg| {
                        try out.appendSlice(allocator, "<parameter=");
                        try out.appendSlice(allocator, arg.name);
                        try out.appendSlice(allocator, ">\n");
                        try out.appendSlice(allocator, arg.value);
                        try out.appendSlice(allocator, "\n</parameter>\n");
                    }
                    try out.appendSlice(allocator, "</function>\n</tool_call>");
                }
                try out.appendSlice(allocator, "<|im_end|>\n");
            },
            .tool => {
                if (i > 0 and messages[i - 1].role != .tool) try out.appendSlice(allocator, "<|im_start|>user");
                try out.appendSlice(allocator, "\n<tool_response>\n");
                try out.appendSlice(allocator, content);
                try out.appendSlice(allocator, "\n</tool_response>");
                const last = i + 1 == messages.len;
                if (last or messages[i + 1].role != .tool) try out.appendSlice(allocator, "<|im_end|>\n");
            },
        }
    }

    if (options.add_generation_prompt) {
        try out.appendSlice(allocator, "<|im_start|>assistant\n");
        if (options.enable_thinking == false) {
            try out.appendSlice(allocator, "<think>\n\n</think>\n\n");
        } else try out.appendSlice(allocator, "<think>\n");
    }
    return out.toOwnedSlice(allocator);
}

/// Python `str.strip()` (Unicode whitespace), i.e. Jinja's `trim` filter.
pub fn trim(s: []const u8) []const u8 {
    var start: usize = 0;
    while (start < s.len) {
        const n = leadingSpace(s[start..]);
        if (n == 0) break;
        start += n;
    }
    var end = s.len;
    while (end > start) {
        const n = trailingSpace(s[start..end]);
        if (n == 0) break;
        end -= n;
    }
    return s[start..end];
}

fn isPySpace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// Byte length of the whitespace codepoint at the start of `s`, or 0.
fn leadingSpace(s: []const u8) usize {
    const n = std.unicode.utf8ByteSequenceLength(s[0]) catch return 0;
    if (n > s.len) return 0;
    const cp = std.unicode.utf8Decode(s[0..n]) catch return 0;
    return if (isPySpace(cp)) n else 0;
}

/// Byte length of the whitespace codepoint at the end of `s`, or 0.
fn trailingSpace(s: []const u8) usize {
    var n: usize = 1;
    while (n <= 4 and n <= s.len) : (n += 1) {
        if (s[s.len - n] & 0xC0 == 0x80) continue;
        const cp = std.unicode.utf8Decode(s[s.len - n ..]) catch return 0;
        return if (isPySpace(cp)) n else 0;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------

const chat_vectors = @import("chat_vectors.zig");
const tokenizer = @import("tokenizer.zig");

test "render matches Python jinja2 rendering of the real template" {
    const allocator = std.testing.allocator;
    for (chat_vectors.cases, 0..) |c, i| {
        const got = try render(allocator, c.messages, c.options);
        defer allocator.free(got);
        std.testing.expectEqualStrings(c.expected, got) catch |err| {
            std.debug.print("chat vector {d} ({s})\n", .{ i, c.name });
            return err;
        };
    }
}

test "render rejects misplaced system messages, empty conversations and bad effort" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.NoMessages, render(allocator, &.{}, .{}));
    try std.testing.expectError(error.SystemMessageNotFirst, render(allocator, &.{
        .{ .role = .user, .content = "a" }, .{ .role = .system, .content = "b" },
    }, .{}));
    try std.testing.expectError(error.UnexpectedReasoningEffort, render(allocator, &.{
        .{ .role = .user, .content = "a" },
    }, .{ .reasoning_effort = "extreme" }));
}

test "trim strips Unicode whitespace like Python" {
    try std.testing.expectEqualStrings("a b", trim(" \u{3000}\x1c a b\u{a0}\n\t"));
    try std.testing.expectEqualStrings("", trim(" \n"));
    try std.testing.expectEqualStrings("é", trim("\u{2003}é\u{2003}"));
}

test "GGUF chat template is the one this port implements, and tokenizes with special tokens" {
    const allocator = std.testing.allocator;
    var header = try tokenizer.TestHeader.load(allocator);
    defer header.deinit(allocator);
    try verifyModelTemplate(header.model);
    var tok = try tokenizer.Tokenizer.init(allocator, header.model);
    defer tok.deinit();
    const text = try render(allocator, &.{.{ .role = .user, .content = "hi" }}, .{ .enable_thinking = false });
    defer allocator.free(text);
    const ids = try tok.encode(allocator, text, true);
    defer allocator.free(ids);
    try std.testing.expectEqual(tok.tokenId("<|im_start|>").?, ids[0]);
    try std.testing.expectEqual(tok.tokenId("<think>").?, ids[ids.len - 4]);
}
