//! OpenAI-compatible HTTP API for one model session.
//! Requests run sequentially on the single loaded sequence; prompt prefixes are reused.
const std = @import("std");
const core = @import("gyro_dx");
const cli = @import("cli.zig");
const runtime = core.runtime;
const chat = core.chat;
const Io = std.Io;
const Allocator = std.mem.Allocator;

const Message = struct {
    role: []const u8,
    content: ?[]const u8 = null,
    reasoning_content: ?[]const u8 = null,
};

const TemplateKwargs = struct { enable_thinking: ?bool = null };

const ChatRequest = struct {
    model: ?[]const u8 = null,
    messages: []const Message = &.{},
    prompt: ?[]const u8 = null,
    max_tokens: ?u32 = null,
    max_completion_tokens: ?u32 = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    min_p: ?f32 = null,
    repeat_penalty: ?f32 = null,
    seed: ?u64 = null,
    stream: bool = false,
    enable_thinking: ?bool = null,
    chat_template_kwargs: ?TemplateKwargs = null,
};

pub const Config = struct {
    model_name: []const u8,
    default_max_tokens: u32 = 2048,
    defaults: core.sampler.Params,
};

/// gyro-dx serve MODEL [--host 127.0.0.1] [--port 8080] [inference flags]
pub fn serve(io: Io, allocator: Allocator, args: []const []const u8, writer: *Io.Writer) !void {
    var f = try cli.Flags.parse(allocator, args);
    if (f.positional.items.len != 1) return error.InvalidArguments;
    const path = f.positional.items[0];
    const rt = try cli.openRuntime(io, allocator, &f, path);
    defer rt.destroy();
    const host = f.str("host") orelse "127.0.0.1";
    const port = try f.int(u16, "port", 8080);
    const address = try Io.net.IpAddress.parseIp4(host, port);
    var listener = try address.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);
    const config = Config{
        .model_name = std.fs.path.basename(path),
        .default_max_tokens = try f.int(u32, "max-tokens", 2048),
        .defaults = try cli.samplerParams(&f),
    };
    try writer.print("gyro-dx serving {s} on http://{s}:{d} (backend {s})\n", .{ config.model_name, host, port, rt.backend.name });
    try writer.flush();
    while (true) {
        const stream = listener.accept(io) catch |err| {
            std.log.err("accept: {s}", .{@errorName(err)});
            continue;
        };
        handleConnection(io, allocator, rt, &config, stream) catch |err| {
            std.log.err("connection: {s}", .{@errorName(err)});
        };
        stream.close(io);
    }
}

fn handleConnection(io: Io, gpa: Allocator, rt: *runtime.Runtime, config: *const Config, stream: Io.net.Stream) !void {
    var in_buf: [1 << 16]u8 = undefined;
    var out_buf: [1 << 16]u8 = undefined;
    var reader = stream.reader(io, &in_buf);
    var writer = stream.writer(io, &out_buf);
    var server = std.http.Server.init(&reader.interface, &writer.interface);
    while (true) {
        var request = server.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return err,
        };
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        try handleRequest(arena.allocator(), rt, config, &request);
        if (!request.head.keep_alive) return;
    }
}

const json_headers = [_]std.http.Header{
    .{ .name = "content-type", .value = "application/json" },
    .{ .name = "access-control-allow-origin", .value = "*" },
};

fn respondJson(request: *std.http.Server.Request, status: std.http.Status, body: []const u8) !void {
    try request.respond(body, .{ .status = status, .extra_headers = &json_headers });
}

fn respondError(allocator: Allocator, request: *std.http.Server.Request, status: std.http.Status, message: []const u8) !void {
    const body = try std.fmt.allocPrint(allocator, "{f}", .{std.json.fmt(.{ .@"error" = .{ .message = message, .type = "invalid_request_error" } }, .{})});
    try respondJson(request, status, body);
}

fn handleRequest(allocator: Allocator, rt: *runtime.Runtime, config: *const Config, request: *std.http.Server.Request) !void {
    const target = request.head.target;
    const path = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[0..q] else target;
    if (request.head.method == .OPTIONS) {
        return request.respond("", .{ .status = .no_content, .extra_headers = &.{
            .{ .name = "access-control-allow-origin", .value = "*" },
            .{ .name = "access-control-allow-headers", .value = "*" },
            .{ .name = "access-control-allow-methods", .value = "GET, POST, OPTIONS" },
        } });
    }
    if (request.head.method == .GET and std.mem.eql(u8, path, "/health")) return respondJson(request, .ok, "{\"status\":\"ok\"}");
    if (request.head.method == .GET and std.mem.eql(u8, path, "/v1/models")) {
        const body = try std.fmt.allocPrint(allocator, "{f}", .{std.json.fmt(.{
            .object = "list",
            .data = .{.{ .id = config.model_name, .object = "model", .owned_by = "gyro-dx" }},
        }, .{})});
        return respondJson(request, .ok, body);
    }
    const is_chat = std.mem.eql(u8, path, "/v1/chat/completions");
    const is_completion = std.mem.eql(u8, path, "/v1/completions");
    if (request.head.method != .POST or !(is_chat or is_completion)) return respondError(allocator, request, .not_found, "unknown endpoint");

    var body_buf: [4096]u8 = undefined;
    const body_reader = request.readerExpectNone(&body_buf);
    const body = body_reader.allocRemaining(allocator, .limited(16 << 20)) catch return respondError(allocator, request, .bad_request, "unreadable body");
    const parsed = std.json.parseFromSliceLeaky(ChatRequest, allocator, body, .{ .ignore_unknown_fields = true }) catch
        return respondError(allocator, request, .bad_request, "invalid JSON request");

    // Prompt text: rendered chat template or a raw completion prompt.
    var prompt: []const u8 = undefined;
    if (is_chat) {
        if (parsed.messages.len == 0) return respondError(allocator, request, .bad_request, "messages must not be empty");
        const messages = try allocator.alloc(chat.Message, parsed.messages.len);
        for (parsed.messages, messages) |m, *out| {
            const role = std.meta.stringToEnum(chat.Role, m.role) orelse return respondError(allocator, request, .bad_request, "unsupported role");
            out.* = .{ .role = role, .content = m.content orelse "", .reasoning_content = m.reasoning_content };
        }
        const thinking = parsed.enable_thinking orelse if (parsed.chat_template_kwargs) |k| k.enable_thinking else null;
        prompt = chat.render(allocator, messages, .{ .enable_thinking = thinking }) catch |err|
            return respondError(allocator, request, .bad_request, @errorName(err));
    } else {
        prompt = parsed.prompt orelse return respondError(allocator, request, .bad_request, "prompt is required");
    }
    const tokens = try rt.tok.encode(allocator, prompt, true);
    if (tokens.len + 1 >= rt.backend.capacity()) return respondError(allocator, request, .bad_request, "prompt exceeds the context size");

    var params = config.defaults;
    if (parsed.temperature) |v| params.temperature = v;
    if (parsed.top_p) |v| params.top_p = v;
    if (parsed.top_k) |v| params.top_k = v;
    if (parsed.min_p) |v| params.min_p = v;
    if (parsed.repeat_penalty) |v| params.repeat_penalty = v;
    if (parsed.seed) |v| params.seed = v;
    const max_tokens = parsed.max_completion_tokens orelse parsed.max_tokens orelse config.default_max_tokens;
    const id = "chatcmpl-gyro";
    const object = if (is_chat) "chat.completion" else "text_completion";

    if (parsed.stream) {
        var stream_buf: [4096]u8 = undefined;
        var bw = try request.respondStreaming(&stream_buf, .{ .respond_options = .{ .extra_headers = &.{
            .{ .name = "content-type", .value = "text/event-stream" },
            .{ .name = "cache-control", .value = "no-cache" },
            .{ .name = "access-control-allow-origin", .value = "*" },
        } } });
        const Sse = struct {
            bw: *std.http.BodyWriter,
            allocator: Allocator,
            is_chat: bool,
            model: []const u8,
            failed: bool = false,
            fn emit(ctx: *anyopaque, text: []const u8, _: u32) bool {
                const self: *@This() = @ptrCast(@alignCast(ctx));
                const chunk = if (self.is_chat)
                    std.fmt.allocPrint(self.allocator, "data: {f}\n\n", .{std.json.fmt(.{
                        .id = id, .object = "chat.completion.chunk", .model = self.model,
                        .choices = .{.{ .index = 0, .delta = .{ .content = text }, .finish_reason = null }},
                    }, .{})}) catch return false
                else
                    std.fmt.allocPrint(self.allocator, "data: {f}\n\n", .{std.json.fmt(.{
                        .id = id, .object = "text_completion", .model = self.model,
                        .choices = .{.{ .index = 0, .text = text, .finish_reason = null }},
                    }, .{})}) catch return false;
                self.bw.writer.writeAll(chunk) catch {
                    self.failed = true;
                    return false;
                };
                self.bw.flush() catch {
                    self.failed = true;
                    return false;
                };
                return true;
            }
        };
        var sse = Sse{ .bw = &bw, .allocator = allocator, .is_chat = is_chat, .model = config.model_name };
        const stats = try rt.generate(tokens, params, max_tokens, .{ .ctx = &sse, .func = Sse.emit });
        if (sse.failed) return;
        const finish = if (stats.stop_reason == .eos) "stop" else "length";
        const tail = try std.fmt.allocPrint(allocator, "data: {f}\n\ndata: [DONE]\n\n", .{std.json.fmt(.{
            .id = id, .object = if (is_chat) "chat.completion.chunk" else "text_completion", .model = config.model_name,
            .choices = .{.{ .index = 0, .delta = .{}, .finish_reason = finish }},
            .usage = .{ .prompt_tokens = stats.prompt_tokens, .completion_tokens = stats.generated_tokens,
                .total_tokens = stats.prompt_tokens + stats.generated_tokens },
        }, .{})});
        try bw.writer.writeAll(tail);
        try bw.end();
        return;
    }

    var text: std.ArrayList(u8) = .empty;
    const Collect = struct {
        text: *std.ArrayList(u8),
        allocator: Allocator,
        fn emit(ctx: *anyopaque, piece: []const u8, _: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.text.appendSlice(self.allocator, piece) catch return false;
            return true;
        }
    };
    var collect = Collect{ .text = &text, .allocator = allocator };
    const stats = try rt.generate(tokens, params, max_tokens, .{ .ctx = &collect, .func = Collect.emit });
    const finish = if (stats.stop_reason == .eos) "stop" else "length";
    const usage = .{ .prompt_tokens = stats.prompt_tokens, .completion_tokens = stats.generated_tokens,
        .total_tokens = stats.prompt_tokens + stats.generated_tokens };
    const timings = .{ .prompt_per_second = stats.prefillRate(), .predicted_per_second = stats.decodeRate(),
        .cached_prompt_tokens = stats.reused_tokens };
    var out: []u8 = undefined;
    if (is_chat) {
        // Separate the reasoning block from the answer, as OpenAI-compatible servers do.
        var content: []const u8 = text.items;
        var reasoning: ?[]const u8 = null;
        if (std.mem.indexOf(u8, content, "</think>")) |end| {
            const start = if (std.mem.indexOf(u8, content[0..end], "<think>")) |o| o + "<think>".len else 0;
            reasoning = chat.trim(content[start..end]);
            content = chat.trim(content[end + "</think>".len ..]);
        }
        out = try std.fmt.allocPrint(allocator, "{f}", .{std.json.fmt(.{
            .id = id, .object = object, .model = config.model_name,
            .choices = .{.{ .index = 0, .message = .{ .role = "assistant", .content = content, .reasoning_content = reasoning },
                .finish_reason = finish }},
            .usage = usage, .timings = timings,
        }, .{})});
    } else {
        out = try std.fmt.allocPrint(allocator, "{f}", .{std.json.fmt(.{
            .id = id, .object = object, .model = config.model_name,
            .choices = .{.{ .index = 0, .text = text.items, .finish_reason = finish }},
            .usage = usage, .timings = timings,
        }, .{})});
    }
    try respondJson(request, .ok, out);
}
