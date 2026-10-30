const std = @import("std");
const gguf = @import("gguf.zig");

pub const Error = gguf.ParseError || error{
    UnsupportedArchitecture, IncompleteExpertSet, InvalidBudget, InsufficientBudget,
};

pub const Projection = struct {
    name: []const u8,
    type_id: u32,
    input_width: u64,
    output_width: u64,
    file_offset: u64,
    bytes: u64,
};

pub fn expertBundle(model: gguf.Model, layer: u32, expert: u32) Error![3]Projection {
    if (!std.mem.eql(u8, try model.text("general.architecture"), "qwen4exp")) return error.UnsupportedArchitecture;
    if (layer >= try model.number("qwen4exp.block_count")) return error.InvalidExpert;
    var result: [3]Projection = undefined;
    for ([_][]const u8{ "gate", "up", "down" }, 0..) |kind, i| {
        var name_buffer: [128]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buffer, "blk.{d}.ffn_{s}_exps.weight", .{ layer, kind }) catch unreachable;
        const tensor = model.tensor(name) orelse return error.IncompleteExpertSet;
        if (tensor.dims[2] != try model.number("qwen4exp.expert_count")) return error.IncompleteExpertSet;
        const range = try tensor.expertRange(expert);
        result[i] = .{ .name = tensor.name, .type_id = tensor.type_id, .input_width = tensor.dims[0],
            .output_width = tensor.dims[1], .file_offset = range.file_offset, .bytes = range.bytes };
    }
    return result;
}

pub const Summary = struct {
    total_tensor_bytes: u64 = 0,
    expert_bytes: u64 = 0,
    resident_expert_allocation_bytes: u64 = 0,
    disk_table_bytes: u64 = 0,
    fixed_weight_bytes: u64 = 0,
    max_expert_bundle_bytes: u64 = 0,
    min_expert_bundle_bytes: u64 = std.math.maxInt(u64),
    expert_slots: u64 = 0,
    layers: u32,
    experts_per_layer: u32,
    active_experts: u32,
    cold_bytes_per_token: u64 = 0,
};

pub fn summarize(model: gguf.Model) Error!Summary {
    try model.validateGyro();
    if (!std.mem.eql(u8, try model.text("general.architecture"), "qwen4exp")) return error.UnsupportedArchitecture;
    const layers = try model.number("qwen4exp.block_count");
    const experts = try model.number("qwen4exp.expert_count");
    const active = try model.number("qwen4exp.expert_used_count");
    if (layers == 0 or layers > 4096 or experts == 0 or experts > 65536 or active == 0 or active > experts)
        return error.InvalidMetadata;
    var summary = Summary{ .layers = @intCast(layers), .experts_per_layer = @intCast(experts), .active_experts = @intCast(active) };
    for (model.tensors) |t| {
        const bytes = t.byte_length orelse return error.UnsupportedTensorType;
        summary.total_tensor_bytes = try add(summary.total_tensor_bytes, bytes);
        if (t.isExpert()) summary.expert_bytes = try add(summary.expert_bytes, bytes);
        if (std.mem.eql(u8, t.name, "per_layer_token_embd.weight")) {
            summary.disk_table_bytes = try add(summary.disk_table_bytes, bytes);
        }
    }
    var validated_expert_bytes: u64 = 0;
    for (0..@intCast(layers)) |layer| {
        const bundle = try expertBundle(model, @intCast(layer), 0);
        var bytes: u64 = 0;
        for (bundle) |projection| bytes = try add(bytes, projection.bytes);
        summary.max_expert_bundle_bytes = @max(summary.max_expert_bundle_bytes, bytes);
        summary.min_expert_bundle_bytes = @min(summary.min_expert_bundle_bytes, bytes);
        summary.cold_bytes_per_token = try add(summary.cold_bytes_per_token, try mul(bytes, active));
        const allocated = (try add(bytes, 65535)) & ~@as(u64, 65535);
        summary.resident_expert_allocation_bytes = try add(summary.resident_expert_allocation_bytes, try mul(allocated, experts));
        validated_expert_bytes = try add(validated_expert_bytes, try mul(bytes, experts));
    }
    if (validated_expert_bytes != summary.expert_bytes) return error.IncompleteExpertSet;
    summary.fixed_weight_bytes = summary.total_tensor_bytes - summary.expert_bytes - summary.disk_table_bytes;
    summary.expert_slots = try mul(layers, experts);
    return summary;
}

pub const Budget = struct {
    available_bytes: u64,
    reserve_bytes: u64,
    fixed_weight_bytes: u64,
    resident_weight_bytes: u64,
    all_experts_fit: bool,
    allocation_strategy: enum { sized_resident_buffers, uniform_cache_slots },
    cache_slots: u64,
    cache_slot_bytes: ?u64,
    cache_allocated_bytes: u64,
    minimum_layer_bytes: u64,
    remaining_bytes: u64,
};

/// available_bytes is the usable local budget. reserve_bytes covers all non-weight allocations.
/// Resident buffers use actual layer sizes. Spill slots use the largest bundle.
/// Both modes include 64 KiB committed-buffer granularity.
pub fn budget(summary: Summary, available_bytes: u64, reserve_bytes: u64) Error!Budget {
    if (available_bytes == 0 or reserve_bytes >= available_bytes) return error.InvalidBudget;
    if (summary.max_expert_bundle_bytes == 0 or summary.expert_slots == 0 or
        summary.resident_expert_allocation_bytes == 0 or
        summary.resident_expert_allocation_bytes < summary.expert_bytes) return error.InvalidMetadata;
    const base = try add(reserve_bytes, summary.fixed_weight_bytes);
    if (base >= available_bytes) return error.InsufficientBudget;
    const slot_bytes = (try add(summary.max_expert_bundle_bytes, 65535)) & ~@as(u64, 65535);
    const capacity = available_bytes - base;
    const all_fit = summary.resident_expert_allocation_bytes <= capacity;
    const slots = if (all_fit) summary.expert_slots else @min(summary.expert_slots, capacity / slot_bytes);
    if (slots < summary.active_experts) return error.InsufficientBudget;
    const allocated = if (all_fit) summary.resident_expert_allocation_bytes else try mul(slots, slot_bytes);
    return .{
        .available_bytes = available_bytes,
        .reserve_bytes = reserve_bytes,
        .fixed_weight_bytes = summary.fixed_weight_bytes,
        .resident_weight_bytes = try add(summary.fixed_weight_bytes, summary.expert_bytes),
        .all_experts_fit = all_fit,
        .allocation_strategy = if (all_fit) .sized_resident_buffers else .uniform_cache_slots,
        .cache_slots = slots,
        .cache_slot_bytes = if (all_fit) null else slot_bytes,
        .cache_allocated_bytes = allocated,
        .minimum_layer_bytes = try mul(summary.active_experts, slot_bytes),
        .remaining_bytes = capacity - allocated,
    };
}

fn add(a: u64, b: u64) Error!u64 { return std.math.add(u64, a, b) catch error.IntegerOverflow; }
fn mul(a: u64, b: u64) Error!u64 { return std.math.mul(u64, a, b) catch error.IntegerOverflow; }

test "budget reserves fixed weights and rounds each independent cache buffer" {
    const summary = Summary{ .layers = 48, .experts_per_layer = 256, .active_experts = 10,
        .max_expert_bundle_bytes = 1305600, .expert_slots = 12288,
        .expert_bytes = 16043212800, .resident_expert_allocation_bytes = 16106127360,
        .fixed_weight_bytes = 5000000000 };
    const result = try budget(summary, 24 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
    try std.testing.expect(result.all_experts_fit);
    try std.testing.expect(result.allocation_strategy == .sized_resident_buffers);
    try std.testing.expect(result.cache_allocated_bytes > summary.expert_bytes);
    try std.testing.expectError(error.InsufficientBudget, budget(summary, 6000000000, 1000000000));
    try std.testing.expectError(error.InvalidBudget, budget(summary, 10, 10));
}
