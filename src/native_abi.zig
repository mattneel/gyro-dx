//! Stable C ABI mirror of native/include/gyro_native.h.
//! The current Zig master no longer provides @cImport.
pub const gd_context = opaque {};
pub const gd_file = opaque {};
pub const gd_buffer = opaque {};
pub const gd_result = i32;

pub const gd_options = extern struct {
    adapter_index: u32,
    enable_debug: u32,
    require_linalg: u32,
    staging_bytes: u32,
    queue_capacity: u32,
};

pub const gd_caps = extern struct {
    abi_version: u32,
    vendor_id: u32,
    device_id: u32,
    shader_model: u32,
    linalg_tier: u32,
    wave16_fp16_fp32_flags: u32,
    vector_fp16_fp32_flags: u32,
    experimental_features_enabled: u32,
    dedicated_video_bytes: u64,
    local_budget_bytes: u64,
    local_usage_bytes: u64,
    adapter_name: [128]u8,
};

pub const gd_range = extern struct {
    file_offset: u64,
    buffer_offset: u64,
    bytes: u32,
    reserved: u32,
};

pub extern fn gd_create(options: *const gd_options, context: *?*gd_context) gd_result;
pub extern fn gd_get_caps(context: *gd_context, caps: *gd_caps) gd_result;
pub extern fn gd_open_file(context: *gd_context, utf8_path: [*:0]const u8, file: *?*gd_file) gd_result;
pub extern fn gd_file_size(file: ?*gd_file, bytes: *u64) gd_result;
pub extern fn gd_create_buffer(context: *gd_context, bytes: u64, buffer: *?*gd_buffer) gd_result;
pub extern fn gd_read_ranges(context: *gd_context, file: ?*gd_file, buffer: ?*gd_buffer,
    ranges: [*]const gd_range, count: u32, storage_ticket: *u64) gd_result;
pub extern fn gd_poll(context: *gd_context, storage_completed: *u64, compute_completed: *u64) gd_result;
pub extern fn gd_wait_storage(context: *gd_context, ticket: u64, timeout_ms: u32) gd_result;
pub extern fn gd_readback(context: *gd_context, buffer: ?*gd_buffer, offset: u64,
    bytes: u32, destination: [*]u8, timeout_ms: u32) gd_result;
pub extern fn gd_validate_linalg(context: *gd_context, dxil: [*]const u8, bytes: usize,
    max_abs_error: *f32, timeout_ms: u32) gd_result;
pub extern fn gd_validate_gyro(context: *gd_context, dxil: [*]const u8, dxil_bytes: usize,
    codebook: [*]const u8, codebook_bytes: usize, encoded: [*]const u8, packed_bytes: usize,
    type_id: u32, expected: [*]const f32, value_count: u32,
    max_abs_error: *f32, timeout_ms: u32) gd_result;
pub extern fn gd_release_buffer(context: *gd_context, buffer: *gd_buffer, timeout_ms: u32) gd_result;
pub extern fn gd_release_file(context: *gd_context, file: *gd_file, timeout_ms: u32) gd_result;
pub extern fn gd_destroy(context: *gd_context, timeout_ms: u32) gd_result;

comptime {
    const assert = @import("std").debug.assert;
    assert(@sizeOf(gd_options) == 20);
    assert(@sizeOf(gd_caps) == 184);
    assert(@sizeOf(gd_range) == 24);
    assert(@offsetOf(gd_caps, "adapter_name") == 56);
}
