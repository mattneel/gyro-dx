#ifndef GYRO_NATIVE_H
#define GYRO_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(GYRO_NATIVE_EXPORTS)
#define GD_API __declspec(dllexport)
#elif defined(_WIN32)
#define GD_API __declspec(dllimport)
#else
#define GD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_context gd_context;
typedef struct gd_file gd_file;
typedef struct gd_buffer gd_buffer;
typedef struct gd_pipeline gd_pipeline;
typedef int32_t gd_result;

typedef struct gd_options {
    uint32_t adapter_index;
    uint32_t enable_debug;
    uint32_t require_linalg;
    uint32_t staging_bytes;
    uint32_t queue_capacity;
} gd_options;

typedef struct gd_caps {
    uint32_t abi_version;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t shader_model;
    uint32_t linalg_tier;
    uint32_t wave16_fp16_fp32_flags;
    uint32_t vector_fp16_fp32_flags;
    uint32_t experimental_features_enabled;
    uint64_t dedicated_video_bytes;
    uint64_t local_budget_bytes;
    uint64_t local_usage_bytes;
    char adapter_name[128];
} gd_caps;

typedef struct gd_range {
    uint64_t file_offset;
    uint64_t buffer_offset;
    uint32_t bytes;
    uint32_t reserved;
} gd_range;

enum { GD_RESOURCE_DEVICE = 0, GD_RESOURCE_HOST = 1 };

typedef struct gd_buffer_info {
    uint32_t srv_index;
    uint32_t uav_index;
    void *mapped;
    uint64_t gpu_address;
    uint64_t allocated_bytes;
} gd_buffer_info;

/* All methods require serialized access to each context.
 * Results use HRESULT semantics. Negative values indicate failure.
 * Buffers remain in COMMON between complete operations.
 * A failed or timed-out destroy/release retains the handle for a later retry.
 * Use separate buffers for cache slots with independent resource lifetimes.
 */
GD_API gd_result gd_create(const gd_options *options, gd_context **out_context);
GD_API gd_result gd_get_caps(gd_context *context, gd_caps *out_caps);
GD_API gd_result gd_open_file(gd_context *context, const char *utf8_path, gd_file **out_file);
GD_API gd_result gd_file_size(gd_file *file, uint64_t *out_bytes);
GD_API gd_result gd_create_buffer(gd_context *context, uint64_t bytes, gd_buffer **out_buffer);

/* This method enqueues raw byte transfers. It does not decode Gyro weights.
 * Ranges must not overlap in the destination. Each range must fit staging_bytes.
 * ERROR_BUSY means that this buffer still has an earlier storage request.
 * The batch waits for this buffer's last compute fence before any GPU work.
 */
GD_API gd_result gd_read_ranges(gd_context *context, gd_file *file, gd_buffer *buffer,
    const gd_range *ranges, uint32_t count, uint64_t *out_storage_ticket);
GD_API gd_result gd_poll(gd_context *context, uint64_t *storage_completed, uint64_t *compute_completed);
GD_API gd_result gd_wait_storage(gd_context *context, uint64_t ticket, uint32_t timeout_ms);
GD_API gd_result gd_readback(gd_context *context, gd_buffer *buffer, uint64_t offset,
    uint32_t bytes, void *destination, uint32_t timeout_ms);

/* The DXIL must come from shaders/linalg_smoke.hlsl with the pinned preview DXC.
 * The result verifies one 16x16x16 wave GEMM against an exact CPU reference.
 * Unsupported hardware returns DXGI_ERROR_UNSUPPORTED.
 */
GD_API gd_result gd_validate_linalg(gd_context *context, const void *dxil, size_t dxil_bytes,
    float *out_max_abs_error, uint32_t timeout_ms);
/* The DXIL must come from shaders/gyro_decode.hlsl.
 * The codebook contains exactly 8192 bytes of normative binary16 values.
 * value_count must be a positive multiple of 128, at most 16777216.
 * packed_bytes must exactly match the selected type and block count.
 * expected contains finite float32 values in the stored, rotated basis.
 * This method returns failure if any output differs bitwise.
 */
GD_API gd_result gd_validate_gyro(gd_context *context, const void *dxil, size_t dxil_bytes,
    const void *codebook, size_t codebook_bytes, const void *packed, size_t packed_bytes,
    uint32_t type_id, const float *expected, uint32_t value_count,
    float *out_max_abs_error, uint32_t timeout_ms);
/* Bindless compute API (requires Shader Model 6.6).
 * Every resource receives a raw SRV and a raw UAV in one shader-visible heap; shaders index
 * ResourceDescriptorHeap with the returned indices. Pipelines share one root signature of
 * 60 root constants at b0. GD_RESOURCE_DEVICE allocates local video memory. GD_RESOURCE_HOST
 * allocates persistently mapped write-back system memory that shaders can read and write.
 * Buffers rely on implicit state promotion: within one submission, read a buffer only through
 * its SRV or only through its UAV. gd_barrier inserts a global UAV barrier.
 * A buffer may receive DirectStorage writes into ranges that no in-flight compute reads.
 */
GD_API gd_result gd_create_resource(gd_context *context, uint64_t bytes, uint32_t kind,
    gd_buffer **out_buffer, gd_buffer_info *out_info);
GD_API gd_result gd_create_pipeline(gd_context *context, const void *dxil, size_t dxil_bytes,
    gd_pipeline **out_pipeline);
GD_API gd_result gd_release_pipeline(gd_context *context, gd_pipeline *pipeline, uint32_t timeout_ms);
GD_API gd_result gd_begin(gd_context *context, uint32_t timeout_ms);
GD_API gd_result gd_dispatch(gd_context *context, gd_pipeline *pipeline, const uint32_t *constants,
    uint32_t constant_count, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
GD_API gd_result gd_barrier(gd_context *context);
GD_API gd_result gd_timestamp(gd_context *context, uint32_t *out_index);
GD_API gd_result gd_submit(gd_context *context, uint64_t *out_compute_ticket);
GD_API gd_result gd_wait_compute(gd_context *context, uint64_t ticket, uint32_t timeout_ms);
GD_API gd_result gd_read_timestamps(gd_context *context, uint32_t count, uint64_t *out_ticks,
    uint64_t *out_frequency);
/* Makes later compute submissions wait on the GPU for a storage ticket. */
GD_API gd_result gd_queue_wait_storage(gd_context *context, uint64_t storage_ticket);
/* Arena variant of gd_read_ranges: the caller names the compute ticket that must finish before
 * DirectStorage writes, normally the last use of the evicted slots. No per-buffer busy check. */
GD_API gd_result gd_read_ranges_after(gd_context *context, gd_file *file, gd_buffer *buffer,
    const gd_range *ranges, uint32_t count, uint64_t compute_ticket, uint64_t *out_storage_ticket);
GD_API gd_result gd_release_buffer(gd_context *context, gd_buffer *buffer, uint32_t timeout_ms);
GD_API gd_result gd_release_file(gd_context *context, gd_file *file, uint32_t timeout_ms);
GD_API gd_result gd_destroy(gd_context *context, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif
#endif
