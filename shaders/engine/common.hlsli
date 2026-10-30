// Shared helpers for the Gyro DX inference kernels.
// Every kernel uses the bindless root signature: 60 root constants at b0 and
// ResourceDescriptorHeap indices for raw buffers. Waves have 32 lanes.
#ifndef GYRO_COMMON_HLSLI
#define GYRO_COMMON_HLSLI

#define WAVE 32

ByteAddressBuffer srv(uint index) { return ResourceDescriptorHeap[NonUniformResourceIndex(index)]; }
RWByteAddressBuffer uav(uint index) { return ResourceDescriptorHeap[NonUniformResourceIndex(index)]; }

float ldf(RWByteAddressBuffer b, uint element) { return asfloat(b.Load(element * 4)); }
void stf(RWByteAddressBuffer b, uint element, float value) { b.Store(element * 4, asuint(value)); }
float ldf_ro(ByteAddressBuffer b, uint element) { return asfloat(b.Load(element * 4)); }

float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
float silu(float x) { return x / (1.0f + exp(-x)); }
float softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// Reads one byte at an arbitrary byte offset of a raw buffer.
uint load_byte(ByteAddressBuffer b, uint offset) {
    return (b.Load(offset & ~3u) >> ((offset & 3u) * 8u)) & 0xffu;
}

// Reads a little-endian binary16 at a 2-byte aligned offset.
float load_half(ByteAddressBuffer b, uint offset) {
    uint word = b.Load(offset & ~3u);
    return f16tof32((offset & 2u) ? (word >> 16) : (word & 0xffffu));
}

int sext8(uint v) { return int(v << 24) >> 24; }

// Applies an epilogue selected by a root constant.
float epilogue(float v, uint act, float scale) {
    v *= scale;
    if (act == 1) return silu(v);
    if (act == 2) return sigmoid(v);
    return v;
}

#endif
