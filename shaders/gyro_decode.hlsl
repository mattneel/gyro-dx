// Scalar Gyro reference decoder. This kernel does not apply Hadamard rotation.
// Format: AgentionAI SPEC.md v1.1, revision 587a5269848eb07996c9c1433f9ad638604eb4d8.
// Copyright 2026 AgentionAI (Laurent Zuijdwijk). Apache-2.0 formulas and codebook.
// Preserve third_party/agention-gyro/LICENSE and NOTICE with derivative distributions.
//
// t0: Packed blocks with zero padding to a multiple of four bytes.
// t1: The normative 8192-byte codebook, as little-endian binary16 values.
// u0: Float32 output, 512 bytes per block.
// b0: Four uint32 root constants in the order below.
// Dispatch: ceil(block_count / 64) groups on the X axis.
// Each thread decodes one block. Unsupported IDs or invalid spans produce no writes.
// The host must validate input arguments before dispatch.

ByteAddressBuffer Input : register(t0);
ByteAddressBuffer Codebook : register(t1);
RWByteAddressBuffer Output : register(u0);

cbuffer DecodeConstants : register(b0)
{
    uint input_byte_offset;
    uint block_count;
    uint type_id;
    uint output_byte_offset;
};

bool Format(uint id, out uint bits, out uint steps, out uint block_bytes)
{
    bits = 0;
    steps = 0;
    block_bytes = 0;
    switch (id)
    {
        case 144: bits = 8; steps = 32; block_bytes = 34; return true;
        case 145: bits = 6; steps = 32; block_bytes = 26; return true;
        case 146: bits = 7; steps = 32; block_bytes = 30; return true;
        case 147: bits = 5; steps = 64; block_bytes = 42; return true;
        case 148: bits = 6; steps = 64; block_bytes = 50; return true;
        case 149: bits = 7; steps = 64; block_bytes = 58; return true;
        case 150: bits = 5; steps = 32; block_bytes = 22; return true;
        case 151: bits = 4; steps = 32; block_bytes = 18; return true;
        default: return false;
    }
}

uint LoadByte(uint byte_offset)
{
    uint word = Input.Load(byte_offset & ~3u);
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}

uint State(uint stream_offset, uint bits, uint steps, uint step)
{
    if (type_id == 144u)
    {
        return (LoadByte(stream_offset + ((step + 31u) % 32u)) << 8u) |
               LoadByte(stream_offset + step);
    }
    uint stream_bytes = steps * bits / 8u;
    uint bit_offset = (steps - 1u - step) * bits;
    uint first = bit_offset >> 3u;
    uint word = LoadByte(stream_offset + first) |
        (LoadByte(stream_offset + ((first + 1u) % stream_bytes)) << 8u) |
        (LoadByte(stream_offset + ((first + 2u) % stream_bytes)) << 16u);
    return (word >> (bit_offset & 7u)) & 65535u;
}

float CodebookValue(uint half_index)
{
    uint word = Codebook.Load((half_index >> 1u) * 4u);
    uint half_bits = (word >> ((half_index & 1u) * 16u)) & 65535u;
    return f16tof32(half_bits);
}

[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    uint block = thread_id.x;
    if (block >= block_count) return;

    uint bits, steps, block_bytes;
    if (!Format(type_id, bits, steps, block_bytes)) return;

    uint input_bytes, codebook_bytes, output_bytes;
    Input.GetDimensions(input_bytes);
    Codebook.GetDimensions(codebook_bytes);
    Output.GetDimensions(output_bytes);
    if ((input_bytes & 3u) != 0u || codebook_bytes < 8192u) return;
    if (input_byte_offset > input_bytes || output_byte_offset > output_bytes) return;
    if ((output_byte_offset & 3u) != 0u) return;
    if (block_count > (input_bytes - input_byte_offset) / block_bytes) return;
    if (block_count > (output_bytes - output_byte_offset) / 512u) return;

    uint base = input_byte_offset + block * block_bytes;
    uint half_scale = LoadByte(base) | (LoadByte(base + 1u) << 8u);
    precise float scale = f16tof32(half_scale);
    uint destination = output_byte_offset + block * 512u;
    uint pair_count = steps == 64u ? 1u : 2u;
    for (uint step = 0u; step < steps; ++step)
    {
        uint state = State(base + 2u, bits, steps, step);
        uint hash = (state & 32767u) * 0x9e3779b1u;
        for (uint pair = 0u; pair < pair_count; ++pair)
        {
            uint index = pair == 0u ? hash >> 21u : (hash >> 10u) & 2047u;
            precise float first = scale * CodebookValue(index * 2u);
            precise float second = scale * CodebookValue(index * 2u + 1u);
            Output.Store(destination, asuint(first));
            Output.Store(destination + 4u, asuint(second));
            destination += 8u;
        }
    }
}
