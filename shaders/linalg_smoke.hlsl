// DXC v1.10.2605.37: -T cs_6_10 -E main -HV 2021 -enable-16bit-types
// This shader validates LinAlg. It does not implement a model layer.
#include <hlsl/dx/linalg.h>

ByteAddressBuffer input_a : register(t0);
ByteAddressBuffer input_b : register(t1);
RWByteAddressBuffer output_c : register(u0);

using namespace dx::linalg;
using MatrixA = Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave>;
using MatrixB = Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave>;

[WaveSize(32)]
[numthreads(32, 1, 1)]
void main()
{
    MatrixA a = MatrixA::Load(input_a, 0, 32, MatrixLayout::RowMajor);
    MatrixB b = MatrixB::Load(input_b, 0, 32, MatrixLayout::RowMajor);
    Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave> c =
        Multiply<ComponentType::F32>(a, b);
    c.Store(output_c, 0, 64, MatrixLayout::RowMajor);
}
