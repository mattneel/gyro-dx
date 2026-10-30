cmake_minimum_required(VERSION 3.24)
project(gyro_native LANGUAGES CXX)
if(NOT WIN32)
    message(FATAL_ERROR "The native backend requires Windows.")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "The native backend requires a 64-bit target.")
endif()

set(GYRO_AGILITY_DIR "" CACHE PATH "Extracted Microsoft.Direct3D.D3D12 1.722.0-preview package")
set(GYRO_DIRECTSTORAGE_DIR "" CACHE PATH "Extracted Microsoft.Direct3D.DirectStorage 1.3.0 package")
set(GYRO_DXC_DIR "" CACHE PATH "Extracted Windows DXC v1.10.2605.37 release")
foreach(required_path
    "${GYRO_AGILITY_DIR}/build/native/include/d3d12.h"
    "${GYRO_DIRECTSTORAGE_DIR}/native/include/dstorage.h")
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR "Missing SDK header: ${required_path}")
    endif()
endforeach()

add_library(gyro_native SHARED gyro_native.cpp)
target_compile_features(gyro_native PRIVATE cxx_std_20)
target_compile_definitions(gyro_native PRIVATE GYRO_NATIVE_EXPORTS)
target_include_directories(gyro_native BEFORE PRIVATE
    "${GYRO_AGILITY_DIR}/build/native/include"
    "${GYRO_DIRECTSTORAGE_DIR}/native/include")
target_include_directories(gyro_native PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(gyro_native PRIVATE d3d12 dxgi
    "${GYRO_DIRECTSTORAGE_DIR}/native/lib/x64/dstorage.lib")
if(MSVC)
    target_compile_options(gyro_native PRIVATE /W4 /EHsc)
endif()

add_executable(gyro-native-smoke smoke.cpp)
target_compile_features(gyro-native-smoke PRIVATE cxx_std_20)
target_link_libraries(gyro-native-smoke PRIVATE gyro_native)
if(MSVC)
    target_compile_options(gyro-native-smoke PRIVATE /W4 /EHsc)
endif()

add_custom_command(TARGET gyro-native-smoke POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:gyro-native-smoke>/D3D12"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${GYRO_AGILITY_DIR}/build/native/bin/x64/D3D12Core.dll"
        "$<TARGET_FILE_DIR:gyro-native-smoke>/D3D12/D3D12Core.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${GYRO_AGILITY_DIR}/build/native/bin/x64/d3d12SDKLayers.dll"
        "$<TARGET_FILE_DIR:gyro-native-smoke>/D3D12/d3d12SDKLayers.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${GYRO_DIRECTSTORAGE_DIR}/native/bin/x64/dstorage.dll"
        "${GYRO_DIRECTSTORAGE_DIR}/native/bin/x64/dstoragecore.dll"
        "$<TARGET_FILE_DIR:gyro-native-smoke>")

if(GYRO_DXC_DIR)
    set(GYRO_SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/linalg_smoke.dxil")
    add_custom_command(OUTPUT "${GYRO_SHADER_OUT}"
        COMMAND "${GYRO_DXC_DIR}/bin/x64/dxc.exe"
            -T cs_6_10 -E main -HV 2021 -enable-16bit-types
            -I "${GYRO_DXC_DIR}/inc"
            "${CMAKE_CURRENT_SOURCE_DIR}/../shaders/linalg_smoke.hlsl"
            -Fo "${GYRO_SHADER_OUT}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../shaders/linalg_smoke.hlsl"
        VERBATIM)
    add_custom_target(gyro-linalg-shader ALL DEPENDS "${GYRO_SHADER_OUT}")
    set(GYRO_DECODE_OUT "${CMAKE_CURRENT_BINARY_DIR}/gyro_decode.dxil")
    add_custom_command(OUTPUT "${GYRO_DECODE_OUT}"
        COMMAND "${GYRO_DXC_DIR}/bin/x64/dxc.exe"
            -T cs_6_0 -E main -Ges -Gis -O3
            "${CMAKE_CURRENT_SOURCE_DIR}/../shaders/gyro_decode.hlsl"
            -Fo "${GYRO_DECODE_OUT}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../shaders/gyro_decode.hlsl"
        VERBATIM)
    add_custom_target(gyro-decode-shader ALL DEPENDS "${GYRO_DECODE_OUT}")
endif()
