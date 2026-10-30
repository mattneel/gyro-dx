param(
    [string]$Zig = "zig",
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release"
)
$ErrorActionPreference = "Stop"
$project = Split-Path $PSScriptRoot -Parent
Push-Location $project
try {
    $expectedVersion = (Get-Content -Raw ".zig-version").Trim()
    $actualVersion = (& $Zig version).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualVersion -ne $expectedVersion) {
        throw "Use Zig $expectedVersion. The selected compiler reports $actualVersion."
    }
    & ".\native\build.ps1" -Configuration $Configuration
    if ($LASTEXITCODE -ne 0) { throw "The native build failed." }
    $nativeOutput = Join-Path $project "native/build/$Configuration"
    & $Zig build -Dtarget=x86_64-windows-gnu "-Dnative-lib-dir=$nativeOutput" -Doptimize=ReleaseSafe
    if ($LASTEXITCODE -ne 0) { throw "The Zig build failed." }
    $output = Join-Path $project "zig-out/bin"
    foreach ($name in @("gyro_native.dll", "dstorage.dll", "dstoragecore.dll")) {
        Copy-Item -Force (Join-Path $nativeOutput $name) $output
    }
    Copy-Item -Force -Recurse (Join-Path $nativeOutput "D3D12") $output
    Copy-Item -Force ".\native\build\linalg_smoke.dxil" $output
    Copy-Item -Force ".\native\build\gyro_decode.dxil" $output
    Write-Output (Join-Path $output "gyro-dx.exe")
} finally {
    Pop-Location
}
