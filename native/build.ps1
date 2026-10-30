param(
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release"
)
$ErrorActionPreference = "Stop"
$packageRoot = Join-Path $PSScriptRoot "packages"
$buildRoot = Join-Path $PSScriptRoot "build"
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-PinnedPackage([string]$Name, [string]$Url, [string]$Sha256) {
    $archive = Join-Path $packageRoot ($Name + ".zip")
    $destination = Join-Path $packageRoot $Name
    if (-not (Test-Path $archive)) {
        Invoke-WebRequest -Uri $Url -OutFile $archive
    }
    # .NET hashing avoids a dependency on the Microsoft.PowerShell.Utility module path.
    $stream = [System.IO.File]::OpenRead($archive)
    try {
        $digest = [System.Security.Cryptography.SHA256]::Create().ComputeHash($stream)
    } finally {
        $stream.Dispose()
    }
    $actual = ([System.BitConverter]::ToString($digest) -replace "-", "").ToLowerInvariant()
    if ($actual -ne $Sha256) {
        throw "Checksum mismatch for $archive. Remove this archive before a retry."
    }
    if (-not (Test-Path $destination)) {
        [System.IO.Compression.ZipFile]::ExtractToDirectory($archive, $destination)
    }
    return $destination
}

$agility = Get-PinnedPackage "agility-1.722.0-preview" `
    "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/1.722.0-preview/microsoft.direct3d.d3d12.1.722.0-preview.nupkg" `
    "bf9915dbb39f7c97018d42b36e65469f11c864b805e3bc9a773eba364208cdc3"
$storage = Get-PinnedPackage "directstorage-1.3.0" `
    "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.directstorage/1.3.0/microsoft.direct3d.directstorage.1.3.0.nupkg" `
    "8ab6c1082537565388b79050d94fdd7f5f437cd0f1d909e7b19f786042b97177"
$dxc = Get-PinnedPackage "dxc-1.10.2605.37" `
    "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.10.2605.37/dxc_2026_08_11.zip" `
    "18da85ccac4d1736209bad960f0a124928a5c5a6046882e331c605d66744a889"

cmake -S $PSScriptRoot -B $buildRoot -A x64 `
    "-DGYRO_AGILITY_DIR=$agility" `
    "-DGYRO_DIRECTSTORAGE_DIR=$storage" `
    "-DGYRO_DXC_DIR=$dxc"
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }
cmake --build $buildRoot --config $Configuration
if ($LASTEXITCODE -ne 0) { throw "Native compilation failed." }
Write-Output (Join-Path $buildRoot "$Configuration/gyro-native-smoke.exe")
