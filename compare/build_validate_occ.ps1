# Build OCC BREP validator
# Prerequisites: vcpkg install opencascade --triplet x64-windows
param(
    [string]$VcpkgRoot = "D:\repos\vcpkg",
    [string]$Triplet = "x64-windows",
    [string]$Output = "target\debug\validate_brep_occ.exe"
)

$ErrorActionPreference = "Stop"

$OccRoot = "$VcpkgRoot\installed\$Triplet"
$OccInc = "$OccRoot\include"
$OccLib = "$OccRoot\lib"
$OccBin = "$OccRoot\bin"

if (-not (Test-Path $OccInc)) {
    Write-Error "OCC not found at $OccRoot. Run: vcpkg install opencascade --triplet $Triplet"
    exit 1
}

Write-Host "OCC found at $OccRoot"
Write-Host "Compiling validate_brep_occ.cpp..."

$Libs = @(
    "TKBRep.lib", "TKTopAlgo.lib", "TKGeomBase.lib", "TKGeomAlgo.lib",
    "TKG2d.lib", "TKG3d.lib", "TKMath.lib", "TKernel.lib"
)
$LibFlags = ($Libs | ForEach-Object { "$OccLib\$_" }) -join " "

$Cmd = "cl /EHsc /std:c++17 /MD /I`"$OccInc`" compare\validate_brep_occ.cpp /link /LIBPATH:`"$OccLib`" $LibFlags /OUT:`"$Output`""
Write-Host $Cmd
Invoke-Expression $Cmd

if ($LASTEXITCODE -ne 0) {
    Write-Error "Compilation failed"
    exit 1
}

Write-Host "Build OK: $Output"

# Copy OCC DLLs alongside the exe
$OutDir = Split-Path $Output -Parent
Copy-Item "$OccBin\*.dll" $OutDir -ErrorAction SilentlyContinue

Write-Host "Ready. Run: $Output compare/out/step-*.brep"
