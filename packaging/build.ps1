<#
.SYNOPSIS
    Builds the Image Mosaic Generator installer.

.DESCRIPTION
    Two stages: PyInstaller freezes the app into dist\ImageMosaic, then Inno
    Setup packs that folder into dist\ImageMosaic-<version>-Setup.exe.

.PARAMETER SkipApp
    Reuse the existing dist\ImageMosaic and only recompile the installer.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipApp
)

$ErrorActionPreference = 'Stop'

$PackagingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot  = Split-Path -Parent $PackagingDir
$DistDir      = Join-Path $ProjectRoot 'dist'
$AppDir       = Join-Path $DistDir 'ImageMosaic'
$Python       = Join-Path $ProjectRoot 'venv\Scripts\python.exe'

function Write-Stage($Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

if (-not (Test-Path $Python)) {
    throw "No virtual environment at $Python. Create one and install requirements.txt first."
}

# --- Stage 1: freeze the application --------------------------------------
if (-not $SkipApp) {
    Write-Stage 'Checking build dependencies'
    # -c rather than `-m PyInstaller --version`: redirecting a native
    # command's stderr in PowerShell 5.1 turns a clean exit into an error.
    & $Python -c 'import PyInstaller' | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host '    installing pyinstaller'
        & $Python -m pip install pyinstaller
        if ($LASTEXITCODE -ne 0) { throw 'pip install pyinstaller failed.' }
    } else {
        Write-Host '    pyinstaller present'
    }

    Write-Stage 'Freezing application (PyInstaller)'
    & $Python -m PyInstaller (Join-Path $PackagingDir 'ImageMosaic.spec') `
        --noconfirm `
        --distpath $DistDir `
        --workpath (Join-Path $ProjectRoot 'build')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
}

$AppExe = Join-Path $AppDir 'ImageMosaic.exe'
if (-not (Test-Path $AppExe)) {
    throw "Expected $AppExe. Run without -SkipApp to build it."
}

# --- Stage 2: compile the installer ---------------------------------------
Write-Stage 'Locating Inno Setup'
$IsccCandidates = @(
    (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $Iscc) {
    throw "Inno Setup 6 not found. Install it with: winget install JRSoftware.InnoSetup"
}
Write-Host "    $Iscc"

Write-Stage 'Compiling installer (Inno Setup)'
& $Iscc (Join-Path $PackagingDir 'installer.iss')
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup failed.' }

$Setup = Get-ChildItem (Join-Path $DistDir 'ImageMosaic-*-Setup.exe') |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1

Write-Stage 'Done'
Write-Host ("    {0}  ({1:N1} MB)" -f $Setup.FullName, ($Setup.Length / 1MB))
