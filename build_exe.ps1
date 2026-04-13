$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$releaseDir = Join-Path $projectRoot "release"
$distDir = Join-Path $projectRoot "dist\\ConvexOptAgent"

py -m PyInstaller --noconfirm --clean convexopt_tutor_agent.spec

if (Test-Path $releaseDir) {
    Remove-Item -Recurse -Force $releaseDir
}

New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
Copy-Item -Path (Join-Path $distDir "*") -Destination $releaseDir -Recurse -Force

Write-Host "Release updated at: $releaseDir"
