# bruv verified installer for Windows (PowerShell).
# Downloads the artifact plus SHA256SUMS, verifies the checksum, then moves the
# binary atomically into a user-owned directory. No administrator required.

param(
  [string]$ArtifactBase = "https://github.com/bruv/bruv/releases/latest/download"
)

$ErrorActionPreference = "Stop"
$DestDir = "$env:LOCALAPPDATA\bruv\bin"
$Artifact = "bruv-windows-x86_64.exe"

Write-Output "Installing bruv for Windows/x86_64 to $DestDir"
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

$TempDir = New-Item -ItemType Directory -Temp
$ArtifactPath = Join-Path $TempDir.FullName $Artifact
$ChecksumsPath = Join-Path $TempDir.FullName "SHA256SUMS"

Write-Output "Downloading $Artifact..."
Invoke-WebRequest -Uri "$ArtifactBase/$Artifact" -OutFile $ArtifactPath

Write-Output "Downloading SHA256SUMS..."
Invoke-WebRequest -Uri "$ArtifactBase/SHA256SUMS" -OutFile $ChecksumsPath

$Checksums = Get-Content $ChecksumsPath
$Line = $Checksums | Where-Object { $_ -match $Artifact }
if (-not $Line) {
  Write-Error "$Artifact missing from SHA256SUMS"
  exit 1
}
$Expected = ($Line -split '\s+')[0]
$Actual = (Get-FileHash $ArtifactPath -Algorithm SHA256).Hash.ToLower()
if ($Expected.ToLower() -ne $Actual) {
  Write-Error "checksum mismatch for $Artifact`n  expected: $Expected`n  actual:   $Actual"
  exit 1
}
Write-Output "Checksum verified."

$DestPath = Join-Path $DestDir "bruv.exe"
Move-Item -Force $ArtifactPath $DestPath

Write-Output "Installed: $DestPath"
Write-Output "Add $DestDir to your PATH if it is not already there."
