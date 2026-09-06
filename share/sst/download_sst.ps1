# =============================================================================
# download_sst.ps1 - Download 504 MODIS-Aqua SST granules from NASA OB.DAAC
# Usage:  powershell -ExecutionPolicy Bypass -File .\download_sst.ps1
#
# WHAT CHANGED (fixes over the old version):
#   * Follows redirects  (curl -L) - NASA's getfile links 301 to the real
#     file host; the old script saved the HTML redirect page as the .nc file.
#   * Earthdata Login cookie session (-b/-c) + -u so the redirect flow is
#     authenticated end-to-end.
#   * NO hardcoded credentials - you are prompted for YOUR OWN account.
#   * Deletes leftover stub/HTML files before resume (so -C - cannot append
#     real bytes onto a 162-byte HTML page).
#   * Verifies every file after download and cleans up anything that is not
#     a real NetCDF granule (HTML pages, files < 1 MB, zero-byte files).
#
# Anything that still fails is listed in sst_failures.txt - just re-run the
# script to retry (it is resumable).
# =============================================================================

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# --- Output folders ---------------------------------------------------------
$OutDir  = Join-Path $PSScriptRoot "nc"
$Cookie  = Join-Path $PSScriptRoot "cookies.txt"
$FailLog = Join-Path $PSScriptRoot "sst_failures.txt"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# --- Credentials (prompt; password is masked on screen) ---------------------
$USER = Read-Host "NASA Earthdata username (create one free at https://urs.earthdata.nasa.gov)"
$SecurePass = Read-Host "NASA Earthdata password" -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecurePass)
$PASS = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($BSTR)
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)

if (-not $USER -or -not $PASS) {
    Write-Host "No credentials entered - aborting." -ForegroundColor Red
    exit 1
}

# A real MODIS L3m 8-day 9km global granule is ~10-20 MB. Anything below this
# is almost certainly an HTML error page / stub or a 0-byte file.
$MinValidBytes = 1MB

$urls = @(Get-Content (Join-Path $PSScriptRoot "sst_urls.txt") |
        Where-Object { $_.Trim() -ne "" })
if ($urls.Count -eq 0) {
    Write-Host "sst_urls.txt is empty or missing - aborting." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Found $($urls.Count) granules to fetch (resumable)."
Write-Host ""

$i = 0
$ok = 0
$failed = New-Object System.Collections.Generic.List[string]

foreach ($url in $urls) {
    $i++
    $base = [System.IO.Path]::GetFileName(($url -replace "[?#].*$", ""))
    $dest = Join-Path $OutDir $base

    # Skip files that are already complete; only retry broken/small ones.
    if (Test-Path $dest) {
        $size = (Get-Item $dest).Length
        if ($size -ge $MinValidBytes) {
            $ok++
            Write-Host "[$i/$($urls.Count)] $base  already downloaded (skip)" -ForegroundColor DarkGray
            continue
        }
        # A previous attempt left a stub/HTML/empty file: remove it so curl's
        # resume (-C -) restarts cleanly instead of corrupting the file.
        Remove-Item $dest -Force
    }

    Write-Host "[$i/$($urls.Count)] $base" -NoNewline

    # -L : follow redirects (THE critical fix)
    # --location-trusted : keep sending credentials when the redirect jumps hosts
    # -b/-c : Earthdata Login cookie session so the EDL redirect stays logged in
    # -C - : resume an interrupted download
    # --retry --retry-all-errors : transient network errors are retried
    curl.exe --fail -sS --location --location-trusted --connect-timeout 20 `
        --retry 5 --retry-all-errors -C - `
        -u "$($USER):$($PASS)" -b $Cookie -c $Cookie `
        -o "$dest" $url
    $code = $LASTEXITCODE

    if ($code -ne 0) {
        Write-Host "  FAILED (curl exit $code)" -ForegroundColor Yellow
        $failed.Add($url)
        continue
    }

    $size = if (Test-Path $dest) { (Get-Item $dest).Length } else { 0 }
    if ($size -lt $MinValidBytes) {
        Write-Host "  FAILED (got only $size bytes - not a valid granule)" -ForegroundColor Yellow
        Remove-Item $dest -Force -ErrorAction SilentlyContinue
        $failed.Add($url)
        continue
    }

    $ok++
    Write-Host "  ok ($([math]::Round($size / 1MB, 1)) MB)" -ForegroundColor Green
}

# --- Final validation pass -------------------------------------------------
Write-Host ""
Write-Host "==============================================="
Write-Host "Downloaded $ok of $($urls.Count) files"
$sizeMB = (Get-ChildItem $OutDir -Filter *.nc | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Total size: {0:N1} MB" -f $sizeMB)

$badFiles = @(Get-ChildItem $OutDir -Filter *.nc | Where-Object { $_.Length -lt $MinValidBytes })
if ($badFiles.Count -gt 0) {
    $badFiles | ForEach-Object { Remove-Item $_.FullName -Force }
    Write-Host "Removed $($badFiles.Count) leftover stub/HTML file(s)." -ForegroundColor Yellow
}

if ($failed.Count -gt 0) {
    $failed | Set-Content -Path $FailLog
    Write-Host "Wrote $($failed.Count) failed URL(s) to sst_failures.txt"
    Write-Host "Re-run this script to retry them (it resumes)." -ForegroundColor Yellow
} elseif (Test-Path $FailLog) {
    Remove-Item $FailLog -Force
}

# --- Package for upload ----------------------------------------------------
Write-Host ""
$zip = Join-Path $PSScriptRoot "sst_all.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path (Join-Path $OutDir "*.nc") -DestinationPath $zip -CompressionLevel Fastest
Write-Host "Zipped to: $zip"
Write-Host ""
Write-Host "NEXT: upload '$zip' (or the 'nc' folder) to Google Drive, share the link"
Write-Host "('Anyone with the link'), and send it to Ricci."
Write-Host ""