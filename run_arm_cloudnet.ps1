# Cloudnet droplet number + ARM aerosol at ARM sites without NDROP:
# Oliktok (UHSAS, 2018-2021) and Bankhead (SMPS, 2025-2026). Run from ARM_Data:
#     conda activate arm; Set-ExecutionPolicy -Scope Process Bypass; .\run_arm_cloudnet.ps1
# One site:        .\run_arm_cloudnet.ps1 -Only oliktok
# Skip downloads:  .\run_arm_cloudnet.ps1 -NoDownload
# Needs ARM_USER / ARM_TOKEN in the environment (as for the other ARM downloads).
# Cloudnet Nd is uncalibrated here: report S and the LWP-controlled S, not the closure diameter.

param([string]$Only = "", [switch]$NoDownload)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$ProgressPreference = 'SilentlyContinue'

python -c "import netCDF4, xarray" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "This python has no netCDF4: $((Get-Command python).Source)"
  Write-Host "Run 'conda activate arm' in this window first."
  exit 1
}

$sites = @(
  @{ id = "oliktok"; cloudnet = "arm-oliktok"; arm = "oli"; ds = "oliaosuhsasM1.b1"; kind = "uhsas";
     start = "2018-01-01"; end = "2021-03-31"; label = "ARM Oliktok Point (Cloudnet + UHSAS)" },
  @{ id = "bankhead"; cloudnet = "arm-bankhead"; arm = "bnf"; ds = "bnfaossmpsM1.b1"; kind = "smps";
     start = "2025-01-01"; end = "2026-10-31"; label = "ARM Bankhead National Forest (Cloudnet + SMPS)" }
)

foreach ($s in $sites) {
  if ($Only -and $s.id -ne $Only) { continue }
  $id = $s.id
  Write-Host "`n=================== $id ==================="
  $psdDir = "data\arm\$($s.arm)\$($s.kind)"
  New-Item -ItemType Directory -Force -Path $psdDir, "results\$id" | Out-Null

  if (-not $NoDownload) {
    python cloudnet_download.py download --site $s.cloudnet --product der --start $s.start --end $s.end --out data\cloudnet
    python cloudnet_download.py download --site $s.cloudnet --product lwc --start $s.start --end $s.end --out data\cloudnet
    python arm_ena_nd_pnsd.py download --ds $s.ds --start $s.start --end $s.end --out $psdDir
  }

  $out = "results\$id\${id}_cloudnet.nc"
  python cloudnet_build.py --der-dir "data\cloudnet\$($s.cloudnet)\der" --lwc-dir "data\cloudnet\$($s.cloudnet)\lwc" `
    --psd $psdDir --site-name $s.label --out $out
  if (Test-Path $out) {
    python arm_ena_nd_pnsd.py quicklook $out --min-cloud-frac 0.5 --max-cloud-base 2000 --max-cloud-top 3000 2>&1 |
      Tee-Object -FilePath "results\$id\${id}_quicklook.txt"
  }
}
Write-Host "`nAll done. Summaries in results\<site>\<site>_quicklook.txt"
