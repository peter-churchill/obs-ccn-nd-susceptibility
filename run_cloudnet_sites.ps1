# Cloudnet droplet number + EBAS size distributions at seven European sites:
# download (resumable), build, quicklook. Run from ARM_Data:
#     conda activate arm; Set-ExecutionPolicy -Scope Process Bypass; .\run_cloudnet_sites.ps1
# Only one site:   .\run_cloudnet_sites.ps1 -Only palaiseau
# Skip downloads:  .\run_cloudnet_sites.ps1 -NoDownload
#
# Cloudnet Nd reads ~1.8x NDROP at SGP: the closure diameter printed here is raw
# (uncalibrated); rerun quicklook with --nd-scale 1.8 for the calibrated value.
# Results go to results\<site>\.

param([string]$Only = "", [switch]$NoDownload)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$ProgressPreference = 'SilentlyContinue'

# The builds need the 'arm' environment (netCDF4). A separate powershell.exe
# can start in conda's base environment, so check before doing anything.
python -c "import netCDF4, xarray" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "This python has no netCDF4: $((Get-Command python).Source)"
  Write-Host "Activate the environment and run the script in the same window:"
  Write-Host "    conda activate arm"
  Write-Host "    Set-ExecutionPolicy -Scope Process Bypass"
  Write-Host "    .\run_cloudnet_sites.ps1 -NoDownload"
  exit 1
}
$thredds = "https://thredds.nilu.no/thredds/fileServer/ebas/"

# Cloudnet id, label, Cloudnet date range (overlap with EBAS), EBAS files
$sites = @(
  @{ id = "palaiseau"; label = "Palaiseau SIRTA (Cloudnet + GRIMM SMPS)"; start = "2014-01-01"; end = "2019-12-31";
     ebas = @(
       "FR0020R.20140101000000.20161019074121.smps.particle_number_size_distribution.aerosol.1y.1h.FR01L_smps_GRIMM_SMPS+5416.FR01L_GRIMM_SMPS+5416_GIF.lev2.nc",
       "FR0020R.20150101000000.20191220115038.smps.particle_number_size_distribution.aerosol.1y.1h.FR01L_smps_GRIMM_SMPS+5416.FR01L_GRIMM_SMPS+5416_GIF.lev2.nc",
       "FR0020R.20170101000000.20191008154251.smps.particle_number_size_distribution.aerosol.1y.1h.FR01L_smps_GRIMM_SMPS+5416.FR01L_GRIMM_SMPS+5416_GIF.lev2.nc",
       "FR0020R.20190101000000.20200525181020.smps.particle_number_size_distribution.aerosol.1y.1h.FR01L_smps_GRIMM_SMPS+5416.FR01L_GRIMM_SMPS+5416_GIF.lev2.nc") },
  @{ id = "cabauw"; label = "Cabauw (Cloudnet + TROPOS MPSS)"; start = "2022-01-01"; end = "2024-12-31";
     ebas = @("NL0011R.20220101000000.20251010110756.smps.particle_number_size_distribution.pm10.3y.1h.NL05L_TROPOS_Cabauw_TNO.NL05L_MPSS_TNO.lev2.nc") },
  @{ id = "granada"; label = "Granada (Cloudnet + UGR SMPS)"; start = "2018-01-01"; end = "2023-12-31";
     ebas = @("ES0020U.20170101000000.20250107152204.smps...7y.1h.ES08L_TSI_3938_UGR.ES08L_smps_01.lev2.nc") },
  @{ id = "schneefernerhaus"; label = "Schneefernerhaus (Cloudnet + UBA SMPS)"; start = "2014-01-01"; end = "2023-12-31";
     ebas = @(
       "DE0054R.20140101000000.20160121000000.smps...2y.1h.DE08L_SMPS_UBA_ZUGSPITZE.DE08L_TSI3080_MODIFIED_BY_DE08L.lev2.nc",
       "DE0054R.20160101000140.20180903000000.smps.particle_number_size_distribution.pm10.76w.1h.DE08L_TSI_SMPS_UBA_ZUGSPITZE.DE08L_TSI_MODIFIED.lev2.nc",
       "DE0054R.20170714000000.20180508000000.smps...6mo.1h.DE08L_TROPOS_SMPS_UBA_ZUGSPITZE.DE08L_TROPOS.lev2.nc",
       "DE0054R.20190101000000.20240402075650.smps...5y.1h.DE08L_TROPOS_SMPS_UBA_ZUGSPITZE.DE08L_TROPOS_2019.lev2.nc") },
  @{ id = "mace-head"; label = "Mace Head (Cloudnet + SMPS)"; start = "2020-01-01"; end = "2021-12-31";
     ebas = @("IE0031R.20200101000000.20220915091513.smps.particle_number_size_distribution.aerosol.2y.1h.IE04L_smps_01.IE04L_smps.lev2.nc") },
  # Radar at Ny-Alesund village (19 m); DMPS at Zeppelin (475 m, ~2 km away; Stockholm University)
  @{ id = "ny-alesund"; label = "Ny-Alesund (Cloudnet + Zeppelin SU DMPS)"; start = "2017-02-01"; end = "2025-12-31";
     ebas = @("NO0042G.20110101000000.20260915091959.dmps...15y.1h.SE02L_TINTIN.SE02L_SOP_TINTIN_Rev_2026.lev2.nc") },
  # Radar at Kenttarova (345 m); DMPS at Pallas Sammaltunturi (565 m, ~6 km away; total inlet)
  @{ id = "kenttarova"; label = "Kenttarova (Cloudnet + Pallas total DMPS)"; start = "2022-08-01"; end = "2025-12-31";
     ebas = @("FI0096G.20230104080000.20251027122806.dmps...2y.1h.FI01L_FMI_DMPS_total.FI01L_dmps_PAL_1.lev2.nc") }
)

foreach ($s in $sites) {
  if ($Only -and $s.id -ne $Only) { continue }
  $id = $s.id
  Write-Host "`n=================== $id ==================="
  $ebasDir = "data\ebas\$id"
  New-Item -ItemType Directory -Force -Path $ebasDir, "results\$id" | Out-Null

  if (-not $NoDownload) {
    python cloudnet_download.py download --site $id --product der --start $s.start --end $s.end --out data\cloudnet
    python cloudnet_download.py download --site $id --product lwc --start $s.start --end $s.end --out data\cloudnet
    foreach ($f in $s.ebas) {
      $dest = Join-Path $ebasDir $f
      if (Test-Path -LiteralPath $dest) { Write-Host "  EBAS have: $f"; continue }
      Write-Host "  EBAS get:  $f"
      # download to .part and rename only when complete, so an interrupted file is never kept
      $part = "$dest.part"
      try { Invoke-WebRequest ($thredds + $f) -OutFile $part; Move-Item -LiteralPath $part $dest -Force }
      catch { Write-Host "  EBAS download failed: $f  ($($_.Exception.Message))"; Remove-Item -LiteralPath $part -ErrorAction SilentlyContinue }
    }
  }

  $psd = Get-ChildItem -LiteralPath $ebasDir -Filter "*.nc" | ForEach-Object { $_.FullName }
  if (-not $psd) { Write-Host "  no EBAS files for $id, skipping build"; continue }
  $out = "results\$id\${id}_cloudnet.nc"
  python cloudnet_build.py --der-dir "data\cloudnet\$id\der" --lwc-dir "data\cloudnet\$id\lwc" --psd $psd --site-name $s.label --out $out
  if (Test-Path $out) {
    python arm_ena_nd_pnsd.py quicklook $out --min-cloud-frac 0.5 --max-cloud-base 2000 --max-cloud-top 3000 2>&1 |
      Tee-Object -FilePath "results\$id\${id}_quicklook.txt"
  }
}
Write-Host "`nAll done. Each site's printed summary is in results\<site>\<site>_quicklook.txt"
