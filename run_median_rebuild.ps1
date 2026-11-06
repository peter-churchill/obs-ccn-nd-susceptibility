# Rebuild ENA and SGP with the hourly median Nd and quicklook old (mean) vs new (median).
# Run from ARM_Data:   conda activate arm ;  .\run_median_rebuild.ps1
$ErrorActionPreference = "Continue"
python -c "import netCDF4" ; if ($LASTEXITCODE -ne 0) { Write-Host "netCDF4 missing: run 'conda activate arm' first"; exit 1 }
function QL($nc) {
  $b = $nc -replace '\.nc$',''
  python arm_ena_nd_pnsd.py quicklook $nc --min-cloud-frac 0.5 --max-rel-err 0.5 *> "${b}_quicklook.txt"
  python arm_ena_nd_pnsd.py quicklook $nc --min-cloud-frac 0.5 --n-boot 100 *> "${b}_samehours.txt"   # no error screen: same hours in old and new
}
$jobs = @(
  @("ena","smps", "results/ena/ena_2022_2026_smps.nc"),
  @("ena","uhsas","results/ena/ena_2014_2023_uhsas.nc"),
  @("sgp","smps", "results/sgp/sgp_2016_2025_smps.nc")
)
foreach ($j in $jobs) {
  $s,$p,$old = $j; $new = $old -replace '\.nc$','_median.nc'
  Write-Host "$(Get-Date -Format T) building $new"
  python arm_ena_nd_pnsd.py build --nd-dir data/arm/$s/ndrop --psd-dir data/arm/$s/$p --met-dir data/arm/$s/met --nd-hourly median --out $new
  QL $new ; QL $old
  Write-Host "$(Get-Date -Format T) done $new"
  Select-String -Path ($new -replace '\.nc$','_quicklook.txt'),($old -replace '\.nc$','_quicklook.txt') -Pattern '^S\(|^D_Delta'
}
Write-Host "ALL DONE"
