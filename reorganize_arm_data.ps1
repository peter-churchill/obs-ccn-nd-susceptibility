# Reorganize the ARM_Data folder.
#
#   Dry run (shows what would move, changes nothing):
#       powershell -ExecutionPolicy Bypass -File .\reorganize_arm_data.ps1 -DryRun
#   Do it:
#       powershell -ExecutionPolicy Bypass -File .\reorganize_arm_data.ps1
#
# Nothing is deleted. Code, README, environment.yml, .gitignore and .git stay
# where they are, so the GitHub repository is unaffected.
#
# Layout afterwards:
#   ARM_Data\
#     *.py, README.md, environment.yml      code (the git repository)
#     data\arm\{ena,lasic,sgp,epc}\...      raw ARM files
#     data\cloudnet\<site>\{der,lwc}        raw Cloudnet files
#     data\ebas\                            EBAS size distributions
#     data\epcape_soledad\                  UCSD fog monitor, SEMS zips, SEMS logs
#     results\<site>\                       built .nc, quicklook .png/.csv, result workbooks
#     docs\                                 observation_options.xlsx
#     logs\                                 download logs
#     _old\                                 superseded files, for you to delete

param([switch]$DryRun)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$moved = 0

function Move-One([string]$src, [string]$dstDir) {
    # Move a file or folder into $dstDir (created if needed). Skips if missing
    # or if something with the same name is already there.
    if (-not (Test-Path -LiteralPath $src)) { return }
    $target = Join-Path $dstDir (Split-Path -Leaf $src)
    if (Test-Path -LiteralPath $target) { Write-Host "  skip (exists): $target"; return }
    if ($DryRun) { Write-Host "  would move $src -> $dstDir\" ; $script:moved++; return }
    New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    Move-Item -LiteralPath $src -Destination $dstDir
    $script:moved++
}

function Move-Pattern([string]$pattern, [string]$dstDir) {
    Get-ChildItem -Path $root -File -Filter $pattern -ErrorAction SilentlyContinue |
        ForEach-Object { Move-One $_.FullName $dstDir }
}

Write-Host "Raw data"
foreach ($d in "ndrop", "smps", "uhsas", "met") { Move-One "data\$d" "data\arm\ena" }
foreach ($s in "lasic", "sgp", "epc")           { Move-One "data\$s" "data\arm" }
foreach ($f in "sems", "sems_logs", "epcape_fm120_5min_compact.csv", "README_fm120_compact.txt") {
    Move-One "epcape\$f" "data\epcape_soledad"
}

Write-Host "Results"
Move-Pattern "sgp_2018_*"  "results\sgp\pilot_2018"
Move-Pattern "ena_*"       "results\ena"
Move-Pattern "lasic_*"     "results\lasic"
Move-Pattern "sgp_*"       "results\sgp"
Move-Pattern "epc_*"       "results\epcape_pier"
Move-Pattern "hyytiala_*"  "results\hyytiala"
Get-ChildItem -Path (Join-Path $root "epcape") -File -Filter "epcape_soledad_hourly*" -ErrorAction SilentlyContinue |
    ForEach-Object { Move-One $_.FullName "results\epcape_soledad" }

Write-Host "Documents"
Move-One "observation_options.xlsx"            "docs"
Move-One "observation_options_2026-11-04.xlsx" "_old"
Move-One "reorganize_petchu.sh"                "_old"

# The old epcape folder should now be empty; remove it only if it is.
$ep = Join-Path $root "epcape"
if ((Test-Path $ep) -and -not $DryRun -and -not (Get-ChildItem $ep -Force)) { Remove-Item $ep }

if ($DryRun) { Write-Host "`nDry run: $moved items would move. Run again without -DryRun to do it." }
else         { Write-Host "`nDone: $moved items moved." }
