#!/bin/bash
# Rebuild NDROP / FM-120 sites with the hourly median Nd, then quicklook old (mean) and new (median) side by side.
cd "$(dirname "$0")"
PY=python3
ql () { $PY arm_ena_nd_pnsd.py quicklook "$1" --min-cloud-frac 0.5 --max-rel-err 0.5 > "${1%.nc}_quicklook.txt" 2>&1; }
run () { # site psd old_name
  s=$1; p=$2; old=$3; new=${old%.nc}_median.nc
  echo "$(date +%T) START $new"
  $PY arm_ena_nd_pnsd.py build --nd-dir data/arm/$s/ndrop --psd-dir data/arm/$s/$p --met-dir data/arm/$s/met --nd-hourly median --out $new && ql $new
  [ -f "${old%.nc}_quicklook.txt" ] || ql $old
  echo "$(date +%T) DONE $new"
}
$PY epcape_build.py --fm data/epcape_soledad/epcape_fm120_5min_compact.csv --sems-dir data/epcape_soledad/sems --logs-dir data/epcape_soledad/sems_logs --nd-hourly median --out results/epcape_soledad/epcape_soledad_hourly_median.nc && ql results/epcape_soledad/epcape_soledad_hourly_median.nc
ql results/epcape_soledad/epcape_soledad_hourly.nc; echo "$(date +%T) DONE soledad"
run epc smps results/epcape_pier/epc_2023_2024_smps.nc
run lasic smps results/lasic/lasic_2016_2017_smps.nc
run lasic uhsas results/lasic/lasic_2016_2017_uhsas.nc
run ena smps results/ena/ena_2022_2026_smps.nc
run ena uhsas results/ena/ena_2014_2023_uhsas.nc
run sgp smps results/sgp/sgp_2016_2025_smps.nc
echo "$(date +%T) ALL DONE"
