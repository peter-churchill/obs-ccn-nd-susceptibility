#!/bin/bash
# Solar-zenith sensitivity for the NDROP ground sites: quicklook with --max-sza 90 (all daylight hours; NDROP is
# daytime only) and --max-sza 60. Outputs get _sza90/_sza60 suffixes, so the standard quicklooks are not overwritten.
# Run from the repository root:  bash run_sza_check.sh   (or sbatch, see below)
#SBATCH -A naiss2026-1-3-cpu -p cpu -n 4 -t 01:00:00 -J sza_check -o logs/sza_check_%j.log
set -u
mkdir -p logs
FILES="results/ena/ena_2022_2026_smps.nc results/sgp/sgp_2016_2025_smps.nc
       results/lasic/lasic_2016_2017_smps.nc results/epcape_pier/epc_2023_2024_smps.nc"
SUM=results/sza_check_summary.txt
: > "$SUM"
for f in $FILES; do
  [ -f "$f" ] || { echo "missing $f" | tee -a "$SUM"; continue; }
  b=$(basename "$f" .nc)
  python -u arm_ena_nd_pnsd.py quicklook "$f" --min-cloud-frac 0.5 --max-sza 90 > "logs/${b}_qlook_all.txt" 2>&1 &
  python -u arm_ena_nd_pnsd.py quicklook "$f" --min-cloud-frac 0.5 --max-sza 60 > "logs/${b}_qlook_sza60.txt" 2>&1 &
done
wait
for f in $FILES; do
  b=$(basename "$f" .nc)
  for v in all sza60; do
    [ -f "logs/${b}_qlook_${v}.txt" ] || continue
    echo "== $b ($v)" >> "$SUM"
    grep -E '^Using|^S\(100|^D_Delta|solar zenith' "logs/${b}_qlook_${v}.txt" >> "$SUM"
  done
done
cat "$SUM"
