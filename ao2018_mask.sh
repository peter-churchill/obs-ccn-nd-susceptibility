#!/bin/bash
# AO2018 Oden: rebuild FSSP fog, CVI residual and Cloudnet pairs without and with the ship-exhaust pollution mask,
# run the quicklooks and collect S(100 nm) and D_Delta in results/ao2018/mask_summary.txt.
#   sbatch ao2018_mask.sh        (from the repository root, with the analysis env active)
#SBATCH -A naiss2026-1-3-cpu -p cpu -n 4 -t 01:00:00 -J ao2018_mask -o logs/ao2018_mask_%j.log
set -u
mkdir -p logs results/ao2018
D=data/ao2018
MASK=$D/ao2018-aerosol-pollution-mask.csv
CVI=$(ls $D/*cvi*.nc* 2>/dev/null | head -1)
QL="--min-cloud-frac 0.5"
QLCN="--min-cloud-frac 0.5 --max-cloud-base 1000 --max-cloud-top 2000"
[ -f "$MASK" ] || { echo "missing $MASK"; exit 1; }

for v in raw clean; do
  M=""; S=""
  [ $v = clean ] && { M="--pollution-mask $MASK"; S="_clean"; }
  ( python -u ao2018_build.py --fssp-dir $D/fssp --dmps $D/ao2018-aerosol-dmps.csv --out results/ao2018/ao2018_fog$S.nc $M \
      && python -u arm_ena_nd_pnsd.py quicklook results/ao2018/ao2018_fog$S.nc $QL ) > logs/ao2018_fog$S.txt 2>&1 &
  if [ -n "$CVI" ]; then
    ( python -u ao2018_cvi_build.py "$CVI" --out results/ao2018/ao2018_cvi_residual$S.nc $M \
        && python -u arm_ena_nd_pnsd.py quicklook results/ao2018/ao2018_cvi_residual$S.nc $QL ) > logs/ao2018_cvi$S.txt 2>&1 &
  fi
done
# Cloudnet runs one after the other: both share the generated der folder and its temporary file
( for S in "" _clean; do
    M=""; [ "$S" = _clean ] && M="--pollution-mask $MASK"
    ( python -u ao2018_cloudnet_build.py --categorize-dir $D/cloudnet/categorize --dmps $D/ao2018-aerosol-dmps.csv \
        --out results/ao2018/ao2018_cloudnet$S.nc $M \
        && python -u arm_ena_nd_pnsd.py quicklook results/ao2018/ao2018_cloudnet$S.nc $QLCN ) > logs/ao2018_cloudnet$S.txt 2>&1
  done ) &
wait

SUM=results/ao2018/mask_summary.txt
: > $SUM
for b in fog cvi cloudnet; do
  for S in "" _clean; do
    f=logs/ao2018_$b$S.txt
    [ -f $f ] || continue
    echo "== $b ${S:-_raw}" >> $SUM
    grep -E '^  *[0-9.]+ +[0-9.]+% |clean flag|^Pollution mask|^Paired|^Using|^S\(100|^D_Delta|Traceback|Error' $f >> $SUM
  done
done
cat $SUM
