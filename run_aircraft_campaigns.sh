#!/bin/bash
# Build + quicklook every ARM aircraft campaign under data/aircraft/ (except ace-ena and activate, done separately).
# Cloud probe: merged cloud SD if present, else FCDP, else CAS. Aerosol: every probe found
# (merged aerosol SD, UHSAS, PCASP, FIMS; FIMS read as dN/dlogDp). Summary -> results/aircraft/campaign_summary.txt
cd "$(dirname "$0")"; mkdir -p results/aircraft logs
python -c "import numpy, xarray, netCDF4" 2>/dev/null || { echo "Wrong python ($(which python)): run conda activate analysis first"; exit 1; }
summary=results/aircraft/campaign_summary.txt; : > $summary
declare -A SITE=([hi-scale]="HI-SCALE (SGP, G-1)" [cacti]="CACTI (Argentina, G-1)" [acapex]="ACAPEX (Pacific, G-1)" [acme-v]="ACME-V (Alaska, G-1)" [goamazon]="GoAmazon (Manaus, G-1)")
for camp in hi-scale cacti acapex acme-v goamazon; do
  base=data/aircraft/$camp
  [ -d $base ] || { echo "$camp: no folder $base" | tee -a $summary; continue; }
  cld=$(ls -d $base/*mergedcldsd* 2>/dev/null | head -1)
  [ -z "$cld" ] && cld=$(ls -d $base/*fcdp* 2>/dev/null | head -1)
  [ -z "$cld" ] && cld=$(ls -d $base/*cas* 2>/dev/null | grep -v pcasp | head -1)
  [ -z "$cld" ] && { echo "$camp: no cloud-probe folder in $base ($(ls $base | tr '\n' ' '))" | tee -a $summary; continue; }
  for pat in mergedaerosolsd uhsas pcasp fims; do
    aer=$(ls -d $base/*${pat}* 2>/dev/null | head -1); [ -z "$aer" ] && continue
    [ "$(ls $aer | wc -l)" -eq 0 ] && continue
    form=auto; [ $pat = fims ] && form=dndlogdp
    out=results/aircraft/${camp}_aircraft_${pat}.nc; log=logs/aircraft_${camp}_${pat}.log
    echo "=== $camp  cloud=$(basename $cld)  aerosol=$(basename $aer)  form=$form  $(date +%H:%M)"
    if ! python -u aircraft_build.py --cloud-dir $cld --aerosol-dir $aer --aerosol-form $form \
         --site-name "${SITE[$camp]} $pat" --out $out > $log 2>&1; then
      echo "$camp $pat: build failed -> $(tail -1 $log)" | tee -a $summary; continue
    fi
    python -u arm_ena_nd_pnsd.py quicklook $out --min-cloud-frac 0.5 > ${out%.nc}_quicklook.txt 2>&1
    { echo "$camp $pat: $(grep -h 'Paired' $log)"; grep -h 'median N_tot' $log | sed 's/^/    /'
      grep -hE '^S\(|^D_Delta' ${out%.nc}_quicklook.txt | sed 's/^/    /'; } | tee -a $summary
  done
done
echo; echo "Summary in $summary"
