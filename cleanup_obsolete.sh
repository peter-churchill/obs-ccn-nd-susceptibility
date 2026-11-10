#!/bin/bash
# List (default) or delete files made obsolete by the Nov 2026 work. Run from the repository root.
#   bash cleanup_obsolete.sh            # list candidates with sizes; nothing is deleted
#   bash cleanup_obsolete.sh --delete   # delete exactly the listed files and folders
# Kept on purpose: *_median* builds (median Nd sensitivity), unmasked AO2018 builds (cited in the synthesis notes),
# UHSAS data and builds (parked, not discarded), _sza*/_relerr quicklooks, raw ARM/EBAS/Cloudnet der data,
# and everything a queued job still needs (arm-ascension categorize, BNF SMPS, ENA SMPS, arm-graciosa der).
# Obsolete scripts are removed with git rm on the laptop, not here.
set -u
DEL=0; [ "${1:-}" = "--delete" ] && DEL=1
total=0
item(){  # item "reason" path...
  local reason="$1"; shift
  for p in "$@"; do
    [ -e "$p" ] || continue
    local kb; kb=$(du -sk "$p" 2>/dev/null | cut -f1)
    total=$((total + kb))
    printf "%9.1f MB  %-60s %s\n" "$(echo "$kb/1024" | bc -l)" "$p" "$reason"
    [ $DEL = 1 ] && rm -rf -- "$p"
  done
}
shopt -s nullglob globstar

echo "== Superseded results"
item "pre-mean-rebuild backup (hourly mean is now standard)" results/**/*_orig.nc results/**/*_orig_quicklook.* results/**/*_orig.*
item "editor/backup copy" results/**/*.bak *.py.bak *.sh.bak
item "2023-only test builds (superseded by 2014-23 / 2022-26)" results/ena/ena_2023_smps* results/ena/ena_2023_uhsas*
item "2018 SGP pilot (superseded by 2016-25 build)" results/sgp/pilot_2018
item "ENA Cloudnet 2015-19 + UHSAS (SMPS version used instead)" results/ena/ena_cloudnet.nc results/ena/ena_cloudnet_quicklook.*
item "FIMS aircraft builds (judged unreliable)" results/aircraft/*_fims*
item "AO2018 fog D_min sensitivity tests" results/ao2018/ao2018_fog_dmin*

echo "== Data that can be re-downloaded and is no longer read"
item "arm-graciosa categorize 2015-19 (der generated; re-downloadable)" data/cloudnet/arm-graciosa/categorize
item "unfinished download" data/**/*.part
item "CloudnetPy temporary file" data/**/_tmp_categorize.nc

printf "\nTotal: %.1f GB %s\n" "$(echo "$total/1048576" | bc -l)" "$([ $DEL = 1 ] && echo deleted || echo '(nothing deleted; rerun with --delete)')"
