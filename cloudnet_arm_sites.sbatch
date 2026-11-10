#!/bin/bash
# Cloudnet droplet number at ARM sites paired with the ARM SMPS (SMPS preferred over UHSAS):
#   Ascension (LASIC 2016-17): der generated from portal categorize files (no portal der) + LASIC SMPS
#   Bankhead (BNF 2024-26):    portal der + BNF SMPS (the earlier Bankhead build had 98 h)
# Download first on the login node (see the commands in the chat / README), then: sbatch cloudnet_arm_sites.sbatch
#SBATCH -A naiss2026-1-3-cpu -p cpu -n 8 -t 03:00:00 -J cn_arm -o logs/cloudnet_arm_sites_%j.log
set -u
QL="--min-cloud-frac 0.5 --max-cloud-base 2000 --max-cloud-top 3000"
mkdir -p results/lasic results/bankhead

python -u cloudnet_make_der.py --cat-dir data/cloudnet/arm-ascension/categorize \
    --der-dir data/cloudnet/arm-ascension/der_generated --jobs 8
python -u cloudnet_build.py --der-dir data/cloudnet/arm-ascension/der_generated --psd data/arm/lasic/smps \
    --site-name "ARM LASIC Ascension (Cloudnet der generated + SMPS)" --out results/lasic/lasic_cloudnet_smps.nc \
  && python -u arm_ena_nd_pnsd.py quicklook results/lasic/lasic_cloudnet_smps.nc $QL > results/lasic/lasic_cloudnet_smps_quicklook.txt 2>&1

python -u cloudnet_build.py --der-dir data/cloudnet/arm-bankhead/der --psd data/arm/bnf/smps \
    --site-name "ARM Bankhead BNF (Cloudnet + SMPS)" --out results/bankhead/bankhead_cloudnet_smps.nc \
  && python -u arm_ena_nd_pnsd.py quicklook results/bankhead/bankhead_cloudnet_smps.nc $QL > results/bankhead/bankhead_cloudnet_smps_quicklook.txt 2>&1

for f in results/lasic/lasic_cloudnet_smps_quicklook.txt results/bankhead/bankhead_cloudnet_smps_quicklook.txt; do
  echo "== $f"; grep -E '^Using|^S\(100|^D_Delta' $f
done
