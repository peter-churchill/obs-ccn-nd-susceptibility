#!/bin/bash
# ARM ENA: Cloudnet der from the portal (2026) paired with the ENA SMPS (SMPS preferred over UHSAS), quicklook with
# the Cloudnet screens, then the S(D_Delta) figure.   sbatch ena_cloudnet_smps.sbatch   (repository root, analysis env)
#SBATCH -A naiss2026-1-3-cpu -p cpu -n 1 -t 02:00:00 -J ena_cn_smps -o logs/ena_cloudnet_smps_%j.log
set -eu
python -u cloudnet_build.py --der-dir data/cloudnet/arm-graciosa/der --psd data/arm/ena/smps \
    --site-name "ARM ENA (Cloudnet 2026 + SMPS)" --out results/ena/ena_cloudnet_smps.nc
python -u arm_ena_nd_pnsd.py quicklook results/ena/ena_cloudnet_smps.nc --min-cloud-frac 0.5 \
    --max-cloud-base 2000 --max-cloud-top 3000 > results/ena/ena_cloudnet_smps_quicklook.txt 2>&1
grep -E '^Using|^S\(100|^D_Delta' results/ena/ena_cloudnet_smps_quicklook.txt
python -u plot_s_at_ddelta.py 2>&1 | grep -E 'ENA|Saved'
