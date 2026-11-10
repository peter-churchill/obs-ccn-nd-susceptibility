#!/bin/bash
# ARM ENA: Cloudnet der generated for 2015-19 (cloudnet_make_der.py) paired with the ENA UHSAS, quicklook with the
# Cloudnet screens, then the S(D_Delta) figure.   sbatch ena_cloudnet.sbatch   (repository root, analysis env active)
#SBATCH -A naiss2026-1-3-cpu -p cpu -n 1 -t 03:00:00 -J ena_cloudnet -o logs/ena_cloudnet_%j.log
set -eu
python -u cloudnet_build.py --der-dir data/cloudnet/arm-graciosa/der_generated --psd data/arm/ena/uhsas \
    --site-name "ARM ENA (Cloudnet der generated 2015-19 + UHSAS)" --out results/ena/ena_cloudnet.nc
python -u arm_ena_nd_pnsd.py quicklook results/ena/ena_cloudnet.nc --min-cloud-frac 0.5 \
    --max-cloud-base 2000 --max-cloud-top 3000 > results/ena/ena_cloudnet_quicklook.txt 2>&1
grep -E '^Using|^S\(100|^D_Delta' results/ena/ena_cloudnet_quicklook.txt
python -u plot_s_at_ddelta.py 2>&1 | grep -E 'ENA|Saved'
