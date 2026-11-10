#!/bin/bash
# Generate Cloudnet der for ARM Graciosa (ENA) from portal categorize files, then Cloudnet vs NDROP by sun angle.
# Download the categorize files first on the login node (cloudnet_download.py, see cloudnet_make_der.py).
#SBATCH -A naiss2026-1-3-cpu -p cpu -n 8 -t 04:00:00 -J ena_der -o logs/ena_der_%j.log
set -u
python -u cloudnet_make_der.py --cat-dir data/cloudnet/arm-graciosa/categorize \
    --der-dir data/cloudnet/arm-graciosa/der_generated --jobs 8
python -u nd_compare.py --ndrop-dir data/arm/ena/ndrop --der-dir data/cloudnet/arm-graciosa/der_generated \
    --site-lat 39.0916 --site-lon -28.0257 --out results/ena/ena_cloudnet_vs_ndrop.csv
