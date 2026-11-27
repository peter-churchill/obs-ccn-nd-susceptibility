#!/bin/bash
# ARM ENA 2026: Cloudnet Nd from the portal der files plus der generated from the 2026 categorize files for the days the
# portal does not cover, paired with the ENA SMPS. Replaces results/ena/ena_cloudnet_smps.nc (the Jan-Jun build is kept
# as ena_cloudnet_smps_janjun.nc). Downloads first: bash ena_cloudnet_2026_get.sh (login node).
#   sbatch ena_cloudnet_2026.sbatch        (repo root)
#SBATCH -A naiss2026-1-3-cpu
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J ena_cn26
#SBATCH -o logs/ena_cloudnet_2026_%j.log
set -eu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate analysis
SITE=data/cloudnet/arm-graciosa
GEN=data/cloudnet_2026/arm-graciosa/der_generated
ALL=$SITE/der_2026_all

# 1. der from categorize, all 2026 days (CloudnetPy, 8 workers; existing outputs skipped)
python -u cloudnet_make_der.py --cat-dir data/cloudnet_2026/arm-graciosa/categorize --der-dir $GEN --tag arm-graciosa --jobs 8

# 2. one folder of links: the portal der where it exists, the generated der for the other 2026 days
mkdir -p $ALL && rm -f $ALL/*_der.nc
n_portal=0; n_gen=0
for f in $SITE/der/2026*_der.nc; do ln -s "$(realpath "$f")" $ALL/; n_portal=$((n_portal + 1)); done
for f in $GEN/2026*_der.nc; do
  d=$(basename "$f" | cut -c1-8)
  if ! ls $ALL/${d}_*_der.nc >/dev/null 2>&1; then ln -s "$(realpath "$f")" $ALL/; n_gen=$((n_gen + 1)); fi
done
echo "der 2026: $n_portal portal days + $n_gen generated days"

# 3. build, quicklook (Cloudnet screens), synthesis figure
[ -f results/ena/ena_cloudnet_smps.nc ] && [ ! -f results/ena/ena_cloudnet_smps_janjun.nc ] && \
  cp results/ena/ena_cloudnet_smps.nc results/ena/ena_cloudnet_smps_janjun.nc && \
  cp results/ena/ena_cloudnet_smps_quicklook.txt results/ena/ena_cloudnet_smps_janjun_quicklook.txt
python -u cloudnet_build.py --der-dir $ALL --lwc-dir $SITE/lwc --psd data/arm/ena/smps \
    --site-name "ARM ENA (Cloudnet 2026, portal + generated der, SMPS)" --out results/ena/ena_cloudnet_smps.nc
python -u arm_ena_nd_pnsd.py quicklook results/ena/ena_cloudnet_smps.nc --min-cloud-frac 0.5 \
    --max-cloud-base 2000 --max-cloud-top 3000 > results/ena/ena_cloudnet_smps_quicklook.txt 2>&1
echo "== Jan-Jun build (before)"; grep -E '^Using|^S\(100|^D_Delta' results/ena/ena_cloudnet_smps_janjun_quicklook.txt || true
echo "== 2026 build (now)";       grep -E '^Using|^S\(100|^D_Delta' results/ena/ena_cloudnet_smps_quicklook.txt || true
python -u plot_s_at_ddelta.py 2>&1 | grep -E 'ENA|Saved'
