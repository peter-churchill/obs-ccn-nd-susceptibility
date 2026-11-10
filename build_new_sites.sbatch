#!/bin/bash
#SBATCH -A naiss2026-1-3-cpu
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J newsites
#SBATCH -o logs/build_new_sites_%j.log
source ~/miniconda3/etc/profile.d/conda.sh
conda activate analysis
cd /nobackup/proj/disk/bolinc/personal/petchu/code/obs-ccn-nd-susceptibility
mkdir -p results/mosaic
QL="--min-cloud-frac 0.5 --max-cloud-base 2000 --max-cloud-top 3000"
for p in smps uhsas; do
  echo "=== MOSAiC $p"
  python -u cloudnet_build.py --der-dir data/cloudnet/rv-polarstern/der --lwc-dir data/cloudnet/rv-polarstern/lwc --psd data/arm/mos/$p --site-name "MOSAiC Polarstern (Cloudnet + $p)" --out results/mosaic/mosaic_cloudnet_$p.nc \
    && python -u arm_ena_nd_pnsd.py quicklook results/mosaic/mosaic_cloudnet_$p.nc $QL > results/mosaic/mosaic_cloudnet_${p}_quicklook.txt 2>&1
  grep -E "^Using|^S\(|^D_Delta" results/mosaic/mosaic_cloudnet_${p}_quicklook.txt
done
python -u cloudnet_ebas_add.py --sites norunda bucharest maido --stage build
