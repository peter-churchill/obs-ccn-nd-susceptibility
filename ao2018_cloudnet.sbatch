#!/bin/bash
#SBATCH -A naiss2026-1-3-cpu
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J ao2018cn
#SBATCH -o logs/ao2018_cloudnet_%j.log
source ~/miniconda3/etc/profile.d/conda.sh
conda activate analysis
cd /nobackup/proj/disk/bolinc/personal/petchu/code/obs-ccn-nd-susceptibility
python -u ao2018_cloudnet_build.py --categorize-dir data/ao2018/cloudnet --dmps data/ao2018/ao2018-aerosol-dmps.csv --out results/ao2018/ao2018_cloudnet.nc --force
python -u arm_ena_nd_pnsd.py quicklook results/ao2018/ao2018_cloudnet.nc --min-cloud-frac 0.5 --max-cloud-base 1000 --max-cloud-top 2000 > results/ao2018/ao2018_cloudnet_quicklook.txt 2>&1
grep -E "^Using|^S\(|^D_Delta" results/ao2018/ao2018_cloudnet_quicklook.txt
