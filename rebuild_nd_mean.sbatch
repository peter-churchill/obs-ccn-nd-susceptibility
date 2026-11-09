#!/bin/bash
#SBATCH -A naiss2026-1-3-cpu
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J ndmean
#SBATCH -o logs/nd_mean_rebuild_%j.log
source ~/miniconda3/etc/profile.d/conda.sh
conda activate analysis
cd /nobackup/proj/disk/bolinc/personal/petchu/code/obs-ccn-nd-susceptibility
python -u rebuild_nd_mean.py --jobs 8
