#!/bin/bash
# NSF/NCAR aircraft campaigns (EOL low-rate files): SOCRATES 2018, CSET 2015, VOCALS-REx 2008.
# Builds cloud-pass Nd + below-cloud N(>D) files with aircraft_build.py --format ncar, then the quicklooks.
#   SOCRATES: wing UHSAS (LWII) forced; the cabin UHSAS (CVIU) sits behind the CVI inlet
#   CSET:     wing UHSAS (RWOOU), chosen automatically (the PCASP total is empty)
#   VOCALS:   PCASP (117 nm - 2.9 um), no UHSAS on the 2008 C-130
# Data: data/aircraft/<campaign>/lrt/*.nc from the EOL archive. Submit from the repo root: sbatch ncar_aircraft.sbatch
#SBATCH -A naiss2026-1-3-cpu
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J ncar_air
#SBATCH -o logs/ncar_aircraft_%j.log
set -u
mkdir -p results/aircraft logs

build() {   # campaign, site name, extra options
  local c=$1 name=$2; shift 2
  echo "=================== $c"
  python -u aircraft_build.py --format ncar --cloud-dir data/aircraft/$c/lrt --aerosol-dir data/aircraft/$c/lrt \
      --site-name "$name" --out results/aircraft/${c}_aircraft.nc "$@" \
    && python -u arm_ena_nd_pnsd.py quicklook results/aircraft/${c}_aircraft.nc --min-cloud-frac 0.5 \
         > results/aircraft/${c}_aircraft_quicklook.txt 2>&1
}

build socrates "SOCRATES (NSF/NCAR GV, CDP + wing UHSAS)" --ncar-aerosol-var CUHSAS_LWII
build cset     "CSET (NSF/NCAR GV, CDP + wing UHSAS)"
build vocals   "VOCALS-REx (NSF/NCAR C-130, CDP + PCASP)" --ncar-aerosol-probe pcasp

echo "=================== summary"
for c in socrates cset vocals; do
  echo "== $c"
  grep -E "^Using|^S\(100|^D_Delta" results/aircraft/${c}_aircraft_quicklook.txt 2>/dev/null || echo "   (no quicklook - see the build output above)"
done
