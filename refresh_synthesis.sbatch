#!/bin/bash
# Re-run everything downstream of the source list after adding observations:
#   scripts:   source_ntot.py, plot_s_at_ddelta.py, data_timeline.py
#   notebooks: susceptibility_estimators, aircraft_overview, model_obs_s_dsb, site_timeseries_pnsd (executed in place)
# Submit from the repo root: sbatch refresh_synthesis.sbatch      (log: logs/refresh_synthesis_<jobid>.log)
# site_timeseries_pnsd builds hourly PNSD caches for sites it has not seen before, so the first run is the slow one.
#SBATCH -A naiss2026-1-3-cpu
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -J refresh
#SBATCH -o logs/refresh_synthesis_%j.log
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate analysis
mkdir -p logs

echo "=== scripts  $(date +%H:%M)"
python -u source_ntot.py      | tail -5
python -u plot_s_at_ddelta.py | grep -E "Saved|no N_total|no quicklook"
python -u data_timeline.py    | grep -E "Saved|-> built"

for nb in susceptibility_estimators aircraft_overview model_obs_s_dsb site_timeseries_pnsd; do
  echo "=== $nb  $(date +%H:%M)"
  if jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb > logs/nb_$nb.log 2>&1; then
    echo "   ok"
  else
    echo "   FAILED - last lines of logs/nb_$nb.log:"; tail -15 logs/nb_$nb.log
  fi
done
echo "=== done  $(date +%H:%M)"
