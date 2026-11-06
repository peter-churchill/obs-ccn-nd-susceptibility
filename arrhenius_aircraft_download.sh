#!/bin/bash
# Download ARM Aerial Facility (G-1 aircraft) campaign data with cloud droplet and
# aerosol size distributions, for in-situ Nd vs below-cloud CCN.
#
# Run on Arrhenius from the repository folder (data/ is the symlink to data/obs):
#     tmux new -s aircraft
#     conda activate analysis
#     cd /nobackup/proj/disk/bolinc/personal/petchu/code/obs-ccn-nd-susceptibility
#     bash arrhenius_aircraft_download.sh 2>&1 | tee -a logs/aircraft_download_$(date +%Y%m%d).log
# Detach with Ctrl-b d. Resumable: files already present are skipped.
# Needs ARM_USER / ARM_TOKEN (read from ~/.arm_credentials if not set).
#
# Campaigns (G-1, 1-Hz data, ~20-40 research flights each):
#   ena  ACE-ENA     Azores marine Sc, Jun-Jul 2017 + Jan-Feb 2018 (over our ENA ground site)
#   sgp  HI-SCALE    Oklahoma continental Cu, Apr-May + Aug-Sep 2016 (over SGP)
#   cor  CACTI       Argentina orographic/continental, Nov-Dec 2018
#   acx  ACAPEX      California coast marine, Jan-Mar 2015
#   nsa  ACME-V      Alaska Arctic, Jun-Sep 2015
#   mao  GoAmazon    Amazon, Feb-Mar + Sep-Oct 2014
# Per campaign: the merged 1-Hz product, merged cloud and aerosol size distributions
# where they exist, the individual droplet (FCDP) and aerosol (FIMS/UHSAS/PCASP)
# probes, and navigation (position, altitude).

set -u
cd "$(dirname "$0")"
mkdir -p logs
if [ -z "${ARM_TOKEN:-}" ] && [ -f ~/.arm_credentials ]; then source ~/.arm_credentials; fi
if [ -z "${ARM_USER:-}" ] || [ -z "${ARM_TOKEN:-}" ]; then
  echo "ARM_USER / ARM_TOKEN not set (put them in ~/.arm_credentials, chmod 600)."; exit 1
fi

get() {  # campaign folder, start, end, datastreams...
  local camp=$1 start=$2 end=$3; shift 3
  for ds in "$@"; do
    echo; echo "=== $camp $ds  ($start to $end)  $(date +%H:%M)"
    mkdir -p "data/aircraft/$camp/$ds"
    python -u arm_ena_nd_pnsd.py download --ds "$ds" --start "$start" --end "$end" \
      --out "data/aircraft/$camp/$ds" || echo "!!! failed: $ds"
  done
}

# Smallest, most useful products first for every campaign; the large merged files last.
get ace-ena  2017-06-01 2018-03-01 enaaafmergedcldsdF1.c1 enaaafmergedaerosolsdF1.c1 enaaaffcdpF1.c1 \
    enaaaffimsF1.b1 enaaafpcaspF1.b1 enaaafcasF1.b1 enaaafnavF1.b1 enaaafccn2colaF1.b1 enaaafinletcviF1.c1
get hi-scale 2016-04-01 2016-10-01 sgpaafmergedcldsdF1.c1 sgpaafmergedaerosolsdF1.c1 sgpaaffcdpF1.c1 \
    sgpaaffimsF1.b1 sgpaafuhsasF1.b1 sgpaafpcaspF1.b1 sgpaafnavF1.b1
get cacti    2018-09-01 2018-12-31 coraafmergedcldsdF1.c1 coraafmergedaerosolsdF1.c1 coraaffcdpF1.c1 \
    coraafpcaspF1.b1 coraafcasF1.b1 coraafnavF1.b1
get acapex   2015-01-01 2015-03-31 acxaaffcdpF1.c1 acxaafuhsasF1.b1 acxaafpcaspF1.b1 acxaafnavF1.b1
get acme-v   2015-05-01 2015-10-31 nsaaaffcdpF1.c1 nsaaafuhsasF1.b1 nsaaafpcaspF1.b1 nsaaafnavF1.b1
get goamazon 2014-02-01 2014-10-31 maoaaffcdpF1.c1 maoaaffimsF1.b1 maoaafuhsasF1.b1 maoaafpcaspF1.b1 maoaafnavF1.b1

# Merged 1-Hz files (0.6-3 GB per campaign): all probes on one time axis
get ace-ena  2017-06-01 2018-03-01 enaaafmergedF1.c1
get hi-scale 2016-04-01 2016-10-01 sgpaafmergedF1.c1
get cacti    2018-09-01 2018-12-31 coraafmergedF1.c1
get acapex   2015-01-01 2015-03-31 acxaafmergedF1.c1
get acme-v   2015-05-01 2015-10-31 nsaaafmergedF1.c1
get goamazon 2014-02-01 2014-10-31 maoaafmergedF1.c1

echo; echo "All done $(date). Sizes:"; du -sh data/aircraft/*
