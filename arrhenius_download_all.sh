#!/bin/bash
# Re-download all raw ARM and Cloudnet data on Arrhenius (same streams and periods
# as on the laptop), then build and quicklook Oliktok and Bankhead.
#
# Run from the repository folder, which has the symlinks data/ and results/:
#     cd /nobackup/proj/disk/bolinc/personal/petchu/code/obs-ccn-nd-susceptibility
#     tmux new -s obsdl
#     conda activate analysis
#     bash arrhenius_download_all.sh 2>&1 | tee -a logs/download_all_$(date +%Y%m%d).log
# Detach with Ctrl-b d; reattach with: tmux attach -t obsdl
#
# Every download is resumable (existing files are skipped), so the script can be
# rerun after an interruption. Needs ARM_USER and ARM_TOKEN, read from
# ~/.arm_credentials (chmod 600) if they are not already set.

set -u
cd "$(dirname "$0")"
mkdir -p logs

if [ -z "${ARM_TOKEN:-}" ] && [ -f ~/.arm_credentials ]; then
  # shellcheck disable=SC1090
  source ~/.arm_credentials
fi
if [ -z "${ARM_USER:-}" ] || [ -z "${ARM_TOKEN:-}" ]; then
  echo "ARM_USER / ARM_TOKEN not set (put them in ~/.arm_credentials, chmod 600)."; exit 1
fi
python -c "import netCDF4, xarray, pandas" || { echo "This python lacks netCDF4/xarray: $(which python)"; exit 1; }

arm() {  # folder, datastream, start, end
  echo; echo "=== ARM $2 -> data/arm/$1  ($3 to $4)  $(date +%H:%M)"
  mkdir -p "data/arm/$1"
  python -u arm_ena_nd_pnsd.py download --ds "$2" --start "$3" --end "$4" --out "data/arm/$1" \
    || echo "!!! failed: $2"
}
cn() {  # site, start, end, products...
  local site=$1 start=$2 end=$3; shift 3
  for p in "$@"; do
    echo; echo "=== Cloudnet $site $p  ($start to $end)  $(date +%H:%M)"
    python -u cloudnet_download.py download --site "$site" --product "$p" --start "$start" --end "$end" \
      --out data/cloudnet || echo "!!! failed: $site $p"
  done
}

# --- ARM (smallest and newest first; ENA UHSAS, 23 GB, last) ----------------
arm oli/uhsas   oliaosuhsasM1.b1     2018-01-01 2021-03-31
arm bnf/smps    bnfaossmpsM1.b1      2025-01-01 2026-10-31
arm epc/met     epcmetM1.b1          2023-02-15 2024-02-14
arm epc/ndrop   epcndropmfrsrM1.c1   2023-02-15 2024-02-14
arm epc/smps    epcaossmpsM1.b1      2023-02-15 2024-02-14
arm lasic/met   asimetM1.b1          2016-05-01 2017-11-01
arm lasic/ndrop asindropmfrsrM1.c1   2016-05-01 2017-11-01
arm lasic/smps  asiaossmpsM1.b1      2016-05-01 2017-11-01
arm lasic/uhsas asiaosuhsasM1.b1     2016-04-01 2017-11-01
arm sgp/met     sgpmetE13.b1         2016-11-15 2025-12-31
arm sgp/ndrop   sgpndropmfrsrC1.c1   2016-11-15 2025-12-31
arm sgp/smps    sgpaossmpsE13.b1     2016-11-15 2025-12-31
arm ena/met     enametC1.b1          2014-01-01 2026-04-30
arm ena/ndrop   enandropmfrsrC1.c1   2014-06-01 2026-04-30
arm ena/smps    enaaossmpsC1.b1      2022-11-01 2026-04-30

# --- Cloudnet droplet number (der) and LWP (lwc) ----------------------------
cn arm-oliktok      2018-01-01 2021-03-31 der lwc
cn arm-bankhead     2025-01-01 2026-10-31 der lwc
cn kenttarova       2022-08-01 2025-12-31 der lwc
cn ny-alesund       2017-02-01 2025-12-31 der lwc
cn mace-head        2020-01-01 2021-12-31 der lwc
cn cabauw           2022-01-01 2024-12-31 der lwc
cn granada          2018-01-01 2023-12-31 der lwc
cn palaiseau        2014-01-01 2019-12-31 der lwc
cn schneefernerhaus 2014-01-01 2023-12-31 der lwc
cn hyytiala         2017-11-01 2026-09-08 der lwc
cn arm-sgp          2020-01-01 2025-12-31 der lwc
cn arm-graciosa     2026-01-01 2026-06-30 der

arm ena/uhsas   enaaosuhsasC1.b1     2014-02-01 2023-12-31

# --- Build Oliktok and Bankhead (new sites) ----------------------------------
build() {  # id, cloudnet site, ARM psd folder, label
  echo; echo "=== build $1  $(date +%H:%M)"
  mkdir -p "results/$1"
  python -u cloudnet_build.py --der-dir "data/cloudnet/$2/der" --lwc-dir "data/cloudnet/$2/lwc" \
    --psd "data/arm/$3" --site-name "$4" --out "results/$1/$1_cloudnet.nc" || { echo "!!! build failed: $1"; return; }
  python -u arm_ena_nd_pnsd.py quicklook "results/$1/$1_cloudnet.nc" --min-cloud-frac 0.5 \
    --max-cloud-base 2000 --max-cloud-top 3000 2>&1 | tee "results/$1/$1_quicklook.txt"
}
build oliktok  arm-oliktok  oli/uhsas "ARM Oliktok Point (Cloudnet + UHSAS)"
build bankhead arm-bankhead bnf/smps  "ARM Bankhead National Forest (Cloudnet + SMPS)"

echo; echo "All done $(date)."
