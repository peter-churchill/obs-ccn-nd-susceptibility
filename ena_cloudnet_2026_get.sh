#!/bin/bash
# ARM ENA 2026: downloads for extending the Cloudnet + SMPS pairing beyond the portal der files (Jan-Jun 2026).
# Run on the LOGIN node (needs internet), from the repo root, with ARM credentials loaded:
#   source ~/.arm_credentials; bash ena_cloudnet_2026_get.sh
# Then: sbatch ena_cloudnet_2026.sbatch
set -u
START=2026-01-01; END=2026-12-31

# Cloudnet portal der (re-check: the portal may have added months since the last download; existing files are skipped)
python -u cloudnet_download.py download --site arm-graciosa --product der --start $START --end $END --out data/cloudnet
python -u cloudnet_download.py download --site arm-graciosa --product lwc --start $START --end $END --out data/cloudnet
# Cloudnet categorize for 2026, to generate der where the portal has none (kept apart from the 2015-19 categorize files)
python -u cloudnet_download.py download --site arm-graciosa --product categorize --start $START --end $END --out data/cloudnet_2026
# ARM SMPS and NDROP after the last downloaded day (existing files are skipped)
python -u arm_ena_nd_pnsd.py download --ds enaaossmpsC1.b1 --start 2026-04-01 --end $END --out data/arm/ena/smps
python -u arm_ena_nd_pnsd.py download --ds enandropmfrsrC1.c1 --start 2026-04-01 --end $END --out data/arm/ena/ndrop

echo "== on disk now"
for d in data/cloudnet/arm-graciosa/der data/cloudnet_2026/arm-graciosa/categorize data/arm/ena/smps data/arm/ena/ndrop; do
  echo "$d: $(ls $d | wc -l) files, last $(ls $d | tail -1)"
done
