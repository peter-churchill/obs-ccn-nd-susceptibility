#!/bin/bash
# Compare Arrhenius file counts with the laptop (counts from 7 Nov 2026)
while read d n; do
  m=$(ls data/$d 2>/dev/null | wc -l)
  s="ok"; [ "$m" -lt "$n" ] && s="MISSING $((n-m))"
  printf "%-32s laptop %5d  arrhenius %5d  %s\n" "$d" "$n" "$m" "$s"
done <<'L'
arm/bnf/smps 227
arm/ena/met 4506
arm/ena/ndrop 3970
arm/ena/smps 1226
arm/ena/uhsas 3103
arm/epc/met 368
arm/epc/ndrop 354
arm/epc/smps 363
arm/lasic/met 553
arm/lasic/ndrop 516
arm/lasic/smps 518
arm/lasic/uhsas 561
arm/oli/uhsas 956
arm/sgp/met 3346
arm/sgp/ndrop 3121
arm/sgp/smps 3174
cloudnet/arm-bankhead/der 588
cloudnet/arm-bankhead/lwc 588
cloudnet/arm-graciosa/der 181
cloudnet/arm-oliktok/der 895
cloudnet/arm-oliktok/lwc 895
cloudnet/arm-sgp/der 1925
cloudnet/arm-sgp/lwc 1925
cloudnet/cabauw/der 410
cloudnet/cabauw/lwc 410
cloudnet/granada/der 1501
cloudnet/granada/lwc 1501
cloudnet/hyytiala/der 2818
cloudnet/hyytiala/lwc 2818
cloudnet/kenttarova/der 603
cloudnet/kenttarova/lwc 603
cloudnet/mace-head/der 381
cloudnet/mace-head/lwc 381
cloudnet/ny-alesund/der 2908
cloudnet/ny-alesund/lwc 2908
cloudnet/palaiseau/der 1927
cloudnet/palaiseau/lwc 1927
cloudnet/schneefernerhaus/der 2371
cloudnet/schneefernerhaus/lwc 2371
L
echo; echo "--- failures in download logs:"; grep -h '!!!' logs/download_all_*.log 2>/dev/null | sort | uniq -c || true
echo "--- last log lines:"; tail -n 3 $(ls -t logs/download_all_*.log | head -1)
echo "--- Oliktok / Bankhead builds:"
for s in oliktok bankhead; do ls -la results/$s/${s}_cloudnet.nc 2>/dev/null || echo "$s: not built"; grep -h '^S(\|^D_Delta' results/$s/${s}_quicklook.txt 2>/dev/null; done
