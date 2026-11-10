#!/bin/bash
# Download the cloud-probe, size-distribution, housekeeping and 1-s merge files for NAAMES, ORACLES, CAMP2Ex and ATom.
#   bash nasa_aircraft_get.sh --dry-run    # sizes only (run this first)
#   bash nasa_aircraft_get.sh              # download (login node, in tmux; resumable)
# Selection: droplet probes (CDP/FCDP/CAS/PDI), cloud flags, SMPS/DMA first plus LAS/UHSAS for large particles,
# inlet flags, housekeeping (altitude, T, p) and the 1-s merges.
DRY="$1"
get(){ python nasa_aircraft_download.py get "$1" --short-name "$2" --match "$3" ${4:+--exclude "$4"} $DRY 2>&1 \
         | grep -vE 'FutureWarning|self\[|size = float'; }

get NAAMES  NAAMES_Cloud_AircraftInSitu_Data      "CDP|CLOUDFLAG|LARGE-CAS"
get NAAMES  NAAMES_Aerosol_AircraftInSitu_Data    "SMPS|LARGE-LAS|MICROPHYSICAL"
get NAAMES  NAAMES_MetNav_AircraftInSitu_Data     "HSKPING"
get NAAMES  NAAMES_Merge_Data                     "MRG01"

get ORACLES ORACLES_Cloud_AircraftInSitu_Data     "UND-CDP|PDI-CDNC|UND-SUMMARY"
get ORACLES ORACLES_Aerosol_AircraftInSitu_Data   "ORACLES-DMA|LDMA|HOWELL-CORRECTED-UHSAS"
get ORACLES ORACLES_MetNav_AircraftInSitu_Data    "Hskping_P3"
get ORACLES ORACLES_Merge_Data                    "MRG1_P3"

get CAMP2Ex CAMP2Ex_Cloud_AircraftInSitu_P3_Data   "FCDP|CLOUDFLAG"
get CAMP2Ex CAMP2Ex_Aerosol_AircraftInSitu_P3_Data "LARGE-SMPS|LARGE-LAS|INLETFLAG|MICROPHYSICAL"
get CAMP2Ex CAMP2Ex_MetNav_AircraftInSitu_P3_Data  "CAMP2EX-METNAV_P3B"
get CAMP2Ex CAMP2Ex_Merge_Data                     "mrg01-P3B" "NC_merge"

get ATom    ATom_CAPSVienna_Data_1981             "."
get ATom    ATom_Aerosol_Properties_V2_2111       "."
get ATom    ATom_merge_V2_1925                    "MER-1HZ|MER10_"
get ATom    ATom_nav_1613                         "csv$"
