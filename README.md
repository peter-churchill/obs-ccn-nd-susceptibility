# Observed CCN–Nd susceptibility

Tools to build hourly matched cloud droplet number (Nd) and aerosol size
distributions from observations, and to compute the CCN–Nd susceptibility
S = d log Nd / d log N(D > D_cut) as a function of cutoff diameter, for
comparison with NorESM and EC-Earth station output.

## Scripts

| Script | What it does |
|---|---|
| `arm_ena_nd_pnsd.py` | ARM sites (ENA, SGP, LASIC). Subcommands `discover`, `download`, `build`, `quicklook`. Nd from the NDROP retrieval (MFRSR + MWR, daytime overcast); aerosol from surface SMPS or UHSAS; optional surface met for a coupling (cloud base minus LCL) screen. |
| `cloudnet_download.py`, `cloudnet_build.py` | Cloudnet (ACTRIS) droplet number (`der` product, scaled-Frisch N_scaled) from the open API, matched with an EBAS size distribution or an ARM aerosol folder; same output layout. |
| `epcape_build.py` | EPCAPE Mt. Soledad (La Jolla, 2023–24): FM-120 fog-monitor Nd paired with clear-air SEMS spectra within ±3 h, excluding logged instrument periods. Output is read by `arm_ena_nd_pnsd.py quicklook`. |

## Method

- Hourly means; an hour counts as cloudy when the Nd retrieval covers at least the chosen fraction of it (`--min-cloud-frac`).
- CCN(D) = integral of dN/dlogDp above each cutoff, on the NorESM station grid (radii 5–150 nm, diameters 10–300 nm), using the instrument's bin bounds.
- S from binned medians of log10 Nd vs log10 CCN (equal-number bins, n/15 clamped to 5–20), with 95% day-block bootstrap intervals.
- D_Delta: the cutoff diameter where mean log Nd equals mean log CCN (closure diameter).
- Optional screens: NDROP relative error (`--max-rel-err`), minimum LWP, coupled hours only, cloud-base split.

## Setup

```
conda env create -f environment.yml
conda activate arm
```

ARM credentials are read from environment variables, never from the code:

```
setx ARM_USER your_arm_username
setx ARM_TOKEN your_armlive_token
```

## Example (ARM ENA)

```
python arm_ena_nd_pnsd.py discover
python arm_ena_nd_pnsd.py download --ds enandropmfrsrC1.c1 --start 2023-01-01 --end 2023-12-31 --out data/arm/ena/ndrop
python arm_ena_nd_pnsd.py download --ds enaaossmpsC1.b1 --start 2023-01-01 --end 2023-12-31 --out data/arm/ena/smps
python arm_ena_nd_pnsd.py download --ds enametC1.b1 --start 2023-01-01 --end 2023-12-31 --out data/arm/ena/met
python arm_ena_nd_pnsd.py build --nd-dir data/arm/ena/ndrop --psd-dir data/arm/ena/smps --met-dir data/arm/ena/met --out results/ena/ena_2023_smps.nc
python arm_ena_nd_pnsd.py quicklook results/ena/ena_2023_smps.nc --min-cloud-frac 0.5 --max-rel-err 0.5
```

Other sites: `discover --site sgp` or `--site asi` (LASIC), and `build --site-name "..."`.

## Data and citation

Data are not included. ARM data: cite each datastream DOI (shown on its ARM
Data Discovery page) and acknowledge the ARM user facility (US DOE Office of
Science). EPCAPE Mt. Soledad data (UCSD Library, CC BY 4.0): the authors ask to
be consulted before publication.

## Folder layout

```
data/arm/<site>/{ndrop,smps,uhsas,met}   raw ARM files
data/cloudnet/<site>/{der,lwc}           raw Cloudnet files
data/ebas/                               EBAS size distributions
data/epcape_soledad/                     UCSD fog monitor and SEMS
results/<site>/                          built files and quicklooks (not in git)
docs/                                    tracking spreadsheet (not in git)
```
