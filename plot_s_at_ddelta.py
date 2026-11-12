#!/usr/bin/env python
"""Susceptibility at the closure diameter, S(D_Delta), vs D_Delta for every source whose quicklook has a crossing.

    python plot_s_at_ddelta.py                       # all results/**/*_quicklook.csv
    python plot_s_at_ddelta.py --exclude uhsas fims   # extra regexes to skip

For each quicklook table (written by `arm_ena_nd_pnsd.py quicklook`, columns D_nm, S, S_lo, S_hi, delta_dex):
  * D_Delta = cutoff where delta_dex (mean log Nd - mean log N(>D)) crosses zero, interpolated in log D;
  * S(D_Delta) = S interpolated to that cutoff; its interval is the fixed-cutoff day-block CI (S_lo, S_hi)
    interpolated the same way (it does not include the uncertainty of D_Delta itself);
  * the D_Delta interval is read from the matching quicklook .txt where one exists.
The tables are the ones on disk: re-run the quicklook with the synthesis settings first if they were overwritten.
Quality (alpha) and the instrument used per source follow the synthesis (SOURCES below).
Writes figures/synthesis/s_at_ddelta.png and results/s_at_ddelta.csv.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# file-name regex -> (label, regime, method, quality); first match wins. Cloudnet sites with >= 100 good hours are
# 'moderate' (low: fewer hours, or Arctic sites where in-situ checks show Nd biased high). Quality as in the Synthesis sheet of
# docs/observation_options.xlsx (docs/ is not in git). Instrument choices follow the synthesis rows; the NDROP ground
# sites use the NDROP error screen as in the synthesis (quicklook --max-rel-err 0.5, files *_relerr_quicklook.*).
SOURCES = [
    (r"/ao2018_fog_clean_", "AO2018 fog (FSSP)", "Arctic", "Ship in situ", "slope only"),
    (r"/ao2018_cvi_residual_clean_", "AO2018 fog (residuals)", "Arctic", "Ship in situ", "slope only"),
    (r"/ao2018_cloudnet_clean_", "AO2018 (Cloudnet)", "Arctic", "Cloudnet", "low"),
    (r"/activate_winter_quicklook", "ACTIVATE winter", "Marine CAO", "Aircraft", "robust"),
    (r"/activate_summer_quicklook", "ACTIVATE summer", "Marine", "Aircraft", "robust"),
    (r"/ace-ena_aircraft_quicklook", "ACE-ENA", "Marine", "Aircraft", "robust"),
    (r"/acapex_aircraft_uhsas_quicklook", "ACAPEX", "Marine", "Aircraft", "moderate"),
    (r"/acme-v_aircraft_pcasp_quicklook", "ACME-V", "Arctic", "Aircraft", "moderate"),
    (r"/hi-scale_aircraft_mergedaerosolsd_quicklook", "HI-SCALE", "Continental", "Aircraft", "robust"),
    (r"/cacti_aircraft_mergedaerosolsd_quicklook", "CACTI", "Continental", "Aircraft", "robust"),
    (r"/goamazon_aircraft_pcasp_quicklook", "GoAmazon", "Continental", "Aircraft", "moderate"),
    (r"/ena_cloudnet_smps_quicklook", "ENA (Cloudnet, SMPS)", "Marine", "Cloudnet", "low"),       # one half-year (2026)
    (r"/sgp_cloudnet_quicklook", "SGP (Cloudnet)", "Continental", "Cloudnet", "moderate"),
    (r"/ena_2014_2023_uhsas_relerr_quicklook", "ENA (UHSAS)", "Marine", "Ground NDROP", "parked"),
    (r"/ena_2022_2026_smps_relerr_quicklook", "ENA (SMPS 2022-26)", "Marine", "Ground NDROP", "moderate"),
    (r"/sgp_2016_2025_smps_relerr_quicklook", "SGP", "Continental", "Ground NDROP", "moderate"),
    (r"/lasic_2016_2017_smps_quicklook", "LASIC (SMPS)", "Marine", "Ground NDROP", "slope only"),
    (r"/lasic_2016_2017_uhsas_quicklook", "LASIC (UHSAS)", "Marine", "Ground NDROP", "parked"),
    (r"/epc_2023_2024_smps_relerr_quicklook", "EPCAPE pier", "Marine", "Ground NDROP", "moderate"),
    (r"/epcape_soledad_hourly_quicklook", "EPCAPE Mt Soledad", "Marine", "Ground in situ", "low"),
    (r"/puijo", "Puijo", "Boreal", "Ground in situ", "robust"),
    (r"/pallas", "Pallas", "Boreal", "Ground in situ", "robust"),
    (r"/zeppelin", "Zeppelin", "Arctic", "Ground in situ", "robust"),
    (r"/hyytiala_cloudnet", "Hyytiala", "Boreal", "Cloudnet", "moderate"),
    (r"/kenttarova_cloudnet", "Kenttarova", "Boreal", "Cloudnet", "moderate"),
    (r"/norunda_cloudnet", "Norunda", "Boreal", "Cloudnet", "moderate"),
    (r"/mace-head_cloudnet", "Mace Head", "Marine", "Cloudnet", "moderate"),
    (r"/maido_cloudnet", "Maido", "Marine", "Cloudnet", "low"),
    (r"/ny-alesund_cloudnet", "Ny-Alesund", "Arctic", "Cloudnet", "low"),
    (r"/oliktok_cloudnet", "Oliktok", "Arctic", "Cloudnet", "low"),
    (r"/mosaic_cloudnet_smps", "MOSAiC (Cloudnet, SMPS)", "Arctic", "Cloudnet", "low"),
    (r"/andoya_cloudnet", "Andoya (Cloudnet)", "Arctic", "Cloudnet", "low"),
    (r"/bankhead_cloudnet_smps_quicklook", "Bankhead", "Continental", "Cloudnet", "low"),           # 98 h
    (r"/lasic_cloudnet_smps_quicklook", "LASIC (Cloudnet, SMPS)", "Marine", "Cloudnet", "low"),   # 8 h with cloud_frac >= 0.5
    (r"/cabauw_cloudnet", "Cabauw", "Continental", "Cloudnet", "moderate"),
    (r"/granada_cloudnet", "Granada", "Continental", "Cloudnet", "moderate"),
    (r"/palaiseau_cloudnet", "Palaiseau", "Continental", "Cloudnet", "low"),
    (r"/schneefernerhaus_cloudnet", "Schneefernerhaus", "Continental", "Cloudnet", "low"),
    (r"/bucharest_cloudnet", "Bucharest", "Continental", "Cloudnet", "moderate"),
    (r"/melpitz_cloudnet", "Melpitz", "Continental", "Cloudnet", "moderate"),
]
SKIP = [r"pilot_", r"_median", r"_orig", r"_sza\d", r"/ena_2023_", r"pnsd_"]
REG_COL = {"Marine": "#1f77b4", "Marine CAO": "#17becf", "Boreal": "#2ca02c", "Arctic": "#9467bd", "Continental": "#d62728"}
MK = {"Ground NDROP": "o", "Ground in situ": "s", "Cloudnet": "^", "Aircraft": "D", "Ship in situ": "P"}
LABEL_OFF = {"Puijo": (-6, 6), "GoAmazon": (6, 6), "LASIC (SMPS)": (-6, -12), "Zeppelin": (6, 4), "Melpitz": (-6, -12),
             "AO2018 fog (FSSP)": (6, -2), "Norunda": (6, -10), "Pallas": (6, 4), "EPCAPE pier": (6, 6),
             "ACTIVATE summer": (-6, 6), "Hyytiala": (6, -10), "Kenttarova": (-6, -12), "ACE-ENA": (6, -2),
             "CACTI": (6, 4), "ACME-V": (6, -4), "SGP (Cloudnet)": (6, 4), "MOSAiC (Cloudnet)": (6, -10),
             "EPCAPE Mt Soledad": (6, 6), "Bankhead": (6, -10), "HI-SCALE": (-6, 6)}
QA = {"robust": 1.0, "moderate": 0.8, "slope only": 0.55, "Nd unverified": 0.55, "low": 0.3}


def crossing(d, y):
    """First zero crossing of y(d) from below, interpolated in log d; NaN if none."""
    ok = np.isfinite(y)
    d, y = d[ok], y[ok]
    for i in range(len(y) - 1):
        if y[i] <= 0 < y[i + 1] or y[i] >= 0 > y[i + 1]:
            f = y[i] / (y[i] - y[i + 1])
            return float(np.exp(np.log(d[i]) + f * (np.log(d[i + 1]) - np.log(d[i]))))
    return np.nan


def at(d, y, x):
    ok = np.isfinite(y)
    return float(np.interp(np.log(x), np.log(d[ok]), y[ok])) if ok.sum() >= 2 else np.nan


def txt_ci(csv, dd):
    """D_Delta 95% CI from the quicklook text output next to the table, if its D_Delta matches (within 3 nm)."""
    stem = csv[: -len("_quicklook.csv")]
    site = os.path.basename(os.path.dirname(csv))
    for c in (stem + "_quicklook.txt", os.path.join(os.path.dirname(csv), f"{site}_quicklook.txt")):
        if not os.path.exists(c):
            continue
        for line in open(c, encoding="utf-8", errors="replace"):
            m = re.match(r"D_Delta = ([\d.]+) nm\s+95% CI \[([\d.]+|nan), ([\d.]+|nan)\]", line)
            if m and abs(float(m.group(1)) - dd) <= 3:
                return float(m.group(2)), float(m.group(3))
    return np.nan, np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude", nargs="*", default=[], help="extra path regexes to skip")
    ap.add_argument("--out", default="figures/synthesis/s_at_ddelta.png")
    a = ap.parse_args()
    rows, seen, unmatched = [], set(), []
    for csv in sorted(glob.glob("results/**/*_quicklook.csv", recursive=True)):
        p = csv.replace("\\", "/")
        if any(re.search(s, p) for s in SKIP + a.exclude):
            continue
        src = next((s for s in SOURCES if re.search(s[0], p)), None)
        if src is None:
            unmatched.append(p)
            continue
        _, label, regime, method, quality = src
        if quality == "parked":                              # UHSAS where an SMPS/DMPS exists (SMPS preferred)
            continue
        if label in seen:
            print(f"  (second file for {label}, skipped: {p})")
            continue
        t = pd.read_csv(csv)
        d = t.D_nm.to_numpy(float)
        dd = crossing(d, t.delta_dex.to_numpy(float))
        s100 = at(d, t.S.to_numpy(float), 100.0)
        if not np.isfinite(dd):
            print(f"  {label:24s} no D_Delta crossing ({p})")
            continue
        seen.add(label)
        dlo, dhi = txt_ci(csv, dd)
        rows.append(dict(label=label, regime=regime, method=method, quality=quality, file=p,
                         D_Delta=dd, D_lo=dlo, D_hi=dhi, S_at_D=at(d, t.S.to_numpy(float), dd),
                         S_lo=at(d, t.S_lo.to_numpy(float), dd), S_hi=at(d, t.S_hi.to_numpy(float), dd),
                         S100=s100, n=int(t.n.max())))
    if unmatched:
        print(f"  not used (other instruments, variants or unlisted): {', '.join(os.path.basename(u) for u in unmatched)}")
    nofile = [s[1] for s in SOURCES if not any(re.search(s[0], c.replace("\\", "/"))
                                               for c in glob.glob("results/**/*_quicklook.csv", recursive=True))]
    if nofile:
        print(f"  no quicklook table found for: {', '.join(nofile)}")
    r = pd.DataFrame(rows).sort_values("D_Delta")
    os.makedirs("results", exist_ok=True)
    r.to_csv("results/s_at_ddelta.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(r[["label", "quality", "n", "D_Delta", "D_lo", "D_hi", "S_at_D", "S_lo", "S_hi", "S100", "file"]]
              .round(2).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(10, 7))
    for _, x in r.iterrows():
        xe = [[max(x.D_Delta - x.D_lo, 0)], [max(x.D_hi - x.D_Delta, 0)]] if np.isfinite(x.D_lo) else None
        ye = [[max(x.S_at_D - x.S_lo, 0)], [max(x.S_hi - x.S_at_D, 0)]] if np.isfinite(x.S_lo) else None
        al = QA.get(x.quality, 0.55)
        ax.errorbar(x.D_Delta, x.S_at_D, xerr=xe, yerr=ye, fmt=MK[x.method], color=REG_COL[x.regime], ms=8,
                    mec="k", mew=0.6, alpha=al, elinewidth=1, capsize=2)
        ax.annotate(x.label, (x.D_Delta, x.S_at_D), xytext=LABEL_OFF.get(x.label, (5, 4)), textcoords="offset points",
                    fontsize=7.5, alpha=max(al, 0.6), ha="right" if LABEL_OFF.get(x.label, (5, 4))[0] < 0 else "left")
    ax.axhline(0, color="0.6", lw=0.8); ax.axhline(1, color="0.6", lw=0.8, ls=":")
    ax.set(xscale="log", xlabel="closure diameter D$_\\Delta$ (nm)", ylabel="S(D$_\\Delta$) = dln N$_d$ / dln N(>D$_\\Delta$)",
           title="Susceptibility at the closure diameter vs the closure diameter (faded: not robust/moderate)")
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([20, 30, 50, 70, 100, 150, 200, 300]))
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.3)
    h = ([Line2D([], [], ls="", marker="o", color=c, mec="k", ms=8, label=k) for k, c in REG_COL.items()]
         + [Line2D([], [], ls="", marker=m, color="0.7", mec="k", ms=8, label=k) for k, m in MK.items()])
    ax.legend(handles=h, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"\nSaved {a.out} and results/s_at_ddelta.csv ({len(r)} sources)")


if __name__ == "__main__":
    main()
