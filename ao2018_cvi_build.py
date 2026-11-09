#!/usr/bin/env python
"""AO2018 cloud residuals as a droplet-number source: write the GCVI residual number (Nd proxy) and the
whole-air N(D > cutoff) per scan in the standard layout, so `arm_ena_nd_pnsd.py quicklook` gives S and D_Delta.

    python ao2018_cvi_build.py data/ao2018/oden-ao-2018-aerosol-cvi-1.nc4 --out results/ao2018/ao2018_cvi_residual.nc
    python arm_ena_nd_pnsd.py quicklook results/ao2018/ao2018_cvi_residual.nc --min-cloud-frac 0.5

The residual number carries the GCVI sampling-efficiency correction (a constant factor in Karlsson et al. 2022),
which cancels in the log-log slope S. D_Delta from this file inherits that factor and is not independent of it.
Each row is one residual DMPS scan (~9 min) paired with the nearest whole-air scan within --max-dt minutes;
the quicklook's day-block bootstrap resamples whole days, so the several scans per event are not over-counted.
Data: Karlsson & Zieger, doi:10.17043/oden-ao-2018-aerosol-cvi-1.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_ena_nd_pnsd import RADII_NM, cumulative_above  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--out", default="results/ao2018/ao2018_cvi_residual.nc")
    p.add_argument("--max-dt", type=float, default=10.0, help="minutes between residual and whole-air scan")
    p.add_argument("--min-residual", type=float, default=0.5, help="cm-3; scans with fewer residuals are dropped")
    a = p.parse_args()
    d = xr.open_dataset(a.file).load()
    d.close()
    D = d["aerosol_diameters"].values.astype(float)
    dnm = D * 1e9 if np.nanmax(D) < 1e-3 else (D if np.nanmax(D) > 5 else D * 1000.0)
    res = d["cloud_residual_size_distribution"].values.astype(float)
    wh = d["aerosol_particle_size_distribution"].values.astype(float)
    tr = pd.DatetimeIndex(d["time_DMPS"].values)
    tw = pd.DatetimeIndex(d["time_DMPS_whlr"].values)
    cut = 2.0 * RADII_NM
    rows, ccn, psd = [], [], []
    for i, t in enumerate(tr):
        j = int(np.argmin(np.abs((tw - t).total_seconds())))
        if abs((tw[j] - t).total_seconds()) > a.max_dt * 60:
            continue
        nres = float(cumulative_above(res[i][None], dnm, np.array([dnm.min()]), None)[0, 0])
        if not np.isfinite(nres) or nres < a.min_residual:
            continue
        rows.append((t, nres))
        ccn.append(cumulative_above(wh[j][None], dnm, cut, None)[0])
        psd.append(wh[j])
    if not rows:
        sys.exit("No paired scans.")
    t = pd.DatetimeIndex([r[0] for r in rows])
    nd = np.array([r[1] for r in rows])
    ccn, psd = np.array(ccn), np.array(psd)
    ds = xr.Dataset(
        {"CDNC": ("time", nd, {"units": "cm-3", "long_name": "GCVI cloud-residual number (droplet-number proxy, efficiency-corrected as archived)"}),
         "CCN": (("time", "radius"), ccn, {"units": "cm-3", "long_name": "whole-air N(D > 2*radius), DMPS"}),
         "cloud_frac": ("time", np.ones(len(t)), {"units": "1", "long_name": "1 for every residual scan (in cloud/fog by construction)"}),
         "dNdlogDp": (("time", "aerosol_diameter"), psd, {"units": "cm-3"})},
        coords={"time": t.values, "radius": ("radius", RADII_NM, {"units": "nm"}),
                "diameter": ("radius", cut, {"units": "nm"}), "aerosol_diameter": ("aerosol_diameter", dnm, {"units": "nm"})},
        attrs={"site": "AO2018 Oden fog (GCVI residuals + whole-air DMPS)",
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "settings": f"max_dt={a.max_dt} min_residual={a.min_residual}",
               "source": "Karlsson & Zieger, doi:10.17043/oden-ao-2018-aerosol-cvi-1",
               "note": "Residual number carries a constant GCVI efficiency factor: S is independent of it, D_Delta is not."})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    ds.to_netcdf(a.out)
    print(f"Paired scans: {len(t)} on {t.normalize().nunique()} days; median residual number {np.median(nd):.1f}, "
          f"whole-air N(>100 nm) {np.nanmedian(ccn[:, np.argmin(abs(cut - 100))]):.1f} cm-3")
    print(f"Wrote {a.out}. Run: python arm_ena_nd_pnsd.py quicklook {a.out} --min-cloud-frac 0.5")


if __name__ == "__main__":
    main()
