#!/usr/bin/env python
"""Build hourly in-situ Nd + CCN(D) files from the Virtanen et al. (2025) station data.

Inputs (one pair per station, as shared by the authors):
    <Station>_Measured_Nd_CCN.csv   Time, CCN (= N(D > 70 nm), total inlet), Nd (in-cloud hours only)
    <Station>_PNSD.csv              Time, X<diameter nm> ... hourly dN/dlogDp (cm-3), total inlet

Output has the same layout as the ARM and Cloudnet builds (CDNC, CCN(radius),
cloud_frac), so `arm_ena_nd_pnsd.py quicklook` runs on it unchanged. Every
hour in the Nd file is an in-cloud hour, so cloud_frac is set to 1.

Example
-------
python virtanen_build.py --nd data\\virtanen\\Pallas_Measured_Nd_CCN.csv \\
    --pnsd data\\virtanen\\Pallas_PNSD.csv --site-name "Pallas (in situ)" --out results\\pallas_insitu\\pallas_insitu.nc
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


def read_pnsd_csv(path):
    p = pd.read_csv(path, index_col="Time", na_values=["NA", "NaN", ""])
    p.index = pd.to_datetime(p.index, format="mixed")
    d = np.array([float(c.lstrip("X")) for c in p.columns])
    return p, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nd", required=True)
    ap.add_argument("--pnsd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--site-name", required=True)
    ap.add_argument("--min-bin-valid", type=float, default=0.95,
                    help="keep edge bins valid in at least this fraction of the Nd hours")
    a = ap.parse_args()

    nc = pd.read_csv(a.nd, parse_dates=["Time"], index_col="Time").sort_index()
    nc = nc[~nc.index.duplicated()]
    p, d = read_pnsd_csv(a.pnsd)
    pm = p.reindex(nc.index)
    valid = pm.notna().mean().to_numpy()
    good = valid >= a.min_bin_valid
    i0, i1 = np.argmax(good), len(good) - np.argmax(good[::-1])   # contiguous span
    arr, d = pm.to_numpy(dtype=float)[:, i0:i1], d[i0:i1]
    arr[arr < 0] = np.nan
    cut_nm = 2.0 * RADII_NM
    ccn = cumulative_above(arr, d, cut_nm)
    band = cumulative_above(arr, d, np.array([d[0], 20.0, 70.0, 100.0, 500.0]))
    n70 = band[:, 2]
    ratio = np.nanmedian(nc["CCN"].to_numpy() / n70)
    print(f"{a.site_name}: {len(nc)} in-cloud hours {nc.index.min():%Y-%m-%d} to {nc.index.max():%Y-%m-%d}; "
          f"PNSD bins {d[0]:.1f}-{d[-1]:.0f} nm; provided CCN / our N(>70 nm) = {ratio:.2f}")

    ds = xr.Dataset(
        {"CDNC": ("time", nc["Nd"].to_numpy(float), {"units": "cm-3", "long_name": "in-situ Nd (total minus interstitial / CVI), hourly"}),
         "CCN": (("time", "radius"), ccn, {"units": "cm-3", "long_name": "N(D > 2*radius) from the hourly total-inlet PNSD"}),
         "CCN_provided": ("time", nc["CCN"].to_numpy(float), {"units": "cm-3", "long_name": "CCN as provided (N > 70 nm)"}),
         "cloud_frac": ("time", np.ones(len(nc)), {"units": "1", "long_name": "all hours are in-cloud hours"}),
         "N_tot": ("time", band[:, 0], {"units": "cm-3"}),
         "N_aitken_20_100": ("time", band[:, 1] - band[:, 3], {"units": "cm-3"}),
         "N70_500": ("time", band[:, 2] - np.nan_to_num(band[:, 4]), {"units": "cm-3"})},
        coords={"time": nc.index.values,
                "radius": ("radius", RADII_NM, {"units": "nm", "long_name": "cutoff radius"}),
                "diameter": ("radius", cut_nm, {"units": "nm", "long_name": "cutoff diameter"})},
        attrs={"site": a.site_name, "source": "Virtanen et al. 2025, Nat. Geosci., doi:10.1038/s41561-025-01662-y (data from the authors)",
               "nd_file": os.path.basename(a.nd), "pnsd_file": os.path.basename(a.pnsd),
               "pnsd_range_nm": f"{d[0]:.1f}-{d[-1]:.0f}",
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ds.to_netcdf(a.out, encoding={v: {"zlib": True} for v in ds.data_vars})
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
