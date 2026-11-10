#!/usr/bin/env python
"""Compare hourly droplet number from two retrievals on the same hours, without
needing an aerosol record: ARM NDROP vs Cloudnet N_scaled.

Prints the ratio (Cloudnet / NDROP) with a day-block bootstrap interval, the
log-log correlation, and how the ratio depends on LWP and cloud thickness.
Use the median ratio as --nd-scale in `arm_ena_nd_pnsd.py quicklook`.

Example
-------
python nd_compare.py --ndrop-dir data\\arm\\ena\\ndrop --der-dir data\\cloudnet\\arm-graciosa\\der \
    --start 2026-01-01 --end 2026-06-30 --out results\\ena\\ena_cloudnet_vs_ndrop.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_ena_nd_pnsd import ND_VAR, read_nd, hourly_nd  # noqa: E402
from cloudnet_build import read_der  # noqa: E402
from arm_ena_nd_pnsd import solar_zenith  # noqa: E402

EXTRA = r"(^lwp|cloud_base|cloud_top|toterror)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ndrop-dir", required=True)
    p.add_argument("--der-dir", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--min-cloud-frac", type=float, default=0.5, help="for both retrievals")
    p.add_argument("--max-base", type=float, default=2000.0)
    p.add_argument("--out", default=None, help="CSV of matched hours")
    p.add_argument("--site-lat", type=float, default=None, help="with --site-lon: ratio vs solar zenith angle")
    p.add_argument("--site-lon", type=float, default=None)
    a = p.parse_args()

    nd = hourly_nd(read_nd(a.ndrop_dir, ND_VAR, "strict", EXTRA), 0.0)
    cn = read_der(a.der_dir, 3, 2500.0, [1.0, 3000.0])
    if a.start:
        nd, cn = nd[nd.index >= a.start], cn[cn.index >= a.start]
    if a.end:
        nd, cn = nd[nd.index <= a.end], cn[cn.index <= a.end]
    j = pd.merge(nd, cn, left_index=True, right_index=True, how="inner", suffixes=("_ndrop", "_cloudnet"))
    m = ((j["cloud_frac_ndrop"] >= a.min_cloud_frac) & (j["cloud_frac_cloudnet"] >= a.min_cloud_frac)
         & (j["Nd_ndrop"] > 0) & (j["Nd_cloudnet"] > 0))
    if "cloud_base_height_ndrop" in j:
        m &= j["cloud_base_height_ndrop"] <= a.max_base
    j = j[m].copy()
    if len(j) < 10:
        sys.exit(f"Only {len(j)} matched hours; nothing to compare.")
    j["ratio"] = j["Nd_cloudnet"] / j["Nd_ndrop"]
    lr = np.log10(j["ratio"].to_numpy())
    day = j.index.normalize().to_numpy()
    groups = [np.where(day == d)[0] for d in np.unique(day)]
    rng = np.random.default_rng(0)
    boot = [10 ** np.median(lr[np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])])
            for _ in range(1000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    r = np.corrcoef(np.log10(j["Nd_ndrop"]), np.log10(j["Nd_cloudnet"]))[0, 1]
    print(f"Matched hours: {len(j)} on {len(groups)} days")
    print(f"Median Nd: NDROP {j['Nd_ndrop'].median():.0f}, Cloudnet {j['Nd_cloudnet'].median():.0f} cm-3")
    print(f"Cloudnet / NDROP: median {10 ** np.median(lr):.2f}  95% CI [{lo:.2f}, {hi:.2f}]  "
          f"IQR [{np.percentile(j['ratio'], 25):.2f}, {np.percentile(j['ratio'], 75):.2f}]  r(log) = {r:.2f}")
    if "drop_number_conc_toterror" in j:
        rel = j["drop_number_conc_toterror"] / j["Nd_ndrop"]
        s = j[rel <= 0.5]
        if len(s) >= 10:
            print(f"  NDROP error <= 50%: {len(s)} h, median ratio {s['ratio'].median():.2f}, "
                  f"r(log) = {np.corrcoef(np.log10(s['Nd_ndrop']), np.log10(s['Nd_cloudnet']))[0, 1]:.2f}")
    for col, bins in (("lwp_meas", [0, 0.05, 0.1, 0.2, 5]), ):
        if col in j:
            g = j.groupby(pd.cut(j[col], bins), observed=True)["ratio"].agg(["size", "median"])
            print("  by LWP (kg m-2):", {str(k): (int(v["size"]), round(v["median"], 2)) for k, v in g.iterrows()})
    if a.site_lat is not None and a.site_lon is not None:
        sza = solar_zenith(j.index + pd.Timedelta("30min"), a.site_lat, a.site_lon)
        j["solar_zenith"] = sza
        print("  by solar zenith angle (hour centre):")
        for b0, b1 in ((0, 30), (30, 45), (45, 60), (60, 75), (75, 90)):
            q = (sza > b0) & (sza <= b1)
            if q.sum() >= 5:
                print(f"    {b0:2d}-{b1:2d} deg: {q.sum():4d} h, median ratio {10 ** np.median(lr[q]):.2f} "
                      f"(IQR {10 ** np.percentile(lr[q], 25):.2f}-{10 ** np.percentile(lr[q], 75):.2f})")
        lw = np.log10(np.clip(j["lwp_meas"].to_numpy() * 1000, 1, None)) if "lwp_meas" in j else None

        def fit(i, with_lwp=False):
            cols = [np.ones(len(i)), sza[i]] + ([lw[i]] if with_lwp else [])
            return np.linalg.lstsq(np.column_stack(cols), lr[i], rcond=None)[0][1]
        for wl in ([False, True] if lw is not None else [False]):
            b = [fit(np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]), wl) for _ in range(500)]
            print(f"  slope vs zenith{' (log LWP held fixed)' if wl else ''}: {fit(np.arange(len(lr)), wl):+.4f} dex per degree "
                  f"[{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]  (SGP: +0.008)")
    if a.out:
        j.to_csv(a.out)
        print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
