#!/usr/bin/env python
"""Aircraft in-situ droplet number vs ARM NDROP, read straight from the NDROP files, so no ground aerosol record
is needed (more matched hours than aircraft_vs_ground.py, and works where the built ground file has gaps, e.g.
SGP during HI-SCALE). Main purpose: does NDROP / aircraft depend on the solar zenith angle?

    python aircraft_vs_ndrop.py --aircraft results/aircraft/ace-ena_aircraft.nc --ndrop-dir data/arm/ena/ndrop \\
        --site-lat 39.0916 --site-lon -28.0257 --label ace-ena_vs_ndrop
    python aircraft_vs_ndrop.py --aircraft results/aircraft/hi-scale_aircraft.nc --ndrop-dir data/arm/sgp/ndrop \\
        --site-lat 36.607 --site-lon -97.488 --radius 100 --min-cloud-frac 0.1 --label hi-scale_vs_ndrop

Matching: aircraft passes within --radius km, mean Nd per UTC hour; NDROP hourly mean (same as the builds) with
cloud_frac >= --min-cloud-frac. Intervals: 95% flight-block (day) bootstrap. In-situ Nd does not depend on the sun,
so a trend of the ratio with zenith angle belongs to NDROP (MFRSR optical depth, N ~ tau^3).
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm_ena_nd_pnsd as m  # noqa: E402
from aircraft_vs_ground import km, day_boot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aircraft", required=True)
    ap.add_argument("--ndrop-dir", required=True)
    ap.add_argument("--site-lat", type=float, required=True)
    ap.add_argument("--site-lon", type=float, required=True)
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--min-cloud-frac", type=float, default=0.3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()

    air = xr.open_dataset(a.aircraft)
    ta = pd.DatetimeIndex(air["time"].values)
    near = km(a.site_lat, a.site_lon, air["lat"].values, air["lon"].values) <= a.radius
    pa = pd.Series(air["CDNC"].values[near], index=ta[near].floor("h"))
    pa = pa[pa > 0].groupby(level=0).mean()
    print(f"Aircraft: {len(ta)} passes, {near.sum()} within {a.radius:.0f} km, {len(pa)} hours on "
          f"{pa.index.normalize().nunique()} flights ({pa.index.min():%Y-%m-%d} to {pa.index.max():%Y-%m-%d})")

    # only read NDROP files on flight days (filename contains YYYYMMDD)
    days = {d.strftime("%Y%m%d") for d in pa.index.normalize()}
    files = [f for f in sorted(glob.glob(os.path.join(a.ndrop_dir, "*.nc")) + glob.glob(os.path.join(a.ndrop_dir, "*.cdf")))
             if any(d in os.path.basename(f) for d in days)]
    print(f"NDROP files on flight days: {len(files)} (of {len(days)} flight days)")
    if not files:
        sys.exit("No NDROP files on the flight days in --ndrop-dir; download them first.")
    tmp = tempfile.mkdtemp(prefix="ndrop_")                  # links to the flight-day files only
    for f in files:
        try:
            os.symlink(os.path.abspath(f), os.path.join(tmp, os.path.basename(f)))
        except OSError:
            shutil.copy(f, tmp)
    try:
        nd = m.hourly_nd(m.read_nd(tmp, m.ND_VAR, "strict", r"(^lwp|toterror)"), a.min_cloud_frac, "mean")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    common = pa.index.intersection(nd.index)
    if len(common) < 5:
        sys.exit(f"Only {len(common)} matched hours (NDROP hours on flight days with cloud_frac >= "
                 f"{a.min_cloud_frac}: {nd.index.normalize().isin(pa.index.normalize()).sum()}).")
    tm = pd.DatetimeIndex(common)
    nd_a, nd_g = pa.loc[common].to_numpy(), nd.loc[common, "Nd"].to_numpy()
    lwp = nd.loc[common, "lwp_meas"].to_numpy() * 1000 if "lwp_meas" in nd else np.full(len(tm), np.nan)
    sza = m.solar_zenith(tm + pd.Timedelta("30min"), a.site_lat, a.site_lon)
    lr = np.log10(nd_g / nd_a)
    lo, hi = day_boot(tm, lambda i: 10 ** np.median(lr[i]), a.n_boot)
    print(f"\nMatched hours: {len(tm)} on {tm.normalize().nunique()} flights")
    print(f"NDROP / aircraft: median {10 ** np.median(lr):.2f} [{lo:.2f}, {hi:.2f}], r(log) "
          f"{np.corrcoef(np.log10(nd_a), np.log10(nd_g))[0, 1]:.2f}; medians aircraft {np.median(nd_a):.0f}, NDROP {np.median(nd_g):.0f} cm-3")

    print("\nBy solar zenith angle (hour centre):")
    for b0, b1 in ((0, 30), (30, 45), (45, 60), (60, 75), (75, 90)):
        q = (sza > b0) & (sza <= b1)
        if q.sum() >= 3:
            print(f"  {b0:2d}-{b1:2d} deg: {q.sum():3d} h on {tm[q].normalize().nunique():2d} flights, median ratio "
                  f"{10 ** np.median(lr[q]):.2f} (IQR {10 ** np.percentile(lr[q], 25):.2f}-{10 ** np.percentile(lr[q], 75):.2f})")
    idx = np.arange(len(lr))
    slope = lambda i: np.polyfit(sza[i], lr[i], 1)[0]
    s0, (l0, h0) = slope(idx), day_boot(tm, slope, a.n_boot)
    print(f"  slope {s0:+.4f} dex per degree [{l0:+.4f}, {h0:+.4f}]  (SGP Cloudnet / NDROP: +0.008)")
    okl = np.isfinite(lwp) & (lwp > 0)
    if okl.sum() >= 10:
        def slope_lwp(i):
            i = i[okl[i]]
            X = np.column_stack([np.ones(len(i)), sza[i], np.log10(lwp[i])])
            return np.linalg.lstsq(X, lr[i], rcond=None)[0][1]
        l1, h1 = day_boot(tm, slope_lwp, a.n_boot)
        print(f"  with log LWP held fixed: {slope_lwp(idx):+.4f} [{l1:+.4f}, {h1:+.4f}]")
        print("By LWP (g m-2):")
        for b0, b1 in ((0, 50), (50, 100), (100, 200), (200, 5000)):
            q = okl & (lwp > b0) & (lwp <= b1)
            if q.sum() >= 3:
                print(f"  {b0:3d}-{b1:<4d}: {q.sum():3d} h, median ratio {10 ** np.median(lr[q]):.2f}")

    out = pd.DataFrame({"Nd_aircraft": nd_a, "Nd_ndrop": nd_g, "lwp_gm2": lwp, "solar_zenith": sza}, index=tm)
    os.makedirs("results/aircraft", exist_ok=True)
    out.to_csv(f"results/aircraft/{a.label}_hours.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    lim = [min(nd_a.min(), nd_g.min()) * 0.7, max(nd_a.max(), nd_g.max()) * 1.4]
    sc = ax[0].scatter(nd_a, nd_g, c=sza, s=22, cmap="viridis")
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="aircraft Nd (cm$^{-3}$)", ylabel="NDROP Nd (cm$^{-3}$)")
    fig.colorbar(sc, ax=ax[0], label="solar zenith (deg)")
    ax[1].scatter(sza, lr, s=22, color="#3B6FB6")
    xx = np.array([sza.min(), sza.max()])
    ax[1].plot(xx, np.polyval(np.polyfit(sza, lr, 1), xx), "k-", lw=1, label=f"fit {s0:+.4f} /deg")
    ax[1].plot(xx, np.median(lr) + 0.008 * (xx - np.median(sza)), "k:", lw=1, label="SGP Cloudnet/NDROP +0.008 /deg")
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set(xlabel="solar zenith angle (deg)", ylabel="log$_{10}$(NDROP / aircraft)")
    ax[1].legend(fontsize=8, frameon=False)
    for x_ in ax:
        x_.grid(alpha=0.3)
    fig.suptitle(f"{a.label}: {len(tm)} matched hours on {tm.normalize().nunique()} flights", x=0.01, ha="left")
    fig.tight_layout()
    fdir = "figures/aircraft" if os.path.isdir("figures") else "results/aircraft"
    os.makedirs(fdir, exist_ok=True)
    fig.savefig(f"{fdir}/{a.label}.png", dpi=150)
    print(f"\nSaved {fdir}/{a.label}.png and results/aircraft/{a.label}_hours.csv")


if __name__ == "__main__":
    main()
