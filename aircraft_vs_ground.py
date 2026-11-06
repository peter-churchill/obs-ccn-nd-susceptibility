#!/usr/bin/env python
"""Compare aircraft in-situ cloud passes with a ground site's retrieved Nd and surface aerosol.

For ACE-ENA vs the ENA ground site (and HI-SCALE vs SGP) this checks, on the same hours:
  1. droplet number: aircraft (in situ, near cloud base) vs NDROP (retrieved);
  2. aerosol: aircraft below-cloud N(>D) vs the ground UHSAS/SMPS N(>D);
  3. where a difference in the closure diameter comes from, by computing D_Delta for all
     four combinations of droplet source (aircraft / ground) and aerosol source
     (aircraft / ground) on the same matched hours;
  4. susceptibility by campaign period (IOP), aircraft vs ground for the same months.

Matching: aircraft passes within --radius km of the site are grouped by UTC hour (median Nd and
median N(>D) per hour) and joined to the ground hour with the same timestamp. Intervals are
95% day-block bootstrap (whole flights resampled).

Example (ACE-ENA)
-----------------
python aircraft_vs_ground.py --aircraft results/aircraft/ace-ena_aircraft.nc \\
    --ground results/ena/ena_2014_2023_uhsas.nc --site-lat 39.0916 --site-lon -28.0257 \\
    --periods "IOP1 summer:2017-06-01:2017-08-01" "IOP2 winter:2018-01-01:2018-03-01" \\
    --label ace-ena_vs_ena
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm_ena_nd_pnsd as m  # noqa: E402


def km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    h = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def day_boot(times, fn, n_boot=500, seed=0):
    """Day-block bootstrap of a scalar statistic fn(index_array)."""
    day = pd.DatetimeIndex(times).normalize().to_numpy()
    groups = [np.where(day == d)[0] for d in np.unique(day)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        vals.append(fn(idx))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals.size >= 20 else (np.nan, np.nan)


def s_and_dd(t, ccn, y, diam, i100, n_boot):
    """S(100 nm) and D_Delta with day-block CIs; NaNs if too few samples."""
    if len(y) < 10:
        return dict(n=len(y), days=pd.DatetimeIndex(t).normalize().nunique())
    tab, dd = m.susceptibility_table(ccn, y, diam)
    sb, db = m.bootstrap_by_day(t, ccn, y, diam, n_boot, cut_idx=[i100])
    s_lo, s_hi = m._ci(sb[:, i100])
    d_lo, d_hi = m._ci(db)
    return dict(n=len(y), days=pd.DatetimeIndex(t).normalize().nunique(), S=tab.S.iloc[i100],
                S_lo=s_lo, S_hi=s_hi, D=dd, D_lo=d_lo, D_hi=d_hi)


def fmt(r):
    if "S" not in r:
        return f"n={r['n']:4d} ({r['days']} days)  too few"
    d = f"{r['D']:.0f} [{r['D_lo']:.0f}, {r['D_hi']:.0f}]" if np.isfinite(r["D"]) else "no crossing"
    return (f"n={r['n']:4d} ({r['days']:3d} days)  S(100 nm) {r['S']:.2f} [{r['S_lo']:.2f}, {r['S_hi']:.2f}]"
            f"  closure {d} nm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aircraft", required=True, help="output of aircraft_build.py")
    ap.add_argument("--ground", required=True, help="ground built file (NDROP + surface aerosol)")
    ap.add_argument("--site-lat", type=float, required=True)
    ap.add_argument("--site-lon", type=float, required=True)
    ap.add_argument("--radius", type=float, default=50.0, help="km from the site for aircraft passes")
    ap.add_argument("--min-cloud-frac", type=float, default=0.3, help="ground hours")
    ap.add_argument("--periods", nargs="*", default=[], help='"name:start:end" campaign periods')
    ap.add_argument("--label", required=True, help="output name stem")
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "results", "aircraft")
    fig_dir = os.path.join(root, "figures", "aircraft") if os.path.isdir(os.path.join(root, "figures")) else out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    air = xr.open_dataset(a.aircraft)
    gnd = xr.open_dataset(a.ground)
    diam = air["diameter"].values
    if not np.allclose(diam, gnd["diameter"].values):
        sys.exit("Aircraft and ground files use different cutoff grids.")
    i100 = int(np.argmin(np.abs(diam - 100)))
    ta = pd.DatetimeIndex(air["time"].values)
    tg = pd.DatetimeIndex(gnd["time"].values)

    # ---- 4 (first, uses all passes): susceptibility by period, aircraft vs ground same months
    print(f"\n=== Susceptibility by period: aircraft (all passes) vs ground ({os.path.basename(a.ground)}, cloud_frac >= {a.min_cloud_frac})")
    cf = gnd["cloud_frac"].values >= a.min_cloud_frac
    rows = []
    periods = [("all", str(ta.min().date()), str((ta.max() + pd.Timedelta(days=1)).date()))]
    periods += [tuple(p.split(":")) for p in a.periods]
    for name, t0, t1 in periods:
        ma = (ta >= t0) & (ta < t1)
        mg = cf & (tg >= t0) & (tg < t1)
        ra = s_and_dd(ta[ma], air["CCN"].values[ma], air["CDNC"].values[ma], diam, i100, a.n_boot)
        rg = s_and_dd(tg[mg], gnd["CCN"].values[mg], gnd["CDNC"].values[mg], diam, i100, a.n_boot)
        print(f"  {name:14s} aircraft {fmt(ra)}\n  {'':14s} ground   {fmt(rg)}")
        rows += [{"period": name, "source": "aircraft", **ra}, {"period": name, "source": "ground", **rg}]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{a.label}_by_period.csv"), index=False)

    # ---- match passes near the site to ground hours
    near = km(a.site_lat, a.site_lon, air["lat"].values, air["lon"].values) <= a.radius
    print(f"\nAircraft passes within {a.radius:.0f} km of the site: {near.sum()} of {len(ta)}")
    hour = ta[near].floor("h")
    pa = pd.DataFrame({"Nd_air": air["CDNC"].values[near]}, index=hour)
    ccn_a = pd.DataFrame(air["CCN"].values[near], index=hour)
    pa = pa.groupby(level=0).median()
    ccn_a = ccn_a.groupby(level=0).median()
    gi = pd.Series(np.arange(len(tg)), index=tg)
    common = pa.index.intersection(tg[cf])
    if len(common) < 5:
        sys.exit(f"Only {len(common)} matched hours; try a larger --radius or lower --min-cloud-frac.")
    ig = gi.loc[common].to_numpy()
    nd_a, nd_g = pa.loc[common, "Nd_air"].to_numpy(), gnd["CDNC"].values[ig]
    ca, cg = ccn_a.loc[common].to_numpy(), gnd["CCN"].values[ig]
    tm = pd.DatetimeIndex(common)
    print(f"Matched hours: {len(common)} on {tm.normalize().nunique()} flights")

    # ---- 1. droplet number
    ok = (nd_a > 0) & (nd_g > 0)
    lr = np.log10(nd_g[ok] / nd_a[ok])
    lo, hi = day_boot(tm[ok], lambda i: 10 ** np.median(lr[i]), a.n_boot)
    print(f"\n=== 1. Droplet number, NDROP / aircraft: median {10 ** np.median(lr):.2f} [{lo:.2f}, {hi:.2f}], "
          f"r(log) {np.corrcoef(np.log10(nd_a[ok]), np.log10(nd_g[ok]))[0, 1]:.2f}; "
          f"medians aircraft {np.median(nd_a[ok]):.0f}, NDROP {np.median(nd_g[ok]):.0f} cm-3")
    if "drop_number_conc_toterror" in gnd:
        rel = gnd["drop_number_conc_toterror"].values[ig] / nd_g
        q = ok & (rel <= 0.5)
        if q.sum() >= 5:
            print(f"    NDROP error <= 50%: {q.sum()} h, ratio {10 ** np.median(np.log10(nd_g[q] / nd_a[q])):.2f}, "
                  f"r(log) {np.corrcoef(np.log10(nd_a[q]), np.log10(nd_g[q]))[0, 1]:.2f}")
    if "lwp_meas" in gnd:
        lwp = gnd["lwp_meas"].values[ig] * 1000
        for lo_, hi_ in ((0, 50), (50, 100), (100, 200), (200, 5000)):
            q = ok & (lwp > lo_) & (lwp <= hi_)
            if q.sum() >= 3:
                print(f"    LWP {lo_:>3}-{hi_:<4} g m-2: {q.sum():3d} h, NDROP / aircraft {10 ** np.median(np.log10(nd_g[q] / nd_a[q])):.2f}")

    # ---- 2. aerosol
    print("\n=== 2. Aerosol, ground / aircraft below-cloud N(>D):")
    for d in (70, 100, 150, 200):
        i = int(np.argmin(np.abs(diam - d)))
        q = (ca[:, i] > 0) & (cg[:, i] > 0)
        if q.sum() >= 5:
            r = np.log10(cg[q, i] / ca[q, i])
            lo, hi = day_boot(tm[q], lambda k: 10 ** np.median(r[k]), a.n_boot)
            print(f"    D > {d:3d} nm: {q.sum():3d} h, median {10 ** np.median(r):.2f} [{lo:.2f}, {hi:.2f}], "
                  f"r(log) {np.corrcoef(np.log10(ca[q, i]), np.log10(cg[q, i]))[0, 1]:.2f}")

    # ---- 3. closure diameter by source combination
    print("\n=== 3. Closure diameter on the matched hours, by source of droplets x aerosol:")
    combos = {"aircraft Nd + aircraft aerosol": (nd_a, ca), "aircraft Nd + ground aerosol": (nd_a, cg),
              "NDROP Nd + aircraft aerosol": (nd_g, ca), "NDROP Nd + ground aerosol": (nd_g, cg)}
    deltas = {}
    for name, (y, c) in combos.items():
        delta, okc = m.closure_offsets(c, y)
        dd = m.d_delta_from(delta, diam)
        lo, hi = day_boot(tm, lambda i: m.d_delta_from(m.closure_offsets(c[i], y[i])[0], diam), a.n_boot)
        deltas[name] = delta
        dtxt = f"{dd:.0f} [{lo:.0f}, {hi:.0f}] nm" if np.isfinite(dd) else "no crossing"
        print(f"    {name:32s} {dtxt}  (n = {okc.sum()})")

    # ---- table and figure
    pairs = pd.DataFrame({"Nd_aircraft": nd_a, "Nd_ndrop": nd_g, "N100_aircraft": ca[:, i100],
                          "N100_ground": cg[:, i100]}, index=tm)
    pairs.to_csv(os.path.join(out_dir, f"{a.label}_matched_hours.csv"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        for k, (x, y, lab) in enumerate(((nd_a, nd_g, "Nd, cm$^{-3}$"), (ca[:, i100], cg[:, i100], "N(D > 100 nm), cm$^{-3}$"))):
            q = (x > 0) & (y > 0)
            ax[k].loglog(x[q], y[q], "o", ms=4, alpha=0.6, color="#3B6FB6")
            lim = [min(x[q].min(), y[q].min()) * 0.7, max(x[q].max(), y[q].max()) * 1.4]
            ax[k].plot(lim, lim, "k--", lw=1)
            ax[k].set_xlim(lim); ax[k].set_ylim(lim)
            ax[k].set_xlabel(f"aircraft {lab}"); ax[k].set_ylabel(f"ground {lab}")
            ax[k].set_title(["Droplet number: NDROP vs in situ", "Aerosol: surface vs below cloud"][k], loc="left", fontsize=10)
        cols = ["#3B6FB6", "#7FA7D9", "#D9822B", "#F0B97A"]
        for (name, delta), col in zip(deltas.items(), cols):
            ax[2].semilogx(diam, delta, color=col, lw=1.8, label=name)
        ax[2].axhline(0, color="k", lw=0.8)
        ax[2].set_xlabel("Cutoff diameter, nm"); ax[2].set_ylabel("mean log Nd − mean log N(>D), dex")
        ax[2].set_title("Closure offset by source", loc="left", fontsize=10)
        ax[2].legend(fontsize=7, frameon=False)
        for x_ in ax:
            x_.grid(alpha=0.3)
        fig.suptitle(f"{a.label}: {len(common)} matched hours", x=0.01, ha="left")
        fig.tight_layout()
        png = os.path.join(fig_dir, f"{a.label}.png")
        fig.savefig(png, dpi=150)
        print(f"\nSaved {png}")
    except Exception as e:  # plotting is optional
        print(f"(figure skipped: {e})")
    print(f"Tables: {out_dir}/{a.label}_matched_hours.csv and _by_period.csv")


if __name__ == "__main__":
    main()
