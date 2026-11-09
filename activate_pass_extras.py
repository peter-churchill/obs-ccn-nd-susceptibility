#!/usr/bin/env python
"""Add in-cloud vertical velocity and a 2D-S drizzle measure to each ACTIVATE cloud pass, then test
whether they explain the Nd-N100 slope.

    python activate_pass_extras.py results/aircraft/activate_winter.nc
    python activate_pass_extras.py results/aircraft/activate_aircraft.nc --merge-dir data/aircraft/activate/merge

For every pass in a file written by `aircraft_build.py --format activate`, the 1-s merge of that day is read
over the pass (centre +/- half its duration, at least +/- 5 s) and these are stored per pass:
    sigma_w    standard deviation of vertical wind w (m/s), a proxy for in-cloud updraft strength
    w_up       mean of the positive w values (m/s)
    lwc_2ds    median 2D-S liquid water content (g/m3; 2D-S sees drops from ~10 um, so it weights large drops)
    n_2ds      median 2D-S liquid drop number (per litre)
    f_2ds      lwc_2ds / FCDP LWC, a drizzle-fraction proxy
Then it prints, with day-block bootstrap 95% CIs (unbinned least squares on log values):
    - S = d ln Nd / d ln N100, pooled
    - S with ln sigma_w as a second predictor, and the slope on ln sigma_w
    - how strongly ln N100 and ln sigma_w covary
    - S in terciles of the drizzle proxy (scavenging test: S should fall in non-drizzling passes if
      drizzle removal of aerosol steepens the slope)
A CSV with one row per pass is written next to the input file.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icartt import read_ict  # noqa: E402
from activate_reader import latest  # noqa: E402

COLS = {"w": "w_ms-1_THORNHILL", "lwc2": "LWC_VOIGT_2DS-HArm", "n2": "N-liquid_VOIGT_2DS-HArm",
        "lwcf": "LWC_VOIGT_FCDP"}


def pass_extras(nc, merge_dir):
    d = xr.open_dataset(nc).load()
    d.close()
    t = pd.DatetimeIndex(d.time.values)
    dur = d["duration_s"].values.astype(float)
    merges = latest(merge_dir, "ACTIVATE-mrg01-HU25_merge_*.ict")
    cache, rows = {}, []
    for ti, du in zip(t, dur):
        key = ti.strftime("%Y%m%d")
        if key not in cache:
            f = merges.get(key)
            m = read_ict(f, usecols=["Time_Start"] + list(COLS.values())) if f else None
            if m is not None:
                m = m[~m.index.duplicated()].sort_index()
            cache[key] = m
        m = cache[key]
        r = dict(sigma_w=np.nan, w_up=np.nan, lwc_2ds=np.nan, n_2ds=np.nan, f_2ds=np.nan, n_w=0)
        if m is not None:
            h = pd.Timedelta(seconds=max(du / 2.0, 5.0))
            s = m.loc[ti - h: ti + h]
            if COLS["w"] in s and s[COLS["w"]].notna().sum() >= 5:
                w = s[COLS["w"]].dropna().to_numpy()
                r.update(sigma_w=float(np.std(w)), w_up=float(w[w > 0].mean()) if (w > 0).any() else np.nan, n_w=len(w))
            if COLS["lwc2"] in s:
                lw2 = np.nanmedian(s[COLS["lwc2"]]) * 1000.0             # kg/m3 -> g/m3
                lwf = np.nanmedian(s[COLS["lwcf"]]) * 1000.0 if COLS["lwcf"] in s else np.nan
                r.update(lwc_2ds=lw2, n_2ds=np.nanmedian(s[COLS["n2"]]) / 1000.0 if COLS["n2"] in s else np.nan,
                         f_2ds=lw2 / lwf if lwf and lwf > 0 else np.nan)
        rows.append(r)
    out = pd.DataFrame(rows, index=t)
    out["Nd"] = d["CDNC"].values
    out["N100"] = d["CCN"].sel(radius=50, method="nearest").values
    out["day"] = t.normalize()
    return out


def boot(df, fn, n=300, seed=0):
    rng = np.random.default_rng(seed)
    days = df["day"].unique()
    g = {k: v for k, v in df.groupby("day")}
    est = fn(df)
    reps = []
    for _ in range(n):
        s = pd.concat([g[k] for k in rng.choice(days, len(days))])
        try:
            reps.append(fn(s))
        except Exception:
            pass
    reps = np.array(reps, dtype=float)
    reps = reps[np.all(np.isfinite(reps), axis=1)] if reps.ndim == 2 else reps[np.isfinite(reps)]
    if len(reps) >= 20:
        lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    else:
        lo = hi = np.full(np.shape(est), np.nan)
    return est, lo, hi


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nc")
    p.add_argument("--merge-dir", default="data/aircraft/activate/merge")
    p.add_argument("--n-boot", type=int, default=300)
    a = p.parse_args()
    df = pass_extras(a.nc, a.merge_dir)
    csv = a.nc.replace(".nc", "_pass_extras.csv")
    df.to_csv(csv)
    ok = np.isfinite(np.log(df.Nd)) & np.isfinite(np.log(df.N100))
    df = df[ok].copy()
    df["x"], df["y"] = np.log(df.N100), np.log(df.Nd)
    print(f"{a.nc}: {len(df)} passes on {df.day.nunique()} days; sigma_w for {df.sigma_w.notna().sum()}, "
          f"2D-S for {df.lwc_2ds.notna().sum()}")
    print(f"  median sigma_w {df.sigma_w.median():.2f} m/s, w_up {df.w_up.median():.2f} m/s; "
          f"median 2D-S LWC {df.lwc_2ds.median():.3f} g/m3, 2D-S/FCDP LWC {df.f_2ds.median():.2f}")

    s1 = lambda s: np.polyfit(s.x, s.y, 1)[0]
    e, lo, hi = boot(df, s1, a.n_boot)
    print(f"\nS pooled (unbinned)                       {e:5.2f}  [{lo:.2f}, {hi:.2f}]")

    w = df[df.sigma_w > 0].copy()
    w["lw"] = np.log(w.sigma_w)

    def s2(s):
        X = np.column_stack([np.ones(len(s)), s.x, s.lw])
        b = np.linalg.lstsq(X, s.y, rcond=None)[0]
        return np.array([b[1], b[2]])

    (ex, ew), (lx, lw_), (hx, hw) = boot(w, s2, a.n_boot)
    print(f"S same passes, sigma_w not controlled     {s1(w):5.2f}")
    print(f"S controlling for ln sigma_w              {ex:5.2f}  [{lx:.2f}, {hx:.2f}]")
    print(f"  slope on ln sigma_w                     {ew:5.2f}  [{lw_:.2f}, {hw:.2f}]")
    print(f"  corr(ln N100, ln sigma_w)               {np.corrcoef(w.x, w.lw)[0, 1]:5.2f}  "
          f"(slope of ln sigma_w on ln N100: {np.polyfit(w.x, w.lw, 1)[0]:.2f})")

    z = df[df.f_2ds.notna()].copy()
    if len(z) > 30:
        q = np.nanpercentile(z.f_2ds, [33.3, 66.7])
        print("\nDrizzle-proxy terciles (2D-S / FCDP LWC):")
        for i, (lo_, hi_) in enumerate([(-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf)]):
            s = z[(z.f_2ds > lo_) & (z.f_2ds <= hi_)]
            e, l, h = boot(s, s1, a.n_boot)
            print(f"  T{i + 1}  ratio {s.f_2ds.median():6.2f}  n {len(s):4d} on {s.day.nunique():3d} days   "
                  f"S {e:5.2f}  [{l:.2f}, {h:.2f}]   median Nd {s.Nd.median():5.0f}  N100 {s.N100.median():5.0f}")
    print(f"\nWrote {csv}")


if __name__ == "__main__":
    main()
