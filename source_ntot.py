#!/usr/bin/env python
"""Median total aerosol number for every synthesis source, on the same screened hours/passes as its quicklook.

    python source_ntot.py                 # writes results/source_ntot.csv

For each source in plot_s_at_ddelta.SOURCES (parked rows included), the built file next to the quicklook table is
opened and the quicklook screens are re-applied (read from the quicklook .txt, otherwise the synthesis defaults; the
hour count at 100 nm must match the table). "N_total" is N(>D_min), where D_min is the smallest cutoff with data in at
least half of the screened hours - so it is a true total (D_min = 10-20 nm) for SMPS/DMPS sources but only N(>~60 nm)
for UHSAS and N(>~100 nm) for PCASP. D_min is written out so these can be flagged. Also written: median N(>100 nm)
and median Nd, for comparison.

Periods: first/last date and number of days/hours with both Nd and a size distribution, before the cloud screens
(paired_*) and after them (screened_*), plus the calendar years covered, so the table doubles as a data inventory.
Writes results/source_ntot.csv and prints a period table.
"""
import glob
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

import arm_ena_nd_pnsd as ae
from plot_s_at_ddelta import SOURCES, SKIP

RESULTS = "results"
OUT = "results/source_ntot.csv"
D_REF = 100.0


def built_file(csv):
    stem = csv[: -len("_quicklook.csv")]
    for c in (stem, re.sub(r"_(relerr|sza\d+)$", "", stem), re.sub(r"_relerr(_sza\d+)?$", "", stem)):
        if os.path.exists(c + ".nc"):
            return c + ".nc"
    return None


def _ns(**kw):
    ns = SimpleNamespace(max_rel_err=None, min_lwp=None, max_cloud_base=None, max_cloud_top=None, max_sza=None,
                         coupled_only=False, dz_coupled=150.0, latlon=None, min_cloud_frac=0.5)
    ns.__dict__.update(kw)
    return ns


def screen_options(csv, method):
    d = {}
    if method == "Cloudnet":
        d.update(max_cloud_base=1000, max_cloud_top=2000) if "ao2018" in csv else d.update(max_cloud_base=2000, max_cloud_top=3000)
    if "_relerr" in csv:
        d["max_rel_err"] = 0.5
    cands = []
    txt = csv[:-4] + ".txt"
    if os.path.exists(txt):
        s = open(txt, encoding="utf-8", errors="replace").read()
        mcf = re.search(r"Using cloud_frac >= ([\d.]+)", s)
        if mcf:
            scr = (re.search(r"Screens applied to every table: (.*)", s) or [None, ""])[1]
            t = {"min_cloud_frac": float(mcf.group(1))}
            for pat, k in [(r"Nd error/Nd <= ([\d.]+)", "max_rel_err"), (r"LWP >= ([\d.]+)", "min_lwp"),
                           (r"cloud base <= ([\d.]+) m", "max_cloud_base"), (r"cloud top <= ([\d.]+) m", "max_cloud_top")]:
                mm = re.search(pat, scr)
                if mm:
                    t[k] = float(mm.group(1))
            cands.append(("txt", _ns(**t)))
    cands.append(("default", _ns(**d)))
    return cands


rows, seen = [], set()
for csv in sorted(glob.glob(f"{RESULTS}/**/*_quicklook.csv", recursive=True)):
    p = csv.replace("\\", "/")
    if any(re.search(s, p) for s in SKIP):
        continue
    src = next((s for s in SOURCES if re.search(s[0], p)), None)
    if src is None or src[1] in seen:
        continue
    label, regime, method, quality = src[1:]
    nc = built_file(p)
    if nc is None:
        print(f"  {label:26s} no built .nc for {p}")
        continue
    seen.add(label)
    ql = pd.read_csv(p)
    n_ql = int(ql.n.iloc[np.argmin(np.abs(ql.D_nm.to_numpy() - D_REF))])
    ds = xr.open_dataset(nc)
    diam = ds["diameter"].values
    i100 = int(np.argmin(np.abs(diam - D_REF)))
    for how, ns in screen_options(p, method):
        try:
            base, cf, _ = ae._base_mask(ds, ns)
        except SystemExit:
            continue
        m = base & (cf >= ns.min_cloud_frac)
        ccn, y = ds["CCN"].values[m], ds["CDNC"].values[m]
        n100 = int((np.isfinite(ccn[:, i100]) & np.isfinite(y) & (ccn[:, i100] > 0) & (y > 0)).sum())
        if n100 == n_ql:
            break
    tm = pd.DatetimeIndex(ds["time"].values)
    allc, ally = ds["CCN"].values, ds["CDNC"].values
    paired = np.isfinite(ally) & (ally > 0) & np.isfinite(allc[:, i100])
    tp, ts = tm[paired], tm[m][np.isfinite(y) & (y > 0) & np.isfinite(ccn[:, i100])]
    yrs = sorted(set(ts.year))
    period = dict(paired_start=tp.min().date() if len(tp) else None, paired_end=tp.max().date() if len(tp) else None,
                  paired_hours=int(paired.sum()), paired_days=int(tp.normalize().nunique()),
                  screened_start=ts.min().date() if len(ts) else None, screened_end=ts.max().date() if len(ts) else None,
                  screened_days=int(ts.normalize().nunique()),
                  years=",".join(str(v) for v in yrs) if len(yrs) <= 3 or yrs[-1] - yrs[0] + 1 != len(yrs) else f"{yrs[0]}-{yrs[-1]}")
    ok = np.isfinite(y) & (y > 0)
    frac = np.isfinite(ccn[ok]).mean(axis=0)
    j = int(np.argmax(frac >= 0.5)) if (frac >= 0.5).any() else None
    ntot = np.nanmedian(ccn[ok, j]) if j is not None else np.nan
    rows.append(dict(label=label, regime=regime, method=method, quality=quality, hours=int(ok.sum()),
                     D_min_nm=diam[j] if j is not None else np.nan, N_total_median=ntot,
                     N100_median=np.nanmedian(ccn[ok, i100]), Nd_median=np.nanmedian(y[ok]),
                     hours_match=n100 == n_ql, **period, file=nc))
    ds.close()
    print(f"  {label:26s} {ok.sum():5d} h  N(>{rows[-1]['D_min_nm']:.0f} nm) {ntot:7.0f}  N100 {rows[-1]['N100_median']:6.0f}  "
          f"Nd {rows[-1]['Nd_median']:6.0f}  {'ok' if n100 == n_ql else 'HOURS DIFFER from quicklook'}")

r = pd.DataFrame(rows)
r.to_csv(OUT, index=False, float_format="%.1f")
with pd.option_context("display.width", 250, "display.max_rows", 200):
    print("\nPeriods with both Nd and a size distribution (paired) and after the cloud screens (screened):")
    print(r.sort_values(["regime", "label"])[["label", "regime", "method", "quality", "paired_start", "paired_end", "paired_hours",
                                              "screened_start", "screened_end", "hours", "screened_days", "years"]].to_string(index=False))
print(f"\nSaved {OUT} ({len(r)} sources)")
