#!/usr/bin/env python
"""Activation diameters from the AO2018 cloud-residual data (Karlsson et al. 2022, JGR, doi:10.1029/2021JD036383;
data doi:10.17043/oden-ao-2018-aerosol-cvi-1).

    python ao2018_cvi.py data/ao2018/oden-ao-2018-aerosol-cvi-1.nc4 --out-dir results/ao2018

For each residual DMPS scan (GCVI inlet: dried fog/cloud droplets) the nearest whole-air DMPS scan within
--max-dt minutes is taken, and:
  AF(D)   = residual dN/dlogD / whole-air dN/dlogD                      activated fraction per size
  plateau = median AF over --plateau-nm (accumulation mode, assumed fully activated)
  D50     = smallest diameter where AF / plateau rises through 0.5 (log-interpolated)
  D_closure = diameter where whole-air N(>D) equals the total residual number (the residual-based analogue of
              D_Delta: if all residuals are droplets, N(>D_closure) = Nd)
The absolute residual number depends on the GCVI sampling-efficiency correction (a constant in the paper), so
D50 (shape only) is the robust quantity; D_closure and the plateau carry that factor.
Writes a per-scan CSV, a per-day summary and a figure of the median AF curve with IQR.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_ena_nd_pnsd import cumulative_above  # noqa: E402


def crossing_up(x_log, y, level):
    for i in range(len(y) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] < level <= y[i + 1]:
            return 10 ** (x_log[i] + (level - y[i]) / (y[i + 1] - y[i]) * (x_log[i + 1] - x_log[i]))
    return np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--out-dir", default="results/ao2018")
    p.add_argument("--max-dt", type=float, default=10.0, help="minutes between residual and whole-air scan")
    p.add_argument("--plateau-nm", type=float, nargs=2, default=[150.0, 500.0])
    p.add_argument("--min-whole", type=float, default=1.0, help="cm-3 per dlogD; bins below are not used for AF")
    a = p.parse_args()
    d = xr.open_dataset(a.file).load()
    d.close()
    D = d["aerosol_diameters"].values.astype(float)
    dnm = D * 1e9 if np.nanmax(D) < 1e-3 else (D if np.nanmax(D) > 5 else D * 1000.0)
    res = d["cloud_residual_size_distribution"].values.astype(float)
    wh = d["aerosol_particle_size_distribution"].values.astype(float)
    tr = pd.DatetimeIndex(d["time_DMPS"].values)
    tw = pd.DatetimeIndex(d["time_DMPS_whlr"].values)
    print(f"diameters {dnm.min():.0f}-{dnm.max():.0f} nm ({len(dnm)} bins); residual scans {len(tr)}, whole-air scans {len(tw)}")
    for v in ("cloud_residual_size_distribution", "aerosol_particle_size_distribution"):
        print(f"  {v}: units '{d[v].attrs.get('units', '?')}', {str(d[v].attrs.get('long_name', ''))[:90]}")
    lx = np.log10(dnm)
    rows, curves = [], []
    for i, t in enumerate(tr):
        j = np.argmin(np.abs((tw - t).total_seconds()))
        if abs((tw[j] - t).total_seconds()) > a.max_dt * 60:
            continue
        r, w = res[i], wh[j]
        af = np.where(w >= a.min_whole, r / w, np.nan)
        pl = (dnm >= a.plateau_nm[0]) & (dnm <= a.plateau_nm[1])
        plateau = np.nanmedian(af[pl])
        afn = af / plateau if np.isfinite(plateau) and plateau > 0 else np.full_like(af, np.nan)
        d50 = crossing_up(lx, afn, 0.5)
        nres = float(cumulative_above(r[None], dnm, np.array([dnm.min()]), None)[0, 0])
        cut = np.geomspace(dnm.min(), dnm.max(), 200)
        nw = cumulative_above(w[None], dnm, cut, None)[0]
        k = np.where(np.isfinite(nw) & (nw <= nres))[0]
        dcl = cut[k[0]] if len(k) and k[0] > 0 else np.nan
        rows.append({"time": t, "dt_min": (tw[j] - t).total_seconds() / 60, "N_residual": nres,
                     "N_whole_tot": float(nw[0]), "plateau_AF": plateau, "D50_nm": d50, "D_closure_nm": dcl,
                     "N_whole_gt_D50": float(cumulative_above(w[None], dnm, np.array([d50]), None)[0, 0]) if np.isfinite(d50) else np.nan})
        curves.append(afn)
    if not rows:
        sys.exit("No residual scan has a whole-air scan within --max-dt.")
    df = pd.DataFrame(rows).set_index("time")
    curves = np.array(curves)
    os.makedirs(a.out_dir, exist_ok=True)
    df.to_csv(os.path.join(a.out_dir, "ao2018_cvi_scans.csv"))
    day = df.groupby(df.index.normalize()).agg(n=("D50_nm", "size"), D50_nm=("D50_nm", "median"),
                                               D_closure_nm=("D_closure_nm", "median"), N_residual=("N_residual", "median"),
                                               N_whole_tot=("N_whole_tot", "median"), plateau_AF=("plateau_AF", "median"))
    day.to_csv(os.path.join(a.out_dir, "ao2018_cvi_days.csv"))
    q = lambda s: f"{np.nanmedian(s):.0f} (IQR {np.nanpercentile(s, 25):.0f}-{np.nanpercentile(s, 75):.0f})"
    print(f"\nPaired scans: {len(df)} on {df.index.normalize().nunique()} days (median |dt| {df.dt_min.abs().median():.1f} min)")
    print(f"  D50 (shape of AF)          {q(df.D50_nm)} nm   [{df.D50_nm.notna().mean():.0%} of scans have a crossing]")
    print(f"  D_closure (N_whole(>D) = N_residual)  {q(df.D_closure_nm)} nm")
    print(f"  residual number            {q(df.N_residual)} cm-3;  whole-air total {q(df.N_whole_tot)} cm-3")
    print(f"  plateau AF {np.nanmedian(df.plateau_AF):.2f}  (should be ~1 if the residual number is already efficiency-corrected)")
    print(f"  whole-air N(>D50) vs residual number: median ratio {np.nanmedian(df.N_whole_gt_D50 / df.N_residual):.2f}")
    print("\nPer day:\n" + day.round(2).to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        med = np.nanmedian(curves, axis=0)
        ax[0].fill_between(dnm, np.nanpercentile(curves, 25, axis=0), np.nanpercentile(curves, 75, axis=0), color="0.8", label="IQR")
        ax[0].plot(dnm, med, "k-", label="median")
        ax[0].axhline(0.5, color="0.5", lw=0.8, ls=":")
        ax[0].axvline(np.nanmedian(df.D50_nm), color="C0", ls="--", label=f"median D50 {np.nanmedian(df.D50_nm):.0f} nm")
        ax[0].axvline(np.nanmedian(df.D_closure_nm), color="C1", ls="--", label=f"median D_closure {np.nanmedian(df.D_closure_nm):.0f} nm")
        ax[0].set(xscale="log", xlabel="dry diameter (nm)", ylabel="activated fraction / plateau", ylim=(0, 1.5), title="AO2018 fog: activated fraction")
        ax[0].legend(fontsize=8)
        ax[1].hist(df.D50_nm.dropna(), bins=20, color="C0", alpha=0.7, label="D50")
        ax[1].hist(df.D_closure_nm.dropna(), bins=20, color="C1", alpha=0.5, label="D_closure")
        ax[1].set(xlabel="diameter (nm)", ylabel="scans", title="per-scan activation diameters")
        ax[1].legend()
        fig.tight_layout()
        os.makedirs("figures/ao2018", exist_ok=True)
        fig.savefig("figures/ao2018/ao2018_cvi_activation.png", dpi=150)
        print("\nSaved figures/ao2018/ao2018_cvi_activation.png")
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"Wrote {a.out_dir}/ao2018_cvi_scans.csv and ao2018_cvi_days.csv")


if __name__ == "__main__":
    main()
