#!/usr/bin/env python
"""Build hourly Nd + N(>D) files from the Zeppelin GCVI dataset (Karlsson et al. 2021, doi:10.17043/zeppelin-cloud-aerosol-1).

    python zeppelin_gcvi_build.py --part1 .../zeppelin-cloud-aerosol-1_part1.nc4 --part2 .../zeppelin-cloud-aerosol-1_part2.nc4 \
        --out-dir results/zeppelin_gcvi [--virtanen data/virtanen/Zeppelin_Nd_CCN.csv]

Dataset layout:
  part1 (one DMPS, 30 bins, ~5-7 min scans): switches inlet. cloud_flag == "yes": DMPS behind the GCVI -> cloud RESIDUAL
        size distribution (already divided by the enrichment factor and multiplied by the transmission factor); "no":
        whole-air inlet. Also the FM-120 fog-monitor droplet spectrum (43 bins, ~2-50 um), updraft, T, wind, visibility.
  part2 (two DMPS on the whole-air inlet, 57 bins with duplicated "pairs" after a settings change, 5-809 nm): total aerosol
        (interstitial + evaporated droplets), used for N(>D) in cloud.

Writes two files with the standard layout (CDNC, CCN(radius) with coord diameter, cloud_frac), so
`arm_ena_nd_pnsd.py quicklook` runs unchanged:
  zeppelin_gcvi_fm120.nc     CDNC = FM-120 droplet number (D >= --fm-min-um), mean over the GCVI ("yes") scans of the hour
  zeppelin_gcvi_residual.nc  CDNC = CPC residual number behind the GCVI (droplets above the GCVI cut, transmission-corrected)
Both carry Nd_fm120, Nd_residual, cloud_frac (fraction of the hour's part1 scans with cloud_flag yes), updraft/T/wind/visibility,
and, for the activation analysis, the hourly residual and whole-air spectra on the part1 grid, the activated fraction AF(D)
= residual / whole-air and the hourly D50 (first upward crossing of AF = 0.5).
"""
import argparse
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_ena_nd_pnsd import RADII_NM, cumulative_above  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def merge_pairs(d_nm, psd, tol=0.05):
    """Merge bins whose diameters differ by < tol (the part2 'pairs': only one of each has data in a given scan)."""
    ok = np.isfinite(d_nm)
    d_nm, psd = d_nm[ok], psd[:, ok]
    o = np.argsort(d_nm)
    d, p = d_nm[o], psd[:, o]
    groups = [[0]]
    for i in range(1, len(d)):
        (groups[-1].append(i) if d[i] / d[groups[-1][0]] - 1 < tol else groups.append([i]))
    dc = np.array([np.exp(np.log(d[g]).mean()) for g in groups])
    pc = np.column_stack([np.nanmean(p[:, g], axis=1) for g in groups])
    return dc, pc


def log_widths(d):
    """log10 bin widths from geometric-midpoint edges."""
    lg = np.log10(d)
    e = np.r_[lg[0] - (lg[1] - lg[0]) / 2, (lg[1:] + lg[:-1]) / 2, lg[-1] + (lg[-1] - lg[-2]) / 2]
    return np.diff(e)


def d50(af, d):
    """First upward crossing of AF = 0.5, interpolated in log D; NaN if none."""
    out = np.full(af.shape[0], np.nan)
    for t in range(af.shape[0]):
        a = af[t]
        ok = np.isfinite(a)
        x, y = np.log(d[ok]), a[ok]
        idx = np.where((y[:-1] < 0.5) & (y[1:] >= 0.5))[0]
        if idx.size:
            i = idx[0]
            out[t] = np.exp(x[i] + (0.5 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return out


def hourly(df):
    return df.groupby(df.index.floor("h")).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part1", required=True)
    ap.add_argument("--part2", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fm-min-um", type=float, default=2.0, help="lower droplet diameter for the FM-120 Nd (um)")
    ap.add_argument("--match-min", type=float, default=10.0,
                    help="a part2 scan counts as in-cloud if its midpoint is within this many minutes of a GCVI scan")
    ap.add_argument("--min-bin-valid", type=float, default=0.9, help="keep part2 bins valid in this fraction of in-cloud scans")
    ap.add_argument("--virtanen", default=None, help="Zeppelin_Nd_CCN.csv from Virtanen et al. for a consistency check")
    a = ap.parse_args()

    p1 = xr.open_dataset(a.part1)
    p2 = xr.open_dataset(a.part2)
    t1 = pd.DatetimeIndex(p1.time.values)
    cloud = np.char.strip(np.asarray(p1.cloud_flag.values).astype(str)) == "yes"
    print(f"part1: {len(t1)} scans, {cloud.sum()} with the GCVI (cloud_flag yes); part2: {p2.sizes['time']} scans")

    # ---- part1: droplets (FM-120), residuals (CPC + DMPS), met
    dc_um = p1.cloud_diameters.values * 1e6
    fm = p1.cloud_particle_size_distribution.values
    sel = dc_um >= a.fm_min_um
    w = log_widths(dc_um)
    nd_fm = np.where(np.isfinite(fm[:, sel]).any(axis=1), np.nansum(fm[:, sel] * w[sel], axis=1), np.nan)
    nd_res = p1.total_aerosol_particle_concentration.values.astype(float)
    dr_nm = p1.aerosol_diameters.values * 1e9
    o1 = np.argsort(dr_nm)
    dr_nm, res = dr_nm[o1], p1.aerosol_particle_size_distribution.values[:, o1]
    met = {k: p1[v].values.astype(float) for k, v in [("updraft", "updraft_velocity"), ("temperature", "temperature"),
                                                     ("wind_speed", "wind_speed"), ("visibility", "visibility_mean"),
                                                     ("enrichment_factor", "enrichment_factor")]}
    print(f"FM-120 bins {dc_um.min():.1f}-{dc_um.max():.1f} um (Nd from >= {a.fm_min_um} um); "
          f"residual DMPS {dr_nm.min():.0f}-{dr_nm.max():.0f} nm")

    s1 = pd.DataFrame({"cloud": cloud.astype(float)}, index=t1)
    sc = pd.DataFrame({"Nd_fm120": nd_fm, "Nd_residual": nd_res, **met}, index=t1)[cloud]
    rs = pd.DataFrame(res, index=t1)[cloud]

    # ---- part2: whole-air size distribution, in-cloud scans only
    t2 = pd.DatetimeIndex(p2.time.values)
    d2, w2 = merge_pairs(p2.aerosol_diameters.values * 1e9, p2.aerosol_particle_size_distribution.values)
    w2[w2 < 0] = np.nan
    tc = t1[cloud]
    j = np.clip(tc.searchsorted(t2), 1, len(tc) - 1) if len(tc) > 1 else np.zeros(len(t2), int)
    gap = np.minimum(np.abs((t2 - tc[j - 1]).total_seconds()), np.abs((t2 - tc[j]).total_seconds())) / 60
    inc = gap <= a.match_min
    valid = np.isfinite(w2[inc]).mean(axis=0)
    good = valid >= a.min_bin_valid
    i0, i1 = np.argmax(good), len(good) - np.argmax(good[::-1])
    d2, w2 = d2[i0:i1], w2[:, i0:i1]
    print(f"part2: {len(np.unique(np.round(p2.aerosol_diameters.values * 1e9, 1)))} bins merged to {len(d2)} "
          f"({d2[0]:.1f}-{d2[-1]:.0f} nm kept); {inc.sum()} scans within {a.match_min:g} min of a GCVI scan")
    cut_nm = 2.0 * RADII_NM
    ccn = cumulative_above(w2[inc], d2, cut_nm)
    n70 = cumulative_above(w2[inc], d2, np.array([70.0]))[:, 0]
    s2 = pd.DataFrame(ccn, index=t2[inc])
    s2["N70"] = n70
    # whole-air spectrum on the residual grid (interpolated in log D, log values) for AF(D)
    lw = np.log(np.where(w2[inc] > 0, w2[inc], np.nan))
    tot_on_res = np.exp(np.array([np.interp(np.log(dr_nm), np.log(d2), r, left=np.nan, right=np.nan) for r in lw]))
    s2t = pd.DataFrame(tot_on_res, index=t2[inc])

    # ---- hourly
    cf = hourly(s1)["cloud"]
    h1, hr, h2, h2t = hourly(sc), hourly(rs), hourly(s2), hourly(s2t)
    idx = h1.index.intersection(h2.index)
    idx = idx[np.isfinite(h2.loc[idx].N70.to_numpy())]
    h1, hr, h2, h2t = h1.loc[idx], hr.reindex(idx), h2.loc[idx], h2t.reindex(idx)
    af = hr.to_numpy() / h2t.to_numpy()
    D50 = d50(af, dr_nm)
    print(f"{len(idx)} hours with GCVI cloud sampling and whole-air N(>D); cloud_frac >= 0.5 in {(cf.loc[idx] >= 0.5).sum()}; "
          f"median Nd FM-120 {h1.Nd_fm120.median():.0f}, residual {h1.Nd_residual.median():.0f} cm-3 "
          f"(median ratio residual/FM-120 {np.nanmedian(h1.Nd_residual / h1.Nd_fm120):.2f}); "
          f"hourly D50 found in {np.isfinite(D50).sum()} h, median {np.nanmedian(D50):.0f} nm")

    if a.virtanen:
        v = pd.read_csv(a.virtanen, parse_dates=["Time"], index_col="Time").sort_index()
        print("Consistency with Virtanen et al. Zeppelin hours (shift = Virtanen time - our hour start):")
        for sh in (-1, 0, 1):
            vv = v.copy()
            vv.index = vv.index - pd.Timedelta(hours=sh)
            m = vv.join(h1[["Nd_fm120", "Nd_residual"]], how="inner").join(h2[["N70"]], how="inner").dropna()
            if len(m) < 10:
                print(f"  shift {sh:+d} h: {len(m)} matched hours")
                continue
            lc = lambda x, y: np.corrcoef(np.log10(x), np.log10(y))[0, 1]
            print(f"  shift {sh:+d} h: {len(m)} of {len(v)} hours matched; Nd/FM-120 median {np.median(m.Nd / m.Nd_fm120):.2f} "
                  f"(r {lc(m.Nd, m.Nd_fm120):.2f}); Nd/residual {np.median(m.Nd / m.Nd_residual):.2f} (r {lc(m.Nd, m.Nd_residual):.2f}); "
                  f"CCN/N70 {np.median(m.CCN / m.N70):.2f} (r {lc(m.CCN, m.N70):.2f})")

    os.makedirs(a.out_dir, exist_ok=True)
    common = {
        "Nd_fm120": ("time", h1.Nd_fm120.to_numpy(), {"units": "cm-3", "long_name": f"FM-120 droplet number, D >= {a.fm_min_um} um, mean of GCVI scans"}),
        "Nd_residual": ("time", h1.Nd_residual.to_numpy(), {"units": "cm-3", "long_name": "CPC cloud-residual number behind the GCVI (EF- and transmission-corrected)"}),
        "CCN": (("time", "radius"), h2[list(range(len(cut_nm)))].to_numpy(), {"units": "cm-3", "long_name": "N(D > 2*radius), whole-air DMPS (part2), in-cloud scans"}),
        "N70": ("time", h2.N70.to_numpy(), {"units": "cm-3"}),
        "cloud_frac": ("time", cf.reindex(idx).to_numpy(), {"units": "1", "long_name": "fraction of the hour's part1 scans sampled through the GCVI"}),
        "residual_psd": (("time", "res_diameter"), hr.to_numpy(), {"units": "cm-3", "long_name": "cloud-residual dN/dlogD (part1 DMPS behind GCVI)"}),
        "wholeair_psd_on_res": (("time", "res_diameter"), h2t.to_numpy(), {"units": "cm-3", "long_name": "whole-air dN/dlogD interpolated to the residual grid"}),
        "activated_fraction": (("time", "res_diameter"), af, {"units": "1", "long_name": "residual / whole-air dN/dlogD"}),
        "D50": ("time", D50, {"units": "nm", "long_name": "first upward crossing of activated fraction = 0.5"}),
    }
    for k in met:
        common[k] = ("time", h1[k].to_numpy(), {"long_name": f"{k}, mean of GCVI scans", "units": p1[{"updraft": "updraft_velocity", "temperature": "temperature", "wind_speed": "wind_speed", "visibility": "visibility_mean", "enrichment_factor": "enrichment_factor"}[k]].attrs.get("units", "1")})
    coords = {"time": idx.values, "radius": ("radius", RADII_NM, {"units": "nm", "long_name": "cutoff radius"}),
              "diameter": ("radius", cut_nm, {"units": "nm", "long_name": "cutoff diameter"}),
              "res_diameter": ("res_diameter", dr_nm, {"units": "nm"})}
    for tag, var, desc in [("fm120", "Nd_fm120", "FM-120 droplet number"), ("residual", "Nd_residual", "GCVI residual number")]:
        dv = dict(common)
        dv["CDNC"] = ("time", h1[var].to_numpy(), {"units": "cm-3", "long_name": f"Nd = {desc}"})
        ds = xr.Dataset(dv, coords=coords, attrs={
            "site": "Zeppelin (GCVI, Karlsson et al. 2021)", "latitude": 78.9, "longitude": 11.9,
            "source": "Karlsson et al. 2021, ACP, doi:10.5194/acp-21-8933-2021; data doi:10.17043/zeppelin-cloud-aerosol-1 (CC BY 4.0)",
            "nd_source": var, "part2_range_nm": f"{d2[0]:.1f}-{d2[-1]:.0f}",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        out = os.path.join(a.out_dir, f"zeppelin_gcvi_{tag}.nc")
        ds.to_netcdf(out, encoding={v: {"zlib": True} for v in ds.data_vars})
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
