#!/usr/bin/env python
"""Arctic Ocean 2018 (icebreaker Oden): hourly fog droplet number (Leeds FSSP-100) paired with the
aerosol size distribution (SU DMPS, 10-921 nm), in the same layout as the other builds, so that
`arm_ena_nd_pnsd.py quicklook` runs on the output unchanged.

    python ao2018_build.py --fssp-dir data/ao2018/fssp --dmps data/ao2018/ao2018-aerosol-dmps.csv \\
        --out results/ao2018/ao2018_fog.nc
    python arm_ena_nd_pnsd.py quicklook results/ao2018/ao2018_fog.nc --min-cloud-frac 0.5

Data: Brooks, Brooks & Achtert (2020), doi:10.17043/oden-ao-2018-aerosol-fssp-1 (acknowledge the University
of Leeds); Karlsson & Zieger (2020), doi:10.17043/oden-ao-2018-aerosol-dmps-1.

Method
 1. FSSP (1-min): droplet number = sum of the per-bin concentrations with bin centre in [--nd-dmin, --nd-dmax] um,
    using only bins whose qc_flag is "good" (meaning read from the file). LWC from the same bins.
    A minute is "in fog" when Nd >= --min-nd and LWC >= --min-lwc.
 2. Hourly: Nd = median over in-fog minutes; cloud_frac = in-fog minutes / 60.
 3. Aerosol: DMPS scans (9 min) with the hour, median dN/dlogDp, then N(D > cutoff) on the common grid.
    --aerosol same-hour uses scans inside the fog hour (the DMPS samples whole air, so activated particles are
    included as dried residuals; the inlet's transmission of large fog droplets is not characterised).
    --aerosol clear uses fog-free scans within +/- --window-h hours instead (as for EPCAPE Mt. Soledad).
 4. Ship exhaust: scans whose total number exceeds --spike-factor times the running 6-h median are dropped
    (0 = off); with --pollution-mask, only scans entirely within clean minutes of the AO2018 pollution mask are kept.
"""
import argparse
import glob
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_ena_nd_pnsd import RADII_NM, cumulative_above  # noqa: E402


def _good_values(qc):
    """qc flag values whose meaning says good (NCAS-AMF convention: 1 = good_data)."""
    vals = qc.attrs.get("flag_values")
    means = str(qc.attrs.get("flag_meanings", "")).split()
    if vals is not None and len(means) == len(np.atleast_1d(vals)):
        good = [int(v) for v, m in zip(np.atleast_1d(vals), means) if "good" in m.lower()]
        if good:
            return good
    return [1]


def _to_um(x, units):
    u = str(units).lower()
    if u in ("m", "metre", "meter", "meters", "metres"):
        return x * 1e6
    if "nm" in u:
        return x / 1000.0
    return x


def _to_cm3(x, units):
    u = str(units).lower().replace(" ", "")
    if "cm" in u or "cc" in u:                 # cm-3, cm^-3, #/cm3, cc-1: already per cm3
        return x
    if "m-3" in u or "/m3" in u or "m^-3" in u:
        return x / 1e6
    if "l-1" in u or "/l" in u:
        return x / 1000.0
    return x


def read_fssp(folder, dmin, dmax):
    files = sorted(glob.glob(os.path.join(folder, "**", "*size-distribution*.nc"), recursive=True))
    if not files:
        sys.exit(f"No FSSP *size-distribution*.nc files under {folder}")
    parts, report = [], None
    for f in files:
        with xr.open_dataset(f) as ds:
            D = _to_um(ds["ambient_aerosol_particle_diameter"].values.astype(float),
                       ds["ambient_aerosol_particle_diameter"].attrs.get("units", "um"))
            nb = ds["ambient_aerosol_number_concentration_per_bin"]
            n = _to_cm3(nb.values.astype(float), nb.attrs.get("units", "cm-3"))
            qc = ds["qc_flag"]
            good = np.isin(qc.values, _good_values(qc))
            if report is None:
                report = (f"FSSP: per-bin units '{nb.attrs.get('units', '?')}', diameter units "
                          f"'{ds['ambient_aerosol_particle_diameter'].attrs.get('units', '?')}', "
                          f"qc good = {_good_values(qc)} ({qc.attrs.get('flag_meanings', 'no meanings')})")
            t = pd.DatetimeIndex(ds["time"].values)
        n = np.where(good & np.isfinite(n) & (n >= 0), n, np.nan)
        inr = (D >= dmin) & (D <= dmax)
        valid = (good & inr).any(axis=1)                          # minute has at least one good bin in range
        nd = np.where(valid, np.nansum(np.where(inr, n, 0.0), axis=1), np.nan)
        lwc = np.where(valid, np.nansum(np.where(inr, n * (np.pi / 6) * (D * 1e-4) ** 3, 0.0), axis=1) * 1e6, np.nan)
        # n cm-3 * (cm)^3 = cm3 water per cm3 air; * 1e6 g/m3 per (cm3/cm3) for water
        parts.append(pd.DataFrame({"Nd": nd, "LWC": lwc, "good_frac": good.mean(axis=1)}, index=t))
    print(report)
    c = pd.concat(parts).sort_index()
    return c[~c.index.duplicated()]


def read_dmps(path):
    raw = pd.read_csv(path, header=None, low_memory=False)
    diam_row = raw.iloc[1].astype(str).tolist()
    dcols, dnm = [], []
    for j, s in enumerate(diam_row):
        m = re.search(r"([0-9.]+[eE][-+]?\d+)\s*m", s)
        if m:
            dcols.append(j)
            dnm.append(float(m.group(1)) * 1e9)
    t = pd.to_datetime(raw.iloc[2:, 0], errors="coerce")
    ntot = pd.to_numeric(raw.iloc[2:, 1], errors="coerce").to_numpy()
    a = np.array(raw.iloc[2:, dcols].apply(pd.to_numeric, errors="coerce").to_numpy(float), copy=True)
    a[a < 0] = np.nan
    keep = t.notna().to_numpy()
    df = pd.DataFrame(a[keep], index=pd.DatetimeIndex(t[keep]))
    df["Ntot"] = ntot[keep]
    return df.sort_index(), np.array(dnm)


# ---- Ship-exhaust mask (Baccarini, Schmale & Dommen, doi:10.17043/ao2018-aerosol-pollution-mask): 1-min
# "Pollution flag" for the 4th-deck inlet, 2018-08-01 to 2018-09-19. A scan is kept when all mask minutes in its
# window are clean (--max-polluted-frac 0) and the mask covers it; scans outside the mask period are dropped.

def read_pollution_mask(path):
    m = pd.read_csv(path)
    t = pd.to_datetime(m.iloc[:, 0], errors="coerce")
    f = pd.to_numeric(m.iloc[:, 1], errors="coerce")
    ok = t.notna().to_numpy()
    return pd.Series(f.to_numpy()[ok], index=pd.DatetimeIndex(t[ok])).sort_index()


def polluted_fraction(times, mask, clean_values, scan_min=9.0):
    """Fraction of mask minutes in [t, t + scan_min) that are not clean; NaN where the mask has no data."""
    times = pd.DatetimeIndex(times)
    v = mask.to_numpy(float)
    ok = np.isfinite(v)
    pol = ok & ~np.isin(v, clean_values)
    c = np.concatenate([[0], np.cumsum(pol)])
    n = np.concatenate([[0], np.cumsum(ok)])
    ti = mask.index.values
    i0 = np.searchsorted(ti, times.values)
    i1 = np.searchsorted(ti, (times + pd.Timedelta(minutes=scan_min)).values)
    cov = n[i1] - n[i0]
    return np.where(cov > 0, (c[i1] - c[i0]) / np.maximum(cov, 1), np.nan)


def clean_flag_auto(mask, times, ntot, scan_min=9.0):
    """Print the total particle number in scans that lie entirely within each flag value; the clean value is the
    one with the lowest median (exhaust adds particles). Returns [value]."""
    vals = sorted(pd.unique(mask.dropna()))
    print(f"Pollution mask: {len(mask)} min, {mask.index.min():%Y-%m-%d %H:%M} to {mask.index.max():%Y-%m-%d %H:%M}")
    print("  flag  share of minutes  scans entirely in flag  median N_tot (cm-3)")
    best, best_n = None, np.inf
    for v in vals:
        inside = polluted_fraction(times, mask, [v], scan_min) == 0
        med = float(np.nanmedian(np.asarray(ntot)[inside])) if inside.any() else np.nan
        print(f"  {v:4g}  {np.mean(mask == v):16.1%}  {int(inside.sum()):22d}  {med:19.0f}")
        if np.isfinite(med) and med < best_n:
            best, best_n = v, med
    print(f"  -> clean flag = {best:g} (lowest particle number; override with --clean-flag)")
    return [float(best)]


def apply_pollution_mask(times, ntot, path, clean="auto", max_frac=0.0, scan_min=9.0):
    """Boolean keep-array for scans at `times` (ntot = total number per scan, used for the auto choice)."""
    mask = read_pollution_mask(path)
    vals = clean_flag_auto(mask, times, ntot, scan_min) if clean == "auto" else [float(x) for x in str(clean).split(",")]
    fr = polluted_fraction(times, mask, vals, scan_min)
    keep = np.isfinite(fr) & (fr <= max_frac)
    print(f"Pollution mask (clean = {vals}, max polluted fraction {max_frac:g}): keeps {int(keep.sum())} of {len(keep)} "
          f"scans; {int((~np.isfinite(fr)).sum())} outside the mask period, {int((np.isfinite(fr) & (fr > max_frac)).sum())} polluted")
    return keep


def add_mask_args(p):
    p.add_argument("--pollution-mask", default=None, help="ao2018-aerosol-pollution-mask.csv (Bolin Centre); off if unset")
    p.add_argument("--clean-flag", default="auto", help="flag value(s) meaning clean, e.g. 1 or 0,1 (default: auto)")
    p.add_argument("--max-polluted-frac", type=float, default=0.0, help="largest polluted share of a scan's minutes")
    p.add_argument("--scan-min", type=float, default=9.0, help="DMPS scan length (min) used to window the mask")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fssp-dir", required=True)
    p.add_argument("--dmps", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--site-name", default="AO2018 Oden fog (FSSP + DMPS)")
    p.add_argument("--nd-dmin", type=float, default=2.0, help="um, smallest droplet counted (haze below)")
    p.add_argument("--nd-dmax", type=float, default=50.0, help="um")
    p.add_argument("--min-nd", type=float, default=5.0, help="cm-3, in-fog minute threshold")
    p.add_argument("--min-lwc", type=float, default=0.01, help="g m-3, in-fog minute threshold")
    p.add_argument("--min-cloud-frac", type=float, default=0.1, help="floor for keeping an hour")
    p.add_argument("--aerosol", choices=["same-hour", "clear"], default="same-hour")
    p.add_argument("--window-h", type=float, default=3.0, help="hours, for --aerosol clear")
    p.add_argument("--min-scans", type=int, default=2)
    p.add_argument("--nd-stat", choices=["mean", "median"], default="mean",
                   help="how in-fog minutes are combined into hourly Nd (mean = in-cloud average as in the models; at ENA the hourly-mean NDROP matched the aircraft, the median was about 40 per cent low)")
    p.add_argument("--spike-factor", type=float, default=3.0, help="drop DMPS scans above this x 6-h median (0 = off)")
    add_mask_args(p)
    a = p.parse_args()

    c = read_fssp(a.fssp_dir, a.nd_dmin, a.nd_dmax)
    fog = (c.Nd >= a.min_nd) & (c.LWC >= a.min_lwc)
    print(f"FSSP: {len(c)} min on {c.index.normalize().nunique()} days, {c.Nd.notna().sum()} with good bins; "
          f"in fog: {int(fog.sum())} min; median in-fog Nd {c.Nd[fog].median():.0f} cm-3, LWC {c.LWC[fog].median():.3f} g m-3")
    g = c[fog].groupby(c[fog].index.floor("h"))
    nvalid = c.Nd.notna().groupby(c.index.floor("h")).sum()
    hourly = pd.DataFrame({"Nd": g.Nd.mean() if a.nd_stat == "mean" else g.Nd.median(),
                           "LWC": g.LWC.mean() if a.nd_stat == "mean" else g.LWC.median(), "n_fog_min": g.Nd.size()})
    hourly["cloud_frac"] = hourly.n_fog_min / 60.0
    hourly["valid_min"] = nvalid.reindex(hourly.index)
    hourly["nd_iqr_ratio"] = g.Nd.quantile(0.75) / g.Nd.quantile(0.25)
    hourly = hourly[hourly.cloud_frac >= a.min_cloud_frac]
    print(f"Fog hours (cloud_frac >= {a.min_cloud_frac}): {len(hourly)}; >= 0.5: {(hourly.cloud_frac >= 0.5).sum()}")

    dm, dnm = read_dmps(a.dmps)
    if a.spike_factor > 0:
        base = dm["Ntot"].rolling("6h", center=True, min_periods=5).median()
        spike = dm["Ntot"] > a.spike_factor * base
        print(f"DMPS: {len(dm)} scans, {dnm.min():.0f}-{dnm.max():.0f} nm; spike screen removes {spike.mean():.1%} (ship exhaust proxy)")
        dm = dm[~spike.fillna(False)]
    if a.pollution_mask:
        dm = dm[apply_pollution_mask(dm.index, dm["Ntot"], a.pollution_mask, a.clean_flag, a.max_polluted_frac, a.scan_min)]
    bins = dm.drop(columns="Ntot")
    fog_hours = set(hourly.index)
    cut = 2.0 * RADII_NM
    rows = []
    for h in hourly.index:
        if a.aerosol == "same-hour":
            s = bins[(bins.index >= h) & (bins.index < h + pd.Timedelta(hours=1))]
        else:
            w = (bins.index >= h - pd.Timedelta(hours=a.window_h)) & (bins.index < h + pd.Timedelta(hours=1 + a.window_h))
            clear = ~bins.index.floor("h").isin(fog_hours)
            s = bins[w & clear]
        if len(s) < a.min_scans:
            rows.append(None)
            continue
        rows.append(np.nanmedian(s.to_numpy(), axis=0))
    keep = [i for i, r in enumerate(rows) if r is not None]
    if not keep:
        sys.exit("No fog hour has enough DMPS scans.")
    hourly = hourly.iloc[keep]
    psd = np.array([rows[i] for i in keep])
    ccn = cumulative_above(psd, dnm, cut, None)
    band = cumulative_above(psd, dnm, np.array([dnm.min(), 20.0, 70.0, 100.0, 500.0]), None)
    print(f"Paired fog hours: {len(hourly)} on {hourly.index.normalize().nunique()} days "
          f"(cloud_frac >= 0.5: {(hourly.cloud_frac >= 0.5).sum()}); median Nd {hourly.Nd.median():.0f}, "
          f"N(>100 nm) {np.nanmedian(ccn[:, np.argmin(abs(cut - 100))]):.0f} cm-3")

    t = hourly.index
    ds = xr.Dataset(
        {"CDNC": ("time", hourly.Nd.to_numpy(), {"units": "cm-3", "long_name": f"hourly {a.nd_stat} FSSP fog droplet number, {a.nd_dmin:g}-{a.nd_dmax:g} um"}),
         "CCN": (("time", "radius"), ccn, {"units": "cm-3", "long_name": f"N(D > 2*radius), DMPS ({a.aerosol})"}),
         "cloud_frac": ("time", hourly.cloud_frac.to_numpy(), {"units": "1", "long_name": "fraction of the hour in fog (FSSP)"}),
         "dNdlogDp": (("time", "aerosol_diameter"), psd, {"units": "cm-3"}),
         "LWC": ("time", hourly.LWC.to_numpy(), {"units": "g m-3"}),
         "nd_iqr_ratio": ("time", hourly.nd_iqr_ratio.to_numpy()),
         "N_tot": ("time", band[:, 0]), "N_aitken_20_100": ("time", band[:, 1] - band[:, 3]),
         "N70_500": ("time", band[:, 2] - np.nan_to_num(band[:, 4]))},
        coords={"time": t.values, "radius": ("radius", RADII_NM, {"units": "nm"}),
                "diameter": ("radius", cut, {"units": "nm"}), "aerosol_diameter": ("aerosol_diameter", dnm, {"units": "nm"})},
        attrs={"site": a.site_name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "settings": " ".join(f"{k}={v}" for k, v in vars(a).items() if k != "out"),
               "nd_source": "Leeds FSSP-100, doi:10.17043/oden-ao-2018-aerosol-fssp-1 (University of Leeds)",
               "psd_source": "SU DMPS, doi:10.17043/oden-ao-2018-aerosol-dmps-1",
               "note": ("Ship-level fog; DMPS scans screened with the AO2018 pollution mask (doi:10.17043/ao2018-aerosol-pollution-mask)"
                        if a.pollution_mask else "Ship-level fog; spike screen only (no pollution mask)")})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    ds.to_netcdf(a.out)
    print(f"Wrote {a.out}. Run: python arm_ena_nd_pnsd.py quicklook {a.out} --min-cloud-frac 0.5")


if __name__ == "__main__":
    main()
