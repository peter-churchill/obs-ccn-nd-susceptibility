#!/usr/bin/env python
"""Pair in-cloud droplet number with below-cloud aerosol from ARM G-1 aircraft data.

Inputs (one folder each, files from arm_ena_nd_pnsd.py download):
    --cloud-dir    <site>aafmergedcldsdF1.c1  merged FCDP+2DS+HVPS size distribution, 1 Hz
                   (or <site>aaffcdpF1.c1 for campaigns without the merged product)
    --aerosol-dir  <site>aafmergedaerosolsdF1.c1  merged FIMS/PCASP/CAS size distribution, 1 Hz
                   with cloud_flag and cvi_flag; or a single probe for campaigns without it:
                   <site>aafuhsasF1.b1 (60-1000 nm), <site>aaffimsF1.b1 (10-~500 nm, heated_flag
                   used), <site>aafpcaspF1.b1 (~100-3000 nm, cloud_flag and data_flag used)

Method, per flight:
 1. In-cloud seconds: droplet number (2-50 um, so drizzle is excluded) >= --min-nd and
    liquid water content (from the same distribution) >= --min-lwc.
 2. Cloud passes: runs of in-cloud seconds (gaps up to --gap s), at least --seg-min s long.
    Nd of a pass = median over its seconds.
 3. Local cloud base = 5th percentile of in-cloud altitudes within +/- --max-dt of the pass.
    Keep passes no more than --max-above-base above that base (Nd is best defined near base).
 4. Below-cloud aerosol: seconds with cloud_flag == 0 and cvi_flag == 0 (isokinetic inlet),
    below base - --below-margin, within +/- --max-dt and --max-dist km of the pass.
    Need at least --min-aer s. The aerosol of a pass = median size distribution of those seconds.
 5. N(D > cutoff) on the same cutoff grid as the ground sites.

Output has the same layout as the other builds (CDNC, CCN(radius), cloud_frac = 1), one row
per cloud pass, so `arm_ena_nd_pnsd.py quicklook` runs on it unchanged. The day-block
bootstrap there resamples whole flights (one flight per day).

Aerosol seconds near droplets (>= --clear-nd cm-3 within 2 s, from the cloud probe) are
always dropped, so single probes without a cloud flag are usable.

Examples
--------
python aircraft_build.py --cloud-dir data/aircraft/ace-ena/enaaafmergedcldsdF1.c1 \\
    --aerosol-dir data/aircraft/ace-ena/enaaafmergedaerosolsdF1.c1 \\
    --site-name "ACE-ENA aircraft (G-1)" --out results/aircraft/ace-ena_aircraft.nc
python aircraft_build.py --cloud-dir data/aircraft/acapex/acxaaffcdpF1.c1 \\
    --aerosol-dir data/aircraft/acapex/acxaafuhsasF1.b1 \\
    --site-name "ACAPEX aircraft (G-1, UHSAS)" --out results/aircraft/acapex_aircraft_uhsas.nc
python aircraft_build.py --format activate --cloud-dir data/aircraft/activate/cloud \\
    --aerosol-dir data/aircraft/activate/aerosol --site-name "ACTIVATE (HU-25 Falcon)" \\
    --out results/aircraft/activate_aircraft.nc
(ACTIVATE: FCDP 3-50 um droplets; SMPS 3-94 nm + LAS 94-3490 nm dry aerosol converted to ambient;
 CVI-inlet seconds dropped; see activate_reader.py)
"""
import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_ena_nd_pnsd import RADII_NM, cumulative_above  # noqa: E402


def _diam_um(ds, dim):
    """Bin edges and centres in micrometres from <dim>_bounds (units guessed if missing)."""
    b = ds[f"{dim}_bounds"].values.astype(float)
    units = str(ds[dim].attrs.get("units", ds[f"{dim}_bounds"].attrs.get("units", ""))).lower()
    if "nm" in units:
        b = b / 1000.0
    elif "um" in units or "µm" in units or "micro" in units:
        pass
    elif "m" == units.strip():
        b = b * 1e6
    elif np.nanmax(b) > 5000:          # no units: nm if the top edge is huge for an aerosol probe
        b = b / 1000.0
    return b, np.sqrt(b[:, 0] * b[:, 1])


def read_cloud(folder, dmin, dmax):
    """1-Hz droplet number (cm-3) and LWC (g m-3) for dmin-dmax um, plus position."""
    files = sorted(glob.glob(os.path.join(folder, "*.nc")) + glob.glob(os.path.join(folder, "*.cdf")))
    if not files:
        sys.exit(f"No files in {folder}")
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            b, dc = _diam_um(ds, "optical_diameter")
            n = ds["number_concentration"].values.astype(float)        # count/L/um
            n[n < 0] = np.nan
            units = str(ds["number_concentration"].attrs.get("units", "")).lower()
            per_um = "/um" in units or "um-1" in units or "/µm" in units
            lo, hi = np.maximum(b[:, 0], dmin), np.minimum(b[:, 1], dmax)
            w = np.clip(hi - lo, 0, None)                              # um of each bin inside the range
            if not per_um:                                             # per bin: weight by overlap fraction
                w = w / (b[:, 1] - b[:, 0])
            nd = np.nansum(n * w, axis=1) / 1000.0                     # count/L -> cm-3
            v = (np.pi / 6.0) * (dc * 1e-6) ** 3                       # m3 per droplet
            lwc = np.nansum(n * w * v, axis=1) * 1000.0 * 1e6          # (count/L)*m3 -> m3/m3 -> g/m3
            parts.append(pd.DataFrame({"Nd": nd, "LWC": lwc, "alt": ds["alt"].values,
                                       "lat": ds["lat"].values, "lon": ds["lon"].values},
                                      index=pd.DatetimeIndex(ds["time"].values)))
    c = pd.concat(parts).sort_index()
    return c[~c.index.duplicated()]


AER_FLAGS = ("cloud_flag", "cvi_flag", "heated_flag", "data_flag")   # 0 = keep


def read_aerosol(folder, cloud=None, clear_nd=1.0, pad_s=2, form="auto"):
    """1-Hz dN/dlogDp (cm-3) for clear-air seconds, plus position, on one diameter grid.

    Reads the merged aerosol product (number_concentration on geometric_diameter) and the
    single probes (UHSAS: size_distribution on diameter_optical; PCASP: size_distribution on
    optical_diameter; FIMS: number_concentration on geometric_diameter). Whether values are
    per bin or dN/dlogDp is decided per file from total_number_concentration when present,
    else from the long name. Seconds are dropped where any of AER_FLAGS is non-zero, and,
    when `cloud` (the droplet-probe series) is given, within pad_s of a second with droplet
    number >= clear_nd cm-3 (for probes without a cloud flag of their own).
    """
    files = sorted(glob.glob(os.path.join(folder, "*.nc")) + glob.glob(os.path.join(folder, "*.cdf")))
    if not files:
        sys.exit(f"No files in {folder}")
    frames, meta, grid, kept = [], [], None, {}
    for f in files:
        with xr.open_dataset(f) as ds:
            var = next(v for v in ("number_concentration", "size_distribution") if v in ds)
            dim = ds[var].dims[1]
            b, dc = _diam_um(ds, dim)
            a = ds[var].values.astype(float)
            a[a < 0] = np.nan
            dlog = np.log10(b[:, 1] / b[:, 0])
            name = (str(ds[var].attrs.get("long_name", "")) + " " + str(ds[var].attrs.get("units", ""))).lower()
            per_bin = "dlog" not in name
            if form != "auto":
                per_bin = form == "per-bin"
            elif "total_number_concentration" in ds:                      # decide from the file's own total
                tot = ds["total_number_concentration"].values.astype(float)
                good = np.isfinite(tot) & (tot > 0)
                if good.sum() > 10:
                    r_bin = np.nanmedian(np.nansum(a[good], axis=1) / tot[good])
                    r_dlog = np.nanmedian(np.nansum(a[good] * dlog, axis=1) / tot[good])
                    per_bin = abs(np.log(r_bin)) < abs(np.log(r_dlog))
            if per_bin:
                a = a / dlog[None, :]
            ok = np.ones(len(a), bool)
            for flag in AER_FLAGS:
                if flag in ds:
                    fk = np.nan_to_num(ds[flag].values.astype(float), nan=1) == 0
                    if fk.mean() > 0.05:                                # ignore a flag with an unexpected sense
                        ok &= fk
                        kept.setdefault(flag, []).append(fk.mean())
            t = pd.DatetimeIndex(ds["time"].values)
            if grid is None:
                grid = dc * 1000.0                                      # nm, from the first file
            if len(dc) != len(grid) or not np.allclose(dc * 1000.0, grid, rtol=0.02):           # interpolate onto the first grid
                a = np.array([np.interp(np.log10(grid), np.log10(dc * 1000.0), row, left=np.nan, right=np.nan) for row in a])
            frames.append(pd.DataFrame(a[ok], index=t[ok]))
            meta.append(pd.DataFrame({"alt": ds["alt"].values[ok], "lat": ds["lat"].values[ok],
                                      "lon": ds["lon"].values[ok]}, index=t[ok]))
    a = pd.concat(frames).sort_index()
    m = pd.concat(meta).sort_index()
    keep = ~a.index.duplicated()
    a, m = a[keep], m[keep]
    msg = ", ".join(f"{k} keeps {np.mean(v):.0%}" for k, v in kept.items()) or "no flags in file"
    how = "set by --aerosol-form" if form != "auto" else "auto-detected"
    print(f"Aerosol probe: {os.path.basename(os.path.normpath(folder))}; "
          f"values read as {'per-bin counts' if per_bin else 'dN/dlogDp'} ({how}); {msg}")
    if cloud is not None:
        wet = cloud.index[cloud["Nd"] >= clear_nd]
        if len(wet):
            ws = pd.Series(1, index=wet).resample("1s").max().fillna(0)
            ws = ws.rolling(2 * pad_s + 1, center=True, min_periods=1).max()
            hit = ws.reindex(a.index.floor("s")).fillna(0).to_numpy() > 0
            print(f"  droplet-probe cloud mask (Nd >= {clear_nd} cm-3, +/- {pad_s} s) removes {hit.mean():.0%} of aerosol seconds")
            a, m = a[~hit], m[~hit]
    return a, m, grid


def _km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    h = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def cloud_passes(c, a):
    inc = (c["Nd"] >= a.min_nd) & (c["LWC"] >= a.min_lwc)
    s = c[inc]
    if s.empty:
        return pd.DataFrame()
    gap = s.index.to_series().diff().dt.total_seconds().fillna(1e9) > a.gap
    sid = gap.cumsum().to_numpy()
    rows = []
    for k in np.unique(sid):
        g = s[sid == k]
        dur = (g.index[-1] - g.index[0]).total_seconds() + 1
        if len(g) < a.seg_min:
            continue
        rows.append({"time": g.index[0] + (g.index[-1] - g.index[0]) / 2, "Nd": g["Nd"].mean() if a.nd_stat == "mean" else g["Nd"].median(),
                     "Nd_iqr_ratio": g["Nd"].quantile(.75) / max(g["Nd"].quantile(.25), 1e-9),
                     "LWC": g["LWC"].median(), "alt": g["alt"].median(), "lat": g["lat"].median(),
                     "lon": g["lon"].median(), "duration_s": dur, "n_s": len(g)})
    p = pd.DataFrame(rows)
    if p.empty:
        return p
    inc_alt = s["alt"]
    base = []
    for t in p["time"]:
        w = inc_alt[(inc_alt.index >= t - pd.Timedelta(seconds=a.max_dt)) &
                    (inc_alt.index <= t + pd.Timedelta(seconds=a.max_dt))]
        base.append(np.nanpercentile(w, 5))
    p["cloud_base"] = base
    p["above_base"] = p["alt"] - p["cloud_base"]
    return p[p["above_base"] <= a.max_above_base].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud-dir", required=True)
    ap.add_argument("--aerosol-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--site-name", required=True)
    ap.add_argument("--nd-dmin", type=float, default=2.0, help="um")
    ap.add_argument("--nd-dmax", type=float, default=50.0, help="um (excludes drizzle)")
    ap.add_argument("--min-nd", type=float, default=10.0, help="cm-3, in-cloud threshold")
    ap.add_argument("--min-lwc", type=float, default=0.05, help="g m-3, in-cloud threshold")
    ap.add_argument("--gap", type=float, default=5.0, help="s, gap allowed inside a cloud pass")
    ap.add_argument("--seg-min", type=int, default=10, help="s, minimum in-cloud seconds per pass")
    ap.add_argument("--max-above-base", type=float, default=300.0, help="m above local cloud base")
    ap.add_argument("--max-dt", type=float, default=1800.0, help="s, time window for base and aerosol")
    ap.add_argument("--max-dist", type=float, default=50.0, help="km between pass and aerosol samples")
    ap.add_argument("--below-margin", type=float, default=50.0, help="m below cloud base for aerosol")
    ap.add_argument("--min-aer", type=int, default=30, help="s of below-cloud aerosol required")
    ap.add_argument("--aerosol-form", choices=["auto", "per-bin", "dndlogdp"], default="auto",
                    help="how the aerosol file stores its size distribution; auto uses the file's total "
                         "concentration when present, else the long name")
    ap.add_argument("--clear-nd", type=float, default=1.0,
                    help="cm-3; drop aerosol seconds within 2 s of droplets above this (0 = off)")
    ap.add_argument("--format", choices=["arm", "activate", "naames", "camp2ex", "oracles"], default="arm",
                    help="arm: ARM netCDF (default); activate: NASA ACTIVATE ICARTT files "
                         "(FCDP in --cloud-dir, LAS/SMPS/InletFlag in --aerosol-dir, 1-s merges in --merge-dir); "
                         "naames / camp2ex: NASA LaRC LARGE files (CDP + CloudFlag in --cloud-dir, SMPS/LAS in --aerosol-dir, "
                         "HSKPING in --merge-dir), see larc_reader.py")
    ap.add_argument("--merge-dir", default=None,
                    help="ACTIVATE 1-s merge folder for position (default: <cloud-dir>/../merge)")
    ap.add_argument("--nd-stat", choices=["mean", "median"], default="mean",
                    help="how the 1-Hz in-cloud seconds of a pass are combined into its Nd (mean = in-cloud average as in the models; at ENA the hourly-mean NDROP matched the aircraft, the median was about 40 per cent low)")
    ap.add_argument("--activate-stp", action="store_true",
                    help="ACTIVATE: keep aerosol at STP instead of converting to ambient")
    a = ap.parse_args()

    if a.format == "activate":
        from activate_reader import read_cloud_activate, read_aerosol_activate
        mdir = a.merge_dir or os.path.join(os.path.dirname(os.path.normpath(a.cloud_dir)), "merge")
        c = read_cloud_activate(a.cloud_dir, a.nd_dmin, a.nd_dmax, mdir)
        aer, am, dnm = read_aerosol_activate(a.aerosol_dir, mdir, cloud=c if a.clear_nd > 0 else None,
                                             clear_nd=a.clear_nd, stp=a.activate_stp)
    elif a.format in ("naames", "camp2ex"):
        from larc_reader import read_cloud_larc, read_aerosol_larc, CAMPAIGNS
        mdir = a.merge_dir or os.path.join(os.path.dirname(os.path.normpath(a.cloud_dir)), CAMPAIGNS[a.format]["nav_dir"])
        c = read_cloud_larc(a.format, a.cloud_dir, a.nd_dmin, a.nd_dmax, mdir)
        aer, am, dnm = read_aerosol_larc(a.format, a.aerosol_dir, mdir, cloud_dir=a.cloud_dir,
                                         cloud=c if a.clear_nd > 0 else None, clear_nd=a.clear_nd, stp=a.activate_stp)
    elif a.format == "oracles":
        from oracles_reader import read_cloud_oracles, read_aerosol_oracles
        mdir = a.merge_dir or os.path.join(os.path.dirname(os.path.normpath(a.cloud_dir)), "ORACLES_MetNav_AircraftInSitu_Data")
        c = read_cloud_oracles(a.cloud_dir, a.nd_dmin, a.nd_dmax, mdir)
        aer, am, dnm = read_aerosol_oracles(a.aerosol_dir, mdir, cloud=c if a.clear_nd > 0 else None,
                                            clear_nd=a.clear_nd, stp=a.activate_stp)
    else:
        c = read_cloud(a.cloud_dir, a.nd_dmin, a.nd_dmax)
        aer, am, dnm = read_aerosol(a.aerosol_dir, cloud=c if a.clear_nd > 0 else None, clear_nd=a.clear_nd,
                                    form=a.aerosol_form)
    tot = aer.to_numpy() * np.gradient(np.log10(dnm))[None, :]
    print(f"Cloud probe: {len(c)} s on {c.index.normalize().nunique()} flights; "
          f"in cloud (Nd >= {a.min_nd}, LWC >= {a.min_lwc}): {((c.Nd >= a.min_nd) & (c.LWC >= a.min_lwc)).sum()} s; "
          f"median in-cloud Nd {c.Nd[(c.Nd >= a.min_nd) & (c.LWC >= a.min_lwc)].median():.0f} cm-3")
    print(f"Aerosol: {len(aer)} clear-air s, {dnm.min():.1f}-{dnm.max():.0f} nm, "
          f"median N_tot {np.nanmedian(np.nansum(tot, axis=1)):.0f} cm-3 over the probe's range")

    p = cloud_passes(c, a)
    if p.empty:
        sys.exit("No cloud passes found.")
    print(f"Cloud passes within {a.max_above_base:.0f} m of base: {len(p)}")
    cut_nm = 2.0 * RADII_NM
    rows, ccn_rows, pnsd = [], [], []
    for _, r in p.iterrows():
        tw = (am.index >= r.time - pd.Timedelta(seconds=a.max_dt)) & (am.index <= r.time + pd.Timedelta(seconds=a.max_dt))
        below = am["alt"].to_numpy() <= r.cloud_base - a.below_margin
        near = _km(r.lat, r.lon, am["lat"].to_numpy(), am["lon"].to_numpy()) <= a.max_dist
        sel = tw & below & near
        if sel.sum() < a.min_aer:
            continue
        med = np.nanmedian(aer.to_numpy()[sel], axis=0)
        ccn = cumulative_above(med[None, :], dnm, cut_nm, None)[0]
        band = cumulative_above(med[None, :], dnm, np.array([dnm.min(), 20.0, 70.0, 100.0, 500.0]), None)[0]
        dts = np.abs((am.index[sel] - r.time).total_seconds())
        rows.append({**r.to_dict(), "n_aer_s": int(sel.sum()), "aer_alt": float(np.nanmedian(am["alt"].to_numpy()[sel])),
                     "aer_dt_s": float(np.median(dts)),
                     "aer_dist_km": float(np.median(_km(r.lat, r.lon, am["lat"].to_numpy()[sel], am["lon"].to_numpy()[sel]))),
                     "N_tot": band[0], "N_aitken_20_100": band[1] - band[3],
                     "N70_500": band[2] - np.nan_to_num(band[4])})
        ccn_rows.append(ccn)
        pnsd.append(med)
    if not rows:
        sys.exit("No cloud pass has enough below-cloud aerosol nearby; relax --max-dt / --max-dist / --min-aer.")
    d = pd.DataFrame(rows)
    t = pd.DatetimeIndex(d["time"])
    order = np.argsort(t.values)
    d, t = d.iloc[order].reset_index(drop=True), t[order]
    ccn = np.array(ccn_rows)[order]
    pnsd = np.array(pnsd)[order]
    print(f"Paired passes: {len(d)} on {t.normalize().nunique()} flights; median Nd {d.Nd.median():.0f}, "
          f"median N(>100 nm) {np.nanmedian(ccn[:, np.argmin(abs(cut_nm - 100))]):.0f} cm-3; "
          f"aerosol median {d.aer_dt_s.median() / 60:.0f} min and {d.aer_dist_km.median():.0f} km from the pass")

    ds = xr.Dataset(
        {"CDNC": ("time", d["Nd"].to_numpy(), {"units": "cm-3", "long_name": f"{a.nd_stat} in-cloud droplet number, {a.nd_dmin:g}-{a.nd_dmax:g} um, per cloud pass"}),
         "CCN": (("time", "radius"), ccn, {"units": "cm-3", "long_name": "N(D > 2*radius), median below-cloud size distribution"}),
         "cloud_frac": ("time", np.ones(len(d)), {"units": "1", "long_name": "every row is an in-cloud pass"}),
         "dNdlogDp": (("time", "aerosol_diameter"), pnsd, {"units": "cm-3", "long_name": "median below-cloud dN/dlogDp"}),
         **{k: ("time", d[k].to_numpy(dtype=float)) for k in
            ["LWC", "alt", "lat", "lon", "cloud_base", "above_base", "duration_s", "n_s", "Nd_iqr_ratio",
             "n_aer_s", "aer_alt", "aer_dt_s", "aer_dist_km", "N_tot", "N_aitken_20_100", "N70_500"]}},
        coords={"time": t.values,
                "radius": ("radius", RADII_NM, {"units": "nm", "long_name": "cutoff radius"}),
                "diameter": ("radius", cut_nm, {"units": "nm", "long_name": "cutoff diameter"}),
                "aerosol_diameter": ("aerosol_diameter", dnm, {"units": "nm"})},
        attrs={"site": a.site_name, "cloud_dir": os.path.abspath(a.cloud_dir), "aerosol_dir": os.path.abspath(a.aerosol_dir),
               "settings": " ".join(f"{k}={v}" for k, v in vars(a).items() if k not in ("cloud_dir", "aerosol_dir", "out", "site_name")),
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "note": "One row per near-base cloud pass; aerosol from clear-air isokinetic seconds below cloud base nearby."})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    ds.to_netcdf(a.out, encoding={v: {"zlib": True} for v in ds.data_vars})
    print(f"Wrote {a.out}. Run: python arm_ena_nd_pnsd.py quicklook {a.out} --min-cloud-frac 0.5")


if __name__ == "__main__":
    main()
