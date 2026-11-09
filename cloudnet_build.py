#!/usr/bin/env python
"""Hourly Cloudnet droplet number (N_scaled) matched with EBAS size distributions.

Droplet number comes from the Cloudnet 'der' product: the scaled Frisch et al. (2002)
retrieval (radar reflectivity, scaled to MWR liquid water path), one value per
profile (N is constant through the liquid layer). Aerosol comes from an EBAS
NetCDF size distribution (DMPS/SMPS, hourly, dN/dlogDp in cm-3), integrated above
each cutoff exactly as in arm_ena_nd_pnsd.py. The output file has the same layout,
so it runs through `arm_ena_nd_pnsd.py quicklook` unchanged.

Profile screen (per 30-s profile):
  * at least --min-pixels liquid pixels with retrieval status 1 (reliable);
  * no pixel in the profile with status 2 (drops + ice) or 3 (precipitation);
  * lowest reliable pixel below --max-base m above ground;
  * N within --n-range cm-3 (removes non-physical retrievals; count reported).
Hourly: median N over valid profiles; cloud_frac = valid profiles / profiles per hour.

Example
-------
python cloudnet_build.py --der-dir data/cloudnet/hyytiala/der \
    --psd data/ebas/FI0050R.20130101000000.20260204141850.dmps...13y.1h.FI03L_UHEL_DMPS_HYY_01.FI03L__TRY_TDMPS.lev2.nc \
    --site-name "Hyytiälä (Cloudnet + SMEAR II DMPS)" --out hyytiala_cloudnet.nc
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
from arm_ena_nd_pnsd import RADII_NM, cumulative_above, read_psd_hourly  # noqa: E402


def read_der(der_dir, min_pixels, max_base, n_range, stat="mean"):
    files = sorted(glob.glob(os.path.join(der_dir, "*_der.nc")))
    if not files:
        sys.exit(f"No *_der.nc files in {der_dir}")
    rows, removed, dt_s = [], 0, None
    for f in files:
        with xr.open_dataset(f) as ds:
            N = ds["N_scaled"].values.astype(float) / 1e6           # m-3 -> cm-3
            st = ds["der_retrieval_status"].values
            h = ds["height"].values.astype(float)
            alt = float(np.nanmean(ds["altitude"].values))
            t = pd.DatetimeIndex(ds["time"].values)
        if dt_s is None and len(t) > 2:
            dt_s = float(np.median(np.diff(t.values).astype("timedelta64[s]").astype(float)))
        ok = (st == 1) & np.isfinite(N)
        bad = ((st == 2) | (st == 3)).any(axis=1)
        n_ok = ok.sum(axis=1)
        for i in np.where((n_ok >= min_pixels) & ~bad)[0]:
            hz = h[ok[i]] - alt
            base, top = hz.min(), hz.max()
            if base > max_base:
                continue
            n = float(np.nanmedian(N[i, ok[i]]))
            if not (n_range[0] <= n <= n_range[1]):
                removed += 1
                continue
            rows.append((t[i], n, base, top))
    if not rows:
        sys.exit("No valid profiles after screening.")
    prof = pd.DataFrame(rows, columns=["time", "N", "base", "top"]).set_index("time")
    per_hour = 3600.0 / (dt_s or 30.0)
    g = prof.groupby(prof.index.floor("h"))
    hourly = pd.DataFrame({
        "Nd": g["N"].mean() if stat == "mean" else g["N"].median(),
        "cloud_frac": g["N"].size() / per_hour,
        "cloud_base_height": g["base"].median(),
        "cloud_top_height": g["top"].median(),
        "nd_iqr_ratio": g["N"].quantile(0.75) / g["N"].quantile(0.25),
    })
    print(f"Cloudnet: {len(files)} daily files, {len(prof)} valid profiles "
          f"({removed} removed outside {n_range[0]}-{n_range[1]} cm-3), "
          f"{len(hourly)} hours with any; >=0.5: {(hourly.cloud_frac >= 0.5).sum()}")
    return hourly


def read_lwp(lwc_dir):
    """Optional hourly mean LWP (kg m-2) from Cloudnet 'lwc' (or 'mwr') files."""
    files = sorted(glob.glob(os.path.join(lwc_dir, "*.nc")))
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            if "lwp" not in ds:
                continue
            v = ds["lwp"].values.astype(float)
            units = str(ds["lwp"].attrs.get("units", "kg m-2")).replace(" ", "")
            if units.startswith("g"):
                v = v / 1000.0
            parts.append(pd.Series(v, index=pd.DatetimeIndex(ds["time"].values)))
    if not parts:
        return None
    s = pd.concat(parts)
    return s.groupby(s.index.floor("h")).mean()


def read_ebas_psd(paths, cut_nm):
    """Hourly CCN(D) from EBAS NetCDF dN/dlogDp (cm-3); several files are merged."""
    frames, auxes, dmin, dmax = [], [], np.inf, 0
    for p in paths:
        with xr.open_dataset(p) as ds:
            name = next((v for v in ("particle_number_size_distribution_amean",
                                     "particle_number_size_distribution") if v in ds), None)
            if name is None:
                print(f"skip {os.path.basename(p)}: no particle_number_size_distribution")
                continue
            a = ds[name].transpose("time", "D").values.astype(float)
            d = ds["D"].values.astype(float)
            t = pd.DatetimeIndex(ds["time"].values).floor("h")
        good = np.isfinite(a).mean(axis=0) >= 0.5          # bins valid in >= half the hours
        a, d = a[:, good], d[good]
        dmin, dmax = min(dmin, d.min()), max(dmax, d.max())
        ccn = cumulative_above(a, d, cut_nm, None)
        frames.append(pd.DataFrame(ccn, index=t))
        aux = pd.DataFrame({
            "N_tot": cumulative_above(a, d, np.array([d.min()]), None)[:, 0],
            "N_aitken_20_100": (cumulative_above(a, d, np.array([20.0]), None)[:, 0]
                                - cumulative_above(a, d, np.array([100.0]), None)[:, 0]),
        }, index=t)
        auxes.append(aux)
    if not frames:
        sys.exit("No usable EBAS size-distribution files.")
    ccn = pd.concat(frames)
    aux = pd.concat(auxes)
    ccn = ccn[~ccn.index.duplicated(keep="last")]
    aux = aux[~aux.index.duplicated(keep="last")]
    ccn = ccn[ccn.notna().any(axis=1)]
    return ccn, aux.loc[ccn.index], (dmin, dmax)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--der-dir", required=True)
    p.add_argument("--psd", nargs="+", required=True, help="EBAS NetCDF size-distribution file(s), or one ARM SMPS/UHSAS folder")
    p.add_argument("--lwc-dir", default=None, help="optional Cloudnet lwc/mwr files for LWP")
    p.add_argument("--out", required=True)
    p.add_argument("--site-name", default="Cloudnet site")
    p.add_argument("--min-pixels", type=int, default=3)
    p.add_argument("--max-base", type=float, default=2500.0, help="m above ground")
    p.add_argument("--n-range", type=float, nargs=2, default=[1.0, 3000.0], help="cm-3")
    p.add_argument("--min-cloud-frac", type=float, default=0.3)
    p.add_argument("--nd-stat", choices=["mean", "median"], default="mean",
                   help="how valid profiles are combined into hourly Nd (mean = in-cloud average as in the models; at ENA the hourly-mean NDROP matched the aircraft, the median was about 40 per cent low)")
    a = p.parse_args()

    cut_nm = 2.0 * RADII_NM
    nd = read_der(a.der_dir, a.min_pixels, a.max_base, a.n_range, a.nd_stat)
    nd = nd[nd.cloud_frac >= a.min_cloud_frac]
    if len(a.psd) == 1 and os.path.isdir(a.psd[0]):
        # An ARM size-distribution folder (e.g. data/sgp/smps): use the ARM reader.
        ccn, aux, dr = read_psd_hourly(a.psd[0], None, "strict", cut_nm, 0.5)
        ccn.columns = range(ccn.shape[1])
    else:
        ccn, aux, dr = read_ebas_psd(a.psd, cut_nm)
    print(f"PSD: {len(ccn)} valid hours, {dr[0]:.1f}-{dr[1]:.1f} nm")
    hours = nd.index.intersection(ccn.index)
    if len(hours) == 0:
        sys.exit("No overlapping hours.")
    nd, ccn, aux = nd.loc[hours], ccn.loc[hours], aux.loc[hours]

    ds = xr.Dataset(
        {"CDNC": ("time", nd["Nd"].to_numpy(),
                  {"units": "cm-3", "long_name": f"hourly {a.nd_stat} Cloudnet N_scaled (scaled Frisch)"}),
         "CCN": (("time", "radius"), ccn.to_numpy(),
                 {"units": "cm-3", "long_name": "N(D > 2*radius) from hourly dN/dlogDp (EBAS)"}),
         "cloud_frac": ("time", nd["cloud_frac"].to_numpy(),
                        {"units": "1", "long_name": "fraction of the hour with a valid profile"}),
         "cloud_base_height": ("time", nd["cloud_base_height"].to_numpy(),
                               {"units": "m", "long_name": "lowest reliable liquid pixel, above ground"}),
         "cloud_top_height": ("time", nd["cloud_top_height"].to_numpy(),
                              {"units": "m", "long_name": "highest reliable liquid pixel, above ground"}),
         "nd_iqr_ratio": ("time", nd["nd_iqr_ratio"].to_numpy(),
                          {"units": "1", "long_name": "within-hour Q75/Q25 of N_scaled"})},
        coords={"time": hours.values,
                "radius": ("radius", RADII_NM, {"units": "nm", "long_name": "cutoff radius"}),
                "diameter": ("radius", cut_nm, {"units": "nm", "long_name": "cutoff diameter"})},
        attrs={"site": a.site_name, "der_dir": os.path.abspath(a.der_dir),
               "psd_files": ";".join(os.path.basename(x) for x in a.psd),
               "screens": f"status==1 pixels >= {a.min_pixels}; no status 2/3 in profile; "
                          f"base <= {a.max_base} m; N in {a.n_range} cm-3",
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "note": "Cloudnet scaled-Frisch Nd vs surface EBAS aerosol; check coupling."})
    for c in aux.columns:
        ds[c] = ("time", aux[c].to_numpy(), {"units": "cm-3"})
    if a.lwc_dir:
        lwp = read_lwp(a.lwc_dir)
        if lwp is not None:
            ds["lwp_meas"] = ("time", lwp.reindex(hours).to_numpy(),
                              {"units": "kg m-2", "long_name": "hourly mean MWR LWP (Cloudnet)"})
    ds.to_netcdf(a.out, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    cf = ds["cloud_frac"].values
    print(f"Wrote {a.out}: {len(hours)} matched hours (cloud_frac >= 0.5: {(cf >= 0.5).sum()}, "
          f">= 0.8: {(cf >= 0.8).sum()}). Run: python arm_ena_nd_pnsd.py quicklook {a.out}")


if __name__ == "__main__":
    main()
