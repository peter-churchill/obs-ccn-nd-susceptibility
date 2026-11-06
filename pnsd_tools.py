"""Hourly particle number size distributions on one common diameter grid.

Used by site_timeseries_pnsd.ipynb. Reads the raw size-distribution files the
same way the build scripts do (ARM AOS SMPS/UHSAS folders, EBAS NetCDF files,
EPCAPE Mt Soledad SEMS, Virtanen et al. CSVs), averages them to hourly dN/dlogDp, and interpolates
each hour onto GRID_NM (log-spaced, 10-1000 nm). Diameters outside a file's
measured range are NaN, so different instruments can sit on the same grid.

Hourly results are cached as NetCDF next to the site results, so the slow
first read (ENA UHSAS: ~3000 files, 23 GB) is done only once.
"""
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from arm_ena_nd_pnsd import _apply_qc, _find_psd_var, _to_cm3

GRID_NM = np.logspace(1, 3, 81)          # 10-1000 nm, 40 bins per decade


def _to_grid(arr, dp_nm):
    """Interpolate rows of dN/dlogDp (time, bin) from dp_nm onto GRID_NM in log D."""
    order = np.argsort(dp_nm)
    x, arr = np.log10(dp_nm[order]), arr[:, order]
    xg = np.log10(GRID_NM)
    out = np.full((arr.shape[0], GRID_NM.size), np.nan)
    inside = (xg >= x[0]) & (xg <= x[-1])
    for i, row in enumerate(arr):
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            out[i, inside] = np.interp(xg[inside], x[ok], row[ok], left=np.nan, right=np.nan)
    return out


def hourly_pnsd_arm(folder, min_psd_frac=0.5, years=None, progress=True):
    """Hourly dN/dlogDp (cm-3) from an ARM SMPS/UHSAS folder, QC'd as in the builds."""
    files = sorted(glob.glob(os.path.join(folder, "*.nc")) + glob.glob(os.path.join(folder, "*.cdf")))
    if years is not None:
        files = [f for f in files if any(f".{y}" in os.path.basename(f) for y in years)]
    parts = []
    for k, f in enumerate(files, 1):
        if progress and (k % 200 == 0 or k == len(files)):
            print(f"  {os.path.basename(folder)}: {k}/{len(files)} files")
        try:
            with xr.open_dataset(f) as ds:
                name, dp_nm, _ = _find_psd_var(ds, None)
                da = _to_cm3(_apply_qc(ds, name, "strict"), name).astype(float)
                da = da.where(da >= 0)
                df = pd.DataFrame(da.values, index=pd.DatetimeIndex(ds["time"].values))
        except (OSError, ValueError, KeyError, SystemExit) as e:
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        scan_ok = df.notna().any(axis=1)
        if not scan_ok.any():
            continue
        # keep the contiguous span of bins that hold data in most scans (as in the builds)
        good = df[scan_ok].notna().mean(axis=0).to_numpy() >= 0.5
        if not good.any():
            continue
        i0, i1 = np.argmax(good), len(good) - np.argmax(good[::-1])
        df, dp = df.iloc[:, i0:i1], dp_nm[i0:i1]
        dt = np.median(np.diff(df.index.values).astype("timedelta64[s]").astype(float)) if len(df) > 1 else np.nan
        per_hour = 3600.0 / dt if dt and dt > 0 else np.nan
        hourly = df[scan_ok].resample("1h").mean()
        n = df[scan_ok].resample("1h").size()
        if np.isfinite(per_hour):
            hourly = hourly[(n / per_hour) >= min_psd_frac]
        if len(hourly):
            parts.append(pd.DataFrame(_to_grid(hourly.to_numpy(), dp), index=hourly.index))
    return _concat(parts)


def hourly_pnsd_ebas(paths):
    """Hourly dN/dlogDp (cm-3) from EBAS NetCDF files (already hourly)."""
    parts = []
    for p in paths:
        with xr.open_dataset(p) as ds:
            name = next((v for v in ("particle_number_size_distribution_amean",
                                     "particle_number_size_distribution") if v in ds), None)
            if name is None:
                print(f"  skip {os.path.basename(p)}: no particle_number_size_distribution")
                continue
            a = ds[name].transpose("time", "D").values.astype(float)
            d = ds["D"].values.astype(float)
            t = pd.DatetimeIndex(ds["time"].values).floor("h")
        good = np.isfinite(a).mean(axis=0) >= 0.5
        a, d = a[:, good], d[good]
        a[a < 0] = np.nan
        parts.append(pd.DataFrame(_to_grid(a, d), index=t))
    return _concat(parts, keep="last")


def hourly_pnsd_sems(sems_dir, logs_dir=None):
    """Hourly dN/dlogDp (cm-3) from the EPCAPE Mt Soledad SEMS (qc == 1, logged periods removed)."""
    from epcape_build import DATENUM_UNIX, read_sems_logs, sems_files
    logs = read_sems_logs(logs_dir) if logs_dir else None
    parts = []
    for path in sems_files(sems_dir):
        with xr.open_dataset(path, decode_times=False) as ds:
            t = pd.to_datetime(ds.time.values - DATENUM_UNIX, unit="D").round("1s")
            dp = ds.diam_mid.values * 1000.0
            arr = ds.PNSD.values.astype(float)
            ok = ds.qc.values == 1
        arr, t = arr[ok], t[ok]
        if logs is not None and len(logs):
            drop = np.zeros(len(t), bool)
            for a, b, _ in logs.itertuples(index=False):
                drop |= (t >= a) & (t <= b)
            arr, t = arr[~drop], t[~drop]
        g = pd.DataFrame(_to_grid(arr, dp), index=t).resample("1h").mean()
        parts.append(g.dropna(how="all"))
    return _concat(parts)


def hourly_pnsd_csv(path):
    """Hourly dN/dlogDp (cm-3) from a CSV with a Time column and X<diameter nm> columns
    (the Virtanen et al. 2025 station files)."""
    p = pd.read_csv(path, index_col="Time", na_values=["NA", "NaN", ""])
    p.index = pd.to_datetime(p.index, format="mixed")
    d = np.array([float(c.lstrip("X")) for c in p.columns])
    a = p.to_numpy(dtype=float)
    a[a < 0] = np.nan
    return _concat([pd.DataFrame(_to_grid(a, d), index=p.index)])


def _concat(parts, keep="first"):
    if not parts:
        return pd.DataFrame(columns=range(GRID_NM.size), dtype=float)
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep=keep)]
    df = df[df.notna().any(axis=1)]
    df.columns = GRID_NM
    return df


def load_or_build(cache, builder, rebuild=False):
    """Read the hourly PNSD cache if present, else run builder() and write it."""
    if os.path.exists(cache) and not rebuild:
        with xr.open_dataset(cache) as ds:
            return pd.DataFrame(ds["dNdlogDp"].values, index=pd.DatetimeIndex(ds["time"].values),
                                columns=ds["diameter"].values)
    df = builder()
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    xr.Dataset({"dNdlogDp": (("time", "diameter"), df.to_numpy(dtype=float),
                             {"units": "cm-3", "long_name": "hourly mean dN/dlogDp on a common grid"})},
               coords={"time": df.index.values, "diameter": ("diameter", GRID_NM, {"units": "nm"})}
               ).to_netcdf(cache, encoding={"dNdlogDp": {"zlib": True, "complevel": 4}})
    return df


def n_above(pnsd, d_nm):
    """N(D > d_nm) (cm-3) from gridded dN/dlogDp; NaN where the grid above d_nm is not measured."""
    d = pnsd.columns.to_numpy(dtype=float)
    dlog = np.gradient(np.log10(d))
    sel = d >= d_nm
    a = pnsd.to_numpy()[:, sel]
    n = np.nansum(a * dlog[sel], axis=1)
    covered = np.isfinite(a).mean(axis=1)
    return pd.Series(np.where(covered > 0.5, n, np.nan), index=pnsd.index)


def nd_hours(ds, min_cloud_frac=0.5, max_base=2000.0, max_top=3000.0):
    """Hourly Nd and N(>100 nm) for the hours that enter the susceptibility fits."""
    t = pd.DatetimeIndex(ds["time"].values)
    m = np.isfinite(ds["CDNC"].values) & (ds["CDNC"].values > 0)
    if "cloud_frac" in ds:
        m &= ds["cloud_frac"].values >= min_cloud_frac
    if "cloud_base_height" in ds and max_base:
        m &= ~(ds["cloud_base_height"].values > max_base)
    if "cloud_top_height" in ds and max_top:
        m &= ~(ds["cloud_top_height"].values > max_top)
    i100 = int(np.argmin(np.abs(ds["diameter"].values - 100.0)))
    return pd.DataFrame({"Nd": ds["CDNC"].values[m],
                         "N100": ds["CCN"].values[m, i100]}, index=t[m])
