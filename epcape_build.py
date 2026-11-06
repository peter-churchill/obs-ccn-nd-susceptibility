#!/usr/bin/env python3
"""
EPCAPE Mt. Soledad: hourly in-situ Nd (FM-120) matched with CCN(D) from the SEMS.

Scientific question
-------------------
At a coastal hilltop that sits inside stratocumulus (Mt. Soledad, La Jolla,
250 m), how does the in-situ droplet number respond to the aerosol number
above a cutoff diameter, and which cutoff closes Nd and CCN? Unlike ARM ENA,
Nd here is measured in cloud at the same point as the aerosol.

Method
------
* Nd: FM-120 fog monitor, 5-min averages. A 5-min sample is in cloud when
  LWC >= --lwc-cloud (default 0.05 g m-3). For each hour, cloud_frac = in-cloud
  samples / 12, and CDNC = mean Nd over the in-cloud samples.
* CCN(D): SEMS dry number size distributions (10-840 nm), qc == 1 only. During
  cloud the SEMS samples partly through the CVI (droplet residuals) and the
  processed files do not mark which scans; the two cannot be separated reliably
  from the data alone. So CCN is taken from scans in CLEAR air (FM-120 LWC <
  --lwc-clear) within +/- --window-h hours of each cloudy hour (default 3 h,
  at least --min-scans scans). CCN_pre uses only scans before the hour.
* Output mirrors the ARM ENA files (CDNC, CCN(time, radius), cloud_frac,
  diameter coordinate), so `arm_ena_nd_pnsd.py quicklook` works on it.

Data (CC BY 4.0; the authors ask to be consulted before publication):
  FM-120: https://doi.org/10.6075/J0NG4RMV   SEMS/APS: https://doi.org/10.6075/J0Q81D8N

Usage
-----
    python epcape_build.py --fm epcape/epcape_fm120_5min_compact.csv \
        --sems-dir epcape/sems --logs-dir epcape/sems_logs --out epcape_soledad_hourly.nc
    python arm_ena_nd_pnsd.py quicklook epcape_soledad_hourly.nc --min-cloud-frac 0.8
"""

import argparse
import glob
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_ena_nd_pnsd import RADII_NM, cumulative_above   # noqa: E402

DATENUM_UNIX = 719529          # MATLAB datenum of 1970-01-01


def read_fm(path):
    """FM-120 5-min file (compact or original monthly CSV) -> DataFrame."""
    f = pd.read_csv(path)
    f.index = pd.to_datetime(f[f.columns[0]])          # first column: full UTC timestamp
    ren = {c: k for c in f.columns for k, key in
           (("Nd", "Number Conc"), ("LWC", "LWC"), ("ED", "ED (um)"),
            ("MVD", "MVD"), ("N9", "CVI Cutoff")) if c.startswith(key)}
    return f.rename(columns=ren)[["Nd", "LWC", "ED", "MVD", "N9"]].astype(float)


def sems_files(sems_dir):
    """Paths of SEMS NetCDF files; ZIPs in the folder are extracted first."""
    ncs = sorted(glob.glob(os.path.join(sems_dir, "**", "EPCAPE_SEMS_*.nc"), recursive=True))
    if ncs:
        return ncs
    zips = sorted(glob.glob(os.path.join(sems_dir, "*.zip")))
    if not zips:
        sys.exit(f"No EPCAPE_SEMS_*.nc or .zip files in {sems_dir}")
    tmp = tempfile.mkdtemp(prefix="epcape_sems_")
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if os.path.basename(n).startswith("EPCAPE_SEMS_") and n.endswith(".nc"):
                    zf.extract(n, tmp)
    return sorted(glob.glob(os.path.join(tmp, "**", "EPCAPE_SEMS_*.nc"), recursive=True))


def read_sems_logs(logs_dir):
    """Logged SEMS problem periods -> DataFrame(start, end, report).

    The log files (SEMS_log_MMMYYYY.txt, tab-separated, UTC) list periods to
    use with caution or remove: maintenance, instrument errors, CVI flow
    problems and on-site contamination (pollution, forklift, container air).
    Two entries have end dates typed with the wrong year (end before start);
    those ends are moved into the start's year.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "SEMS_log_*.txt"))):
        with open(path, encoding="latin-1") as fh:
            for line in fh.read().splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) < 3 or not parts[0].strip():
                    continue
                a, b = pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
                if b < a:
                    b = b.replace(year=a.year)
                rows.append((a, b, parts[2].strip()))
    return pd.DataFrame(rows, columns=["start", "end", "report"])


def read_sems_ccn(sems_dir, cut_nm, logs=None):
    """Per-scan N(D > D_c), Ntot, N_aitken_20_100, N70_500 for qc == 1 scans.

    Each file is integrated on its own diameter grid (grids differ slightly
    between the quarterly files). Scans inside any logged period are dropped,
    even when their qc is 1.
    """
    parts = []
    for path in sems_files(sems_dir):
        with xr.open_dataset(path, decode_times=False) as ds:
            t = pd.to_datetime(ds.time.values - DATENUM_UNIX, unit="D").round("1s")
            dp = ds.diam_mid.values * 1000.0                     # um -> nm
            arr = ds.PNSD.values.astype(float)
            ok = ds.qc.values == 1
        arr, t = arr[ok], t[ok]
        ccn = cumulative_above(arr, dp, cut_nm)
        band = cumulative_above(arr, dp, np.array([20.0, 70.0, 100.0, 500.0]))
        df = pd.DataFrame(ccn, index=t, columns=[f"c{i}" for i in range(len(cut_nm))])
        df["N_aitken_20_100"] = band[:, 0] - band[:, 2]
        df["N70_500"] = band[:, 1] - band[:, 3]
        parts.append(df)
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if logs is not None and len(logs):
        drop = np.zeros(len(s), bool)
        hits = {}
        for a, b, rep in logs.itertuples(index=False):
            m = (s.index >= a) & (s.index <= b)
            if m.any():
                hits[rep] = hits.get(rep, 0) + int(m.sum())
            drop |= m
        top = sorted(hits.items(), key=lambda kv: -kv[1])[:6]
        print(f"Logs: {len(logs)} periods; removed {drop.sum()} qc==1 scans "
              f"(largest: {', '.join(f'{k} {v}' for k, v in top)})")
        s = s[~drop]
    return s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fm", required=True, help="FM-120 5-min CSV")
    p.add_argument("--sems-dir", required=True, help="folder with SEMS ZIPs or NetCDF files")
    p.add_argument("--logs-dir", default=None,
                   help="folder with SEMS_log_*.txt; scans in logged periods are dropped")
    p.add_argument("--out", required=True)
    p.add_argument("--lwc-cloud", type=float, default=0.05, help="in-cloud LWC, g m-3")
    p.add_argument("--lwc-clear", type=float, default=0.01, help="clear-air LWC, g m-3")
    p.add_argument("--window-h", type=float, default=3.0, help="clear-air window, hours")
    p.add_argument("--min-scans", type=int, default=6)
    p.add_argument("--min-cloud-frac", type=float, default=0.3, help="floor for keeping an hour")
    p.add_argument("--nd-hourly", choices=("median", "geomean", "mean"), default="median",
                   help="how in-cloud 5-min Nd samples are combined into one hourly value "
                        "(median matches the other builds; mean was used before Nov 2026)")
    a = p.parse_args(argv)

    radii = RADII_NM
    cut_nm = 2.0 * radii
    fm = read_fm(a.fm)
    incloud = fm["LWC"] >= a.lwc_cloud
    g = fm[incloud].resample("1h")
    hourly = g.mean() if a.nd_hourly == "mean" else g.median()
    if a.nd_hourly == "geomean":
        hourly["Nd"] = np.exp(np.log(fm.loc[incloud & (fm["Nd"] > 0), "Nd"]).resample("1h").mean())
    hourly["cloud_frac"] = (fm["LWC"].resample("1h").apply(
        lambda s: (s >= a.lwc_cloud).sum() / 12.0)).reindex(hourly.index)
    hourly = hourly[hourly["cloud_frac"] >= a.min_cloud_frac]
    print(f"FM-120: {len(fm)} samples, {int(incloud.sum())} in cloud; hours with "
          f"cloud_frac >= 0.5: {(hourly.cloud_frac >= 0.5).sum()}, >= 0.8: "
          f"{(hourly.cloud_frac >= 0.8).sum()}")

    logs = read_sems_logs(a.logs_dir) if a.logs_dir else None
    s = read_sems_ccn(a.sems_dir, cut_nm, logs)
    clear = fm["LWC"].reindex(s.index, method="nearest", tolerance=pd.Timedelta("5min")) < a.lwc_clear
    s = s[clear.fillna(False).to_numpy()]
    print(f"SEMS: {len(s)} good clear-air scans")

    ccols = [c for c in s.columns if c.startswith("c")]
    win = pd.Timedelta(hours=a.window_h)
    rows, rows_pre, aux = [], [], []
    st = s.index.values
    for h in hourly.index:
        mid = h + pd.Timedelta("30min")
        i0, i1 = np.searchsorted(st, (mid - win).to_datetime64()), np.searchsorted(st, (mid + win).to_datetime64())
        sel = s.iloc[i0:i1]
        pre = sel[sel.index < h]
        rows.append(sel[ccols].mean().to_numpy() if len(sel) >= a.min_scans else np.full(len(ccols), np.nan))
        rows_pre.append(pre[ccols].mean().to_numpy() if len(pre) >= a.min_scans // 2 else np.full(len(ccols), np.nan))
        aux.append([sel["N_aitken_20_100"].mean(), sel["N70_500"].mean(), len(sel),
                    float(np.mean(np.abs((sel.index - mid).total_seconds()))) / 3600 if len(sel) else np.nan])
    ccn, ccn_pre, aux = np.array(rows), np.array(rows_pre), np.array(aux)
    ok = np.isfinite(ccn).any(axis=1) & np.isfinite(hourly["Nd"].to_numpy())
    print(f"Matched: {ok.sum()} of {len(hourly)} cloudy hours have >= {a.min_scans} clear-air "
          f"scans within +/-{a.window_h:g} h")

    hh = hourly[ok]
    ds = xr.Dataset(
        {"CDNC": ("time", hh["Nd"].to_numpy(), {"units": "cm-3", "long_name": f"FM-120 Nd, hourly {a.nd_hourly} of in-cloud 5-min samples", "hourly_statistic": a.nd_hourly}),
         "CCN": (("time", "radius"), ccn[ok], {"units": "cm-3", "long_name": f"N(D > 2*radius), clear-air SEMS scans within +/-{a.window_h:g} h"}),
         "CCN_pre": (("time", "radius"), ccn_pre[ok], {"units": "cm-3", "long_name": f"as CCN, scans in the {a.window_h:g} h before the hour only"}),
         "cloud_frac": ("time", hh["cloud_frac"].to_numpy(), {"units": "1", "long_name": f"fraction of 5-min samples with LWC >= {a.lwc_cloud} g m-3"}),
         "LWC": ("time", hh["LWC"].to_numpy(), {"units": "g m-3"}),
         "ED": ("time", hh["ED"].to_numpy(), {"units": "um", "long_name": "effective diameter"}),
         "N_droplets_gt9um": ("time", hh["N9"].to_numpy(), {"units": "cm-3", "long_name": "FM-120 droplets >= ~9 um (CVI cut size)"}),
         "N_aitken_20_100": ("time", aux[ok, 0], {"units": "cm-3"}),
         "N70_500": ("time", aux[ok, 1], {"units": "cm-3"}),
         "n_clear_scans": ("time", aux[ok, 2]),
         "mean_scan_offset_h": ("time", aux[ok, 3], {"units": "h"})},
        coords={"time": hh.index.values,
                "radius": ("radius", radii, {"units": "nm"}),
                "diameter": ("radius", cut_nm, {"units": "nm"})},
        attrs={"site": "EPCAPE Mt. Soledad (La Jolla, 250 m)",
               "nd_source": "FM-120, doi:10.6075/J0NG4RMV",
               "aerosol_source": "SEMS, doi:10.6075/J0Q81D8N (qc == 1, clear-air scans)",
               "lwc_cloud": a.lwc_cloud, "lwc_clear": a.lwc_clear, "window_h": a.window_h,
               "logged_periods_excluded": "yes" if logs is not None else "no",
               "note": "Data CC BY 4.0; the authors ask users to consult them before publication."})
    ds.to_netcdf(a.out, encoding={v: {"zlib": True} for v in ds.data_vars})
    print(f"Wrote {a.out}: {ds.sizes['time']} hours (cloud_frac >= 0.5: "
          f"{int((ds.cloud_frac >= 0.5).sum())}, >= 0.8: {int((ds.cloud_frac >= 0.8).sum())})")


if __name__ == "__main__":
    main()
