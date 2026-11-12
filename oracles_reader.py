"""Reader for NASA ORACLES (P-3B, south-east Atlantic, 2016-18), returning the same objects as activate_reader so
aircraft_build's pass detection and pairing run unchanged.

Folder layout (from nasa_aircraft_get.sh):
    data/aircraft/oracles/ORACLES_Cloud_AircraftInSitu_Data/    ORACLES-UND-SUMMARY_P3_<date>_R*.ict  1 Hz, CDP_Conc, CDP_LWC
    data/aircraft/oracles/ORACLES_Aerosol_AircraftInSitu_Data/  ORACLES-LDMA_P3_<date>_R*.nc   ~86-s scans, 10-535 nm, STP
                                                                ORACLES-DMA_P3_<date>_R*.nc    ~87-s scans, 10-200 nm, STP
                                                                ORACLES-HOWELL-CORRECTED-UHSAS_P3_<date>_R*.nc  1 Hz, 55-1000 nm, STP
    data/aircraft/oracles/ORACLES_MetNav_AircraftInSitu_Data/   ORACLES-Hskping_P3_<date>_R*.ict  position, static P and T

Choices
- Droplets: CDP number (cm-3) and LWC from the UND summary files (all three years; the stand-alone CDP files exist only
  for 2017-18).
- Aerosol: the long DMA (mobility, 10-535 nm; SMPS-type, so preferred) with the UHSAS above the LDMA range; the short DMA
  (10-200 nm) replaces the LDMA on flights without LDMA. LDMA scans are attached to every UHSAS second inside the scan.
- STP -> ambient with the housekeeping static pressure and temperature: N_amb = N_stp * (p / 1013.25 hPa) * (273.15 K / T)
  (--activate-stp keeps STP).
- The files already carry the PI's manual cloud-shatter mask; aircraft_build additionally drops seconds near droplets.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icartt import read_ict  # noqa: E402
from activate_reader import _rev_key  # noqa: E402

_DATE = re.compile(r"_(20\d{6})_R(\d+|[A-Za-z]+)(?:_L(\d))?\.(?:ict|nc)$", re.I)


def latest(folder, pattern):
    best = {}
    for f in glob.glob(os.path.join(folder, pattern)):
        m = _DATE.search(os.path.basename(f))
        if not m:
            continue
        d = m.group(1) + (f"_L{m.group(3)}" if m.group(3) else "")
        if d not in best or _rev_key(m.group(2)) > _rev_key(best[d][0]):
            best[d] = (m.group(2), f)
    return {d: f for d, (_, f) in sorted(best.items())}


def _nav(nav_dir, flight, cache={}):
    key = (nav_dir, flight)
    if key not in cache:
        f = latest(nav_dir, "ORACLES-Hskping_P3_*.ict").get(flight)
        if f is None:
            cache[key] = None
        else:
            d = read_ict(f)
            n = pd.DataFrame({"alt": d["GPS_Altitude"], "lat": d["Latitude"], "lon": d["Longitude"],
                              "p": d["Static_Pressure"], "T": d["Static_Air_Temp"]})
            cache[key] = n[~n.index.duplicated()]
    return cache[key]


def _psd(path):
    """(time index, end times or None, diameters nm, dN/dlogD [time, bin]) from an ORACLES size-distribution file."""
    with xr.open_dataset(path, decode_times=False) as ds:
        d = ds["diameter"].values.astype(float) * 1000.0
        v = ds["dNdlogd"].transpose(..., "sizebins").values.astype(float) if "sizebins" in ds["dNdlogd"].dims \
            else ds["dNdlogd"].values.astype(float)
        if v.shape[-1] != d.size:
            v = v.T
        v[v < 0] = np.nan
        t0 = ds["time_start"].values if "time_start" in ds else ds["time"].values
        t1 = ds["time_end"].values if "time_end" in ds else None
    t0 = pd.to_datetime(np.asarray(t0, float), unit="s")
    t1 = pd.to_datetime(np.asarray(t1, float), unit="s") if t1 is not None else None
    return t0, t1, d, v


def read_cloud_oracles(cloud_dir, dmin, dmax, nav_dir):
    files = latest(cloud_dir, "ORACLES-UND-SUMMARY_P3_*.ict")
    if not files:
        sys.exit(f"No ORACLES-UND-SUMMARY files in {cloud_dir}")
    parts, no_nav, no_cdp = [], [], []
    for fl, f in files.items():
        nav = _nav(nav_dir, fl)
        if nav is None:
            no_nav.append(fl)
            continue
        d = read_ict(f)
        cn = next((x for x in d.columns if re.fullmatch(r"CDP_?Conc\w*", x, re.I)), None)
        cl = next((x for x in d.columns if re.fullmatch(r"CDP_?LWC\w*", x, re.I)), None)
        if cn is None or cl is None or d[cn].notna().sum() == 0:
            no_cdp.append(fl)
            continue
        c = pd.DataFrame({"Nd": d[cn].to_numpy(float), "LWC": d[cl].to_numpy(float)}, index=d.index)
        c = c[~c.index.duplicated()].join(nav[["alt", "lat", "lon"]], how="inner")
        parts.append(c)
    if no_nav:
        print(f"  CDP: {len(no_nav)} flights skipped without housekeeping ({', '.join(no_nav[:5])})")
    if no_cdp:
        print(f"  CDP: {len(no_cdp)} summary files without CDP number ({', '.join(no_cdp[:8])}{' ...' if len(no_cdp) > 8 else ''})")
    if not parts:
        sys.exit("No flight with CDP number and housekeeping.")
    print("  droplets: CDP (UND summary, CDP_Conc, about 2-50 um); --nd-dmin/--nd-dmax are not applied")
    c = pd.concat(parts).sort_index()
    return c[~c.index.duplicated()]


def read_aerosol_oracles(aerosol_dir, nav_dir, cloud=None, clear_nd=1.0, pad_s=2, stp=False):
    uh = latest(aerosol_dir, "ORACLES-HOWELL-CORRECTED-UHSAS_P3_*.nc")
    ld = latest(aerosol_dir, "ORACLES-LDMA_P3_*.nc")
    dm = latest(aerosol_dir, "ORACLES-DMA_P3_*.nc")
    if not uh:
        sys.exit(f"No ORACLES-HOWELL-CORRECTED-UHSAS files in {aerosol_dir}")
    grid = None
    frames, meta = [], []
    st = dict(flights=0, no_nav=0, ldma=0, dma=0, none=0, scan_cover=[])
    for fl, f in uh.items():
        nav = _nav(nav_dir, fl)
        if nav is None:
            st["no_nav"] += 1
            continue
        tu, _, du, vu = _psd(f)
        keep = ~tu.duplicated()
        tu, vu = tu[keep], vu[keep]
        mob = ld.get(fl) or dm.get(fl)
        if fl in ld:
            st["ldma"] += 1
        elif fl in dm:
            st["dma"] += 1
        else:
            st["none"] += 1
        if grid is None:
            # mobility grid: the LDMA bins (or a DMA file if the first flight has no LDMA); UHSAS above its top
            ref = next(iter(ld.values()), None) or next(iter(dm.values()))
            _, _, dref, _ = _psd(ref)
            top = dref[-1] * (dref[-1] / dref[-2]) ** 0.5
            u_keep = np.where(du > top)[0]
            grid = np.concatenate([dref, du[u_keep]])
            n_m = dref.size
        a = np.full((len(tu), grid.size), np.nan)
        a[:, n_m:] = vu[:, u_keep]
        if mob is not None:
            t0, t1, dmob, vm = _psd(mob)
            # interpolate the scan onto the reference mobility grid in log D (DMA bins differ from LDMA bins)
            vg = np.array([np.interp(np.log(grid[:n_m]), np.log(dmob), row, left=np.nan, right=np.nan) for row in vm])
            j = np.searchsorted(t0.values, tu.values, side="right") - 1
            ok = j >= 0
            if t1 is not None:
                ok[ok] &= tu.values[ok] <= t1.values[j[ok]]
            a[ok, :n_m] = vg[j[ok]]
            st["scan_cover"].append(ok.mean())
        n = nav.reindex(tu.floor("s"))
        if not stp:
            fac = (n["p"].to_numpy(float) / 1013.25) * (273.15 / (n["T"].to_numpy(float) + 273.15))
            a = a * fac[:, None]
        ok = n["alt"].notna().to_numpy() & (np.isfinite(a).any(axis=1))
        if not stp:
            ok &= np.isfinite(fac)
        frames.append(pd.DataFrame(a[ok], index=tu[ok]))
        meta.append(pd.DataFrame(n[["alt", "lat", "lon"]].to_numpy()[ok], columns=["alt", "lat", "lon"], index=tu[ok]))
        st["flights"] += 1
    a = pd.concat(frames).sort_index()
    m = pd.concat(meta).sort_index()
    keep = ~a.index.duplicated()
    a, m = a[keep], m[keep]
    print(f"Aerosol: ORACLES mobility ({grid[0]:.0f}-{grid[n_m - 1]:.0f} nm; LDMA on {st['ldma']} flights, DMA on {st['dma']}, "
          f"none on {st['none']}) + UHSAS ({grid[n_m]:.0f}-{grid[-1]:.0f} nm), {'STP' if stp else 'ambient (housekeeping p, T)'}; "
          f"{st['flights']} flights, {st['no_nav']} without housekeeping; scans cover "
          f"{np.mean(st['scan_cover']) if st['scan_cover'] else 0:.0%} of UHSAS seconds")
    if cloud is not None:
        wet = cloud.index[cloud["Nd"] >= clear_nd]
        if len(wet):
            ws = pd.Series(1, index=wet).resample("1s").max().fillna(0)
            ws = ws.rolling(2 * pad_s + 1, center=True, min_periods=1).max()
            hit = ws.reindex(a.index.floor("s")).fillna(0).to_numpy() > 0
            print(f"  droplet-probe cloud mask (Nd >= {clear_nd} cm-3, +/- {pad_s} s) removes {hit.mean():.0%} of aerosol seconds")
            a, m = a[~hit], m[~hit]
    return a, m, grid
