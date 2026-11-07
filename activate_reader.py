"""Readers for NASA ACTIVATE HU-25 Falcon in-situ data (ICARTT), returning the same objects as
aircraft_build.read_cloud / read_aerosol so the ARM pairing runs on them unchanged.

Folder layout (from activate_download.py):
    data/aircraft/activate/cloud/    ACTIVATE-FCDP_HU25_<date>_R*.ict           1 Hz, 3-50 um, 18 bins, ambient
    data/aircraft/activate/aerosol/  ACTIVATE-LARGE-LAS_HU25_<date>_R*.ict      1 Hz, 94-3490 nm, dry, STP
                                     ACTIVATE-LARGE-SMPS_HU25_<date>_R*.ict     60-s scans, 3-94 nm, dry, STP
                                     ACTIVATE-LARGE-InletFlag_HU25_<date>_R*.ict  0 = isokinetic inlet, 1 = CVI
    data/aircraft/activate/merge/    ACTIVATE-mrg01-HU25_merge_<date>_R*.ict    position (lat, lon, GPS altitude)

Choices
- The highest revision is used when a flight has several (numbers above letters, letters are preliminary).
- Aerosol concentrations are converted from STP to ambient (divided by stdPT), matching the droplet
  number, which the FCDP reports at ambient conditions. --activate-stp keeps them at STP.
- SMPS scans are attached to every LAS second inside the scan (Time_Start..Time_Stop), giving one
  3 nm - 3.5 um spectrum per second; seconds without a scan have NaN below 94 nm only.
- Seconds with InletFlag != 0 (CVI) are dropped. A flight without inlet information is kept with a warning
  (the droplet-probe mask in aircraft_build still removes in-cloud seconds).
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icartt import read_ict, read_header  # noqa: E402

_DATE = re.compile(r"_(20\d{6})_R(\d+|[A-Za-z]+)(?:_L(\d))?\.ict$", re.I)


def _rev_key(r):
    return (1, int(r)) if r.isdigit() else (0, r.upper())


def latest(folder, pattern):
    """{flight: path} with the highest revision per flight. A flight is the date, or date_L1 / date_L2
    on days with two flights (ACTIVATE files end in _R<rev>_L<n>.ict for those)."""
    best = {}
    for f in glob.glob(os.path.join(folder, pattern)):
        m = _DATE.search(os.path.basename(f))
        if not m:
            continue                                         # e.g. *_thru<date>.ict summary files
        d = m.group(1) + (f"_L{m.group(3)}" if m.group(3) else "")
        r = m.group(2)
        if d not in best or _rev_key(r) > _rev_key(best[d][0]):
            best[d] = (r, f)
    return {d: f for d, (r, f) in sorted(best.items())}


def _edges(header, lower_key, upper_key):
    lo = up = None
    for line in header:
        if lower_key in line:
            lo = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", line.split(lower_key, 1)[1])]
        if upper_key in line:
            up = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", line.split(upper_key, 1)[1])]
    if lo is None or up is None or len(lo) != len(up):
        raise ValueError(f"bin edges ({lower_key}/{upper_key}) not found in header")
    return np.array(lo), np.array(up)


def _nav(merge_dir, date, cache={}):
    key = (merge_dir, date)
    if key not in cache:
        f = latest(merge_dir, f"ACTIVATE-mrg01-HU25_merge_{date}_R*.ict").get(date)
        if f is None:
            cache[key] = None
        else:
            d = read_ict(f, usecols=["Time_Start", "Latitude_THORNHILL", "Longitude_THORNHILL", "GPS_Altitude_THORNHILL"])
            cache[key] = d.rename(columns={"Latitude_THORNHILL": "lat", "Longitude_THORNHILL": "lon",
                                           "GPS_Altitude_THORNHILL": "alt"})[["alt", "lat", "lon"]]
            cache[key] = cache[key][~cache[key].index.duplicated()]
    return cache[key]


def read_cloud_activate(cloud_dir, dmin, dmax, merge_dir):
    """1-Hz FCDP droplet number (cm-3) and LWC (g m-3) for dmin-dmax um, plus position."""
    files = latest(cloud_dir, "ACTIVATE-FCDP_HU25_*.ict")
    if not files:
        sys.exit(f"No ACTIVATE-FCDP files in {cloud_dir}")
    parts, no_nav = [], []
    for date, f in files.items():
        nav = _nav(merge_dir, date[:8])            # one merge per day covers both flights
        if nav is None:
            no_nav.append(date)
            continue
        _, h = read_header(f)
        lo, up = _edges(h, "Bin Lower Edges (in um) =", "Bin Upper Edges (in um) =")
        d = read_ict(f)
        cols = [c for c in d.columns if c.startswith("dNdlogD_")]
        n = d[cols].to_numpy(float)                                  # m-3 per dlog10 D
        n[n < 0] = np.nan
        w = np.clip(np.log10(np.minimum(up, dmax)) - np.log10(np.maximum(lo, dmin)), 0, None)  # dlog10 inside range
        nb = n * w[None, :]                                          # m-3 per bin (part in range)
        dc = np.sqrt(lo * up) * 1e-6                                 # m
        nd = np.nansum(nb, axis=1) / 1e6                             # cm-3
        lwc = np.nansum(nb * (np.pi / 6) * dc ** 3, axis=1) * 1e6    # m3/m3 * 1e6 g/m3 (water)
        nd[np.all(np.isnan(n), axis=1)] = np.nan
        c = pd.DataFrame({"Nd": nd, "LWC": lwc}, index=d.index)
        c = c[~c.index.duplicated()].join(nav, how="inner")
        parts.append(c)
    if no_nav:
        print(f"  FCDP: {len(no_nav)} flights skipped, no 1-s merge for position ({', '.join(no_nav[:5])}{' ...' if len(no_nav) > 5 else ''})")
    c = pd.concat(parts).sort_index()
    if dmin < 3.0:
        print(f"  FCDP starts at 3 um, so Nd covers 3-{dmax:g} um (requested {dmin:g})")
    return c[~c.index.duplicated()]


def read_aerosol_activate(aerosol_dir, merge_dir, cloud=None, clear_nd=1.0, pad_s=2, stp=False):
    """1-Hz combined SMPS+LAS dN/dlogDp (cm-3, ambient unless stp) for isokinetic seconds, plus position."""
    las = latest(aerosol_dir, "ACTIVATE-LARGE-LAS_HU25_*.ict")
    smps = latest(aerosol_dir, "ACTIVATE-LARGE-SMPS_HU25_*.ict")
    inl = {**latest(aerosol_dir, "ACTIVATE-LARGE-INLETFLAG_HU25_*.ict"),
           **latest(aerosol_dir, "ACTIVATE-LARGE-InletFlag_HU25_*.ict")}
    if not las:
        sys.exit(f"No ACTIVATE-LARGE-LAS files in {aerosol_dir}")
    grid = s_keep = None
    frames, meta, stats = [], [], {"flights": 0, "no_smps": 0, "no_inlet": 0, "no_nav": 0, "cvi_frac": []}
    for date, f in las.items():
        nav = _nav(merge_dir, date[:8])            # one merge per day covers both flights
        if nav is None:
            stats["no_nav"] += 1
            continue
        _, h = read_header(f)
        llo, lup = _edges(h, "Lower Bounds:", "Upper Bounds:")
        L = read_ict(f)
        lcols = [c for c in L.columns if re.match(r"LAS_Bin\d+", c)]
        lv = L[lcols].to_numpy(float)
        if not stp:
            lv = lv / L["stdPT"].to_numpy(float)[:, None]
        L2 = pd.DataFrame(lv, index=L.index)
        L2 = L2[~L2.index.duplicated()]
        # SMPS scans onto the LAS seconds
        sv = None
        if date in smps:
            _, hs = read_header(smps[date])
            slo, sup = _edges(hs, "Lower Bounds:", "Upper Bounds:")
            S = read_ict(smps[date])
            scols = [c for c in S.columns if re.match(r"SMPS_Bin\d+", c)]
            s = S[scols].to_numpy(float)
            if not stp:
                s = s / S["stdPT"].to_numpy(float)[:, None]
            t0 = pd.Timestamp(S.attrs["date"]) + pd.to_timedelta(S["Time_Start"].to_numpy(float), unit="s")
            t1 = pd.Timestamp(S.attrs["date"]) + pd.to_timedelta(S["Time_Stop"].to_numpy(float), unit="s")
            idx = np.searchsorted(t0.values, L2.index.values, side="right") - 1
            ok = (idx >= 0)
            ok[ok] &= L2.index.values[ok] <= t1.values[idx[ok]]
            sv = np.full((len(L2), s.shape[1]), np.nan)
            sv[ok] = s[idx[ok]]
            if grid is None:
                s_keep = np.where(np.isfinite(s).any(axis=0))[0]    # drop bins never reported (e.g. removed top bin)
                s_edges = (slo[s_keep], sup[s_keep])
        else:
            stats["no_smps"] += 1
        if grid is None:
            if s_keep is None:
                sys.exit("The first flight has no SMPS file; download the SMPS files too.")
            lo = np.concatenate([s_edges[0], llo]); up = np.concatenate([s_edges[1], lup])
            grid = np.sqrt(lo * up)
            n_s = len(s_keep)
        a = np.full((len(L2), len(grid)), np.nan)
        a[:, n_s:] = L2.to_numpy()
        if sv is not None:
            a[:, :n_s] = sv[:, s_keep]
        # inlet flag
        ok = np.ones(len(L2), bool)
        if date in inl:
            fl = read_ict(inl[date]).iloc[:, 1]
            fl = fl[~fl.index.duplicated()].reindex(L2.index.floor("s"))
            if fl.notna().mean() > 0.5:
                ok = (fl == 0).to_numpy()
                stats["cvi_frac"].append(float((fl == 1).mean()))
            else:
                stats["no_inlet"] += 1
        else:
            stats["no_inlet"] += 1
        m = nav.reindex(L2.index.floor("s"))
        ok &= m["alt"].notna().to_numpy()
        frames.append(pd.DataFrame(a[ok], index=L2.index[ok]))
        meta.append(pd.DataFrame(m.to_numpy()[ok], columns=["alt", "lat", "lon"], index=L2.index[ok]))
        stats["flights"] += 1
    a = pd.concat(frames).sort_index()
    m = pd.concat(meta).sort_index()
    keep = ~a.index.duplicated()
    a, m = a[keep], m[keep]
    cvi = np.mean(stats["cvi_frac"]) if stats["cvi_frac"] else 0.0
    print(f"Aerosol: ACTIVATE SMPS ({grid[0]:.1f}-{grid[n_s - 1]:.0f} nm) + LAS ({grid[n_s]:.0f}-{grid[-1]:.0f} nm), "
          f"{'STP' if stp else 'ambient (divided by stdPT)'}; {stats['flights']} flights, "
          f"{stats['no_nav']} skipped without position, {stats['no_smps']} without SMPS, "
          f"{stats['no_inlet']} without inlet flag; CVI inlet removes {cvi:.0%} of seconds")
    if cloud is not None:
        wet = cloud.index[cloud["Nd"] >= clear_nd]
        if len(wet):
            ws = pd.Series(1, index=wet).resample("1s").max().fillna(0)
            ws = ws.rolling(2 * pad_s + 1, center=True, min_periods=1).max()
            hit = ws.reindex(a.index.floor("s")).fillna(0).to_numpy() > 0
            print(f"  droplet-probe cloud mask (Nd >= {clear_nd} cm-3, +/- {pad_s} s) removes {hit.mean():.0%} of aerosol seconds")
            a, m = a[~hit], m[~hit]
    return a, m, grid
