"""Readers for NASA LaRC airborne campaigns whose ICARTT files follow the ACTIVATE layout (LARGE group):
NAAMES (C-130, 2015-18) now; CAMP2Ex to follow. Same return objects as activate_reader, so aircraft_build's
pass detection and pairing run unchanged.

Folder layout (from nasa_aircraft_get.sh), e.g. NAAMES:
    data/aircraft/naames/NAAMES_Cloud_AircraftInSitu_Data/    NAAMES-LARGE-CDP_C130_<date>_R*.ict   1 Hz, 2-50 um, ambient
                                                              NAAMES-LARGE-CLOUDFLAG_C130_<date>_R*.ict
    data/aircraft/naames/NAAMES_Aerosol_AircraftInSitu_Data/  NAAMES-LARGE-SMPS_C130_<date>_R*.ict  scans, mobility, STP
                                                              NAAMES-LARGE-LAS_C130_<date>_R*.ict   1 Hz, 90-7500 nm, STP
    data/aircraft/naames/NAAMES_MetNav_AircraftInSitu_Data/   NAAMES-HSKPING_C130_<date>_R*.ict    position, GPS altitude

Choices (as for ACTIVATE)
- Highest revision per flight.
- Droplets: CDP DropletNum (2-50 um) and LWC from the CDP volume (vCDP, um3 cm-3 -> g m-3 x 1e-6).
- Aerosol: SMPS below the first LAS bin, LAS above; STP -> ambient by dividing by stdPT (--activate-stp keeps STP).
  The bin values are tested against each file's integrated number (nSMPS, IntegN_..._LAS) to decide whether they
  are dN/dlogD or dN per bin; per-bin values are converted to dN/dlogD.
- Seconds with CloudFlag = 1 (possible inlet shatter) are dropped from the aerosol, in addition to the droplet mask
  applied in aircraft_build.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icartt import read_ict, read_header  # noqa: E402
from activate_reader import latest  # noqa: E402

CAMPAIGNS = {
    "naames": dict(platform="C130", cdp="NAAMES-LARGE-CDP_C130_*.ict", cloudflag="NAAMES-LARGE-CLOUDFLAG_C130_*.ict",
                   smps="NAAMES-LARGE-SMPS_C130_*.ict", las="NAAMES-LARGE-LAS_C130_*.ict",
                   nav="NAAMES-HSKPING_C130_*.ict", nd_col="DropletNum", vol_col="vCDP",
                   smps_total="nSMPS", las_total=r"IntegN_\d+to\d+nm.*LAS"),
}


def _bin_diameters(h, nbins, lo_nm, hi_nm):
    """Bin mid-diameters (nm) from an ICARTT header: a run of nbins increasing numbers after a line that mentions
    diameters/bins (values may continue over several lines); lower/upper bound pairs give geometric means."""
    lists = []
    for i, line in enumerate(h):
        if not re.search(r"diam|bin|bound|edge|size", line, re.I):
            continue
        for start in (i, i + 1):                              # numbers on the same line, or only on the next lines
            nums = []
            for j in range(start, min(start + 12, len(h))):
                part = h[j] if j > i else re.split(r"[:=]", line, 1)[-1]
                nums += [float(x) for x in re.findall(r"(?<![A-Za-z_])\d+\.?\d*(?:[eE][-+]?\d+)?", part)]
                if len(nums) >= nbins:
                    break
            nums = nums[:nbins]
            if len(nums) == nbins and np.all(np.diff(nums) > 0):
                lists.append((line, np.array(nums)))
    def scale(x):
        for f in (1.0, 1000.0):                              # nm, or um -> nm
            if lo_nm <= x[0] * f and x[-1] * f <= hi_nm:
                return x * f
        return None
    low = [s for l, x in lists if re.search(r"lower", l, re.I) for s in [scale(x)] if s is not None]
    up = [s for l, x in lists if re.search(r"upper", l, re.I) for s in [scale(x)] if s is not None]
    if low and up:
        return np.sqrt(low[0] * up[0]), "lower/upper bounds"
    mids = [(l, s) for l, x in lists for s in [scale(x)] if s is not None]
    if mids:
        return mids[0][1], mids[0][0].strip()[:60]
    raise ValueError(f"no list of {nbins} bin diameters in {lo_nm:g}-{hi_nm:g} nm found in the header")


def _as_dndlogd(v, d_nm, total, name):
    """Return dN/dlogD. v holds either dN/dlogD or dN per bin; decide from the file's integrated number."""
    dlog = np.gradient(np.log10(d_nm))
    if total is None or not np.isfinite(total).any():
        print(f"  {name}: no integrated number to check the bin convention; assuming dN/dlogD")
        return v, "dN/dlogD (assumed)"
    ok = np.isfinite(total) & (total > 0) & np.isfinite(v).all(axis=1)
    if ok.sum() < 10:
        return v, "dN/dlogD (assumed)"
    r_dlog = np.nanmedian(np.nansum(v[ok] * dlog, axis=1) / total[ok])
    r_bin = np.nanmedian(np.nansum(v[ok], axis=1) / total[ok])
    if abs(np.log(r_bin)) < abs(np.log(r_dlog)):
        return v / dlog[None, :], f"dN per bin -> dN/dlogD (sum/total {r_bin:.2f})"
    return v, f"dN/dlogD (integral/total {r_dlog:.2f})"


def _bins(df, inst, header, lo_nm, hi_nm):
    """Size-bin columns of instrument inst (LAS/SMPS) and their mid-diameters (nm). Two naming schemes occur:
    LAS_Bin01.. with the diameters in a header comment (2016-17 files), or dNdlogDp_PSL_100nm_LAS.. with the
    diameter in the name (2015 files)."""
    named = [(c, float(m.group(1))) for c in df.columns
             for m in [re.search(r"(\d+\.?\d*)\s*nm", c)] if m and inst.lower() in c.lower() and "dndlog" in c.lower()]
    if len(named) >= 5:
        return [c for c, _ in named], np.array([d for _, d in named]), "diameters in the column names"
    cols = _cols(df, rf"{inst}_?Bin_?\d+")
    d, how = _bin_diameters(header, len(cols), lo_nm, hi_nm)
    return cols, d, how


def _cols(df, pat):
    """Bin columns, e.g. LAS_Bin01 (also LAS_Bin_01, las_bin01_...); exits with the column list if none match."""
    cols = [c for c in df.columns if re.match(pat, c, re.I)]
    if not cols:
        sys.exit(f"No columns matching {pat} in {df.attrs.get('file')}. Columns: {list(df.columns)[:40]}")
    return cols


def _nav(nav_dir, pattern, flight, cache={}):
    key = (nav_dir, flight)
    if key not in cache:
        f = latest(nav_dir, pattern).get(flight)
        if f is None:
            cache[key] = None
        else:
            d = read_ict(f)
            lat = next(c for c in d.columns if re.match(r"lat", c, re.I))
            lon = next(c for c in d.columns if re.match(r"lon", c, re.I))
            alt = next(c for c in d.columns if re.match(r"gps_?alt", c, re.I))
            n = d[[alt, lat, lon]].set_axis(["alt", "lat", "lon"], axis=1)
            cache[key] = n[~n.index.duplicated()]
    return cache[key]


def read_cloud_larc(campaign, cloud_dir, dmin, dmax, nav_dir):
    cfg = CAMPAIGNS[campaign]
    files = latest(cloud_dir, cfg["cdp"])
    if not files:
        sys.exit(f"No {cfg['cdp']} files in {cloud_dir}")
    parts, no_nav = [], []
    for fl, f in files.items():
        nav = _nav(nav_dir, cfg["nav"], fl)
        if nav is None:
            no_nav.append(fl)
            continue
        d = read_ict(f)
        c = pd.DataFrame({"Nd": d[cfg["nd_col"]].to_numpy(float),
                          "LWC": d[cfg["vol_col"]].to_numpy(float) * 1e-6}, index=d.index)
        c = c[~c.index.duplicated()].join(nav, how="inner")
        parts.append(c)
    if no_nav:
        print(f"  CDP: {len(no_nav)} flights skipped without housekeeping position ({', '.join(no_nav[:5])})")
    if dmin > 2.0 or dmax < 50.0:
        print(f"  note: {cfg['nd_col']} covers the full CDP range 2-50 um; --nd-dmin/--nd-dmax are not applied")
    c = pd.concat(parts).sort_index()
    return c[~c.index.duplicated()]


def read_aerosol_larc(campaign, aerosol_dir, nav_dir, cloud_dir=None, cloud=None, clear_nd=1.0, pad_s=2, stp=False):
    cfg = CAMPAIGNS[campaign]
    las, smps = latest(aerosol_dir, cfg["las"]), latest(aerosol_dir, cfg["smps"])
    cflag = latest(cloud_dir, cfg["cloudflag"]) if cloud_dir else {}
    if not las:
        sys.exit(f"No {cfg['las']} files in {aerosol_dir}")
    grid = None
    frames, meta = [], []
    st = dict(flights=0, no_nav=0, no_smps=0, no_flag=0, flagged=[])
    notes = set()
    for fl, f in las.items():
        nav = _nav(nav_dir, cfg["nav"], fl)
        if nav is None:
            st["no_nav"] += 1
            continue
        L = read_ict(f)
        _, h = read_header(f)
        lcols, dl, how_l = _bins(L, "LAS", h, 30, 20000)
        tot = next((L[c].to_numpy(float) for c in L.columns if re.fullmatch(cfg["las_total"], c)), None)
        lv, conv = _as_dndlogd(L[lcols].to_numpy(float), dl, tot, "LAS")
        notes.add(f"LAS: {conv}; diameters from '{how_l}'")
        if not stp:
            lv = lv / L["stdPT"].to_numpy(float)[:, None]
        idx = L.index
        keep = ~idx.duplicated()
        lv, idx = lv[keep], idx[keep]
        sv, ds = None, None
        if fl in smps:
            S = read_ict(smps[fl])
            _, hs = read_header(smps[fl])
            scols, ds, how_s = _bins(S, "SMPS", hs, 2, 1500)
            s, conv = _as_dndlogd(S[scols].to_numpy(float), ds, S[cfg["smps_total"]].to_numpy(float)
                                  if cfg["smps_total"] in S else None, "SMPS")
            notes.add(f"SMPS: {conv}; diameters from '{how_s}'")
            if not stp:
                s = s / S["stdPT"].to_numpy(float)[:, None]
            base = pd.Timestamp(S.attrs["date"])
            t0 = base + pd.to_timedelta(S[S.attrs["indep"]].to_numpy(float), unit="s")
            t1 = base + pd.to_timedelta(S["End_UTC"].to_numpy(float), unit="s")
            j = np.searchsorted(t0.values, idx.values, side="right") - 1
            ok = j >= 0
            ok[ok] &= idx.values[ok] <= t1.values[j[ok]]
            sv = np.full((len(idx), s.shape[1]), np.nan)
            sv[ok] = s[j[ok]]
        else:
            st["no_smps"] += 1
        if grid is None:
            if ds is None:
                sys.exit("The first flight has no SMPS file; download the SMPS files too.")
            s_keep = np.where(ds < dl[0] * 10 ** (-0.5 * np.log10(dl[1] / dl[0])))[0]   # SMPS below the first LAS bin
            grid = np.concatenate([ds[s_keep], dl])
            n_s = len(s_keep)
        a = np.full((len(idx), len(grid)), np.nan)
        a[:, n_s:] = lv
        if sv is not None:
            a[:, :n_s] = sv[:, s_keep]
        ok = np.ones(len(idx), bool)
        if fl in cflag:
            cf = read_ict(cflag[fl])["CloudFlag"]
            cf = cf[~cf.index.duplicated()].reindex(idx.floor("s"))
            st["flagged"].append(float((cf == 1).mean()))
            ok &= ~(cf == 1).to_numpy()
        else:
            st["no_flag"] += 1
        m = nav.reindex(idx.floor("s"))
        ok = ok & m["alt"].notna().to_numpy()
        frames.append(pd.DataFrame(a[ok], index=idx[ok]))
        meta.append(pd.DataFrame(m.to_numpy()[ok], columns=["alt", "lat", "lon"], index=idx[ok]))
        st["flights"] += 1
    a = pd.concat(frames).sort_index()
    m = pd.concat(meta).sort_index()
    keep = ~a.index.duplicated()
    a, m = a[keep], m[keep]
    print(f"Aerosol: {campaign.upper()} SMPS ({grid[0]:.1f}-{grid[n_s - 1]:.0f} nm) + LAS ({grid[n_s]:.0f}-{grid[-1]:.0f} nm), "
          f"{'STP' if stp else 'ambient (divided by stdPT)'}; {st['flights']} flights, {st['no_nav']} without position, "
          f"{st['no_smps']} without SMPS, {st['no_flag']} without cloud flag; cloud flag removes "
          f"{np.mean(st['flagged']) if st['flagged'] else 0:.0%} of seconds")
    for n in sorted(notes):
        print("  " + n)
    if cloud is not None:
        wet = cloud.index[cloud["Nd"] >= clear_nd]
        if len(wet):
            ws = pd.Series(1, index=wet).resample("1s").max().fillna(0)
            ws = ws.rolling(2 * pad_s + 1, center=True, min_periods=1).max()
            hit = ws.reindex(a.index.floor("s")).fillna(0).to_numpy() > 0
            print(f"  droplet-probe cloud mask (Nd >= {clear_nd} cm-3, +/- {pad_s} s) removes {hit.mean():.0%} of aerosol seconds")
            a, m = a[~hit], m[~hit]
    return a, m, grid
