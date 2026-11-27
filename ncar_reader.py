#!/usr/bin/env python
"""Reader for NSF/NCAR research-aircraft low-rate (LRT, 1 Hz) files from the EOL archive (GV HIAPER and C-130:
SOCRATES 2018, CSET 2015, VOCALS-REx 2008, ...), returning the same objects as the other aircraft readers, so
aircraft_build's pass detection and pairing run unchanged.

One netCDF file per flight (NCAR/RAF format) holds every probe. The variable names follow RAF conventions,
<prefix>_<probe location suffix>:
    droplets   CDP       CONCD_*  (cm-3, whole probe), PLWCD_* (g m-3), CCDP_*  (per-channel cm-3, CellSizes in um)
               FSSP-100  CONCF_*,  PLWCF_*,  CFSSP_*   (older campaigns, e.g. VOCALS C-130; --ncar-cloud-probe fssp)
    aerosol    UHSAS     CONCU_*,  CUHSAS_*  (~0.06-1 um)
               PCASP     CONCP_*,  CS200_*   (~0.1-3 um; "SPP-200")
    position   GGLAT / GGLON / GGALT (fall back to LAT/LON/ALT and LATC/LONC/GGALTC)
Per-channel arrays are (Time, sps1, VectorN); channel 0 is unused in RAF files. CellSizes holds the bin edges
(N or N+1 values); bins are rebuilt from consecutive edges and converted to dN/dlogDp. Which suffix to use when a
probe appears more than once (e.g. a wing and a cabin UHSAS) is picked as the one with most valid data, or set
with --ncar-cloud-var / --ncar-aerosol-var.

Run `python ncar_reader.py inspect <file.nc>` on one flight first: it lists the probe variables, units, CellSizes
and how per-channel sums compare with the probe totals, so the choices above can be checked per campaign.

Caveats
- Wing-mounted UHSAS/PCASP measure at ambient RH and pressure; values are used as they are (ambient). Hygroscopic
  growth on the wing at high RH below cloud shifts sizes up (check with --max-rh in aircraft_build if needed).
- The UHSAS starts at ~60 nm and the PCASP at ~100 nm: N(>D) is only defined above those, so D_Delta below them
  cannot be found from these campaigns.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

POS = {"lat": ("GGLAT", "LAT", "LATC"), "lon": ("GGLON", "LON", "LONC"), "alt": ("GGALT", "GGALTC", "ALT", "PALT")}
CLOUD = {"cdp": ("CONCD_", "PLWCD_", "CCDP_"), "fssp": ("CONCF_", "PLWCF_", "CFSSP_")}
AEROSOL = {"uhsas": ("CONCU_", "CUHSAS_"), "pcasp": ("CONCP_", "CS200_")}


def files_in(folder):
    f = sorted(glob.glob(os.path.join(folder, "*.nc")) + glob.glob(os.path.join(folder, "*.cdf")))
    if not f:
        sys.exit(f"No .nc files in {folder}")
    return f


def _1hz(da):
    """(Time, sps1[, Vector]) -> (Time[, Vector]); averages the sub-second samples if sps > 1."""
    a = da.values.astype(float)
    a[a <= -32767] = np.nan                                 # RAF missing value
    if a.ndim >= 2 and da.dims[1].startswith("sps"):
        a = np.nanmean(a, axis=1) if a.shape[1] > 1 else a[:, 0]
    return a


def _pos(ds):
    out = {}
    for k, names in POS.items():
        v = next((n for n in names if n in ds), None)
        out[k] = _1hz(ds[v]) if v else np.full(ds.sizes["Time"], np.nan)
    return out


def _edges(da):
    """Bin edges (um) for the channels of a RAF per-channel variable, and the channel indices they belong to."""
    cs = da.attrs.get("CellSizes")
    if cs is None:
        return None, None
    cs = np.asarray(cs, float)
    n = da.shape[-1]
    if len(cs) == n + 1:                         # edges for every channel
        return cs, np.arange(n)
    if len(cs) == n:                             # RAF: channel 0 unused, CellSizes[i-1]..CellSizes[i] is channel i
        return cs, np.arange(1, n)
    return None, None


def _suffixes(ds, prefix):
    return sorted(v[len(prefix):] for v in ds.data_vars if v.startswith(prefix))


def _pick(ds, prefix, override=None):
    if override:
        return override[len(prefix):] if override.startswith(prefix) else override
    sfx = _suffixes(ds, prefix)
    if not sfx:
        return None
    return max(sfx, key=lambda s: np.isfinite(_1hz(ds[prefix + s])).sum())


def read_cloud_ncar(folder, dmin, dmax, probe="cdp", var=None):
    """1-Hz droplet number (cm-3) and LWC (g m-3) in dmin-dmax um, with position."""
    pc, pl, pd_ = CLOUD[probe]
    parts = []
    for f in files_in(folder):
        with xr.open_dataset(f) as ds:
            s = _pick(ds, pc, var)
            if s is None:
                print(f"  (no {pc}* in {os.path.basename(f)}; skipped)")
                continue
            t = pd.DatetimeIndex(ds["Time"].values)
            nd, lwc = _1hz(ds[pc + s]), _1hz(ds[pl + s]) if pl + s in ds else np.full(len(t), np.nan)
            how = "whole probe"
            if pd_ + s in ds:
                e, ch = _edges(ds[pd_ + s])
                if e is not None:
                    a = _1hz(ds[pd_ + s])[:, ch]
                    lo, hi = np.maximum(e[:-1], dmin), np.minimum(e[1:], dmax)
                    w = np.clip(hi - lo, 0, None) / (e[1:] - e[:-1])            # fraction of each bin in range
                    dc = np.sqrt(e[:-1] * e[1:])
                    nd = np.nansum(a * w, axis=1)
                    nd[~np.isfinite(a).any(axis=1)] = np.nan
                    lwc = np.nansum(a * w * (np.pi / 6) * (dc * 1e-4) ** 3, axis=1) * 1e6   # cm-3 * cm3 -> g m-3
                    how = f"bins {dmin:g}-{dmax:g} um"
            p = _pos(ds)
            parts.append(pd.DataFrame({"Nd": nd, "LWC": lwc, **p}, index=t))
            print(f"  {os.path.basename(f)}: droplets from {pc}{s} ({how})")
    if not parts:
        sys.exit(f"No {probe.upper()} data found in {folder}")
    c = pd.concat(parts).sort_index()
    return c[~c.index.duplicated()]


def read_aerosol_ncar(folder, cloud=None, clear_nd=1.0, pad_s=2, probe="uhsas", var=None):
    """1-Hz dN/dlogDp (cm-3) on one diameter grid (nm), clear-air seconds only, with position."""
    ptot, pch = AEROSOL[probe]
    frames, meta, grid = [], [], None
    for f in files_in(folder):
        with xr.open_dataset(f) as ds:
            s = _pick(ds, pch, var)
            if s is None:
                print(f"  (no {pch}* in {os.path.basename(f)}; skipped)")
                continue
            e, ch = _edges(ds[pch + s])
            if e is None:
                print(f"  (no usable CellSizes on {pch}{s} in {os.path.basename(f)}; skipped)")
                continue
            a = _1hz(ds[pch + s])[:, ch]
            a[a < 0] = np.nan
            dlog = np.log10(e[1:] / e[:-1])
            dc = np.sqrt(e[:-1] * e[1:]) * 1000.0                                  # nm
            if ptot + s in ds:                                                     # per-channel vs total check
                tot = _1hz(ds[ptot + s])
                good = np.isfinite(tot) & (tot > 1)
                if good.sum() > 30:
                    r = np.nanmedian(np.nansum(a[good], axis=1) / tot[good])
                    if not 0.7 < r < 1.3:
                        print(f"  !!! {os.path.basename(f)}: sum of {pch}{s} channels / {ptot}{s} = {r:.2f} (expected ~1)")
            a = a / dlog[None, :]
            if grid is None:
                grid = dc
            if len(dc) != len(grid) or not np.allclose(dc, grid, rtol=0.02):
                a = np.array([np.interp(np.log10(grid), np.log10(dc), row, left=np.nan, right=np.nan) for row in a])
            t = pd.DatetimeIndex(ds["Time"].values)
            frames.append(pd.DataFrame(a, index=t))
            meta.append(pd.DataFrame(_pos(ds), index=t))
            print(f"  {os.path.basename(f)}: aerosol from {pch}{s}, {dc.min():.0f}-{dc.max():.0f} nm")
    if not frames:
        sys.exit(f"No {probe.upper()} size distributions found in {folder}")
    a = pd.concat(frames).sort_index()
    m = pd.concat(meta).sort_index()
    keep = ~a.index.duplicated()
    a, m = a[keep], m[keep]
    a = a[np.isfinite(a.to_numpy()).any(axis=1)]
    m = m.loc[a.index]
    if cloud is not None:
        wet = cloud.index[cloud["Nd"] >= clear_nd]
        if len(wet):
            ws = pd.Series(1, index=wet).resample("1s").max().fillna(0)
            ws = ws.rolling(2 * pad_s + 1, center=True, min_periods=1).max()
            hit = ws.reindex(a.index.floor("s")).fillna(0).to_numpy() > 0
            print(f"  droplet-probe cloud mask (Nd >= {clear_nd} cm-3, +/- {pad_s} s) removes {hit.mean():.0%} of aerosol seconds")
            a, m = a[~hit], m[~hit]
    return a, m, grid


def inspect(path):
    """Print what a RAF file holds for this analysis."""
    with xr.open_dataset(path) as ds:
        t = pd.DatetimeIndex(ds["Time"].values)
        print(f"{os.path.basename(path)}: {t[0]} to {t[-1]} ({len(t)} s); project {ds.attrs.get('ProjectName', '?')}, "
              f"platform {ds.attrs.get('Platform', '?')}, flight {ds.attrs.get('FlightNumber', '?')}")
        print("position:", {k: next((n for n in v if n in ds), None) for k, v in POS.items()})
        for group in (CLOUD, AEROSOL):
            for probe, prefixes in group.items():
                for pre in prefixes:
                    for s in _suffixes(ds, pre):
                        da = ds[pre + s]
                        v = _1hz(da)
                        line = (f"  {probe:5s} {pre + s:16s} dims {da.dims} units '{da.attrs.get('units', '')}' "
                                f"valid {np.isfinite(v).mean() if v.ndim == 1 else np.isfinite(v).any(axis=1).mean():.0%}")
                        e, ch = _edges(da)
                        if e is not None:
                            line += f"  CellSizes {len(da.attrs['CellSizes'])} values {e[0]:.3g}-{e[-1]:.3g} um ({len(ch)} bins)"
                        print(line)
        other = [v for v in ds.data_vars if re.match(r"^(CONC|CS|CU|CCDP|C2D|CONCN|RHUM|PSXC|ATX)", v)]
        print("other concentration-like variables:", ", ".join(sorted(other)[:40]))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        for p in sys.argv[2:]:
            inspect(p)
            print()
    else:
        print(__doc__)
