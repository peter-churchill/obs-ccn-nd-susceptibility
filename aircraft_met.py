"""1-Hz vertical wind, static temperature and pressure, and AMS species from an aircraft's met / merge files.

Used by aircraft_build.py --met-dir to give each cloud pass what a parcel model needs: the vertical-wind
samples during the pass, the cloud-base temperature and pressure, and the below-cloud composition.

    from aircraft_met import read_met, default_met_dir
    met = read_met(folder)        # DataFrame on a 1-s UTC index: w [m s-1], T [K], P [hPa],
                                  # ams_org/so4/nh4/no3/chl [ug m-3, as stored], ams_ok (bool)

Works on NetCDF (ARM aafmerged, NSF/NCAR LRT) and ICARTT (NASA merges and MetNav/housekeeping files).
Variables are picked by name from preference lists (below), else by the name/description patterns of
aircraft_inventory.py; the choice and a quick summary of each candidate are printed, so a wrong pick
(e.g. the aircraft's own climb rate instead of the air's vertical wind) is visible in the log.

AMS seconds are dropped where the file flags them (ams_flag != 0, when 0 is the common value) and where
the AMS sat behind the counterflow virtual impactor (CVI enhancement factor > 1.5): in cloud it then
measures droplet residuals, not the aerosol below.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from aircraft_inventory import GROUPS, base_name  # noqa: E402

# preferred variables, compared with base_name() (PI and unit suffixes stripped), case-insensitive
PREFER = {
    "w": ["vert_wind_speed", "vertical_wind_speed", "vertical_wind", "WIC", "WI", "w", "W", "W_FastMeas"],
    "T": ["static_air_temperature", "static_air_temp", "temperature_static", "ATX", "Static_Air_Temp",
          "Ts", "ambient_temperature", "air_temperature", "temperature"],
    "P": ["static_pressure", "pressure_static", "PSXC", "PSX", "Static_Pressure", "ambient_pressure", "pressure"],
}
NOT_AIR_W = re.compile(r"^(vertical_?speed|vertical_velocity_?platform|climb|vspd)$", re.I)  # aircraft climb rate
AMS_SPECIES = {"org": r"org(anics?)?", "so4": r"so4|sulfate|sulphate", "nh4": r"nh4|ammonium",
               "no3": r"no3|nitrate", "chl": r"chl|chloride"}
AMS_EXCLUDE = re.compile(r"err|qc|flag|unc|prec|lod|detect|_alt|_lat|_lon|coarse|calc|mass_?spec_?pm|ratio", re.I)


def default_met_dir(fmt, cloud_dir, merge_dir=None):
    """Where the met variables usually are for each aircraft_build --format."""
    camp = os.path.dirname(os.path.normpath(cloud_dir))
    if fmt == "ncar":
        return cloud_dir
    if fmt == "activate":
        return merge_dir or os.path.join(camp, "merge")
    if fmt in ("naames", "camp2ex", "oracles"):
        hits = sorted(glob.glob(os.path.join(camp, "*Merge_Data*")))
        return hits[0] if hits else None
    hits = [d for d in sorted(glob.glob(os.path.join(camp, "*aafmergedF1*")))     # ARM: the merged met/state file
            if "aerosol" not in d.lower() and "cld" not in d.lower()]
    return hits[0] if hits else None


def _files(folder):
    return sorted(glob.glob(os.path.join(folder, "*.nc")) + glob.glob(os.path.join(folder, "*.cdf"))
                  + glob.glob(os.path.join(folder, "*.ict")))


def _read_nc(path):
    import xarray as xr
    with xr.open_dataset(path) as ds:
        tname = "time" if "time" in ds.coords or "time" in ds else ("Time" if "Time" in ds else None)
        if tname is None:
            raise ValueError("no time coordinate")
        t = pd.DatetimeIndex(ds[tname].values)
        cols, units, desc = {}, {}, {}
        for v in ds.data_vars:
            da = ds[v]
            if not da.dims or da.dims[0] != tname or da.dtype.kind not in "fiu":
                continue
            a = da.values.astype(float)
            if a.ndim >= 2:
                if not da.dims[1].lower().startswith("sps"):
                    continue                                       # size distributions etc.
                a = np.nanmean(a.reshape(a.shape[0], -1), axis=1)
            a[a <= -9990] = np.nan
            cols[v] = a
            units[v] = str(da.attrs.get("units", ""))
            desc[v] = f"{da.attrs.get('long_name', '')} {da.attrs.get('standard_name', '')}"
    df = pd.DataFrame(cols, index=t)
    df.attrs.update(units=units, desc=desc)
    return df


def _read_ict(path):
    from icartt import read_ict
    df = read_ict(path)
    h = df.attrs["header"]
    desc = {}
    for c in df.columns:
        line = next((l for l in h[1:-1] if l.split(",")[0].strip() == c), "")
        desc[c] = " ".join(p.strip() for p in line.split(",")[2:])
    df.attrs["desc"] = desc
    return df


def _pick(df, group):
    names = list(df.columns)
    bn = {n: base_name(n) for n in names}
    for want in PREFER.get(group, []):
        for n in names:
            if bn[n].lower() == want.lower() and not (group == "w" and NOT_AIR_W.match(bn[n])):
                return n
    pat = GROUPS[group]
    desc = df.attrs.get("desc", {})
    for n in names:
        if group == "w" and NOT_AIR_W.match(bn[n]):
            continue
        if re.match(pat["name"], bn[n], re.I) or re.search(pat["desc"], desc.get(n, ""), re.I):
            return n
    return None


def _kelvin(x, units):
    u = (units or "").lower()
    if ("c" in u and "k" not in u) or np.nanmedian(x) < 100:
        return x + 273.15
    return x


def _hpa(x, units):
    u = (units or "").lower()
    if u in ("pa",) or np.nanmedian(x) > 2000:
        return x / 100.0
    if u == "kpa" or np.nanmedian(x) < 200:
        return x * 10.0
    return x


def _ams(df):
    out, used = {}, {}
    for sp, pat in AMS_SPECIES.items():
        for n in df.columns:
            b = base_name(n)
            core = re.sub(r"^ams_", "", b, flags=re.I)
            if AMS_EXCLUDE.search(n) or not re.match(rf"^({pat})(_ave.*|_isok.*|_large|_ams)?$", core, re.I):
                continue
            if b.lower().startswith("ams_") or "ams" in df.attrs.get("desc", {}).get(n, "").lower() \
                    or re.search(r"isok|ziemba|large|jimenez", n, re.I):
                out[sp], used[sp] = df[n].to_numpy(float), n
                break
    ok = np.ones(len(df), bool)
    notes = []
    flag = next((n for n in df.columns if base_name(n).lower() == "ams_flag"), None)
    if flag is not None and out:
        f = df[flag].to_numpy(float)
        if np.mean(f == 0) > 0.05:
            ok &= f == 0
            notes.append(f"{flag}==0 keeps {np.mean(f == 0):.0%}")
    cvi = next((n for n in df.columns if re.search(r"ams_?cvi_?enhancement", n, re.I)
                and not n.lower().startswith("qc")), None)
    if cvi is not None and out:
        e = df[cvi].to_numpy(float)
        bad = np.isfinite(e) & (e > 1.5)
        ok &= ~bad
        notes.append(f"{cvi} > 1.5 (behind the CVI) drops {bad.mean():.0%}")
    return out, used, ok, notes


def read_met(folder, verbose=True):
    """1-s DataFrame with w [m s-1], T [K], P [hPa], ams_<species> and ams_ok, or None if nothing was found."""
    if not folder or not os.path.isdir(folder):
        if verbose:
            print(f"Met: no folder ({folder}); vertical wind, cloud-base T/P and AMS are left empty")
        return None
    parts, chosen, notes_all = [], {}, set()
    for f in _files(folder):
        try:
            df = _read_ict(f) if f.lower().endswith(".ict") else _read_nc(f)
        except Exception as ex:                                          # noqa: BLE001
            print(f"  met: skip {os.path.basename(f)}: {str(ex)[:80]}")
            continue
        units = df.attrs.get("units", {})
        out = pd.DataFrame(index=df.index)
        for g, conv in (("w", None), ("T", _kelvin), ("P", _hpa)):
            n = _pick(df, g)
            if n is None:
                out[g] = np.nan
                continue
            x = df[n].to_numpy(float)
            out[g] = conv(x, units.get(n, "")) if conv else x
            chosen.setdefault(g, set()).add(f"{n} [{units.get(n, '')}]")
        ams, used, ok, notes = _ams(df)
        for sp in AMS_SPECIES:
            out[f"ams_{sp}"] = ams.get(sp, np.nan)
        out["ams_ok"] = ok if ams else False
        for sp, n in used.items():
            chosen.setdefault("ams", set()).add(n)
        notes_all.update(notes)
        parts.append(out)
    if not parts:
        print(f"Met: nothing readable in {folder}")
        return None
    m = pd.concat(parts).sort_index()
    m = m[~m.index.duplicated()]
    if m.index.tz is not None:
        m.index = m.index.tz_convert(None)
    m.index = m.index.floor("s")
    m = m[~m.index.duplicated()]
    if verbose:
        print(f"Met: {os.path.basename(os.path.normpath(folder))}, {len(m)} s")
        for g in ("w", "T", "P", "ams"):
            print(f"  {g:4s} {', '.join(sorted(chosen.get(g, []))) or 'not found'}")
        if m["w"].notna().any():
            w = m["w"].dropna()
            print(f"       w: mean {w.mean():+.2f}, std {w.std():.2f} m s-1 (air vertical wind: mean ~0, std ~0.2-1; "
                  f"a climb rate would show std of several m s-1)")
        if m["T"].notna().any():
            print(f"       T median {m['T'].median():.1f} K, P median {m['P'].median():.0f} hPa")
        for n in sorted(notes_all):
            print(f"       ams: {n}")
    return m
