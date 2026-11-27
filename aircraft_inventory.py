#!/usr/bin/env python
"""Which aircraft campaigns already have what PARSEC needs: vertical wind, static T and p, AMS.

Walks data/aircraft/<campaign>/ (any depth), opens up to --per-folder files in every folder that
holds NetCDF (.nc/.cdf) or ICARTT (.ict) files, and matches their variables (names and long names)
against four groups:

  w     vertical wind / vertical velocity (e.g. ARM AIMMS-20 w, NCAR WIC, NASA Vertical_Wind)
  T     static (ambient) air temperature (e.g. ATX, Static_Air_Temp, temperature)
  p     static (ambient) pressure (e.g. PSXC, Static_Pressure, pressure)
  ams   aerosol mass spectrometer species (org, so4, nh4, no3, chl), or an AMS/ACSM folder

Prints one block per campaign (folder, file count, matching variables) and a summary table, and
writes results/aircraft/inventory.csv (one row per campaign, folder and group). Nothing is changed.

Examples
--------
python aircraft_inventory.py
python aircraft_inventory.py --root data/aircraft --campaigns ace-ena activate
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import netCDF4  # noqa: F401  before numpy, as in the other scripts
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GROUPS = {
    "w": dict(
        name=r"^(w|wi|wic|wz|vw|vwind|w_?wind|vert(ical)?_?wind(_?speed)?|vertical_?velocity|w_?aimms|aimms_?w|"
             r"updraft(_?velocity)?)$",
        desc=r"vertical\s*(wind|velocity|air\s*velocity)|w\s*wind\s*speed|updraft\s*velocity"),
    "T": dict(
        name=r"^(atx|ta|tamb|t_?amb(ient)?|t_?s(tatic)?|static_?air_?temp(erature)?|ambient_?(air_?)?temp(erature)?|"
             r"air_?temp(erature)?|temperature|temp)$",
        desc=r"(static|ambient)\s*(air\s*)?temperature|^\s*air\s*temperature"),
    "p": dict(
        name=r"^(psxc|psx|ps|pstatic|p_?s(tatic)?|static_?pressure|ambient_?pressure|pressure|press|pres)$",
        desc=r"(static|ambient)\s*(air\s*)?pressure"),
    "ams": dict(
        name=r"^(org|so4|nh4|no3|chl|sulfate|sulphate|organics?|nitrate|ammonium|chloride)(_.*)?$",
        desc=r"aerosol mass spectrometer|\bams\b|\bacsm\b|non-refractory"),
}
FOLDER_AMS = re.compile(r"(^|[^a-z])(ams|acsm|hrtofams|toframs)([^a-z]|$)", re.I)


def nc_vars(path):
    import xarray as xr
    with xr.open_dataset(path, decode_times=False) as ds:
        return [(v, str(ds[v].attrs.get("long_name", "")) + " " + str(ds[v].attrs.get("standard_name", "")),
                 str(ds[v].attrs.get("units", ""))) for v in ds.data_vars]


def ict_vars(path):
    from icartt import read_header
    _, h = read_header(path)
    cols = [c.strip() for c in h[-1].split(",")]
    out = []
    for c in cols:
        # ICARTT variable lines start with the column name, then units and descriptions
        line = next((l for l in h[1:-1] if l.split(",")[0].strip() == c), "")
        parts = [p.strip() for p in line.split(",")]
        out.append((c, " ".join(parts[2:]), parts[1] if len(parts) > 1 else ""))
    return out


def base_name(name):
    """Strip ICARTT merge suffixes: the PI name (_THORNHILL) and a unit tag (_ms-1), e.g.
    w_ms-1_THORNHILL -> w, Static_Air_Temp_THORNHILL -> Static_Air_Temp."""
    n = re.sub(r"(_[A-Z][A-Z0-9]{2,})+$", "", name)
    return re.sub(r"_(ms-1|m_?s-?1|deg|degc|k|hpa|mb)$", "", n, flags=re.I)


def match(vars_, folder):
    hits = defaultdict(list)
    for name, desc, units in vars_:
        for g, pat in GROUPS.items():
            if re.match(pat["name"], base_name(name), re.I) or (desc.strip() and re.search(pat["desc"], desc, re.I)):
                hits[g].append(f"{name}[{units}]" if units else name)
    if FOLDER_AMS.search(os.path.basename(folder)):
        hits["ams"].insert(0, "(AMS/ACSM folder)")
    return hits


def scan(campaign_dir, per_folder):
    rows = []
    for folder, _, files in sorted(os.walk(campaign_dir)):
        data = sorted(f for f in files if f.lower().endswith((".nc", ".cdf", ".ict")))
        if not data:
            continue
        vars_, errs = [], []
        for f in data[:per_folder]:
            p = os.path.join(folder, f)
            try:
                vars_ += nc_vars(p) if not f.lower().endswith(".ict") else ict_vars(p)
            except Exception as ex:                                              # noqa: BLE001
                errs.append(f"{f}: {str(ex)[:80]}")
        hits = match(vars_, folder)
        rows.append(dict(folder=os.path.relpath(folder, campaign_dir), files=len(data),
                         nvars=len({v[0] for v in vars_}), errors="; ".join(errs),
                         **{g: ", ".join(dict.fromkeys(hits.get(g, [])))[:200] for g in GROUPS}))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(HERE, "data", "aircraft"))
    ap.add_argument("--campaigns", nargs="*", help="default: every folder under --root")
    ap.add_argument("--per-folder", type=int, default=2, help="files opened per folder (default 2)")
    a = ap.parse_args()
    camps = a.campaigns or sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    allrows, summary = [], []
    for c in camps:
        rows = scan(os.path.join(a.root, c), a.per_folder)
        print(f"\n=== {c}: {len(rows)} data folders")
        for r in rows:
            print(f"  {r['folder']}  ({r['files']} files, {r['nvars']} variables)")
            for g in GROUPS:
                if r[g]:
                    print(f"      {g:4s} {r[g]}")
            if r["errors"]:
                print(f"      could not open: {r['errors']}")
        for r in rows:
            allrows.append(dict(campaign=c, **r))
        summary.append(dict(campaign=c, **{g: "; ".join(r["folder"] for r in rows if r[g]) or "-" for g in GROUPS}))
    out = os.path.join(HERE, "results", "aircraft", "inventory.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(allrows).to_csv(out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\nSummary (folders where each group was found; '-' = not in the downloaded files):")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\n-> {os.path.relpath(out, HERE)}  (check the w and T/p hits by eye: name matching can pick up e.g. "
          f"a probe's internal temperature)")


if __name__ == "__main__":
    main()
