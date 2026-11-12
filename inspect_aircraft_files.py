#!/usr/bin/env python
"""Summarise the downloaded NAAMES / ORACLES / CAMP2Ex / ATom files so the readers can be written:
for one file per instrument, the variable names and units (ICARTT header, or NetCDF variables), bin-edge
comments, and the date range of all files of that instrument.

    python inspect_aircraft_files.py > logs/aircraft_headers.txt
"""
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icartt import read_header  # noqa: E402

ROOT = "data/aircraft"
KEEP_NC = re.compile(r"cloud|cdp|cas|caps|drop|lwc|nd|n_|conc|size|dndlog|bin|diam|alt|lat|lon|time|flag|smps|uhsas|las|nmass|temp|pres|ptot", re.I)


def ict_summary(path):
    n, h = read_header(path)
    nv = int(re.split(r"[,\s]+", h[9].strip())[0])
    print(f"  header lines {n}; PI line: {h[1][:70]}; instrument: {h[3][:70]}")
    print(f"  independent: {h[8][:60]}; missing: {h[11][:40]}")
    names = [v.split(",")[0].strip() for v in h[12:12 + nv]]
    units = [v.split(",")[1].strip() if "," in v else "" for v in h[12:12 + nv]]
    # compress numbered bins (e.g. CDP_Bin01..CDP_Bin30) into one line
    groups = defaultdict(list)
    for nm, u in zip(names, units):
        key = re.sub(r"\d+$", "#", nm)
        groups[(key, u)].append(nm)
    print(f"  {nv} variables:")
    for (key, u), g in groups.items():
        print(f"    {key if len(g) > 1 else g[0]:40s} [{u}]" + (f"  x{len(g)} ({g[0]}..{g[-1]})" if len(g) > 1 else ""))
    for line in h:
        if re.search(r"bin|edge|diam|size|lower|upper|STP|ambient|inlet|LLOD|cloud", line, re.I) and len(line) < 400 \
                and not line.split(",")[0].strip() in names:
            print(f"    | {line[:300]}")


def nc_summary(path):
    import xarray as xr
    with xr.open_dataset(path, decode_times=False) as ds:
        print(f"  dims {dict(ds.sizes)}")
        for v in ds.variables:
            if KEEP_NC.search(v) or KEEP_NC.search(str(ds[v].attrs.get("long_name", ""))):
                print(f"    {v:40s} {str(ds[v].dims):30s} [{ds[v].attrs.get('units', '')}] {str(ds[v].attrs.get('long_name', ''))[:70]}")
        if not ds.variables:
            print("    (groups only?)")


def main():
    for camp in sorted(os.listdir(ROOT)):
        cdir = os.path.join(ROOT, camp)
        if camp == "activate" or not os.path.isdir(cdir):
            continue
        print(f"\n######## {camp}")
        for coll in sorted(os.listdir(cdir)):
            files = sorted(f for f in glob.glob(os.path.join(cdir, coll, "*")) if re.search(r"\.(ict|nc|nc4|h5|csv)$", f, re.I))
            if not files:
                continue
            stems = defaultdict(list)
            for f in files:
                stems[re.sub(r"[_-]?20\d{6}.*$", "", os.path.basename(f))].append(f)
            for stem, fs in sorted(stems.items()):
                dates = sorted({m.group(1) for f in fs for m in [re.search(r"(20\d{6})", os.path.basename(f))] if m})
                print(f"\n=== {coll} / {stem}: {len(fs)} files" + (f", {dates[0]}..{dates[-1]}" if dates else ""))
                f = fs[len(fs) // 2]
                print(f"  example: {os.path.basename(f)}")
                try:
                    if f.lower().endswith(".ict"):
                        ict_summary(f)
                    elif f.lower().endswith(".csv"):
                        print("  " + open(f, encoding="utf-8", errors="replace").readline().strip()[:300])
                    else:
                        nc_summary(f)
                except Exception as e:
                    print(f"  !! could not read: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
