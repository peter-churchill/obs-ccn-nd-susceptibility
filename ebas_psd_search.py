#!/usr/bin/env python
"""Which Cloudnet sites with a droplet product (der) have an EBAS particle size distribution nearby?

    python ebas_psd_search.py --max-km 30
    python ebas_psd_search.py --max-km 30 --sites melpitz leipzig maido      # only some Cloudnet sites

1. Cloudnet API: sites with der files, their coordinates and der date range.
2. EBAS THREDDS catalogue (thredds.nilu.no): all level-2 size-distribution files (SMPS, DMPS, MPSS, ...),
   parsed from the file names (station, instrument, start, period).
3. Station coordinates read from one file per station over OPeNDAP (metadata only, nothing downloaded).
4. For each Cloudnet site: EBAS stations within --max-km, with the years of overlap with the der record.
Writes results/ebas_psd_search.csv and prints a table plus the EBAS file names to download.
"""
import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd
import requests

CN = "https://cloudnet.fmi.fi/api"
THREDDS = "https://thredds.nilu.no/thredds"
PSD_INSTR = ("smps", "dmps", "mpss", "dmps_twin", "twin_dmps", "nanosmps", "nano_smps", "sems", "uhsas", "ufp")


def km(a, b, c, d):
    p = math.pi / 180
    h = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def parse_ebas(name):
    """EBAS name: STATION.START.REVISION.instrument.component.matrix.period.resolution.lab.method.level.nc"""
    f = name.split(".")
    if len(f) < 11 or not re.fullmatch(r"\d{14}", f[1]):
        return None
    instr, comp, period = f[3].lower(), f[4].lower(), f[6]
    is_psd = "particle_number_size_distribution" in comp or instr in PSD_INSTR
    if not is_psd or "lev2" not in f[-2].lower():
        return None
    start = datetime.strptime(f[1], "%Y%m%d%H%M%S")
    m = re.fullmatch(r"(\d+)(y|mo|w|d|h)", period)
    if m:
        n, u = int(m.group(1)), m.group(2)
        end = start + {"y": timedelta(days=365.25 * n), "mo": timedelta(days=30.44 * n), "w": timedelta(weeks=n),
                       "d": timedelta(days=n), "h": timedelta(hours=n)}[u]
    else:
        end = start + timedelta(days=365)
    return dict(file=name, station=f[0], instrument=instr, start=start, end=end, resolution=f[7])


def ebas_catalog(session):
    """All dataset names in the flat ebas THREDDS catalogue (follows nested catalogRefs if present)."""
    names, todo, seen = [], [f"{THREDDS}/catalog/ebas/catalog.xml"], set()
    while todo:
        url = todo.pop()
        if url in seen:
            continue
        seen.add(url)
        r = session.get(url, timeout=600)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for el in root.iter():
            tag = el.tag.split("}")[-1]
            if tag == "dataset" and el.get("urlPath"):
                names.append(el.get("urlPath").split("/")[-1])
            elif tag == "catalogRef":
                href = el.get("{http://www.w3.org/1999/xlink}href")
                if href:
                    todo.append(url.rsplit("/", 1)[0] + "/" + href)
    return names


def station_coords(files, cache_csv):
    """Latitude/longitude per station from one file's global attributes (OPeNDAP)."""
    import netCDF4
    cache = pd.read_csv(cache_csv, index_col=0) if os.path.exists(cache_csv) else pd.DataFrame(columns=["lat", "lon", "name"])
    out = {}
    for st, f in files.items():
        if st in cache.index:
            out[st] = tuple(cache.loc[st, ["lat", "lon", "name"]])
            continue
        try:
            with netCDF4.Dataset(f"{THREDDS}/dodsC/ebas/{f}") as nc:
                at = {k: nc.getncattr(k) for k in nc.ncattrs()}
            lat = at.get("geospatial_lat_min", at.get("Station latitude"))
            lon = at.get("geospatial_lon_min", at.get("Station longitude"))
            name = str(at.get("ebas_station_name", at.get("Station name", st)))
            if lat is None or lon is None:                         # EBAS keeps metadata as JSON in ebas_metadata
                import json
                md = json.loads(at.get("ebas_metadata", "{}"))
                lat, lon = md.get("Station latitude"), md.get("Station longitude")
                name = md.get("Station name", name)
            out[st] = (float(lat), float(lon), name)
        except Exception as e:
            print(f"  ! no coordinates for {st}: {type(e).__name__}: {str(e)[:80]}")
            out[st] = (float("nan"), float("nan"), st)
    pd.DataFrame(out, index=["lat", "lon", "name"]).T.to_csv(cache_csv)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-km", type=float, default=30.0)
    ap.add_argument("--sites", nargs="*", default=None, help="Cloudnet site ids (default: all with der)")
    ap.add_argument("--out", default="results/ebas_psd_search.csv")
    a = ap.parse_args()
    s = requests.Session()

    print("Cloudnet sites with der ...", flush=True)
    sites = {x["id"]: x for x in s.get(f"{CN}/sites", timeout=60).json()}
    cn = []
    for sid in (a.sites or sorted(sites)):
        x = sites.get(sid)
        if x is None or x.get("latitude") is None:
            continue
        f = s.get(f"{CN}/files", params={"site": sid, "product": "der", "showLegacy": "true"}, timeout=120).json()
        if not f:
            continue
        d = sorted(v["measurementDate"] for v in f)
        cn.append(dict(site=sid, name=x.get("humanReadableName", sid), lat=float(x["latitude"]), lon=float(x["longitude"]),
                       der_days=len(d), der_start=pd.Timestamp(d[0]), der_end=pd.Timestamp(d[-1]), der_dates=set(d)))
    print(f"  {len(cn)} sites")

    print("EBAS THREDDS catalogue ...", flush=True)
    names = ebas_catalog(s)
    psd = [p for p in (parse_ebas(n) for n in names) if p]
    print(f"  {len(names)} files, {len(psd)} level-2 size-distribution files at {len({p['station'] for p in psd})} stations")
    first = {}
    for p in psd:
        first.setdefault(p["station"], p["file"])
    coords = station_coords(first, "results/ebas_station_coords.csv")

    rows = []
    for c in cn:
        for p in psd:
            lat, lon, sname = coords[p["station"]]
            if not (lat == lat):
                continue
            dist = km(c["lat"], c["lon"], lat, lon)
            if dist > a.max_km:
                continue
            lo, hi = max(c["der_start"], pd.Timestamp(p["start"])), min(c["der_end"], pd.Timestamp(p["end"]))
            ov = sum(1 for d in c["der_dates"] if lo <= pd.Timestamp(d) <= hi) if lo <= hi else 0
            rows.append(dict(cloudnet=c["site"], ebas_station=p["station"], station_name=sname, km=round(dist, 1),
                             instrument=p["instrument"], ebas_start=p["start"].date(), ebas_end=p["end"].date(),
                             der_days=c["der_days"], overlap_der_days=ov, file=p["file"]))
    t = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if t.empty:
        print("No EBAS size distributions within", a.max_km, "km of any Cloudnet der site.")
        return
    t = t.sort_values(["cloudnet", "overlap_der_days"], ascending=[True, False])
    t.to_csv(a.out, index=False)
    summ = (t.groupby("cloudnet").agg(der_days=("der_days", "first"), best_overlap=("overlap_der_days", "max"),
                                       stations=("ebas_station", lambda x: ",".join(sorted(set(x)))),
                                       km=("km", "min"), instruments=("instrument", lambda x: ",".join(sorted(set(x)))))
            .sort_values("best_overlap", ascending=False))
    with pd.option_context("display.width", 200, "display.max_colwidth", 50):
        print("\n" + summ.to_string())
    none = sorted({c["site"] for c in cn} - set(t.cloudnet))
    print(f"\nNo EBAS size distribution within {a.max_km:g} km: {', '.join(none)}")
    print(f"\nDetails (one row per EBAS file) in {a.out}")


if __name__ == "__main__":
    main()
