#!/usr/bin/env python
"""Search EBAS (thredds.nilu.no) for data that can be paired with Cloudnet droplet number, or for in-situ
cloud droplet measurements.

Size distributions near Cloudnet der sites (default mode):
    python ebas_psd_search.py --max-km 30
    python ebas_psd_search.py --max-km 100 --levels lev2 lev1.5      # wider, and include near-real-time files
    python ebas_psd_search.py --max-km 30 --sites melpitz troll

In-situ cloud droplet data anywhere in EBAS (fog monitor, PVM, CDP, CVI, cloud LWC/Nd), with the size
distributions at the same station:
    python ebas_psd_search.py --cloud

1. Cloudnet API: sites with der files, coordinates, altitude and der date range.
2. EBAS THREDDS catalogue: file names parsed into station, instrument, component, start, period, level.
3. Station coordinates and altitude from one file per station over OPeNDAP (metadata only), cached in
   results/ebas_station_coords.csv.
Writes results/ebas_psd_search.csv (or results/ebas_cloud_search.csv with --cloud).
"""
import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd
import requests

CN = "https://cloudnet.fmi.fi/api"
THREDDS = "https://thredds.nilu.no/thredds"
PSD_INSTR = ("smps", "dmps", "mpss", "dmps_twin", "twin_dmps", "tdmps", "tsmps", "v-smps", "v-tdmps",
             "nanosmps", "nano_smps", "sems", "uhsas")
CLOUD_TOKENS = ("cloud", "droplet", "fog", "pvm", "fm100", "fm-100", "fm120", "cdp", "cvi", "liquid_water",
                "interstitial", "residual")


def km(a, b, c, d):
    p = math.pi / 180
    h = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def parse_ebas(name):
    """EBAS name: STATION.START.REVISION.instrument.component.matrix.period.resolution.lab.method.LEVEL.nc
    (the level may contain a dot, e.g. lev1.5)."""
    m = re.search(r"\.(lev[0-9][0-9.]*)\.nc$", name, re.I)
    if not m:
        return None
    f = name[: m.start()].split(".")
    if len(f) < 10 or not re.fullmatch(r"\d{14}", f[1]):
        return None
    start = datetime.strptime(f[1], "%Y%m%d%H%M%S")
    period = f[6]
    p = re.fullmatch(r"(\d+)(y|mo|w|d|h)", period)
    if p:
        n, u = int(p.group(1)), p.group(2)
        end = start + {"y": timedelta(days=365.25 * n), "mo": timedelta(days=30.44 * n), "w": timedelta(weeks=n),
                       "d": timedelta(days=n), "h": timedelta(hours=n)}[u]
    else:
        end = start + timedelta(days=365)
    instr, comp = f[3].lower(), f[4].lower()
    return dict(file=name, station=f[0], instrument=instr, component=comp, matrix=f[5].lower(), start=start, end=end,
                resolution=f[7], level=m.group(1).lower(),
                is_psd="particle_number_size_distribution" in comp or instr in PSD_INSTR,
                is_cloud=any(t in name.lower().replace("cloud_condensation", "") for t in CLOUD_TOKENS))


def ebas_catalog(session):
    names, todo, seen = [], [f"{THREDDS}/catalog/ebas/catalog.xml"], set()
    while todo:
        url = todo.pop()
        if url in seen:
            continue
        seen.add(url)
        r = session.get(url, timeout=600)
        r.raise_for_status()
        for el in ET.fromstring(r.content).iter():
            tag = el.tag.split("}")[-1]
            if tag == "dataset" and el.get("urlPath"):
                names.append(el.get("urlPath").split("/")[-1])
            elif tag == "catalogRef":
                href = el.get("{http://www.w3.org/1999/xlink}href")
                if href:
                    todo.append(url.rsplit("/", 1)[0] + "/" + href)
    return names


def station_coords(files, cache_csv):
    """lat, lon, altitude (m), name per station from one file's global attributes (OPeNDAP)."""
    import netCDF4
    cols = ["lat", "lon", "alt", "name"]
    cache = pd.read_csv(cache_csv, index_col=0) if os.path.exists(cache_csv) else pd.DataFrame(columns=cols)
    if "alt" not in cache.columns:                       # cache from the first version: refetch
        cache = pd.DataFrame(columns=cols)
    out = {}
    for st, f in files.items():
        if st in cache.index and pd.notna(cache.loc[st, "lat"]):
            out[st] = tuple(cache.loc[st, cols])
            continue
        try:
            with netCDF4.Dataset(f"{THREDDS}/dodsC/ebas/{f}") as nc:
                at = {k: nc.getncattr(k) for k in nc.ncattrs()}
            md = json.loads(at.get("ebas_metadata", "{}")) if "ebas_metadata" in at else {}
            lat = at.get("geospatial_lat_min", md.get("Station latitude"))
            lon = at.get("geospatial_lon_min", md.get("Station longitude"))
            alt = at.get("geospatial_vertical_min", md.get("Station altitude", float("nan")))
            name = str(at.get("ebas_station_name", md.get("Station name", st)))
            out[st] = (float(lat), float(lon), float(alt), name)
        except Exception as e:
            print(f"  ! no coordinates for {st}: {type(e).__name__}: {str(e)[:80]}")
            out[st] = (float("nan"), float("nan"), float("nan"), st)
    pd.DataFrame(out, index=cols).T.to_csv(cache_csv)
    return out


def cloudnet_sites(s, only=None):
    sites = {x["id"]: x for x in s.get(f"{CN}/sites", timeout=60).json()}
    cn = []
    for sid in (only or sorted(sites)):
        x = sites.get(sid)
        if x is None or x.get("latitude") is None:
            continue
        f = s.get(f"{CN}/files", params={"site": sid, "product": "der", "showLegacy": "true"}, timeout=120).json()
        if not f:
            continue
        d = sorted({v["measurementDate"] for v in f})
        cn.append(dict(site=sid, name=x.get("humanReadableName", sid), lat=float(x["latitude"]),
                       lon=float(x["longitude"]), alt=float(x.get("altitude") or float("nan")),
                       der_days=len(d), der_start=pd.Timestamp(d[0]), der_end=pd.Timestamp(d[-1]), der_dates=d))
    return cn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-km", type=float, default=30.0)
    ap.add_argument("--levels", nargs="+", default=["lev2"], help="EBAS levels to accept, e.g. lev2 lev1.5")
    ap.add_argument("--sites", nargs="*", default=None, help="Cloudnet site ids (default: all with der)")
    ap.add_argument("--cloud", action="store_true", help="list in-situ cloud droplet data in EBAS instead")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = requests.Session()
    os.makedirs("results", exist_ok=True)

    print("EBAS THREDDS catalogue ...", flush=True)
    names = ebas_catalog(s)
    parsed = [p for p in (parse_ebas(n) for n in names) if p]
    lev = {x.lower() for x in a.levels}
    print(f"  {len(names)} files, {len(parsed)} parsed; levels present: "
          + ", ".join(f"{k} {v}" for k, v in pd.Series([p['level'] for p in parsed]).value_counts().items()))

    if a.cloud:
        cl = [p for p in parsed if p["is_cloud"]]
        psd = [p for p in parsed if p["is_psd"]]
        first = {}
        for p in cl + psd:
            first.setdefault(p["station"], p["file"])
        coords = station_coords({k: v for k, v in first.items() if k in {p['station'] for p in cl}},
                                "results/ebas_station_coords.csv")
        rows = []
        for p in cl:
            lat, lon, alt, sname = coords.get(p["station"], (float("nan"),) * 3 + (p["station"],))
            sp = [q for q in psd if q["station"] == p["station"] and q["start"] < p["end"] and q["end"] > p["start"]]
            rows.append(dict(station=p["station"], name=sname, lat=lat, lon=lon, alt_m=alt, instrument=p["instrument"],
                             component=p["component"], matrix=p["matrix"], level=p["level"], start=p["start"].date(),
                             end=p["end"].date(), resolution=p["resolution"],
                             psd_same_station=",".join(sorted({q["instrument"] for q in sp})) or "-", file=p["file"]))
        t = pd.DataFrame(rows)
        out = a.out or "results/ebas_cloud_search.csv"
        t.to_csv(out, index=False)
        if t.empty:
            print("No cloud-related files found.")
            return
        g = (t.groupby(["station", "name", "instrument", "component", "matrix"])
             .agg(alt_m=("alt_m", "first"), start=("start", "min"), end=("end", "max"), files=("file", "size"),
                  levels=("level", lambda x: ",".join(sorted(set(x)))),
                  psd_same_station=("psd_same_station", lambda x: ",".join(sorted(set(",".join(x).split(","))))))
             .reset_index().sort_values(["station", "start"]))
        with pd.option_context("display.width", 250, "display.max_colwidth", 45, "display.max_rows", 500):
            print("\n" + g.to_string(index=False))
        print(f"\nAll files in {out}")
        return

    print("Cloudnet sites with der ...", flush=True)
    cn = cloudnet_sites(s, a.sites)
    print(f"  {len(cn)} sites")
    psd = [p for p in parsed if p["is_psd"] and p["level"] in lev]
    print(f"  {len(psd)} size-distribution files at levels {sorted(lev)}, {len({p['station'] for p in psd})} stations")
    first = {}
    for p in psd:
        first.setdefault(p["station"], p["file"])
    coords = station_coords(first, "results/ebas_station_coords.csv")
    rows = []
    for c in cn:
        dates = pd.to_datetime(pd.Series(c["der_dates"]))
        for p in psd:
            lat, lon, alt, sname = coords[p["station"]]
            if lat != lat:
                continue
            dist = km(c["lat"], c["lon"], lat, lon)
            if dist > a.max_km:
                continue
            ov = int(((dates >= pd.Timestamp(p["start"])) & (dates <= pd.Timestamp(p["end"]))).sum())
            rows.append(dict(cloudnet=c["site"], ebas_station=p["station"], station_name=sname, km=round(dist, 1),
                             dalt_m=round(alt - c["alt"]) if alt == alt and c["alt"] == c["alt"] else None,
                             instrument=p["instrument"], level=p["level"], ebas_start=p["start"].date(),
                             ebas_end=p["end"].date(), der_days=c["der_days"], overlap_der_days=ov, file=p["file"]))
    t = pd.DataFrame(rows)
    out = a.out or "results/ebas_psd_search.csv"
    if t.empty:
        print("No EBAS size distributions within", a.max_km, "km of any Cloudnet der site.")
        return
    t = t.sort_values(["cloudnet", "overlap_der_days"], ascending=[True, False])
    t.to_csv(out, index=False)
    summ = (t.groupby("cloudnet").agg(der_days=("der_days", "first"), best_overlap=("overlap_der_days", "max"),
                                       stations=("ebas_station", lambda x: ",".join(sorted(set(x)))),
                                       km=("km", "min"), dalt_m=("dalt_m", "first"),
                                       levels=("level", lambda x: ",".join(sorted(set(x)))),
                                       instruments=("instrument", lambda x: ",".join(sorted(set(x)))))
            .sort_values("best_overlap", ascending=False))
    with pd.option_context("display.width", 220, "display.max_colwidth", 45):
        print("\n" + summ.to_string())
    none = sorted({c["site"] for c in cn} - set(t.cloudnet))
    print(f"\nNo EBAS size distribution within {a.max_km:g} km: {', '.join(none)}")
    print(f"\nDetails (one row per EBAS file) in {out}")


if __name__ == "__main__":
    main()
