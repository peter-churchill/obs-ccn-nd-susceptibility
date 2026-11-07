#!/usr/bin/env python
"""Download NASA ACTIVATE (2020-2022, NW Atlantic) HU-25 Falcon in-situ data from the ASDC via Earthdata.

Step 1 (inventory, a few MB of metadata, no data):
    python activate_download.py --list
  writes data/aircraft/activate/inventory_<collection>.csv (file name, size, date) for each collection,
  so we can see the file naming before downloading anything big.

Step 2 (download), e.g. the 1-s merges plus the droplet and aerosol probe files:
    python activate_download.py --get merge --match "01S|_1s|mrg01"      # adjust after looking at the inventory
    python activate_download.py --get cloud --match "FCDP|2DS"
    python activate_download.py --get aerosol --match "LAS|SMPS|CPC|InletFlag"

Login: needs a free NASA Earthdata account (https://urs.earthdata.nasa.gov). The first run asks for the
user name and password at the prompt and saves them to your own ~/.netrc (_netrc on Windows); they are
never written into this script. Already-downloaded files are skipped, so the script can be restarted.
"""
import argparse, csv, os, re, sys
from pathlib import Path

COLLECTIONS = {  # short_name, version
    "merge":   ("ACTIVATE_Merge_Data", "1"),
    "cloud":   ("ACTIVATE_Cloud_AircraftInSitu_Falcon_Data", "1"),
    "aerosol": ("ACTIVATE_Aerosol_AircraftInSitu_Falcon_Data", "1"),
    "metnav":  ("ACTIVATE_MetNav_AircraftInSitu_Falcon_Data", "1"),
}


def login():
    import earthaccess

    def _try(strategy, **kw):
        try:
            a = earthaccess.login(strategy=strategy, **kw)
            return a if getattr(a, "authenticated", False) else None
        except Exception:      # strategy not available (no netrc file, no env variables)
            return None

    auth = _try("netrc") or _try("environment")   # saved netrc, or EARTHDATA_USERNAME / EARTHDATA_PASSWORD
    if auth is None:
        print("Earthdata login: enter your user name and password (saved to your own netrc file).")
        auth = _try("interactive", persist=True)
    if auth is None:
        sys.exit("Earthdata login failed: check the user name/password at https://urs.earthdata.nasa.gov")
    return earthaccess


def granules(ea, key):
    short, ver = COLLECTIONS[key]
    g = ea.search_data(short_name=short, version=ver, count=-1)
    if not g:  # some ASDC collections are registered without the version string
        g = ea.search_data(short_name=short, count=-1)
    out = []
    for x in g:
        for url in x.data_links():
            name = url.rsplit("/", 1)[-1]
            m = re.search(r"(20\d{6})", name)
            try:
                size = float(x.size())
            except Exception:
                size = float("nan")
            out.append(dict(name=name, url=url, date=m.group(1) if m else "", size_mb=round(size, 3)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default="data/aircraft/activate")
    p.add_argument("--list", action="store_true", help="write the file inventory of every collection")
    p.add_argument("--get", choices=list(COLLECTIONS), help="collection to download")
    p.add_argument("--match", default="", help="regex on the file name (case-insensitive)")
    p.add_argument("--exclude", default="", help="regex of file names to skip")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    ea = login()

    if a.list:
        for key, (short, _) in COLLECTIONS.items():
            rows = granules(ea, key)
            f = dest / f"inventory_{key}.csv"
            with open(f, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["name", "date", "size_mb", "url"]); w.writeheader(); w.writerows(rows)
            tot = sum(r["size_mb"] for r in rows if r["size_mb"] == r["size_mb"])
            print(f"{key:8s} {short}: {len(rows)} files, {tot/1024:.2f} GB -> {f}")
            stems = sorted({re.sub(r"_?20\d{6}.*$", "", r["name"]) for r in rows})
            print("   file name stems:", ", ".join(stems[:40]) + (" ..." if len(stems) > 40 else ""))
        return

    if not a.get:
        p.error("give --list or --get")
    rows = granules(ea, a.get)
    keep = [r for r in rows if (not a.match or re.search(a.match, r["name"], re.I))
            and not (a.exclude and re.search(a.exclude, r["name"], re.I))]
    out = dest / a.get
    out.mkdir(exist_ok=True)
    todo = [r for r in keep if not (out / r["name"]).exists()]
    gb = sum(r["size_mb"] for r in todo if r["size_mb"] == r["size_mb"]) / 1024
    print(f"{a.get}: {len(rows)} files in collection, {len(keep)} match, {len(todo)} still to download ({gb:.2f} GB) -> {out}")
    if a.dry_run or not todo:
        for r in todo[:20]:
            print("  ", r["name"], r["size_mb"], "MB")
        return
    got = ea.download([r["url"] for r in todo], local_path=str(out), threads=a.threads)
    print(f"downloaded {len(got)} of {len(todo)}")
    missing = [r["name"] for r in todo if not (out / r["name"]).exists()]
    if missing:
        print(f"!!! {len(missing)} files missing, rerun the same command to retry, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
