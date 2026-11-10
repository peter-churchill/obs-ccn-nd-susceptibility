#!/usr/bin/env python
"""Find and download NASA airborne campaign data (any DAAC: ASDC, ORNL DAAC, ...) via Earthdata, for campaigns
beyond ACTIVATE: NAAMES, ORACLES, CAMP2Ex, ATom, ...

1. Which collections exist (metadata only):
       python nasa_aircraft_download.py discover NAAMES ORACLES CAMP2Ex ATom
   prints short name, version, DAAC, file count, total size and title for every collection whose metadata matches
   the campaign, and writes data/aircraft/<campaign>/collections.csv.
2. File inventory of chosen collections (names, dates, sizes; nothing downloaded):
       python nasa_aircraft_download.py list NAAMES --short-name NAAMES_Cloud_AircraftInSitu_Data
3. Download (resumable; existing files are skipped):
       python nasa_aircraft_download.py get NAAMES --short-name NAAMES_Cloud_AircraftInSitu_Data --match "CDP|2DS"

Login: the same Earthdata credentials as activate_download.py (~/.netrc, never stored in this script).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activate_download import login  # noqa: E402

AIRBORNE = re.compile(r"aircraft|airborne|in ?situ|merge|insitu|dc-8|p-3|c-130|er-2|falcon|king air|g-1|cloud|aerosol", re.I)


def collections(ea, campaign):
    res = ea.search_datasets(keyword=campaign, count=-1)
    rows = []
    for c in res:
        umm = c.get("umm", {}) if isinstance(c, dict) else c["umm"]
        meta = c.get("meta", {}) if isinstance(c, dict) else c["meta"]
        title = umm.get("EntryTitle", "")
        short = umm.get("ShortName", "")
        projects = " ".join(p.get("ShortName", "") for p in umm.get("Projects", []) or [])
        if campaign.lower() not in (title + " " + short + " " + projects).lower():
            continue
        rows.append(dict(short_name=short, version=umm.get("Version", ""), daac=meta.get("provider-id", ""),
                         concept_id=meta.get("concept-id", ""), title=title,
                         airborne=bool(AIRBORNE.search(title + " " + short))))
    return rows


def granules(ea, short, version=None, concept_id=None):
    kw = {"concept_id": concept_id} if concept_id else {"short_name": short}
    if version and not concept_id:
        kw["version"] = version
    g = ea.search_data(count=-1, **kw)
    if not g and version and not concept_id:
        g = ea.search_data(short_name=short, count=-1)
    out = []
    for x in g:
        try:
            size = float(x.size())
        except Exception:
            size = float("nan")
        links = x.data_links()
        for url in links:
            name = url.rsplit("/", 1)[-1]
            m = re.search(r"(20\d{6})", name)
            out.append(dict(name=name, url=url, date=m.group(1) if m else "", size_mb=round(size / max(len(links), 1), 3)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover"); d.add_argument("campaigns", nargs="+")
    d.add_argument("--all", action="store_true", help="also list non-airborne collections (satellite, ship, ground)")
    d.add_argument("--no-count", action="store_true", help="skip counting files per collection (faster)")
    for name in ("list", "get"):
        s = sub.add_parser(name); s.add_argument("campaign")
        s.add_argument("--short-name", required=True); s.add_argument("--version", default=None)
        s.add_argument("--concept-id", default=None, help="use when a short name exists in several versions")
        s.add_argument("--match", default=""); s.add_argument("--exclude", default="")
        s.add_argument("--dest", default=None, help="default data/aircraft/<campaign>/<short name>")
        if name == "get":
            s.add_argument("--dry-run", action="store_true"); s.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    ea = login()

    if a.cmd == "discover":
        for camp in a.campaigns:
            rows = [r for r in collections(ea, camp) if a.all or r["airborne"]]
            if not a.no_count:
                for r in rows:
                    g = granules(ea, r["short_name"], concept_id=r["concept_id"])
                    r["files"] = len(g)
                    r["size_gb"] = round(sum(x["size_mb"] for x in g if x["size_mb"] == x["size_mb"]) / 1024, 2)
                    r["dates"] = (f"{min(x['date'] for x in g if x['date'])}-{max(x['date'] for x in g if x['date'])}"
                                  if any(x["date"] for x in g) else "")
            out = Path(f"data/aircraft/{camp.lower()}"); out.mkdir(parents=True, exist_ok=True)
            with open(out / "collections.csv", "w", newline="", encoding="utf-8") as fh:
                keys = ["short_name", "version", "daac", "files", "size_gb", "dates", "title", "concept_id", "airborne"]
                w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
            print(f"\n=== {camp}: {len(rows)} collections{'' if a.all else ' (airborne)'} -> {out / 'collections.csv'}")
            for r in sorted(rows, key=lambda r: r["short_name"]):
                cnt = "" if a.no_count else f"{r['files']:5d} files {r['size_gb']:7.2f} GB {r['dates']:17s}"
                print(f"  {r['short_name'][:46]:46s} v{str(r['version'])[:4]:4s} {r['daac'][:9]:9s} {cnt} {r['title'][:70]}")
        return

    rows = granules(ea, a.short_name, a.version, a.concept_id)
    keep = [r for r in rows if (not a.match or re.search(a.match, r["name"], re.I))
            and not (a.exclude and re.search(a.exclude, r["name"], re.I))]
    dest = Path(a.dest or f"data/aircraft/{a.campaign.lower()}/{a.short_name}"); dest.mkdir(parents=True, exist_ok=True)
    if a.cmd == "list":
        f = dest / "inventory.csv"
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "date", "size_mb", "url"]); w.writeheader(); w.writerows(keep)
        print(f"{a.short_name}: {len(keep)} files, {sum(r['size_mb'] for r in keep if r['size_mb'] == r['size_mb'])/1024:.2f} GB -> {f}")
        stems = sorted({re.sub(r"[_-]?20\d{6}.*$", "", r["name"]) for r in keep})
        print("  file name stems:\n   " + "\n   ".join(stems[:60]) + ("\n   ..." if len(stems) > 60 else ""))
        return
    todo = [r for r in keep if not (dest / r["name"]).exists()]
    gb = sum(r["size_mb"] for r in todo if r["size_mb"] == r["size_mb"]) / 1024
    print(f"{a.short_name}: {len(rows)} files, {len(keep)} match, {len(todo)} to download ({gb:.2f} GB) -> {dest}")
    if a.dry_run or not todo:
        for r in todo[:20]:
            print("  ", r["name"], r["size_mb"], "MB")
        return
    got = ea.download([r["url"] for r in todo], local_path=str(dest), threads=a.threads)
    missing = [r["name"] for r in todo if not (dest / r["name"]).exists()]
    print(f"downloaded {len(got)} of {len(todo)}" + (f"; {len(missing)} missing, rerun to retry" if missing else ""))


if __name__ == "__main__":
    main()
