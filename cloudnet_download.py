#!/usr/bin/env python
"""Download Cloudnet (ACTRIS) product files through the open Cloudnet API.

No account is needed. Files are written to <out>/<site>/<product>/ and files
already present with the same size are skipped, so the command can be re-run.

Examples
--------
List what exists for a site and product, by year:
    python cloudnet_download.py list --site hyytiala --product der

Download one month of droplet-effective-radius files (they hold N_scaled):
    python cloudnet_download.py download --site hyytiala --product der \
        --start 2023-10-01 --end 2023-10-31 --out data/cloudnet
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

API = "https://cloudnet.fmi.fi/api"


def _get_json(path, params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def file_list(site, product, start, end):
    """Newest version of each file; the API returns only current versions by default."""
    return _get_json("files", {"site": site, "product": product,
                               "dateFrom": start, "dateTo": end})


def cmd_list(a):
    files = file_list(a.site, a.product, a.start, a.end)
    if not files:
        print(f"No {a.product} files for {a.site} between {a.start} and {a.end}.")
        return
    d = [f["measurementDate"] for f in files]
    years = Counter(x[:4] for x in d)
    size = sum(int(f["size"]) for f in files) / 1e9
    print(f"{a.site} {a.product}: {len(files)} files, {size:.2f} GB, "
          f"{min(d)} -> {max(d)}")
    for y in sorted(years):
        print(f"  {y}: {years[y]} days")


def cmd_download(a):
    files = file_list(a.site, a.product, a.start, a.end)
    out = os.path.join(a.out, a.site, a.product)
    os.makedirs(out, exist_ok=True)
    print(f"{a.site} {a.product}: {len(files)} files -> {out}")
    stats = Counter()
    for i, f in enumerate(sorted(files, key=lambda x: x["measurementDate"]), 1):
        dest = os.path.join(out, f["filename"])
        if os.path.exists(dest) and os.path.getsize(dest) == int(f["size"]):
            stats["skipped"] += 1
        else:
            part = dest + ".part"
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(f["downloadUrl"], timeout=300) as r, open(part, "wb") as fh:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            fh.write(chunk)
                    os.replace(part, dest)
                    stats["downloaded"] += 1
                    break
                except Exception as e:  # network hiccup: retry, then report
                    if attempt == 2:
                        print(f"  failed {f['filename']}: {e}", file=sys.stderr)
                        stats["failed"] += 1
                    time.sleep(5)
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {dict(stats)}", flush=True)
    print("Cloudnet data: cite the file PIDs/DOIs listed on cloudnet.fmi.fi and acknowledge ACTRIS.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "download"):
        s = sub.add_parser(name)
        s.add_argument("--site", required=True, help="Cloudnet site id, e.g. hyytiala, palaiseau, cabauw, arm-sgp")
        s.add_argument("--product", default="der", help="product id (default der; also lwc, classification, mwr)")
        s.add_argument("--start", default="2010-01-01")
        s.add_argument("--end", default="2030-01-01")
        if name == "download":
            s.add_argument("--out", default="data/cloudnet")
    a = p.parse_args()
    {"list": cmd_list, "download": cmd_download}[a.cmd](a)


if __name__ == "__main__":
    main()
