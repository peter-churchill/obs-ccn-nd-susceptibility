#!/usr/bin/env python
"""Add Cloudnet sites that have an EBAS size distribution nearby (from ebas_psd_search.py).

    python cloudnet_ebas_add.py --sites norunda bucharest leipzig maido --stage download   # login node (internet)
    python cloudnet_ebas_add.py --sites norunda bucharest leipzig maido --stage build      # compute node or batch

For each site, from results/ebas_psd_search.csv:
  * the EBAS station + instrument with the largest total overlap with the Cloudnet der record is chosen,
    and all its level-2 files that overlap are used;
  * download: those EBAS files (THREDDS, to data/ebas/<site>/, resumable) and Cloudnet der + lwc for the
    period they cover (cloudnet_download.py, existing files skipped);
  * build: cloudnet_build.py with the usual screens (hourly mean Nd), then the quicklook with the same options
    as the other Cloudnet sites (cloud_frac >= 0.5, base <= 2000 m, top <= 3000 m).
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

THREDDS = "https://thredds.nilu.no/thredds/fileServer/ebas/"
QL = ["--min-cloud-frac", "0.5", "--max-cloud-base", "2000", "--max-cloud-top", "3000"]


def choose(t, site, pick=None):
    s = t[(t.cloudnet == site) & (t.overlap_der_days > 0)]
    if s.empty:
        return None
    tot = s.groupby(["ebas_station", "instrument"]).agg(name=("station_name", "first"), km=("km", "min"),
                                                         overlap=("overlap_der_days", "sum")).sort_values("overlap", ascending=False)
    print(f"  candidates for {site}:")
    for (st, ins), r in tot.iterrows():
        print(f"    {st:9s} {ins:8s} {str(r['name'])[:30]:30s} {r.km:5.1f} km  {int(r.overlap):5d} der days")
    if pick:
        cand = [i for i in tot.index if i[0] == pick.split(":")[0] and (":" not in pick or i[1] == pick.split(":")[1])]
        if not cand:
            print(f"  !!! --pick {pick} not among the candidates")
            return None
        best = cand[0]
    else:
        best = tot.index[0]
    files = s[(s.ebas_station == best[0]) & (s.instrument == best[1])]
    return dict(station=best[0], instrument=best[1], name=files.station_name.iloc[0], km=files.km.min(),
                files=list(files.file), start=pd.to_datetime(files.ebas_start).min(),
                end=min(pd.to_datetime(files.ebas_end).max(), pd.Timestamp.today().normalize()),
                overlap=int(files.overlap_der_days.sum()))


def get(url, dest):
    import requests
    if os.path.exists(dest):
        return "have"
    part = dest + ".part"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(part, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    os.replace(part, dest)
    return "got"


def run(cmd, log=None):
    print("  $", " ".join(cmd), flush=True)
    if log:
        with open(log, "w") as fh:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--csv", default="results/ebas_psd_search.csv")
    ap.add_argument("--stage", choices=["download", "build", "all", "plan"], default="plan")
    ap.add_argument("--pick", nargs="*", default=[], help="site=STATION or site=STATION:instrument to override the choice")
    a = ap.parse_args()
    t = pd.read_csv(a.csv)
    picks = dict(p.split("=", 1) for p in a.pick)
    for site in a.sites:
        print(f"\n--- {site}")
        c = choose(t, site, picks.get(site))
        if c is None:
            print(f"=== {site}: no overlapping EBAS size distribution in {a.csv}; skipped")
            continue
        print(f"=== {site}: EBAS {c['station']} ({c['name']}, {c['km']} km) {c['instrument']}, "
              f"{len(c['files'])} file(s), {c['start'].date()} to {c['end'].date()}, ~{c['overlap']} der days overlap")
        edir = f"data/ebas/{site}"
        if a.stage in ("download", "all"):
            os.makedirs(edir, exist_ok=True)
            for f in c["files"]:
                try:
                    print(f"  EBAS {get(THREDDS + f, os.path.join(edir, f))}: {f}", flush=True)
                except Exception as e:
                    print(f"  !!! EBAS download failed: {f} ({e})")
            for prod in ("der", "lwc"):
                run([sys.executable, "-u", "cloudnet_download.py", "download", "--site", site, "--product", prod,
                     "--start", str(c["start"].date()), "--end", str(c["end"].date()), "--out", "data/cloudnet"])
        if a.stage in ("build", "all"):
            psd = [os.path.join(edir, f) for f in c["files"] if os.path.exists(os.path.join(edir, f))]
            if not psd:
                print("  !!! no EBAS files on disk; run --stage download first")
                continue
            out = f"results/{site}/{site}_cloudnet.nc"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            cmd = [sys.executable, "-u", "cloudnet_build.py", "--der-dir", f"data/cloudnet/{site}/der", "--psd", *psd,
                   "--site-name", f"{c['name']} (Cloudnet + EBAS {c['instrument'].upper()})", "--out", out]
            if os.path.isdir(f"data/cloudnet/{site}/lwc"):
                cmd[cmd.index("--psd"):cmd.index("--psd")] = ["--lwc-dir", f"data/cloudnet/{site}/lwc"]
            if run(cmd, f"logs/{site}_cloudnet_build.log") != 0 or not os.path.exists(out):
                print(f"  !!! build failed, see logs/{site}_cloudnet_build.log")
                continue
            ql = out.replace(".nc", "_quicklook.txt")
            run([sys.executable, "-u", "arm_ena_nd_pnsd.py", "quicklook", out, *QL], ql)
            for line in open(ql, encoding="utf-8", errors="replace"):
                if line.startswith(("Using", "S(", "D_Delta")):
                    print("   ", line.rstrip())


if __name__ == "__main__":
    main()
