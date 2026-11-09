#!/usr/bin/env python
"""Rebuild every Cloudnet, aircraft and AO2018 file with mean and median Nd, and compare.

    python rebuild_nd_mean.py --dry-run          # list the files found and the commands, run nothing
    python rebuild_nd_mean.py --jobs 6           # rebuild (run inside an interactive/batch allocation)

For each built file X.nc it reconstructs the original build command from the file's own attributes
(aircraft and AO2018: the stored 'settings'; Cloudnet: der_dir, psd_files and screens), then builds
    X.nc         with --nd-stat mean    (the new default; the name downstream code already uses)
    X_median.nc  with --nd-stat median  (sensitivity)
from that same command, so the two differ only in the statistic. The previous X.nc is kept as X_orig.nc.
Both are quicklooked with the same options, and a before/after table of S(100 nm) and D_Delta is printed
and written to results/nd_stat_comparison.csv. ACTIVATE winter/summer splits are regenerated from the new files.
NDROP sites need nothing: their existing files already use the hourly mean (median versions are *_median.nc).
"""
import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import xarray as xr

QL_CLOUDNET = "--min-cloud-frac 0.5 --max-cloud-base 2000 --max-cloud-top 3000"
QL_OTHER = "--min-cloud-frac 0.5"
ARM_CODE = {"arm-oliktok": "oli", "arm-bankhead": "bnf", "arm-sgp": "sgp", "arm-graciosa": "ena"}


def settings_args(s, skip=()):
    """'k=v k2=v with spaces ...' -> CLI args; True flags kept, False/None dropped."""
    out = []
    for tok in re.split(r" (?=[a-z_0-9]+=)", s.strip()):
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k in skip or v in ("None", "False", ""):
            continue
        flag = "--" + k.replace("_", "-")
        if v == "True":
            out.append(flag)
        elif v.startswith("[") and v.endswith("]"):
            out += [flag] + [x.strip().strip("'\"") for x in v[1:-1].split(",")]
        else:
            out += [flag, v]
    return out


def _site_from_path(p):
    parts = re.split(r"[\\/]+", p)
    return parts[parts.index("cloudnet") + 1] if "cloudnet" in parts else None


def cloudnet_cmd(f, attrs):
    site = _site_from_path(attrs.get("der_dir", ""))
    if not site:
        return None, "no der_dir attribute"
    cmd = ["python", "-u", "cloudnet_build.py", "--der-dir", f"data/cloudnet/{site}/der"]
    if os.path.isdir(f"data/cloudnet/{site}/lwc"):
        cmd += ["--lwc-dir", f"data/cloudnet/{site}/lwc"]
    psd = []
    for name in [x for x in str(attrs.get("psd_files", "")).split(";") if x]:
        if site in ARM_CODE and os.path.isdir(f"data/arm/{ARM_CODE[site]}/{name}"):
            psd.append(f"data/arm/{ARM_CODE[site]}/{name}")
            continue
        hits = glob.glob(f"data/ebas/**/{name}", recursive=True) or \
            [h for h in glob.glob(f"data/**/{name}", recursive=True) if "/cloudnet/" not in h]
        if not hits:
            return None, f"psd file {name} not found under data/"
        psd.append(hits[0])
    if not psd:
        return None, "no psd_files attribute"
    cmd += ["--psd"] + psd + ["--site-name", attrs.get("site", site)]
    sc = str(attrs.get("screens", ""))
    m = re.search(r"pixels >= (\d+)", sc)
    if m:
        cmd += ["--min-pixels", m.group(1)]
    m = re.search(r"base <= ([0-9.]+)", sc)
    if m:
        cmd += ["--max-base", m.group(1)]
    m = re.search(r"N in \[([0-9.eE+-]+), ([0-9.eE+-]+)\]", sc)
    if m:
        cmd += ["--n-range", m.group(1), m.group(2)]
    return cmd, None


def find_jobs():
    jobs = []
    for f in sorted(glob.glob("results/*/*_cloudnet.nc")):
        with xr.open_dataset(f) as d:
            cmd, err = cloudnet_cmd(f, dict(d.attrs))
        jobs.append(dict(kind="cloudnet", file=f, cmd=cmd, err=err, ql=QL_CLOUDNET))
    for f in sorted(glob.glob("results/aircraft/*_aircraft*.nc")):
        if not re.search(r"_aircraft(_[a-z0-9]+)?\.nc$", f) or re.search(r"_(median|orig)\.nc$", f):
            continue
        with xr.open_dataset(f) as d:
            at = dict(d.attrs)
        if "cloud_dir" not in at:
            jobs.append(dict(kind="aircraft", file=f, cmd=None, err="no cloud_dir attribute", ql=QL_OTHER)); continue
        cmd = (["python", "-u", "aircraft_build.py", "--cloud-dir", at["cloud_dir"], "--aerosol-dir", at["aerosol_dir"],
                "--site-name", at.get("site", os.path.basename(f))] + settings_args(at.get("settings", ""), skip=("nd_stat",)))
        jobs.append(dict(kind="aircraft", file=f, cmd=cmd, err=None, ql=QL_OTHER))
    f = "results/ao2018/ao2018_fog.nc"
    if os.path.exists(f):
        with xr.open_dataset(f) as d:
            at = dict(d.attrs)
        cmd = ["python", "-u", "ao2018_build.py"] + settings_args(at.get("settings", ""), skip=("nd_stat",))
        jobs.append(dict(kind="ao2018", file=f, cmd=cmd, err=None, ql=QL_OTHER))
    return jobs


def run(cmd, log):
    with open(log, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


def parse_ql(txt):
    S = D = n = None
    try:
        s = open(txt, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return None, None, None
    m = re.search(r"^S\(100 nm\) = (\S+)\s+95% CI \[(\S+), (\S+)\]", s, re.M)
    if m:
        S = f"{float(m.group(1)):.2f} [{float(m.group(2)):.2f}, {float(m.group(3)):.2f}]" if m.group(1) != "nan" else "nan"
    m = re.search(r"^D_Delta = (\S+) nm\s+95% CI \[(\S+), (\S+)\]", s, re.M)
    D = f"{m.group(1)} [{m.group(2)}, {m.group(3)}]" if m else ("none" if "D_Delta:" in s else None)
    m = re.search(r"^Using cloud_frac >= \S+: (\d+) hours on (\d+) days", s, re.M)
    n = f"{m.group(1)} ({m.group(2)} d)" if m else None
    return S, D, n


def do_job(j, dry):
    f = j["file"]
    stem = f[:-3]
    out_mean, out_med, orig = f, stem + "_median.nc", stem + "_orig.nc"
    if j["cmd"] is None:
        return j, f"skipped: {j['err']}"
    cmds = {s: j["cmd"] + ["--nd-stat", s, "--out", (out_mean if s == "mean" else out_med)] for s in ("mean", "median")}
    if dry:
        return j, "\n      " + "\n      ".join(shlex.join(c) for c in cmds.values())
    if not os.path.exists(orig):
        os.replace(f, orig)
        for ext in ("_quicklook.txt", "_quicklook.png", "_quicklook.csv"):
            if os.path.exists(stem + ext):
                os.replace(stem + ext, orig[:-3] + ext)
    os.makedirs("logs/nd_mean", exist_ok=True)
    tag = os.path.basename(stem)
    msgs = []
    for s, c in cmds.items():
        rc = run(c, f"logs/nd_mean/{tag}_{s}.log")
        target = out_mean if s == "mean" else out_med
        if rc != 0 or not os.path.exists(target):
            msgs.append(f"{s} build FAILED (logs/nd_mean/{tag}_{s}.log)")
            continue
        run(["python", "-u", "arm_ena_nd_pnsd.py", "quicklook", target] + j["ql"].split(), target[:-3] + "_quicklook.txt")
    return j, "; ".join(msgs) or "ok"


def activate_splits():
    for base in ("results/aircraft/activate_aircraft.nc", "results/aircraft/activate_aircraft_median.nc"):
        if not os.path.exists(base):
            continue
        sfx = "_median" if base.endswith("_median.nc") else ""
        d = xr.open_dataset(base).load(); d.close()
        w = d.time.dt.month.isin([11, 12, 1, 2, 3, 4])
        for name, sel in (("winter", w), ("summer", ~w)):
            out = f"results/aircraft/activate_{name}{sfx}.nc"
            if os.path.exists(out):
                os.remove(out)
            d.sel(time=sel).to_netcdf(out)
            run(["python", "-u", "arm_ena_nd_pnsd.py", "quicklook", out] + QL_OTHER.split(), out[:-3] + "_quicklook.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="regex on file path, e.g. 'aircraft|ao2018'")
    a = ap.parse_args()
    if not os.path.exists("arm_ena_nd_pnsd.py"):
        sys.exit("Run from the repository folder.")
    jobs = [j for j in find_jobs() if not a.only or re.search(a.only, j["file"])]
    print(f"{len(jobs)} built files found ({sum(j['cmd'] is not None for j in jobs)} rebuildable)")
    with ThreadPoolExecutor(max_workers=1 if a.dry_run else a.jobs) as ex:
        for j, msg in ex.map(lambda j: do_job(j, a.dry_run), jobs):
            print(f"  {j['kind']:8s} {j['file']}: {msg}", flush=True)
    if a.dry_run:
        return
    activate_splits()
    rows = []
    files = [j["file"] for j in jobs] + [f"results/aircraft/activate_{s}.nc" for s in ("winter", "summer")]
    for f in files:
        stem = f[:-3]
        o = parse_ql(stem + "_orig_quicklook.txt")
        m = parse_ql(stem + "_median_quicklook.txt")
        n = parse_ql(stem + "_quicklook.txt")
        rows.append(dict(file=f, n_mean=n[2], S_mean=n[0], D_mean=n[1], S_median=m[0], D_median=m[1],
                         S_orig=o[0], D_orig=o[1]))
    import pandas as pd
    t = pd.DataFrame(rows)
    t.to_csv("results/nd_stat_comparison.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print("\n" + t.to_string(index=False))
    print("\nWrote results/nd_stat_comparison.csv")


if __name__ == "__main__":
    main()
