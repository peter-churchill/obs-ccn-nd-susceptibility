#!/usr/bin/env python
"""Generate Cloudnet droplet-number (der) files from categorize files with CloudnetPy, for sites and years where
the portal has categorize but no der (e.g. arm-graciosa 2015-2019). Same retrieval (scaled Frisch, CloudnetPy
default spectral width) as the portal der product, so the output works with cloudnet_build.py and nd_compare.py.

    python cloudnet_download.py download --site arm-graciosa --product categorize --start 2018-01-01 --end 2018-12-31
    python cloudnet_make_der.py --cat-dir data/cloudnet/arm-graciosa/categorize \\
        --der-dir data/cloudnet/arm-graciosa/der_generated --jobs 8

Old categorize files are made compatible on the fly (rain_detected from is_rain, int8 bit fields widened,
LWP g m-2 -> kg m-2), as in ao2018_cloudnet_build.py. Existing der files are skipped unless --force.
"""
import argparse
import glob
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def one(job):
    cat, out = job
    from ao2018_cloudnet_build import _with_rain
    from cloudnetpy.products import generate_der
    tmp = tempfile.mkdtemp(prefix="der_")
    try:
        generate_der(_with_rain(cat, tmp), out)
        return "made", None
    except Exception as e:
        if os.path.exists(out):
            os.remove(out)
        return "failed", f"{os.path.basename(cat)}: {type(e).__name__}: {str(e)[:120]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat-dir", required=True)
    p.add_argument("--der-dir", required=True)
    p.add_argument("--tag", default=None, help="name part of the output files (default: site folder name)")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    tag = a.tag or os.path.basename(os.path.dirname(os.path.abspath(a.cat_dir)))
    os.makedirs(a.der_dir, exist_ok=True)
    jobs, have = [], 0
    for c in sorted(glob.glob(os.path.join(a.cat_dir, "**", "*.nc"), recursive=True)):
        m = re.search(r"(20\d{6})", os.path.basename(c))
        if not m:
            continue
        out = os.path.join(a.der_dir, f"{m.group(1)}_{tag}_der.nc")
        if os.path.exists(out) and not a.force:
            have += 1
        else:
            jobs.append((c, out))
    print(f"{len(jobs) + have} categorize files; {have} der already present; generating {len(jobs)} with {a.jobs} workers",
          flush=True)
    made, errs = 0, []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, (st, err) in enumerate(ex.map(one, jobs), 1):
            made += st == "made"
            if err:
                errs.append(err)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}: made {made}, failed {len(errs)}", flush=True)
    print(f"Done: {made} made, {len(errs)} failed" + (f"; first failure: {errs[0]}" if errs else ""))


if __name__ == "__main__":
    main()
