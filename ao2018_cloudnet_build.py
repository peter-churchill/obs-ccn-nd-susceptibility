#!/usr/bin/env python
"""AO2018 (Oden) Cloudnet droplet number paired with the ship DMPS aerosol.

The Bolin Centre AO2018 Cloudnet archive (Leeds, v4) has categorize files but no droplet-number (der) product,
so this script first generates der files with CloudnetPy (scaled Frisch, the same method as the Cloudnet sites),
then pairs hourly Nd with the DMPS (same-hour scans) in the standard layout for `arm_ena_nd_pnsd.py quicklook`.

    pip install cloudnetpy                      # once, in the analysis environment
    python ao2018_cloudnet_build.py --categorize-dir data/ao2018/cloudnet/categorize \\
        --dmps data/ao2018/ao2018-aerosol-dmps.csv --out results/ao2018/ao2018_cloudnet.nc
    python arm_ena_nd_pnsd.py quicklook results/ao2018/ao2018_cloudnet.nc --min-cloud-frac 0.5 \\
        --max-cloud-base 1000 --max-cloud-top 2000

Profile screens and hourly statistic are those of cloudnet_build.read_der (status 1 pixels, no ice/drizzle in the
profile, base <= --max-base, hourly mean by default). The aerosol is measured at ship level, so low cloud bases
(coupled boundary layer) are the relevant cases; screen them in the quicklook.
Data: Cloudnet AO2018 v4, Bolin Centre Database (University of Leeds); DMPS doi:10.17043/oden-ao-2018-aerosol-dmps-1.
"""
import argparse
import glob
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_ena_nd_pnsd import RADII_NM, cumulative_above  # noqa: E402
from cloudnet_build import read_der  # noqa: E402
from ao2018_build import read_dmps  # noqa: E402


RAIN_NAMES = ("rain_detected", "rainfall_rate", "rain_rate")
_rain_note = set()


def _with_rain(cat, der_dir):
    """Current CloudnetPy needs rain_detected/rainfall_rate/rain_rate in categorize; older files (as AO2018 v4)
    store it as is_rain. Return the original file if it already has one, else a temporary copy with
    rain_detected = is_rain (or 0 if no rain information exists; profiles with drizzle/precipitation are
    rejected later by the status-3 screen anyway)."""
    # Also: old files store category_bits/quality_bits as int8, and current CloudnetPy tests quality bits up
    # to bit 9, which overflows int8 under NumPy 2 ("Python integer 128 out of bounds for int8").
    ds = xr.open_dataset(cat, decode_cf=False, mask_and_scale=False, decode_times=False).load()
    ds.close()
    changed = False
    for name in ("category_bits", "quality_bits"):
        if name in ds and ds[name].dtype.itemsize < 4:
            attrs = {k: v for k, v in ds[name].attrs.items() if k not in ("_FillValue", "valid_range", "valid_min", "valid_max")}
            ds[name] = ds[name].astype("int32")
            ds[name].attrs = attrs
            ds[name].encoding = {}
            changed = True
    # Also: current CloudnetPy reads lwp as kg m-2 without a unit check; the 2018 files store g m-2, which
    # inflates N_scaled (proportional to LWP^2) by 1e6.
    for name in ("lwp", "lwp_error"):
        if name in ds and str(ds[name].attrs.get("units", "")).replace(" ", "").lower() in ("gm-2", "g/m2", "gm^-2", "g/m^2"):
            at = dict(ds[name].attrs)
            if "scale_factor" in at:                         # packed integers: rescale the factor, not the values
                at["scale_factor"] = float(at["scale_factor"]) / 1000.0
                if "add_offset" in at:
                    at["add_offset"] = float(at["add_offset"]) / 1000.0
            else:                                            # plain floats: divide, but keep fill values as they are
                v = ds[name].values.astype("float64")
                fill = at.get("_FillValue", at.get("missing_value"))
                bad = ~np.isfinite(v) | ((v == fill) if fill is not None else False)
                ds[name] = (ds[name].dims, np.where(bad, v, v / 1000.0).astype("float32"))
            at["units"] = "kg m-2"
            ds[name].attrs = at
            ds[name].encoding = {}
            _rain_note.add(f"{name} g m-2 -> kg m-2")
            changed = True
    if not any(n in ds for n in RAIN_NAMES):
        src = next((n for n in ("is_rain", "rain", "rainrate") if n in ds), None)
        r = (np.nan_to_num(ds[src].values.astype(float)) > 0).astype("int8") if src else np.zeros(ds.sizes["time"], "int8")
        ds["rain_detected"] = ("time", r, {"long_name": f"rain detected (from {src})" if src else "rain detected (not available; set to 0)"})
        _rain_note.add(src or "none (set to 0)")
        changed = True
    if not changed:
        return cat
    tmp = os.path.join(der_dir, "_tmp_categorize.nc")
    if os.path.exists(tmp):
        os.remove(tmp)
    ds.to_netcdf(tmp)
    return tmp


def make_der(cat_dir, der_dir, force=False):
    from cloudnetpy.products import generate_der
    os.makedirs(der_dir, exist_ok=True)
    cats = sorted(glob.glob(os.path.join(cat_dir, "**", "*.nc"), recursive=True))
    made = skipped = failed = 0
    first_err = None
    for c in cats:
        m = re.search(r"(20\d{6})", os.path.basename(c))
        if not m:
            continue
        out = os.path.join(der_dir, f"{m.group(1)}_ao2018_der.nc")
        if os.path.exists(out) and not force:
            skipped += 1
            continue
        try:
            generate_der(_with_rain(c, der_dir), out)
            made += 1
        except Exception as e:                                    # old categorize versions may lack variables
            failed += 1
            first_err = first_err or f"{os.path.basename(c)}: {type(e).__name__}: {e}"
            if os.path.exists(out):
                os.remove(out)
    tmp = os.path.join(der_dir, "_tmp_categorize.nc")
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"der: {len(cats)} categorize files; generated {made}, existing {skipped}, failed {failed}"
          + (f"; compatibility fixes: {', '.join(sorted(_rain_note))}" if _rain_note else "")
          + (f"\n  first failure: {first_err}" if first_err else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categorize-dir", required=True)
    p.add_argument("--der-dir", default="data/ao2018/cloudnet/der_generated")
    p.add_argument("--force", action="store_true", help="regenerate der files that already exist")
    p.add_argument("--dmps", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-pixels", type=int, default=3)
    p.add_argument("--max-base", type=float, default=2500.0)
    p.add_argument("--n-range", type=float, nargs=2, default=[1.0, 3000.0])
    p.add_argument("--nd-stat", choices=["mean", "median"], default="mean")
    p.add_argument("--min-cloud-frac", type=float, default=0.1)
    p.add_argument("--min-scans", type=int, default=2)
    p.add_argument("--spike-factor", type=float, default=3.0)
    a = p.parse_args()

    make_der(a.categorize_dir, a.der_dir, a.force)
    nd = read_der(a.der_dir, a.min_pixels, a.max_base, a.n_range, a.nd_stat)
    nd = nd[nd.cloud_frac >= a.min_cloud_frac]
    dm, dnm = read_dmps(a.dmps)
    if a.spike_factor > 0:
        base = dm["Ntot"].rolling("6h", center=True, min_periods=5).median()
        spike = (dm["Ntot"] > a.spike_factor * base).fillna(False)
        print(f"DMPS: {len(dm)} scans; spike screen removes {spike.mean():.1%}")
        dm = dm[~spike]
    bins = dm.drop(columns="Ntot")
    cut = 2.0 * RADII_NM
    keep, psd = [], []
    for h in nd.index:
        s = bins[(bins.index >= h) & (bins.index < h + pd.Timedelta(hours=1))]
        if len(s) >= a.min_scans:
            keep.append(h)
            psd.append(np.nanmedian(s.to_numpy(), axis=0))
    if not keep:
        sys.exit("No Cloudnet hour has DMPS scans.")
    nd = nd.loc[keep]
    psd = np.array(psd)
    ccn = cumulative_above(psd, dnm, cut, None)
    print(f"Paired hours: {len(nd)} on {nd.index.normalize().nunique()} days (cloud_frac >= 0.5: "
          f"{(nd.cloud_frac >= 0.5).sum()}); median Nd {nd.Nd.median():.0f}, N(>100 nm) "
          f"{np.nanmedian(ccn[:, np.argmin(abs(cut - 100))]):.0f} cm-3; median cloud base {nd.cloud_base_height.median():.0f} m")
    ds = xr.Dataset(
        {"CDNC": ("time", nd.Nd.to_numpy(), {"units": "cm-3", "long_name": f"hourly {a.nd_stat} Cloudnet N_scaled (scaled Frisch, generated with CloudnetPy)"}),
         "CCN": (("time", "radius"), ccn, {"units": "cm-3", "long_name": "N(D > 2*radius), ship DMPS, same hour"}),
         "cloud_frac": ("time", nd.cloud_frac.to_numpy(), {"units": "1"}),
         "cloud_base_height": ("time", nd.cloud_base_height.to_numpy(), {"units": "m"}),
         "cloud_top_height": ("time", nd.cloud_top_height.to_numpy(), {"units": "m"}),
         "nd_iqr_ratio": ("time", nd.nd_iqr_ratio.to_numpy()),
         "dNdlogDp": (("time", "aerosol_diameter"), psd, {"units": "cm-3"})},
        coords={"time": nd.index.values, "radius": ("radius", RADII_NM, {"units": "nm"}),
                "diameter": ("radius", cut, {"units": "nm"}), "aerosol_diameter": ("aerosol_diameter", dnm, {"units": "nm"})},
        attrs={"site": "AO2018 Oden (Cloudnet + DMPS)", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "settings": " ".join(f"{k}={v}" for k, v in vars(a).items() if k != "out"),
               "note": "Cloudnet Nd (scaled Frisch) is biased high in the Arctic elsewhere (Ny-Alesund x4.7 vs in situ); use for S."})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    ds.to_netcdf(a.out)
    print(f"Wrote {a.out}. Run: python arm_ena_nd_pnsd.py quicklook {a.out} --min-cloud-frac 0.5 --max-cloud-base 1000 --max-cloud-top 2000")


if __name__ == "__main__":
    main()
