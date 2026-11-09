#!/usr/bin/env python
"""Make the arithmetic mean the default Nd statistic in every build script (median kept as an option).

    python patch_nd_mean.py            # patch in place, keeping <file>.bak copies
    python patch_nd_mean.py --check    # only report what would change

Why: models (NorESM, EC-Earth) give in-cloud mean Nd, and at ENA the hourly-mean NDROP matched the
ACE-ENA aircraft (NDROP/aircraft 1.21) while the hourly median did not (0.62).
Files: arm_ena_nd_pnsd.py, epcape_build.py (--nd-hourly default -> mean); cloudnet_build.py,
aircraft_build.py, ao2018_build.py (new --nd-stat {mean,median}, default mean).
Every edit must match exactly once; a file is left untouched if any of its required edits fails.
"""
import argparse
import shutil
import sys

WHY = "mean = in-cloud average as in the models; at ENA the hourly-mean NDROP matched the aircraft, the median was about 40 per cent low"

EDITS = {
    "arm_ena_nd_pnsd.py": [
        ('def hourly_nd(df, min_cloud_frac, stat="median"):',
         'def hourly_nd(df, min_cloud_frac, stat="mean"):', True),
        ('    b.add_argument("--nd-hourly", choices=ND_HOURLY_STATS, default="median",\n'
         '                   help="how the 20-s Nd retrievals are combined into one hourly value "\n'
         '                        "(median matches Cloudnet and the aircraft; mean was used before Nov 2026)")',
         '    b.add_argument("--nd-hourly", choices=ND_HOURLY_STATS, default="mean",\n'
         '                   help="how the 20-s Nd retrievals are combined into one hourly value "\n'
         f'                        "({WHY}; median kept as a sensitivity option)")', True),
    ],
    "epcape_build.py": [
        ('    p.add_argument("--nd-hourly", choices=("median", "geomean", "mean"), default="median",\n'
         '                   help="how in-cloud 5-min Nd samples are combined into one hourly value "\n'
         '                        "(median matches the other builds; mean was used before Nov 2026)")',
         '    p.add_argument("--nd-hourly", choices=("median", "geomean", "mean"), default="mean",\n'
         '                   help="how in-cloud 5-min Nd samples are combined into one hourly value "\n'
         f'                        "({WHY}; median kept as a sensitivity option)")', True),
    ],
    "cloudnet_build.py": [
        ("def read_der(der_dir, min_pixels, max_base, n_range):",
         'def read_der(der_dir, min_pixels, max_base, n_range, stat="mean"):', True),
        ('        "Nd": g["N"].median(),',
         '        "Nd": g["N"].mean() if stat == "mean" else g["N"].median(),', True),
        ("    nd = read_der(a.der_dir, a.min_pixels, a.max_base, a.n_range)",
         "    nd = read_der(a.der_dir, a.min_pixels, a.max_base, a.n_range, a.nd_stat)", True),
        ('    p.add_argument("--min-cloud-frac", type=float, default=0.3)',
         '    p.add_argument("--min-cloud-frac", type=float, default=0.3)\n'
         '    p.add_argument("--nd-stat", choices=["mean", "median"], default="mean",\n'
         f'                   help="how valid profiles are combined into hourly Nd ({WHY})")', True),
        ('"hourly median Cloudnet N_scaled (scaled Frisch)"',
         'f"hourly {a.nd_stat} Cloudnet N_scaled (scaled Frisch)"', False),
    ],
    "aircraft_build.py": [
        ('"Nd": g["Nd"].median(),',
         '"Nd": g["Nd"].mean() if a.nd_stat == "mean" else g["Nd"].median(),', True),
        ('f"median in-cloud droplet number, {a.nd_dmin:g}-{a.nd_dmax:g} um, per cloud pass"',
         'f"{a.nd_stat} in-cloud droplet number, {a.nd_dmin:g}-{a.nd_dmax:g} um, per cloud pass"', False),
        ('    ap.add_argument("--activate-stp", action="store_true",',
         '    ap.add_argument("--nd-stat", choices=["mean", "median"], default="mean",\n'
         f'                    help="how the 1-Hz in-cloud seconds of a pass are combined into its Nd ({WHY})")\n'
         '    ap.add_argument("--activate-stp", action="store_true",', True),
    ],
    "ao2018_build.py": [
        ('hourly = pd.DataFrame({"Nd": g.Nd.median(), "LWC": g.LWC.median(),',
         'hourly = pd.DataFrame({"Nd": g.Nd.mean() if a.nd_stat == "mean" else g.Nd.median(),\n'
         '                           "LWC": g.LWC.mean() if a.nd_stat == "mean" else g.LWC.median(),', True),
        ('f"hourly median FSSP fog droplet number, {a.nd_dmin:g}-{a.nd_dmax:g} um"',
         'f"hourly {a.nd_stat} FSSP fog droplet number, {a.nd_dmin:g}-{a.nd_dmax:g} um"', False),
        ('    p.add_argument("--spike-factor",',
         '    p.add_argument("--nd-stat", choices=["mean", "median"], default="mean",\n'
         f'                   help="how in-fog minutes are combined into hourly Nd ({WHY})")\n'
         '    p.add_argument("--spike-factor",', True),
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bad = 0
    for fname, edits in EDITS.items():
        try:
            s = open(fname, encoding="utf-8").read()
        except FileNotFoundError:
            print(f"!!! {fname}: not found"); bad += 1; continue
        new, report, ok = s, [], True
        for old, rep, required in edits:
            n = new.count(old)
            if rep in new:                       # already patched (insertions contain the old text)
                report.append("already")
            elif n == 1:
                new = new.replace(old, rep); report.append("ok")
            else:
                report.append(f"{'MISSING' if n == 0 else f'{n} matches'}{'' if required else ' (optional)'}")
                if required:
                    ok = False
        status = "unchanged (a required edit failed)" if not ok else ("no change needed" if new == s else "patched")
        if not ok:
            bad += 1
        print(f"{fname:20s} {status:36s} [{', '.join(report)}]")
        if ok and new != s and not a.check:
            shutil.copy2(fname, fname + ".bak")
            open(fname, "w", encoding="utf-8").write(new)
    if bad:
        print("\nSome files were not patched; paste this output so the edit can be adjusted.")
        sys.exit(1)
    print("\nAll done." if not a.check else "\nCheck only; nothing written.")


if __name__ == "__main__":
    main()
