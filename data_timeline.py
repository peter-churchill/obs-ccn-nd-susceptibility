#!/usr/bin/env python
"""Data-availability timeline: for every source, when there is Nd (CDNC) and when there is a size distribution (PNSD).

    python data_timeline.py                          # all synthesis sources, parked ones included
    python data_timeline.py --no-parked --gap 60     # skip parked rows; join periods separated by < 60 days

Laid out like the EBAS facility view: one block per source, two rows (CDNC, PNSD), a bar for each period with data.

--record raw (default): each instrument's own record, from the raw input files under --data-root (SPECS below):
    a file whose name carries a date (ARM daily files, Cloudnet der, ICARTT flights) counts that day; other files
    (EBAS multi-year NetCDF, CSV) are opened and the days with any finite value are used. Sources with no SPECS entry,
    or whose files are not found, fall back to the built record and are drawn hatched.
--record built: what went into the analysis, from the built hourly file behind each quicklook (an hour counts for
    CDNC if Nd > 0 and for PNSD if N(>D) exists at the smallest cutoff). Builds keep paired hours only, so the two rows
    are then nearly identical.
Days with data are joined into one bar when the gap between them is shorter than --gap days; aircraft campaigns are
drawn at least --min-width days wide so they stay visible.

Writes figures/synthesis/data_timeline.png and results/data_timeline.csv (one row per bar).
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

from plot_s_at_ddelta import SOURCES, SKIP, REG_COL

REG_ORDER = ["Arctic", "Boreal", "Marine CAO", "Marine", "Continental"]
VAR_COL = {"CDNC": "#08425a", "PNSD": "#6a4fa3"}          # EBAS-like dark teal, and purple


# label -> (CDNC globs, PNSD globs), relative to --data-root. Add or fix entries here when a directory differs.
def _cn(site, psd):
    return ([f"cloudnet/{site}/der/*.nc", f"cloudnet/{site}/der_generated/*.nc"], psd)


SPECS = {
    # ARM NDROP ground sites
    "ENA (SMPS 2022-26)": (["arm/ena/ndrop/*.nc"], ["arm/ena/smps/*.nc"]),
    "ENA (UHSAS)": (["arm/ena/ndrop/*.nc"], ["arm/ena/uhsas/*.nc"]),
    "SGP": (["arm/sgp/ndrop/*.nc"], ["arm/sgp/smps/*.nc"]),
    "LASIC (SMPS)": (["arm/lasic/ndrop/*.nc"], ["arm/lasic/smps/*.nc"]),
    "LASIC (UHSAS)": (["arm/lasic/ndrop/*.nc"], ["arm/lasic/uhsas/*.nc"]),
    "EPCAPE pier": (["arm/epc/ndrop/*.nc"], ["arm/epc/smps/*.nc"]),
    # Cloudnet + surface size distribution
    "SGP (Cloudnet)": _cn("arm-sgp", ["arm/sgp/smps/*.nc"]),
    "ENA (Cloudnet, SMPS)": _cn("arm-graciosa", ["arm/ena/smps/*.nc"]),
    "LASIC (Cloudnet, SMPS)": _cn("arm-ascension", ["arm/lasic/smps/*.nc"]),
    "Bankhead": _cn("arm-bankhead", ["arm/bnf/smps/*.nc"]),
    "Oliktok": _cn("arm-oliktok", ["arm/oli/uhsas/*.nc"]),
    "Hyytiala": _cn("hyytiala", ["ebas/FI0050R*.nc", "ebas/hyytiala/*.nc"]),
    **{lab: _cn(site, [f"ebas/{site}/*.nc"]) for lab, site in [
        ("Cabauw", "cabauw"), ("Granada", "granada"), ("Kenttarova", "kenttarova"), ("Mace Head", "mace-head"),
        ("Ny-Alesund", "ny-alesund"), ("Palaiseau", "palaiseau"), ("Schneefernerhaus", "schneefernerhaus"),
        ("Norunda", "norunda"), ("Bucharest", "bucharest"), ("Melpitz", "melpitz"), ("Maido", "maido"),
        ("Andoya (Cloudnet)", "andoya")]},
    # in situ
    "Pallas": (["virtanen/Pallas_Nd_CCN.csv"], ["virtanen/Pallas_PNSD.csv"]),
    "Puijo": (["virtanen/Puijo_Nd_CCN.csv"], ["virtanen/Puijo_PNSD.csv"]),
    "Zeppelin": (["virtanen/Zeppelin_Nd_CCN.csv"], ["virtanen/Zeppelin_PNSD.csv"]),
    # aircraft (flight days)
    "ACE-ENA": (["aircraft/ace-ena/enaaafmergedcldsdF1.c1/*"], ["aircraft/ace-ena/enaaafmergedaerosolsdF1.c1/*"]),
    "ACAPEX": (["aircraft/acapex/acxaaffcdpF1.c1/*"], ["aircraft/acapex/acxaafuhsasF1.b1/*"]),
    "NAAMES": (["aircraft/naames/NAAMES_Cloud_AircraftInSitu_Data/**/*.ict"], ["aircraft/naames/NAAMES_Aerosol_AircraftInSitu_Data/**/*.ict"]),
    "ORACLES": (["aircraft/oracles/ORACLES_Cloud_AircraftInSitu_Data/**/*.ict"], ["aircraft/oracles/ORACLES_Aerosol_AircraftInSitu_Data/**/*.ict"]),
    "CAMP2Ex": (["aircraft/camp2ex/*Cloud*/**/*.ict"], ["aircraft/camp2ex/*Aerosol*/**/*.ict"]),
}
DATE_RE = __import__("re").compile(r"(?<!\d)((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")


def raw_days(root, globs):
    """Days covered by the files matching globs: from a YYYYMMDD in the file name, else from the file's contents."""
    files = sorted({f for g in globs for f in glob.glob(os.path.join(root, g), recursive=True) if os.path.isfile(f)})
    days = []
    for f in files:
        m = DATE_RE.search(os.path.basename(f))
        if m and not os.path.basename(f).lower().endswith(".csv"):
            days.append(np.datetime64(f"{m[1]}-{m[2]}-{m[3]}"))
            continue
        try:
            if f.lower().endswith(".csv"):
                t = pd.read_csv(f, index_col=0, na_values=["NA", "NaN", ""])
                ok = t.notna().any(axis=1).to_numpy()
                days.extend(pd.to_datetime(t.index[ok], format="mixed").normalize().unique().to_numpy())
            elif f.lower().endswith((".nc", ".nc4", ".cdf")):
                with xr.open_dataset(f) as ds:
                    var = max((v for v in ds.data_vars if "time" in ds[v].dims and ds[v].dtype.kind == "f"),
                              key=lambda v: ds[v].size, default=None)
                    if var is None:
                        continue
                    a = ds[var].values
                    ok = np.isfinite(a).reshape(a.shape[0], -1).any(axis=1) if ds[var].dims[0] == "time" else np.ones(ds.sizes["time"], bool)
                    days.extend(pd.DatetimeIndex(ds["time"].values[ok]).normalize().unique().to_numpy())
        except Exception as e:                       # noqa: BLE001 - an unreadable file should not stop the inventory
            print(f"      (skipped {f}: {type(e).__name__})")
    return np.array(days, dtype="datetime64[D]"), len(files)


def built_file(csv):
    stem = csv[: -len("_quicklook.csv")]
    for c in (stem, re.sub(r"_(relerr|sza\d+)$", "", stem), re.sub(r"_relerr(_sza\d+)?$", "", stem)):
        if os.path.exists(c + ".nc"):
            return c + ".nc"
    return None


def segments(days, gap):
    """Join sorted unique days into [start, end] runs where consecutive days are < gap days apart."""
    if len(days) == 0:
        return []
    d = np.sort(np.unique(days))
    br = np.where(np.diff(d) > np.timedelta64(gap, "D"))[0]
    starts = np.r_[d[0], d[br + 1]]
    ends = np.r_[d[br], d[-1]]
    return list(zip(starts, ends))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gap", type=float, default=31, help="join days closer than this (days)")
    ap.add_argument("--min-width", type=float, default=20, help="minimum drawn bar width (days)")
    ap.add_argument("--no-parked", action="store_true", help="leave out parked (UHSAS / alternative) sources")
    ap.add_argument("--record", choices=["raw", "built"], default="raw", help="instrument records (raw) or analysed hours (built)")
    ap.add_argument("--data-root", default="data", help="root of the raw input files (SPECS paths are relative to it)")
    ap.add_argument("--out", default="figures/synthesis/data_timeline.png")
    a = ap.parse_args()

    rows, seen = [], set()
    for csv in sorted(glob.glob("results/**/*_quicklook.csv", recursive=True)):
        p = csv.replace("\\", "/")
        if any(re.search(s, p) for s in SKIP):
            continue
        src = next((s for s in SOURCES if re.search(s[0], p)), None)
        if src is None or src[1] in seen or (a.no_parked and src[4] == "parked"):
            continue
        nc = built_file(p)
        if nc is None:
            continue
        seen.add(src[1])
        ds = xr.open_dataset(nc)
        t = pd.DatetimeIndex(ds["time"].values).normalize().to_numpy()
        y, ccn = ds["CDNC"].values, ds["CCN"].values
        frac = np.isfinite(ccn).mean(axis=0)
        j = int(np.argmax(frac > 0)) if (frac > 0).any() else 0
        have = {"CDNC": np.isfinite(y) & (y > 0), "PNSD": np.isfinite(ccn[:, j])}
        ds.close()
        msg = []
        for k, var in enumerate(["CDNC", "PNSD"]):
            days, how = t[have[var]].astype("datetime64[D]"), "built"
            if a.record == "raw" and src[1] in SPECS:
                rd, nf = raw_days(a.data_root, SPECS[src[1]][k])
                if len(rd):
                    days, how = rd, "raw"
                else:
                    msg.append(f"{var}: no raw files ({nf} matched) -> built")
            ndays = len(np.unique(days))
            for s0, s1 in segments(days, a.gap):
                rows.append(dict(label=src[1], regime=src[2], method=src[3], quality=src[4], variable=var, record=how,
                                 start=pd.Timestamp(s0), end=pd.Timestamp(s1),
                                 days=int(((days >= s0) & (days <= s1)).sum()) if how == "raw" else
                                 int(len(np.unique(days[(days >= s0) & (days <= s1)])))))
            if ndays == 0:
                rows.append(dict(label=src[1], regime=src[2], method=src[3], quality=src[4], variable=var, record=how,
                                 start=pd.NaT, end=pd.NaT, days=0))
            msg.append(f"{var} {how} {ndays} d")
        print(f"  {src[1]:28s} " + "; ".join(msg))

    r = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    r.to_csv("results/data_timeline.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    first = r.groupby("label").start.min()
    lab = r.drop_duplicates("label").set_index("label")
    order = sorted(lab.index, key=lambda l: (REG_ORDER.index(lab.regime[l]) if lab.regime[l] in REG_ORDER else 99,
                                             first[l] if pd.notna(first[l]) else pd.Timestamp.max))
    n = len(order)
    fig, ax = plt.subplots(figsize=(15, 0.42 * n + 1.8))
    ypos = {}
    for i, l in enumerate(order):
        yb = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(yb - 0.5, yb + 0.5, color="0.95", zorder=0, lw=0)
        for k, var in enumerate(["CDNC", "PNSD"]):
            yc = yb + (0.18 if var == "CDNC" else -0.18)
            for _, s in r[(r.label == l) & (r.variable == var) & r.start.notna()].iterrows():
                w = max((s.end - s.start).days + 1, a.min_width)
                hat = s.record == "built" and a.record == "raw"
                ax.barh(yc, w, left=mdates.date2num(s.start), height=0.3, color=VAR_COL[var],
                        alpha=1.0 if lab.quality[l] not in ("low", "parked") else 0.45, zorder=2,
                        hatch="////" if hat else None, edgecolor="white" if hat else None, lw=0)
        ypos[l] = yb
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([f"{l}" for l in ypos], fontsize=8)
    for tl, l in zip(ax.get_yticklabels(), ypos):
        tl.set_color(REG_COL.get(lab.regime[l], "k"))
        if lab.quality[l] in ("low", "parked"):
            tl.set_alpha(0.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", which="both", ls=":", color="0.6", alpha=0.7)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    for s in ("right", "left"):
        ax.spines[s].set_visible(False)
    h = [Patch(color=VAR_COL["CDNC"], label="CDNC (droplet number)"), Patch(color=VAR_COL["PNSD"], label="PNSD (size distribution)")]
    if a.record == "raw" and (r.record == "built").any():
        h += [Patch(facecolor="0.4", hatch="////", edgecolor="white", label="hatched: raw record not found, analysed hours shown")]
    h += [Patch(color=c, label=f"{k} (label colour)") for k, c in REG_COL.items() if k in set(lab.regime)]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=8, frameon=False)
    what = "instrument records" if a.record == "raw" else "hours in the built (analysed) files"
    ax.set_title(f"CDNC and PNSD availability: {what} (bars join days < {a.gap:g} d apart; faded = low quality or parked)",
                 fontsize=10, pad=24)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {a.out} and results/data_timeline.csv ({n} sources, {len(r)} bars)")


if __name__ == "__main__":
    main()
