#!/usr/bin/env python3
"""
ARM ENA: hourly matched cloud droplet number (NDROP) and particle number size
distribution (SMPS / UHSAS), built for CCN–CDNC susceptibility analysis.

Scientific question
-------------------
At the Eastern North Atlantic (ENA, Graciosa, Azores) site, how does the
retrieved cloud droplet number concentration (Nd) respond to the surface-measured
CCN proxy N(D > D_cut), and how does that sensitivity depend on the cutoff
diameter?  The output mirrors the NorESM station files: ``CDNC(time)`` and
``CCN(time, radius)`` on the usual cutoff grid, so the binned-median log-log
recipe can be applied unchanged.

Subcommands
-----------
discover   Check which candidate datastreams return files for your account and
           print the years where an Nd stream and a size-distribution stream
           overlap.
download   Fetch every file of one datastream between two dates (resumable).
build      Hourly-average both streams, apply an in-cloud fraction criterion to
           Nd, integrate dN/dlogDp to CCN at each cutoff radius, match by hour
           and write a NetCDF file.
quicklook  Binned-median log-log susceptibility vs cutoff diameter from the
           built file, plus the D_Delta closure diameter, as a table and a PNG.

Credentials
-----------
Set ``ARM_USER`` (your ARM username, not your e-mail) and ``ARM_TOKEN`` (from
https://adc.arm.gov/armlive/register) as environment variables, so the token
does not end up in your shell history.

Examples
--------
    export ARM_USER=petchu ARM_TOKEN=xxxxxxxx
    python arm_ena_nd_pnsd.py discover
    python arm_ena_nd_pnsd.py download --ds enandropmfrsrC1.c1 \
        --start 2018-01-01 --end 2018-12-31 --out data/ndrop
    python arm_ena_nd_pnsd.py download --ds enaaossmpsC1.b1 \
        --start 2018-01-01 --end 2018-12-31 --out data/smps
    python arm_ena_nd_pnsd.py download --ds enametC1.b1 \
        --start 2018-01-01 --end 2018-12-31 --out data/met
    python arm_ena_nd_pnsd.py build --nd-dir data/ndrop --psd-dir data/smps \
        --met-dir data/met --out ena_nd_ccn_2018.nc
    python arm_ena_nd_pnsd.py quicklook ena_nd_ccn_2018.nc --min-cloud-frac 0.5
    python arm_ena_nd_pnsd.py quicklook ena_nd_ccn_2018.nc --min-cloud-frac 0.5 \
        --max-rel-err 0.5 --min-lwp 50 --coupled-only

quicklook prints a cloud-fraction threshold sweep, a table of S(D) with 95 %
day-block bootstrap intervals, D_Delta with its interval, and a regime
comparison (coupled vs decoupled, low vs high cloud base).

Requires numpy, pandas, xarray, netCDF4, scipy and (for quicklook) matplotlib.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Current ARM Live endpoints (https://adc.arm.gov/armlive): /query and /saveData.
# The older /armlive/data/... paths (still used by ACT 2.4) return a generic
# "please verify that you are authenticated" page even with a valid token.
ARM_LIVE = "https://adc.arm.gov/armlive"
HEADERS = {"user-agent": "arm-ena-nd-pnsd/1.0 (python urllib)"}

# Candidate datastreams. NDROP at ENA is documented for Jun 2014 - Oct 2019
# (DOI 10.5439/1131339). The aerosol stream names are NOT verified - `discover`
# tells you which of these exist for your account. Add more with --ds.
CANDIDATES = {
    "nd": ["enandropmfrsrC1.c1", "enandropmfrsrC1.c2"],
    "psd": [
        "enaaossmpsC1.b1", "enaaossmpsE13.b1", "enaaossmpsC1.a1",
        "enaaosuhsasC1.b1", "enaaosuhsasC1.a1", "enaaosuhsasE13.b1",
        "enamergedsmpsapsmlC1.c1",
    ],
}

# Cutoff grid: radii in nm, as in the NorESM station notebooks
# (RADII = np.linspace(5, 150, 59) -> diameters 10-300 nm).
RADII_NM = np.linspace(5, 150, 59)

# NDROP variable (verified name on the ARM metadata page).
ND_VAR = "drop_number_conc"

# Size-distribution variable names to try first (ARM AOS convention).
PSD_VAR_PREFERRED = ["dN_dlogDp", "dN_dlogD", "number_size_distribution"]

# In-cloud criterion: fraction of the hour with a valid Nd retrieval. This is
# the observational analogue of the FREQL > 0.8 mask and of the "continuous
# cloud for the full hour" requirement in Virtanen et al. `build` keeps every
# hour above a low floor and stores cloud_frac, so the threshold itself is
# chosen (and swept) in `quicklook`. NDROP retrieves every 20 s but often only
# for 50-80 % of an overcast hour, so 0.8 is strict.
MIN_CLOUD_FRAC_FLOOR = 0.3
MIN_CLOUD_FRAC = 0.8

# Minimum fraction of expected size-distribution scans in an hour.
MIN_PSD_FRAC = 0.5

FILE_DATE_RE = re.compile(r"\.(\d{8})\.(\d{6})\.")


# ----------------------------------------------------------------------------
# ARM Live web service
# ----------------------------------------------------------------------------

def _credentials(args):
    """Return (user, token) from CLI arguments or environment variables."""
    user = getattr(args, "user", None) or os.environ.get("ARM_USER")
    token = getattr(args, "token", None) or os.environ.get("ARM_TOKEN")
    if not user or not token:
        sys.exit("Set ARM_USER and ARM_TOKEN (or pass --user/--token).")
    # Stray whitespace or quotes from copy-paste are a common cause of rejection.
    user = user.strip().strip("\"'").strip()
    token = token.strip().strip("\"'").strip()
    return user, token


def _credential_hints(user, token):
    """Describe the credentials (never the token itself) to help debugging."""
    hints = [f"username '{user}' ({len(user)} chars)",
             f"token {len(token)} chars"]
    if "@" in user:
        hints.append("the username contains '@' - use your ARM username, not your e-mail")
    if not re.fullmatch(r"[A-Za-z0-9]+", token):
        hints.append("the token contains characters other than letters/digits")
    return "; ".join(hints)


def _iso(date_str, end=False):
    """Format a YYYY-MM-DD date the way ARM Live expects (ISO, ms, Z)."""
    d = pd.Timestamp(date_str)
    if end and d == d.normalize():
        d = d + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


def arm_query(user, token, ds, start, end):
    """List the file names ARM Live holds for a datastream between two dates.

    Parameters
    ----------
    user, token : str
        ARM username and ARM Live access token.
    ds : str
        Datastream name, e.g. ``enandropmfrsrC1.c1``.
    start, end : str
        Dates as ``YYYY-MM-DD`` (end is inclusive).

    Returns
    -------
    list of str
        File names (empty if the datastream has no files or does not exist).
    """
    url = (f"{ARM_LIVE}/query?user={quote(user)}:{quote(token)}&ds={quote(ds)}"
           f"&start={_iso(start)}&end={_iso(end, end=True)}&wt=json")
    try:
        body = urlopen(Request(url, headers=HEADERS), timeout=180).read().decode(
            "utf-8", "replace")
    except HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        if not body.lstrip().startswith("<"):
            raise
    if body.lstrip().startswith("<"):
        paras = [re.sub(r"<[^>]+>|\s+", " ", p).strip()
                 for p in re.findall(r"<p[^>]*>(.*?)</p>", body, re.S | re.I)]
        title = re.search(r"<title[^>]*>(.*?)</title>", body, re.S | re.I)
        said = " ".join(p for p in paras if p) or (title.group(1).strip() if title else "")
        raise PermissionError(
            "ARM Live returned an HTML page instead of data.\n"
            f"  ARM says: {said.replace(token, '***')}\n"
            f"  Credentials used: {_credential_hints(user, token)}\n"
            "  Check: ARM_USER is your ARM username (not e-mail), the token is the "
            "one shown at https://adc.arm.gov/armlive/register, and the account "
            "has been activated.")
    j = json.loads(body)
    # The service has returned both a bare list of names and a
    # {"status": ..., "files": [...]} object; accept either.
    if isinstance(j, list):
        files = j
    elif isinstance(j, dict):
        if j.get("status", "success") != "success":
            return []
        files = j.get("files", [])
    else:
        files = []
    return sorted(str(f) for f in files)


def arm_download(user, token, fname, outdir, retries=3):
    """Download one file via ARM Live's saveData; skip if already present.

    The file is written to ``<name>.part`` first and only renamed once it is
    complete and starts with a NetCDF/HDF5 signature, so an interrupted run
    can simply be restarted.
    """
    dest = os.path.join(outdir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skipped"
    url = f"{ARM_LIVE}/saveData?user={quote(user)}:{quote(token)}&file={quote(fname)}"
    for attempt in range(1, retries + 1):
        try:
            data = urlopen(Request(url, headers=HEADERS), timeout=600).read()
            break
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt == retries:
                print(f"  failed {fname}: {e}")
                return "failed"
            time.sleep(5 * attempt)
    if not (data[:3] == b"CDF" or data[:4] == b"\x89HDF"):
        print(f"  not a NetCDF file (ARM says: {data[:80]!r}) -> {fname}")
        return "failed"
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return "downloaded"


def _file_date(fname):
    m = FILE_DATE_RE.search(fname)
    return pd.Timestamp(m.group(1)) if m else None


# ----------------------------------------------------------------------------
# Subcommand: discover
# ----------------------------------------------------------------------------

def site_candidates(site, facilities):
    """Datastream names to test at another ARM site, e.g. site='sgp'."""
    nd, psd = [], []
    for f in facilities:
        nd += [f"{site}ndropmfrsr{f}.c1", f"{site}ndropmfrsr{f}.c2"]
        psd += [f"{site}aossmps{f}.b1", f"{site}aossmps{f}.a1",
                f"{site}aosuhsas{f}.b1", f"{site}aosuhsas{f}.a1",
                f"{site}mergedsmpsapsml{f}.c1"]
    return {"nd": nd, "psd": psd}


def cmd_discover(args):
    """Report file counts per candidate datastream and the overlap years."""
    user, token = _credentials(args)
    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    years = {"nd": Counter(), "psd": Counter()}
    extra = {"nd": args.nd_ds or [], "psd": args.psd_ds or []}
    cands = site_candidates(args.site, args.facilities) if args.site else CANDIDATES
    for kind in ("nd", "psd"):
        print(f"\n{'Nd' if kind == 'nd' else 'Size distribution'} candidates "
              f"({args.start} to {end})")
        for ds in cands[kind] + extra[kind]:
            try:
                files = arm_query(user, token, ds, args.start, end)
            except PermissionError as e:
                sys.exit(str(e))
            except Exception as e:                      # noqa: BLE001
                print(f"  {ds:28s} query error: {e}")
                continue
            if not files:
                print(f"  {ds:28s} no files")
                continue
            dates = [d for d in map(_file_date, files) if d is not None]
            per_year = Counter(d.year for d in dates)
            for y, n in per_year.items():   # best stream per year
                years[kind][y] = max(years[kind][y], n)
            print(f"  {ds:28s} {len(files):6d} files  "
                  f"{min(dates):%Y-%m-%d} -> {max(dates):%Y-%m-%d}")
    both = sorted(set(years["nd"]) & set(years["psd"]))
    print("\nYears with both an Nd and a size-distribution stream "
          "(daily files, best stream each):")
    if not both:
        print("  none - try other datastream names with --nd-ds / --psd-ds")
    for y in both:
        print(f"  {y}: Nd {years['nd'][y]:4d} files, PSD {years['psd'][y]:4d} files")


# ----------------------------------------------------------------------------
# Subcommand: download
# ----------------------------------------------------------------------------

def cmd_download(args):
    """Download all files of one datastream into ``--out``."""
    user, token = _credentials(args)
    files = arm_query(user, token, args.ds, args.start, args.end)
    if not files:
        sys.exit(f"No files for {args.ds} between {args.start} and {args.end}.")
    os.makedirs(args.out, exist_ok=True)
    print(f"{args.ds}: {len(files)} files -> {args.out}")
    stats = Counter()
    for i, fname in enumerate(files, 1):
        stats[arm_download(user, token, fname, args.out)] += 1
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {dict(stats)}")
    print("Cite the datastream DOI printed on its ARM Data Discovery page.")


# ----------------------------------------------------------------------------
# Reading and hourly averaging
# ----------------------------------------------------------------------------

def _apply_qc(ds, var, mode):
    """Mask a variable with its ARM ``qc_<var>`` companion.

    ARM QC variables are bit-packed integers; 0 means no test failed. With
    ``mode='strict'`` any set bit removes the value, ``'none'`` skips QC.
    """
    da = ds[var]
    qc_name = f"qc_{var}"
    if mode == "strict" and qc_name in ds:
        qc = ds[qc_name]
        da = da.where(qc == 0)
    return da


def _expected_per_hour(times):
    """Native samples per hour, from the median time step of the record."""
    if len(times) < 2:
        return np.nan
    dt = np.median(np.diff(times.values).astype("timedelta64[s]").astype(float))
    return 3600.0 / dt if dt > 0 else np.nan


def _to_cm3(da, name):
    """Return number concentration in cm^-3, converting from m^-3 if needed."""
    units = str(da.attrs.get("units", "")).replace(" ", "")
    if re.search(r"(m\^?-3|/m\^?3|1/m3)$", units) and "cm" not in units:
        return da * 1e-6
    if units and "cm" not in units:
        warnings.warn(f"{name}: unexpected units '{units}', assumed cm^-3")
    return da


def read_nd(nd_dir, var, qc_mode, extra_pattern):
    """Read all NDROP files into one time series of Nd plus auxiliary fields.

    Parameters
    ----------
    nd_dir : str
        Directory holding the NDROP NetCDF files.
    var : str
        Nd variable name (default ``drop_number_conc``).
    qc_mode : {'strict', 'none'}
        How to use the ``qc_<var>`` field.
    extra_pattern : str
        Regex for auxiliary 1-D variables to carry along (LWP, cloud base,
        adiabaticity, optical depth).

    Returns
    -------
    pandas.DataFrame
        Indexed by time, with column ``Nd`` (cm^-3) and the auxiliary fields.
    """
    files = sorted(glob.glob(os.path.join(nd_dir, "*.nc")) +
                   glob.glob(os.path.join(nd_dir, "*.cdf")))
    if not files:
        sys.exit(f"No NetCDF files in {nd_dir}")
    frames, extra_names, extra_attrs = [], None, {}
    for f in files:
        with xr.open_dataset(f) as ds:
            if var not in ds:
                sys.exit(f"{os.path.basename(f)} has no '{var}'. 1-D variables: "
                         f"{[v for v in ds.data_vars if ds[v].dims == ('time',)]}")
            # NDROP stores Nd in m-3; _to_cm3 converts.
            nd = _to_cm3(_apply_qc(ds, var, qc_mode), var)
            nd = nd.where(nd > 0)                       # -9999 / failed retrievals
            cols = {"Nd": nd.values}
            if extra_names is None:
                extra_names = [v for v in ds.data_vars
                               if ds[v].dims == ("time",) and not v.startswith("qc_")
                               and v != var and re.search(extra_pattern, v)
                               and not re.search(r"type|source", v)]   # categorical
                extra_attrs = {v: {k: ds[v].attrs[k] for k in ("units", "long_name")
                                   if k in ds[v].attrs} for v in extra_names}
            for v in extra_names:
                if v in ds:
                    x = ds[v].astype(float)
                    x = x.where(x > -9000)
                    if re.fullmatch(r"m\s*-3|m\^-3|1/m3", str(ds[v].attrs.get("units", ""))):
                        x = x * 1e-6                     # number concentrations -> cm-3
                        extra_attrs[v]["units"] = "cm-3"
                    cols[v] = x.values
            frames.append(pd.DataFrame(cols, index=pd.DatetimeIndex(ds["time"].values)))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.attrs["extra_attrs"] = extra_attrs
    return df


ND_HOURLY_STATS = ("median", "geomean", "mean")


def hourly_nd(df, min_cloud_frac, stat="mean"):
    """Hourly Nd over valid retrievals, kept only for cloudy hours.

    stat : 'median' (default; matches the Cloudnet and aircraft builds), 'geomean'
        (geometric mean, i.e. the mean in log space used by the fit) or 'mean'
        (arithmetic mean, the behaviour of builds before Nov 2026). Auxiliary
        columns (LWP, error, cloud heights, ...) use the median unless stat is
        'mean', in which case they keep the arithmetic mean.

    The cloud fraction is the number of valid retrievals in the hour divided by
    the number expected from the native time step, so it works whether the
    file stores every time step (with fill values) or only successful ones.
    """
    if stat not in ND_HOURLY_STATS:
        raise ValueError(f"stat must be one of {ND_HOURLY_STATS}")
    per_hour = _expected_per_hour(df.index.to_series())
    valid = df["Nd"].notna()
    g = df[valid].resample("1h")
    out = g.mean() if stat == "mean" else g.median()
    if stat == "geomean":
        out["Nd"] = np.exp(np.log(df.loc[valid, "Nd"]).resample("1h").mean())
    out["cloud_frac"] = (g["Nd"].count() / per_hour).clip(upper=1.0)
    out = out[out["cloud_frac"] >= min_cloud_frac]
    out.attrs["samples_per_hour"] = per_hour
    return out


def _find_psd_var(ds, override):
    """Pick the dN/dlogDp variable, its diameter coordinate and bin edges (nm).

    Returns
    -------
    name : str
    dp_nm : ndarray
        Bin centre diameters (nm).
    edges_nm : ndarray or None
        Bin edges (nm, length nbin+1) from a ``<bin_dim>_bounds`` variable when
        the file provides contiguous bounds (ARM AOS files do), else None.
    """
    if override:
        name = override
    else:
        name = next((v for v in PSD_VAR_PREFERRED if v in ds), None)
        if name is None:   # any 2-D (time, bin) variable with 'dlog' in name/long_name
            for v in ds.data_vars:
                da = ds[v]
                txt = f"{v} {da.attrs.get('long_name', '')}".lower()
                if da.ndim == 2 and da.dims[0] == "time" and "dlog" in txt \
                        and not v.startswith("qc_"):
                    name = v
                    break
    if name is None or name not in ds:
        two_d = [v for v in ds.data_vars if ds[v].ndim == 2]
        sys.exit(f"Could not identify dN/dlogDp. 2-D variables: {two_d}. "
                 f"Pass --psd-var NAME.")
    da = ds[name]
    bin_dim = da.dims[1]
    # Diameter: a 1-D variable on the bin dimension with length units.
    cands = [v for v in list(ds.data_vars) + list(ds.coords)
             if ds[v].dims == (bin_dim,)]
    cands.sort(key=lambda v: ("diameter" not in v.lower(), "bound" in v.lower()))
    def scale(v):
        units = str(ds[v].attrs.get("units", "")).lower()
        if units in ("nm", "nanometers", "nanometer"):
            return 1.0
        if units in ("um", "µm", "μm", "micrometers", "micrometer", "microns", "micron"):
            return 1e3
        return None

    for v in cands:
        s = scale(v)
        if s is not None:
            dp_nm = ds[v].values.astype(float) * s
            edges = None
            bname = f"{bin_dim}_bounds"
            if bname in ds and ds[bname].ndim == 2:
                # UHSAS bounds carry no units attribute: assume the diameter's.
                bs = scale(bname) if scale(bname) is not None else s
                b = ds[bname].values.astype(float) * bs
                b = b[np.argsort(b[:, 0])]
                if np.allclose(b[1:, 0], b[:-1, 1], rtol=1e-3):   # contiguous bins
                    edges = np.concatenate([b[:, 0], b[-1:, 1]])
            return name, dp_nm, edges
    sys.exit(f"Could not find a diameter (nm or um) on dimension '{bin_dim}'. "
             f"Candidates: {[(v, ds[v].attrs.get('units')) for v in cands]}")


def _bin_edges_log(dp_nm):
    """Log10 bin edges from bin centres: geometric midpoints, ends mirrored."""
    lg = np.log10(dp_nm)
    mid = 0.5 * (lg[1:] + lg[:-1])
    return np.concatenate([[lg[0] - (mid[0] - lg[0])], mid,
                           [lg[-1] + (lg[-1] - mid[-1])]])


def cumulative_above(dndlogdp, dp_nm, cut_nm, edges_nm=None):
    """Number concentration of particles larger than each cutoff diameter.

    N(D > D_c) = sum_i dN/dlogDp_i * dlogDp_i * f_i, where f_i is the fraction
    of bin i (in log D) lying above D_c - so a cutoff inside a bin takes the
    matching share of that bin, assuming dN/dlogDp is flat within it.

    Parameters
    ----------
    dndlogdp : ndarray, shape (time, bin)
        Size distribution in cm^-3 per unit log10 Dp.
    dp_nm : ndarray, shape (bin,)
        Bin centre diameters (nm), increasing.
    cut_nm : ndarray, shape (ncut,)
        Cutoff diameters (nm).
    edges_nm : ndarray, shape (bin+1,), optional
        Instrument bin edges (nm). If None, edges are the geometric midpoints
        between bin centres.

    Returns
    -------
    ndarray, shape (time, ncut)
        N(D > D_c) in cm^-3. NaN where the cutoff lies outside the measured
        range (below it the instrument misses particles between the cutoff and
        its lower edge) or where any contributing bin is missing.
    """
    order = np.argsort(dp_nm)
    dp_nm, dndlogdp = dp_nm[order], dndlogdp[:, order]
    edges = np.log10(np.sort(edges_nm)) if edges_nm is not None else _bin_edges_log(dp_nm)
    dlog = np.diff(edges)                                    # (bin,)
    lc = np.log10(cut_nm)[None, :]                           # (1, ncut)
    frac = np.clip((edges[1:, None] - lc) / dlog[:, None], 0.0, 1.0)  # (bin, ncut)
    n_bin = dndlogdp * dlog[None, :]                         # (time, bin)
    # Missing bins that do not contribute (frac == 0) must not poison the sum.
    contrib = frac > 0
    out = np.where(np.isnan(n_bin)[:, :, None] & contrib[None], np.nan,
                   np.nan_to_num(n_bin)[:, :, None] * frac[None]).sum(axis=1)
    out[:, (cut_nm < 10 ** edges[0]) | (cut_nm > 10 ** edges[-1])] = np.nan
    return out


def read_psd_hourly(psd_dir, var_override, qc_mode, cut_nm, min_psd_frac):
    """Hourly size distributions, integrated to N(D > D_c) file by file.

    Integrating per file before concatenating keeps the result independent of
    instrument bin layouts, which can change between deployments.

    Returns
    -------
    ccn : pandas.DataFrame
        Hourly N(D > D_c) (cm^-3), one column per cutoff.
    aux : pandas.DataFrame
        Hourly N_tot, N_aitken (20-100 nm) and N70_500 (70-500 nm band).
    dp_range : tuple
        Overall measured diameter range (nm).
    """
    files = sorted(glob.glob(os.path.join(psd_dir, "*.nc")) +
                   glob.glob(os.path.join(psd_dir, "*.cdf")))
    if not files:
        sys.exit(f"No NetCDF files in {psd_dir}")
    band_cuts = np.array([20.0, 70.0, 100.0, 500.0])
    ccn_parts, aux_parts, ranges = [], [], []
    per_hour = None
    for f in files:
        with xr.open_dataset(f) as ds:
            name, dp_nm, edges_nm = _find_psd_var(ds, var_override)
            da = _to_cm3(_apply_qc(ds, name, qc_mode), name).astype(float)
            da = da.where(da >= 0)
            df = pd.DataFrame(da.values, index=pd.DatetimeIndex(ds["time"].values))
        order = np.argsort(dp_nm)
        dp_nm, df = dp_nm[order], df.iloc[:, order]
        df.columns = range(df.shape[1])
        if edges_nm is not None:
            edges_nm = np.sort(edges_nm)
        # ARM AOS files share one wide bin grid (ENA SMPS: 192 bins, 1-982 nm)
        # but the instrument only scans part of it (~10-500 nm); bins outside
        # are always missing. Keep the contiguous span of bins that hold data
        # in most scans, so those empty bins do not blank every cutoff.
        scan_ok = df.notna().any(axis=1)
        if not scan_ok.any():
            continue
        good = df[scan_ok].notna().mean(axis=0).to_numpy() >= 0.5
        if not good.any():
            continue
        i0, i1 = np.argmax(good), len(good) - np.argmax(good[::-1])
        dp_nm, df = dp_nm[i0:i1], df.iloc[:, i0:i1]
        if edges_nm is not None:
            edges_nm = edges_nm[i0:i1 + 1]
        if per_hour is None:
            per_hour = _expected_per_hour(df.index.to_series())
        hourly = df[scan_ok].resample("1h").mean()
        n_scans = df[scan_ok].resample("1h").size()
        hourly = hourly[(n_scans / per_hour) >= min_psd_frac]
        if hourly.empty:
            continue
        arr = hourly.to_numpy()
        ccn_parts.append(pd.DataFrame(cumulative_above(arr, dp_nm, cut_nm, edges_nm),
                                      index=hourly.index))
        # Cap the 500 nm band edge at the top of the measured range, so an
        # instrument that stops below 500 nm (LASIC SMPS: 470 nm) still gives N70.
        top = edges_nm[-1] if edges_nm is not None else dp_nm[-1]
        cuts = band_cuts.copy()
        cuts[3] = min(cuts[3], top * 0.999)
        b = cumulative_above(arr, dp_nm, cuts, edges_nm)
        lg = np.log10(edges_nm) if edges_nm is not None else _bin_edges_log(dp_nm)
        n_tot = (arr * np.diff(lg)).sum(axis=1)
        aux_parts.append(pd.DataFrame({"N_tot": n_tot,
                                       "N_aitken_20_100": b[:, 0] - b[:, 2],
                                       "N70_500": b[:, 1] - b[:, 3]},
                                      index=hourly.index))
        lo = edges_nm[0] if edges_nm is not None else dp_nm[0]
        hi = edges_nm[-1] if edges_nm is not None else dp_nm[-1]
        ranges.append((lo, hi))
    if not ccn_parts:
        sys.exit("No hour passed the size-distribution completeness check.")
    ccn = pd.concat(ccn_parts).sort_index()
    aux = pd.concat(aux_parts).sort_index()
    keep = ~ccn.index.duplicated(keep="first")
    dp_range = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    return ccn[keep], aux[~aux.index.duplicated(keep="first")], dp_range


MET_T = ["temp_mean", "temp", "air_temperature", "temperature"]
MET_RH = ["rh_mean", "rh", "relative_humidity", "rel_humidity"]


def _pick(ds, names, kind):
    for n in names:
        if n in ds and ds[n].dims == ("time",):
            return n
    sys.exit(f"Met file has no {kind} variable (tried {names}). 1-D variables: "
             f"{[v for v in ds.data_vars if ds[v].dims == ('time',)][:40]}")


def read_met_lcl(met_dir, qc_mode):
    """Hourly lifting condensation level (m above the surface) from surface met.

    Uses the Lawrence (2005) approximation z_LCL ~ 125 m per K of dew-point
    depression, with the dew point from the Magnus formula. The decoupling
    measure is then dz = cloud base - z_LCL: near zero for a well-mixed
    (coupled) boundary layer whose surface air feeds the cloud, large and
    positive when the cloud sits above a decoupled layer.

    Returns
    -------
    pandas.Series
        Hourly mean z_LCL (m), indexed by hour.
    """
    files = sorted(glob.glob(os.path.join(met_dir, "*.nc")) +
                   glob.glob(os.path.join(met_dir, "*.cdf")))
    if not files:
        sys.exit(f"No NetCDF files in {met_dir}")
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            tn, rn = _pick(ds, MET_T, "temperature"), _pick(ds, MET_RH, "relative humidity")
            t = _apply_qc(ds, tn, qc_mode).astype(float).values
            rh = _apply_qc(ds, rn, qc_mode).astype(float).values
            if str(ds[tn].attrs.get("units", "")).strip() in ("K", "kelvin"):
                t = t - 273.15
            if np.nanmax(rh) <= 1.5:                    # fraction -> percent
                rh = rh * 100.0
            parts.append(pd.DataFrame({"t": t, "rh": rh},
                                      index=pd.DatetimeIndex(ds["time"].values)))
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df.rh > 1) & (df.rh <= 105) & (df.t > -60) & (df.t < 50)]
    # Magnus dew point (deg C) and Lawrence LCL height (m).
    a, b = 17.625, 243.04
    gamma = np.log(df.rh.clip(upper=100) / 100.0) + a * df.t / (b + df.t)
    td = b * gamma / (a - gamma)
    lcl = 125.0 * (df.t - td)
    return lcl.resample("1h").mean().rename("lcl_height")

# ----------------------------------------------------------------------------
# Subcommand: build
# ----------------------------------------------------------------------------

def cmd_build(args):
    """Match hourly Nd with hourly CCN(D_c) and write the NetCDF file."""
    radii = RADII_NM
    cut_nm = 2.0 * radii
    nd_raw = read_nd(args.nd_dir, args.nd_var, args.qc, args.nd_extra)
    extra_attrs = nd_raw.attrs.get("extra_attrs", {})
    nd = hourly_nd(nd_raw, args.min_cloud_frac, args.nd_hourly)
    counts = ", ".join(f">={t}: {(nd['cloud_frac'] >= t).sum()}"
                       for t in (0.3, 0.5, 0.8) if t >= args.min_cloud_frac)
    print(f"Nd: {len(nd_raw)} samples (~{nd.attrs['samples_per_hour']:.0f}/h) -> "
          f"hours by cloud_frac {counts}")
    ccn, aux, dp_range = read_psd_hourly(args.psd_dir, args.psd_var, args.qc,
                                         cut_nm, args.min_psd_frac)
    print(f"PSD: {len(ccn)} valid hours, measured {dp_range[0]:.1f}-{dp_range[1]:.1f} nm")

    hours = nd.index.intersection(ccn.index)
    if len(hours) == 0:
        sys.exit("No overlapping hours between Nd and the size distribution.")
    nd, ccn, aux = nd.loc[hours], ccn.loc[hours], aux.loc[hours]
    out_of_range = np.isnan(ccn.to_numpy()).all(axis=0)
    if out_of_range.any():
        print(f"Note: {out_of_range.sum()} cutoffs fall outside the measured range "
              f"and are NaN ({cut_nm[out_of_range].min():.0f}-"
              f"{cut_nm[out_of_range].max():.0f} nm).")

    ds = xr.Dataset(
        {
            "CDNC": ("time", nd["Nd"].to_numpy(),
                     {"units": "cm-3", "long_name": f"hourly {args.nd_hourly} of retrieved Nd (NDROP)",
                      "hourly_statistic": args.nd_hourly}),
            "CCN": (("time", "radius"), ccn.to_numpy(),
                    {"units": "cm-3",
                     "long_name": "N(D > 2*radius) from hourly mean dN/dlogDp"}),
            "cloud_frac": ("time", nd["cloud_frac"].to_numpy(),
                           {"units": "1",
                            "long_name": "fraction of the hour with a valid Nd retrieval"}),
        },
        coords={"time": hours.values,
                "radius": ("radius", radii, {"units": "nm", "long_name": "cutoff radius"}),
                "diameter": ("radius", cut_nm, {"units": "nm", "long_name": "cutoff diameter"})},
        attrs={"site": args.site_name,
               "nd_dir": os.path.abspath(args.nd_dir),
               "psd_dir": os.path.abspath(args.psd_dir),
               "qc": args.qc, "min_cloud_frac": args.min_cloud_frac,
               "min_psd_frac": args.min_psd_frac,
               "psd_diameter_range_nm": f"{dp_range[0]:.1f}-{dp_range[1]:.1f}",
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "note": ("NDROP is a daytime (MFRSR), overcast-only retrieval; aerosol "
                        "is measured at the surface, so check boundary-layer coupling.")},
    )
    for c in aux.columns:
        ds[c] = ("time", aux[c].to_numpy(), {"units": "cm-3"})
    for c in nd.columns:
        if c not in ("Nd", "cloud_frac"):
            a = dict(extra_attrs.get(c, {}))
            a["long_name"] = "hourly mean over valid retrievals: " + a.get("long_name", c)
            ds[c] = ("time", nd[c].to_numpy(), a)
    if args.met_dir:
        lcl = read_met_lcl(args.met_dir, args.qc).reindex(hours)
        ds["lcl_height"] = ("time", lcl.to_numpy(),
                            {"units": "m", "long_name": "hourly mean surface LCL height "
                             "(Lawrence 2005 approximation, Magnus dew point)"})
        if "cloud_base_height" in ds:
            # NDROP cloud base is from ARSCL; ENA is 30 m a.s.l., small next to dz.
            ds["decoupling_dz"] = ("time", ds["cloud_base_height"].values - lcl.to_numpy(),
                                   {"units": "m", "long_name": "cloud base minus surface LCL "
                                    "(small = coupled boundary layer)"})
            dz = ds["decoupling_dz"].values
            print(f"Met: LCL for {np.isfinite(lcl.to_numpy()).sum()} of {len(hours)} hours; "
                  f"cloud base - LCL median {np.nanmedian(dz):.0f} m "
                  f"(IQR {np.nanpercentile(dz, 25):.0f} to {np.nanpercentile(dz, 75):.0f} m)")
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(args.out, encoding=enc)
    cf = ds["cloud_frac"].values
    print(f"Wrote {args.out}: {ds.sizes['time']} matched hours "
          f"(cloud_frac >= 0.5: {(cf >= 0.5).sum()}, >= 0.8: {(cf >= 0.8).sum()}). "
          f"Choose the threshold in quicklook with --min-cloud-frac.")


# ----------------------------------------------------------------------------
# Subcommand: quicklook
# ----------------------------------------------------------------------------

def binned_loglog(x, y, n_bins):
    """Canonical recipe: log10 -> equal-number bins -> bin medians -> linregress.

    Returns
    -------
    tuple
        (slope, intercept, r, n) of the regression through the bin medians.
    """
    from scipy.stats import linregress
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    if lx.size < 2 * n_bins:
        return (np.nan, np.nan, np.nan, int(lx.size))
    order = np.argsort(lx)
    xb = [np.median(c) for c in np.array_split(lx[order], n_bins)]
    yb = [np.median(c) for c in np.array_split(ly[order], n_bins)]
    if np.ptp(xb) == 0:                     # all x in one value (e.g. a cutoff where every hour is 0)
        return (np.nan, np.nan, np.nan, int(lx.size))
    res = linregress(xb, yb)
    return (res.slope, res.intercept, res.rvalue, int(lx.size))


def _n_bins_for(n, override=None):
    """Equal-number bin count: ~15 samples per bin, between 5 and 20."""
    return override or max(5, min(20, int(n) // 15))


def closure_offsets(ccn, y):
    """delta(D) = mean log10 Nd - mean log10 CCN(D) over pairs valid at every cutoff.

    Uses a common-across-cutoffs mask (as in the NorESM cutoff notebooks), so
    the zero crossing is not shifted by different samples at different D.
    """
    finite_cols = ~np.isnan(ccn).all(axis=0)
    ok = np.isfinite(y) & (y > 0) & np.all((ccn[:, finite_cols] > 0), axis=1)
    out = np.full(ccn.shape[1], np.nan)
    if ok.sum() == 0:
        return out, ok
    out[finite_cols] = np.log10(y[ok]).mean() - np.log10(ccn[ok][:, finite_cols]).mean(axis=0)
    return out, ok


def d_delta_from(delta, diam):
    """Interpolated (in log D) zero crossing of delta(D); NaN if none."""
    good = np.isfinite(delta)
    d, v = diam[good], delta[good]
    idx = np.where(np.diff(np.sign(v)) != 0)[0]
    if not idx.size:
        return np.nan
    i = idx[0]
    x0, x1 = np.log10(d[i]), np.log10(d[i + 1])
    return 10 ** (x0 - v[i] * (x1 - x0) / (v[i + 1] - v[i]))


def susceptibility_table(ccn, y, diam, n_bins=None):
    """S, r and the closure offset at every cutoff.

    Returns
    -------
    tab : pandas.DataFrame
        Columns D_nm, S, r, delta_dex (mean log Nd - mean log CCN), n.
    d_delta : float
        D_Delta, the interpolated zero crossing of delta_dex (nm), or NaN.
    """
    delta, ok = closure_offsets(ccn, y)
    rows = []
    for i, d in enumerate(diam):
        s, _, r, n = binned_loglog(ccn[:, i], y, _n_bins_for(ok.sum(), n_bins))
        rows.append((d, s, r, delta[i], n))
    tab = pd.DataFrame(rows, columns=["D_nm", "S", "r", "delta_dex", "n"])
    return tab, d_delta_from(delta, diam)


def _partial_lwp(ds, m, times, x100, y, n_boot, seed=1):
    """Unbinned slope of log Nd on log CCN(100 nm), with and without log LWP as a
    second predictor. Retrievals that scale Nd with LWP (Cloudnet N_scaled, and
    partly NDROP) can share LWP variability with Nd; the partial slope removes it."""
    if "lwp_meas" not in ds:
        return
    lwp = ds["lwp_meas"].values[m] * 1000.0                      # g m-2
    ok = (y > 0) & (x100 > 0) & (lwp > 5) & np.isfinite(y) & np.isfinite(x100)
    if ok.sum() < 30:
        return
    lx, ly, ll = np.log10(x100[ok]), np.log10(y[ok]), np.log10(lwp[ok])
    day = pd.DatetimeIndex(times[ok]).normalize().to_numpy()

    def fit(idx):
        X = np.c_[np.ones(idx.size), lx[idx], ll[idx]]
        c = np.linalg.lstsq(X, ly[idx], rcond=None)[0]
        return np.polyfit(lx[idx], ly[idx], 1)[0], c[1], c[2]

    simple, partial, beta = fit(np.arange(lx.size))
    rng = np.random.default_rng(seed)
    days = np.unique(day)
    groups = [np.where(day == d)[0] for d in days]
    boots = np.array([fit(np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]))
                      for _ in range(max(n_boot, 100))])
    lo, hi = np.nanpercentile(boots[:, 1], [2.5, 97.5])
    print(f"Unbinned slope at 100 nm: {simple:.3f}; controlling for log LWP: {partial:.3f} "
          f"[{lo:.3f}, {hi:.3f}] (LWP exponent {beta:.2f}, r(logNd, logLWP) = "
          f"{np.corrcoef(ly, ll)[0, 1]:.2f}, n = {ok.sum()})")


def bootstrap_by_day(times, ccn, y, diam, n_boot, n_bins=None, cut_idx=None, seed=0):
    """Day-block bootstrap of S(D) and D_Delta.

    Whole days are resampled with replacement, so hours within a day (which
    share air mass and cloud regime) stay together - the observational
    counterpart of the 72-hour moving blocks used for the model.

    Parameters
    ----------
    times : ndarray of datetime64
    ccn : ndarray (time, ncut)
    y : ndarray (time,)
    diam : ndarray (ncut,)
    n_boot : int
    cut_idx : list of int, optional
        Only compute S at these cutoff indices (faster); default all.

    Returns
    -------
    s_boot : ndarray (n_boot, ncut)
        NaN at cutoffs not computed.
    dd_boot : ndarray (n_boot,)
    """
    rng = np.random.default_rng(seed)
    day = pd.DatetimeIndex(times).normalize().to_numpy()
    days = np.unique(day)
    groups = [np.where(day == d)[0] for d in days]
    cut_idx = range(len(diam)) if cut_idx is None else cut_idx
    _, ok_all = closure_offsets(ccn, y)
    nb = _n_bins_for(ok_all.sum(), n_bins)          # keep the bin count fixed
    s_boot = np.full((n_boot, len(diam)), np.nan)
    dd_boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        cb, yb = ccn[idx], y[idx]
        for i in cut_idx:
            s_boot[b, i] = binned_loglog(cb[:, i], yb, nb)[0]
        dd_boot[b] = d_delta_from(closure_offsets(cb, yb)[0], diam)
    return s_boot, dd_boot


def _ci(a):
    a = a[np.isfinite(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if a.size >= 20 else (np.nan, np.nan)


def _base_mask(ds, args):
    """Hours passing the cloud-fraction, noise and LWP screens."""
    cf = ds["cloud_frac"].values
    m = np.ones(cf.size, bool)
    notes = []
    if args.max_rel_err is not None:
        if "drop_number_conc_toterror" in ds:
            rel = ds["drop_number_conc_toterror"].values / ds["CDNC"].values
            m &= np.isfinite(rel) & (rel <= args.max_rel_err)
            notes.append(f"Nd error/Nd <= {args.max_rel_err}")
        else:
            print("Note: no drop_number_conc_toterror in file; --max-rel-err ignored.")
    if args.min_lwp is not None:
        if "lwp_meas" in ds:
            m &= ds["lwp_meas"].values * 1000.0 >= args.min_lwp   # kg m-2 -> g m-2
            notes.append(f"LWP >= {args.min_lwp} g m-2")
        else:
            print("Note: no lwp_meas in file; --min-lwp ignored.")
    for opt, var, label in ((args.max_cloud_base, "cloud_base_height", "cloud base"),
                            (args.max_cloud_top, "cloud_top_height", "cloud top")):
        if opt is not None:
            if var in ds:
                m &= ds[var].values <= opt
                notes.append(f"{label} <= {opt:.0f} m")
            else:
                print(f"Note: no {var} in file; the {label} screen is ignored.")
    if args.coupled_only:
        if "decoupling_dz" in ds:
            m &= ds["decoupling_dz"].values <= args.dz_coupled
            notes.append(f"coupled (cloud base - LCL <= {args.dz_coupled:.0f} m)")
        else:
            print("Note: no decoupling_dz in file (build with --met-dir); --coupled-only ignored.")
    return m, cf, notes


def cmd_quicklook(args):
    """Susceptibility and closure diagnostics vs cutoff diameter, with CIs."""
    ds = xr.open_dataset(args.file)
    diam = ds["diameter"].values
    ccn_all, y_all = ds["CCN"].values, ds["CDNC"].values
    if args.nd_scale != 1.0:
        # Divide Nd by a calibration factor (e.g. Cloudnet / NDROP). S is unchanged;
        # the closure diameter moves to where N(>D) equals the corrected Nd.
        y_all = y_all / args.nd_scale
        print(f"Nd divided by {args.nd_scale:g} before analysis (calibration factor).")
    times = ds["time"].values
    if np.isnan(ccn_all).all():
        sys.exit("Every CCN value is NaN - the size distribution did not cover the "
                 "cutoffs. Rebuild with the current script version.")
    base, cf, notes = _base_mask(ds, args)
    i100 = int(np.argmin(np.abs(diam - 100)))
    if notes:
        print("Screens applied to every table: " + "; ".join(notes))

    # 1. Sensitivity to the in-cloud threshold (cf. the FREQL sweep in the model).
    print("\nCloud-fraction threshold sensitivity:")
    print("  thresh  hours  days   S(100nm)   r(100nm)  D_Delta")
    for thr in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = base & (cf >= thr)
        nd = len(np.unique(pd.DatetimeIndex(times[m]).normalize()))
        if m.sum() < 10:
            print(f"  {thr:5.1f}  {m.sum():5d}  {nd:4d}   too few hours")
            continue
        t, dd = susceptibility_table(ccn_all[m], y_all[m], diam, args.n_bins)
        dds = f"{dd:5.0f} nm" if np.isfinite(dd) else "  none"
        print(f"  {thr:5.1f}  {m.sum():5d}  {nd:4d}   {t.S[i100]:8.3f}   {t.r[i100]:8.3f}  {dds}")

    m = base & (cf >= args.min_cloud_frac)
    n_days = len(np.unique(pd.DatetimeIndex(times[m]).normalize()))
    if m.sum() < 10:
        sys.exit(f"Only {m.sum()} hours pass the screens; relax them.")
    ccn, y, t_m = ccn_all[m], y_all[m], times[m]
    tab, d_delta = susceptibility_table(ccn, y, diam, args.n_bins)

    # 2. Day-block bootstrap for the main selection.
    print(f"\nUsing cloud_frac >= {args.min_cloud_frac}: {m.sum()} hours on {n_days} days. "
          f"Bootstrapping {args.n_boot} day-block resamples ...")
    s_boot, dd_boot = bootstrap_by_day(t_m, ccn, y, diam, args.n_boot, args.n_bins)
    ci = np.array([_ci(s_boot[:, i]) for i in range(len(diam))])
    tab["S_lo"], tab["S_hi"] = ci[:, 0], ci[:, 1]
    dd_lo, dd_hi = _ci(dd_boot)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(tab[["D_nm", "S", "S_lo", "S_hi", "r", "delta_dex", "n"]].iloc[::4]
              .to_string(index=False))
    print(f"S(100 nm) = {tab.S[i100]:.3f}  95% CI [{tab.S_lo[i100]:.3f}, {tab.S_hi[i100]:.3f}]")
    if np.isfinite(d_delta):
        print(f"D_Delta = {d_delta:.0f} nm  95% CI [{dd_lo:.0f}, {dd_hi:.0f}] nm "
              f"({np.isfinite(dd_boot).mean():.0%} of resamples have a crossing)")
    else:
        print("D_Delta: no zero crossing in range")
    _partial_lwp(ds, m, t_m, ccn[:, i100], y, args.n_boot)

    # 3. Regime comparison: coupling and cloud-base height.
    groups = []
    if "decoupling_dz" in ds:
        dz = ds["decoupling_dz"].values[m]
        groups += [(f"coupled (dz <= {args.dz_coupled:.0f} m)", dz <= args.dz_coupled),
                   (f"decoupled (dz > {args.dz_coupled:.0f} m)", dz > args.dz_coupled)]
    if "cloud_base_height" in ds:
        cb = ds["cloud_base_height"].values[m]
        split = args.cb_split if args.cb_split is not None else np.nanmedian(cb)
        groups += [(f"cloud base <= {split:.0f} m", cb <= split),
                   (f"cloud base > {split:.0f} m", cb > split)]
    if groups:
        print("\nRegime comparison (same screens; 95% day-block CIs):")
        print(f"  {'group':30s} hours  S(100nm) [CI]            D_Delta [CI]")
        for name, g in groups:
            if g.sum() < 30:
                print(f"  {name:30s} {g.sum():5d}  too few hours")
                continue
            tg, ddg = susceptibility_table(ccn[g], y[g], diam, args.n_bins)
            sb, db = bootstrap_by_day(t_m[g], ccn[g], y[g], diam, args.n_boot,
                                      args.n_bins, cut_idx=[i100])
            slo, shi = _ci(sb[:, i100])
            dlo, dhi = _ci(db)
            dds = f"{ddg:4.0f} [{dlo:.0f}, {dhi:.0f}] nm" if np.isfinite(ddg) else "none"
            print(f"  {name:30s} {g.sum():5d}  {tg.S[i100]:6.3f} [{slo:6.3f}, {shi:6.3f}]   {dds}")
        if "decoupling_dz" not in ds:
            print("  (Add the surface met data with build --met-dir for the coupling split.)")

    if tab["S"].notna().sum() < 2:
        print("Too few valid cutoffs to plot.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    ax[0].fill_between(tab.D_nm, tab.S_lo, tab.S_hi, color="0.85", lw=0,
                       label="95% day-block CI")
    ax[0].plot(tab.D_nm, tab.S, "k-")
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_ylabel("Susceptibility S")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(tab.D_nm, tab.r, "k-")
    ax[1].set_ylabel("Pearson r (bin medians)")
    ax[2].plot(tab.D_nm, tab.delta_dex, "k-")
    ax[2].axhline(0, color="0.6", lw=0.8)
    if np.isfinite(d_delta):
        for a in ax:
            a.axvline(d_delta, color="tab:blue", lw=0.8, ls="--")
        if np.isfinite(dd_lo):
            ax[2].axvspan(dd_lo, dd_hi, color="tab:blue", alpha=0.12, lw=0)
    ax[2].set_ylabel("mean log Nd - mean log CCN (dex)")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    for a in ax:
        a.set_xscale("log")
        a.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 300]))
        a.xaxis.set_major_formatter(ScalarFormatter())
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xlabel("Cutoff diameter (nm)")
    sub = f"; {'; '.join(notes)}" if notes else ""
    tsel = pd.DatetimeIndex(t_m)
    site = ds.attrs.get("site", "ARM ENA").split(" (")[0]
    fig.suptitle(f"{site} {tsel.min():%b %Y} - {tsel.max():%b %Y}: {m.sum()} h on {n_days} days, "
                 f"cloud_frac >= {args.min_cloud_frac}{sub}", fontsize=10)
    fig.tight_layout()
    png = os.path.splitext(args.file)[0] + "_quicklook.png"
    fig.savefig(png, dpi=150)
    tab.to_csv(os.path.splitext(args.file)[0] + "_quicklook.csv", index=False)
    print(f"\nSaved {png} and the CSV table next to it.")

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def creds(sp):
        sp.add_argument("--user", help="ARM username (default: $ARM_USER)")
        sp.add_argument("--token", help="ARM Live token (default: $ARM_TOKEN)")

    d = sub.add_parser("discover", help="find available datastreams and overlap years")
    creds(d)
    d.add_argument("--start", default="2014-01-01")
    d.add_argument("--end", default=None)
    d.add_argument("--nd-ds", nargs="*", help="extra Nd datastreams to test")
    d.add_argument("--psd-ds", nargs="*", help="extra size-distribution datastreams")
    d.add_argument("--site", default=None,
                   help="ARM site code to scan instead of ENA, e.g. sgp or asi (LASIC)")
    d.add_argument("--facilities", nargs="*", default=["C1", "E13", "M1", "S1"],
                   help="facility codes to try with --site")
    d.set_defaults(func=cmd_discover)

    g = sub.add_parser("download", help="download one datastream")
    creds(g)
    g.add_argument("--ds", required=True)
    g.add_argument("--start", required=True)
    g.add_argument("--end", required=True)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_download)

    b = sub.add_parser("build", help="hourly match Nd and CCN(D_c)")
    b.add_argument("--nd-dir", required=True)
    b.add_argument("--psd-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--nd-var", default=ND_VAR)
    b.add_argument("--psd-var", default=None, help="dN/dlogDp variable (auto if unset)")
    b.add_argument("--qc", choices=["strict", "none"], default="strict")
    b.add_argument("--min-cloud-frac", type=float, default=MIN_CLOUD_FRAC_FLOOR,
                   help="floor for keeping an hour (the analysis threshold is "
                        "chosen in quicklook)")
    b.add_argument("--min-psd-frac", type=float, default=MIN_PSD_FRAC)
    b.add_argument("--nd-hourly", choices=ND_HOURLY_STATS, default="mean",
                   help="how the 20-s Nd retrievals are combined into one hourly value "
                        "(mean = in-cloud average as in the models; at ENA the hourly-mean NDROP matched the aircraft, the median was about 40 per cent low; median kept as a sensitivity option)")
    b.add_argument("--site-name", default="ARM ENA (Graciosa, Azores)",
                   help="label stored in the file and used in plot titles")
    b.add_argument("--met-dir", default=None,
                   help="surface met files (e.g. enametC1.b1) for the LCL / coupling measure")
    b.add_argument("--nd-extra",
                   default=r"(^lwp|cloud_base|cloud_top|cloud_thickness|adiabatic|"
                           r"optical_depth|^beta$|toterror)",
                   help="regex of 1-D NDROP variables to carry along")
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("quicklook", help="susceptibility vs cutoff from a built file")
    q.add_argument("file")
    q.add_argument("--min-cloud-frac", type=float, default=MIN_CLOUD_FRAC,
                   help="in-cloud threshold for the main table and figure")
    q.add_argument("--max-rel-err", type=float, default=None,
                   help="noise screen: keep hours with NDROP error / Nd <= this (e.g. 0.5)")
    q.add_argument("--min-lwp", type=float, default=None,
                   help="keep hours with measured LWP >= this, in g m-2 (e.g. 50)")
    q.add_argument("--nd-scale", type=float, default=1.0,
                   help="divide Nd by this factor first, e.g. 1.8 for Cloudnet vs NDROP at SGP")
    q.add_argument("--max-cloud-base", type=float, default=None,
                   help="keep hours with retrieval cloud base <= this (m), e.g. 2000 for boundary-layer cloud")
    q.add_argument("--max-cloud-top", type=float, default=None,
                   help="keep hours with retrieval cloud top <= this (m), e.g. 3000")
    q.add_argument("--coupled-only", action="store_true",
                   help="keep hours with cloud base - LCL <= --dz-coupled (needs build --met-dir)")
    q.add_argument("--dz-coupled", type=float, default=150.0,
                   help="coupling threshold on cloud base - LCL, m (default 150)")
    q.add_argument("--cb-split", type=float, default=None,
                   help="cloud-base height (m) splitting low/high bases (default: median)")
    q.add_argument("--n-boot", type=int, default=300,
                   help="day-block bootstrap resamples (default 300)")
    q.add_argument("--n-bins", type=int, default=None,
                   help="equal-number bins (default: n/15, between 5 and 20)")
    q.set_defaults(func=cmd_quicklook)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
