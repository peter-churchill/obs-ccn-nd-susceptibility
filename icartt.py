"""Minimal reader for NASA ICARTT (.ict) files, format 1001 (v1.1 and v2.0).

    from icartt import read_ict
    df = read_ict("ACTIVATE-..._20200214_R1.ict")   # pandas DataFrame indexed by UTC time
    df.attrs["units"]["CNgt3nm"], df.attrs["header"]  # units per column, raw header lines

Missing values (line 12 of the header) and the LLOD / ULOD flags declared in the normal comments are
set to NaN. The time index is the first column (Time_Start or Time_Mid, seconds after 00 UTC of the
flight date in header line 7).
"""
import re
import numpy as np
import pandas as pd


def _nums(line):
    return [float(x) for x in re.split(r"[,\s]+", line.strip()) if x]


def read_header(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        nhead = int(re.split(r"[,\s]+", first.strip())[0])
        lines = [first.rstrip("\n")] + [fh.readline().rstrip("\n") for _ in range(nhead - 1)]
    return nhead, lines


def read_ict(path, keep_flags=False, usecols=None):
    nhead, h = read_header(path)
    if "1001" not in h[0]:
        raise ValueError(f"{path}: only ICARTT FFI 1001 is supported (got '{h[0]}')")
    d = [int(x) for x in _nums(h[6])[:3]]
    date = pd.Timestamp(year=d[0], month=d[1], day=d[2], tz=None)
    indep = h[8].split(",")[0].strip()
    nv = int(_nums(h[9])[0])
    scale = _nums(h[10])
    miss = _nums(h[11])
    var_lines = h[12:12 + nv]
    names = [indep] + [v.split(",")[0].strip() for v in var_lines]
    units = {indep: (h[8].split(",")[1].strip() if "," in h[8] else "s")}
    for v in var_lines:
        parts = [x.strip() for x in v.split(",")]
        units[parts[0]] = parts[1] if len(parts) > 1 else ""
    # LLOD / ULOD flags from the normal comments
    flags = set()
    for line in h:
        m = re.match(r"\s*(LLOD_FLAG|ULOD_FLAG)\s*:\s*(-?\d+\.?\d*)", line, re.I)
        if m:
            flags.add(float(m.group(2)))
    # the last header line is the column header; prefer it when it has the right length
    col = [c.strip() for c in h[-1].split(",")]
    if len(col) == len(names):
        names = col
    if usecols is not None:
        usecols = [names[0]] + [c for c in usecols if c != names[0] and c in names]
    df = pd.read_csv(path, skiprows=nhead, header=None, names=names, usecols=usecols,
                     skipinitialspace=True, na_values=[], low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    for i, c in enumerate(names[1:]):
        if c not in df.columns:
            continue
        x = df[c]
        if i < len(miss):
            x = x.mask(np.isclose(x, miss[i]))
        if flags and not keep_flags:
            x = x.mask(x.isin(list(flags)))
        x = x.mask(x <= -9999)  # some files use several missing codes
        if i < len(scale) and scale[i] not in (0, 1):
            x = x * scale[i]
        df[c] = x
    t = df[names[0]]
    df.index = date + pd.to_timedelta(t, unit="s")
    df.index.name = "time"
    df.attrs.update(units=units, header=h, file=str(path), date=str(date.date()), indep=names[0])
    return df


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        df = read_ict(f)
        print(f, df.shape, df.index.min(), "->", df.index.max())
        for c in df.columns[:60]:
            print(f"  {c:40s} {df.attrs['units'].get(c, ''):15s} valid {df[c].notna().mean():5.0%}")
