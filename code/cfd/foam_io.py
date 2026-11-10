"""Minimal readers for OpenFOAM ASCII output (no PyFoam/fluidfoam dependency)."""
import gzip
import re
from pathlib import Path

import numpy as np


def _open(path):
    path = Path(path)
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    if path.suffix == ".gz":
        return gzip.open(path, "rb").read()
    return path.read_bytes()


def field_path(time_dir, name):
    p = Path(time_dir) / name
    return p if p.exists() else Path(str(p) + ".gz")


def read_internal_field(path, n_cells=None):
    """Return internalField as (N,) or (N, 3) float64 from an ASCII field file."""
    raw = _open(path)
    i = raw.find(b"internalField")
    if i < 0:
        raise ValueError(f"{path}: no internalField")
    head = raw[i:i + 200].decode("ascii", "replace")
    m = re.match(r"internalField\s+uniform\s+(\([^)]*\)|[-+0-9.eE]+)", head)
    if m:
        if n_cells is None:
            raise ValueError(f"{path}: uniform field, pass n_cells")
        v = np.array(m.group(1).strip("()").split(), float)
        return np.tile(v, (n_cells, 1)).squeeze()
    m = re.match(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", head)
    if not m:
        raise ValueError(f"{path}: unsupported internalField header {head[:80]!r} "
                         "(binary? set writeFormat ascii)")
    kind, n = m.group(1), int(m.group(2))
    ncomp = {"scalar": 1, "vector": 3}[kind]
    start = i + m.end()
    if ncomp == 1:
        end = raw.index(b")", start)
    else:
        end = raw.index(b"\n)", start)
    vals = np.array(raw[start:end].translate(None, b"()").split(), dtype=np.float64)
    if vals.size != n * ncomp:
        raise ValueError(f"{path}: expected {n * ncomp} values, got {vals.size}")
    return vals.reshape(n, ncomp) if ncomp > 1 else vals


def time_dirs(case, t_min=None):
    """Numeric time directories (excluding 0) sorted by value."""
    out = []
    for d in Path(case).iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
            except ValueError:
                continue
            if t > 0 and (t_min is None or t >= t_min - 1e-9):
                out.append((t, d))
    return sorted(out)


def read_sampled_set(func_dir, time_name):
    """Raw-format sampled set -> (xyz (M,3), values (M,) or (M,3))."""
    d = Path(func_dir) / time_name
    files = sorted(p for p in d.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(d)
    a = np.loadtxt(files[0], comments="#", ndmin=2)
    return a[:, :3], (a[:, 3] if a.shape[1] == 4 else a[:, 3:6])


def match_probes(probes, xyz, vals, tol=1e-5):
    """Scatter sampled values back to the canonical probe order (NaN = not found)."""
    from scipy.spatial import cKDTree
    d, j = cKDTree(xyz).query(probes)
    ncomp = 1 if vals.ndim == 1 else vals.shape[1]
    out = np.full((len(probes), ncomp), np.nan)
    ok = d < tol
    out[ok] = vals[j[ok]].reshape(-1, ncomp)
    return out


def sampled_time_name(func_dir, t):
    """postProcessing/<func>/<time> directory names are formatted by OpenFOAM."""
    for d in Path(func_dir).iterdir():
        try:
            if abs(float(d.name) - t) < 1e-9:
                return d.name
        except ValueError:
            pass
    raise FileNotFoundError(f"{func_dir}: no time {t}")


def read_dat(path):
    """Whitespace table with '#' header; parentheses stripped. Returns (header, array)."""
    header, rows = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            if "Time" in line:
                header = line.lstrip("# ").split()
            continue
        parts = line.replace("(", " ").replace(")", " ").split()
        if parts:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
    return header, np.array(rows)
