"""Exported OpenFOAM runs (cfd/export_case.py) -> runs, point pools, normalization, splits.

Export layout
  B_mXX_<flow>/  transient_points/t_<time>.npy (N, 7) x y z u v w p, xyz.npy, times.npy,
                 freestream.npy (T, 4) = prescribed inflow, probes.npy (T, P, 4)
  A_mXX/         A_mXX_points.npy (N, 7), probes.npy (P, 4), meta.json (freestream)

Conventions (fixed for every dataset, saved in each checkpoint)
  xyz    -> [-1, 1] with the physical domain bounds from the config (not from the data)
  fields -> residuals (u - U_in(t), v - V_in(t), w - W_in(t), p) / scale, scale = per-channel
            std over the training pools. p is kinematic, outlet reference p = 0; nothing is
            subtracted that the model would not know at inference.
  cond   -> [U_in(t) / U_ref, dU_in/dt * L / U_ref^2]: the prescribed inflow and its
            acceleration (unsteady-Bernoulli pressure level ~ L dU/dt).
"""
import fnmatch
import json
import re
from pathlib import Path

import numpy as np
import torch

NAME_B = re.compile(r"B_(m\d+)_(\w+)$")
NAME_A = re.compile(r"A_(m\d+)$")


def parse_name(name):
    m = NAME_B.match(name)
    if m:
        return dict(kind="B", mesh=m.group(1), flow=m.group(2))
    m = NAME_A.match(name)
    if m:
        return dict(kind="A", mesh=m.group(1), flow="steady")
    return None


class Run:
    def __init__(self, path, t_min=0.0):
        self.dir = Path(path)
        self.name = self.dir.name
        info = parse_name(self.name)
        if info is None:
            raise ValueError(f"not an export folder name: {self.name}")
        self.kind, self.mesh, self.flow = info["kind"], info["mesh"], info["flow"]
        probes = self.dir / "probes.npy"
        if self.kind == "B":
            times = np.load(self.dir / "times.npy").astype(np.float64)
            fs = np.load(self.dir / "freestream.npy")[:, :3].astype(np.float32)
            files = [self.dir / "transient_points" / f"t_{t:.3f}.npy" for t in times]
            keep = np.flatnonzero(times >= t_min - 1e-9)
            self.times, self.fs, self.files = times[keep], fs[keep], [files[k] for k in keep]
            self.xyz = np.load(self.dir / "xyz.npy").astype(np.float32)
            self.probe_values = np.load(probes, mmap_mode="r")[keep] if probes.exists() else None
        else:
            meta = json.loads((self.dir / "meta.json").read_text())
            self.times = np.array([float(meta.get("time", 0.0))])
            self.fs = np.array([meta["freestream"][:3]], np.float32)
            self.files = [self.dir / f"{self.name}_points.npy"]
            self.xyz = np.asarray(np.load(self.files[0], mmap_mode="r")[:, :3], np.float32)
            self.probe_values = np.load(probes)[None] if probes.exists() else None
        missing = [f for f in self.files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"{self.name}: {len(missing)} snapshot files missing, e.g. {missing[0]}")
        self.N, self.T = len(self.xyz), len(self.times)

    def frame(self, k):
        return np.load(self.files[k]).astype(np.float32)

    def build_pool(self, n_uniform, n_importance, rng, u_ref):
        """Fixed point subset cached for all snapshots (the mesh is static).
        First n_uniform rows: uniform over cells -> encoder context (keeps the mesh's own
        point density). Remaining rows: weighted by |U - U_in| -> extra queries near the
        body and in the wake."""
        uni = np.sort(rng.choice(self.N, min(n_uniform, self.N), replace=False))
        w = np.zeros(self.N)
        ks = sorted({0, self.T // 2, self.T - 1})
        for k in ks:
            a = self.frame(k)
            w += np.linalg.norm(a[:, 3:6] - self.fs[k], axis=1)
        w = w / len(ks) + 0.05 * u_ref
        rest = np.setdiff1d(np.arange(self.N), uni)
        n_imp = min(n_importance, len(rest))
        imp = np.sort(rng.choice(rest, n_imp, replace=False, p=w[rest] / w[rest].sum())) if n_imp else rest[:0]
        self.pool_idx = np.concatenate([uni, imp])
        self.n_uniform = len(uni)
        self.pool = np.stack([np.asarray(np.load(f, mmap_mode="r")[self.pool_idx]) for f in self.files]
                             ).astype(np.float32)                                   # (T, M, 7)


class Normalizer:
    def __init__(self, bounds, u_ref, accel_length, scale=None):
        self.bounds = np.asarray(bounds, np.float32)
        self.u_ref, self.accel_length = float(u_ref), float(accel_length)
        self.p_ref = 0.5 * self.u_ref ** 2
        self.scale = None if scale is None else np.asarray(scale, np.float32)
        self._t = {}

    def _tensors(self, device):
        if device not in self._t:
            self._t[device] = (torch.tensor(self.bounds[:, 0], device=device),
                               torch.tensor(self.bounds[:, 1], device=device),
                               torch.tensor(self.scale, device=device))
        return self._t[device]

    def fit_scale(self, runs):
        res = np.concatenate([(r.pool[::max(1, r.T // 20), :, 3:7]
                               - np.concatenate([r.fs, np.zeros((r.T, 1), np.float32)], 1)
                               [::max(1, r.T // 20), None]).reshape(-1, 4) for r in runs])
        floor = 1e-3 * np.array([self.u_ref] * 3 + [self.p_ref])
        self.scale = np.maximum(res.std(0), floor).astype(np.float32)
        self._t = {}

    def xyz(self, x):
        lo, hi, _ = self._tensors(x.device)
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def fields(self, a, fs):
        """a (..., 4) physical u v w p, fs (..., 3) inflow -> normalized residual."""
        _, _, s = self._tensors(a.device)
        return torch.cat([a[..., :3] - fs, a[..., 3:]], -1) / s

    def unfields(self, r, fs):
        _, _, s = self._tensors(r.device)
        a = r * s
        return torch.cat([a[..., :3] + fs, a[..., 3:]], -1)

    def cond(self, run):
        """(T, 6): inflow and inflow acceleration, dimensionless."""
        acc = np.gradient(run.fs, run.times, axis=0) if run.T > 1 else np.zeros_like(run.fs)
        return np.concatenate([run.fs / self.u_ref, acc * self.accel_length / self.u_ref ** 2], 1).astype(np.float32)

    def state_dict(self):
        return dict(bounds=self.bounds.tolist(), u_ref=self.u_ref, accel_length=self.accel_length,
                    scale=self.scale.tolist())

    @classmethod
    def from_state(cls, d):
        return cls(d["bounds"], d["u_ref"], d["accel_length"], d["scale"])


def _match(names, patterns, what):
    out = []
    for p in patterns:
        hit = [n for n in names if fnmatch.fnmatch(n, p)]
        if not hit:
            raise SystemExit(f"{what}: pattern {p!r} matches no loaded run. Loaded: {names}")
        out += [n for n in hit if n not in out]
    return out


class Dataset:
    """All runs of one experiment, their pools on the training device, and the splits."""

    def __init__(self, cfg, device, normalizer=None, cases=None, need_splits=True, verbose=True):
        d, ph = cfg["data"], cfg["physics"]
        root = Path(d["root"])
        if not root.is_dir():
            raise SystemExit(f"data.root does not exist: {root}")
        found = sorted(p.name for p in root.iterdir() if p.is_dir() and parse_name(p.name))
        names = _match(found, cases or d["cases"], "data.cases")
        rng = np.random.default_rng(cfg["seed"])
        self.runs = []
        for n in names:
            r = Run(root / n, d["t_min"])
            r.build_pool(d["pool_uniform"], d["pool_importance"], rng, ph["u_ref"])
            self.runs.append(r)
            if verbose:
                print(f"  {r.name:24s} cells {r.N:>9,}  snapshots {r.T:>4d}  "
                      f"t {r.times[0]:.3f}-{r.times[-1]:.3f} s  pool {len(r.pool_idx):,}", flush=True)
        self.index = {r.name: i for i, r in enumerate(self.runs)}

        self.splits = {}
        if need_splits:
            spec = d.get("splits")
            if d.get("splits_file"):                     # written by cfd/generate_cases.py
                spec = json.loads(Path(d["splits_file"]).read_text())
            for s in ("train", "val", "test"):
                self.splits[s] = [self.index[n] for n in _match(names, spec.get(s, []), f"splits.{s}")]
                if s in ("train", "val") and not self.splits[s]:
                    raise SystemExit(f"splits.{s} is empty: refusing to train without it")
            seen = {}
            for s, ids in self.splits.items():
                for i in ids:
                    if i in seen:
                        raise SystemExit(f"{self.runs[i].name} is in both {seen[i]} and {s}")
                    seen[i] = s
        else:
            self.splits["train"] = list(range(len(self.runs)))

        if normalizer is None:
            normalizer = Normalizer(ph["bounds"], ph["u_ref"], ph["accel_length"])
            normalizer.fit_scale([self.runs[i] for i in self.splits["train"]])
        self.norm = normalizer
        if verbose:
            print("  residual scale [u' v' w' p]:", np.round(self.norm.scale, 4).tolist(), flush=True)

        self.device = torch.device(device)
        pool_dev = self.device if d.get("pool_device", "cuda") != "cpu" else torch.device("cpu")
        self.pool = [torch.from_numpy(r.pool).to(pool_dev) for r in self.runs]
        self.fs = [torch.from_numpy(r.fs).to(self.device) for r in self.runs]
        self.cond = [torch.from_numpy(self.norm.cond(r)).to(self.device) for r in self.runs]

        # same flow on different meshes with identical snapshot times -> mesh-consistency pairs
        self.mesh_pairs = []
        for i, a in enumerate(self.runs):
            for j, b in enumerate(self.runs):
                if i < j and a.flow == b.flow and a.mesh != b.mesh and a.T == b.T \
                        and np.allclose(a.times, b.times):
                    self.mesh_pairs.append((i, j))

    def split_name(self, i):
        return next((s for s, ids in self.splits.items() if i in ids), "-")

    # ---------------- sampling (used by every trainer) ----------------
    def context(self, i, k, n, gen):
        """(n, 7) rows from the uniform part of the pool (the mesh's own point density)."""
        r = self.runs[i]
        rows = torch.randint(0, r.n_uniform, (n,), generator=gen, device=gen.device)
        return self.pool[i][k][rows.to(self.pool[i].device)].to(self.device)

    def query(self, i, k, n, gen):
        """(n, 7): half uniform rows, half importance rows."""
        r = self.runs[i]
        M = len(r.pool_idx)
        n_u = n // 2 if M > r.n_uniform else n
        rows = torch.cat([torch.randint(0, r.n_uniform, (n_u,), generator=gen, device=gen.device),
                          torch.randint(r.n_uniform, max(M, r.n_uniform + 1), (n - n_u,),
                                        generator=gen, device=gen.device)])
        return self.pool[i][k][rows.to(self.pool[i].device)].to(self.device)
