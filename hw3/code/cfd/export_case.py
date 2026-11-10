"""Convert one finished OpenFOAM case into training-ready arrays.

Dataset A (steady)  -> <out>/<name>_points.npy  (N, 7) x y z u v w p   [same layout as
                       ellipsoid_*_u34_points.npy used by the original training script]
                       <out>/probes.npy        (P, 4) u v w p at the shared probes (NaN = outside mesh)
Dataset B (transient)-> <out>/transient_points/t_<time>.npy (N, 7) per snapshot, xyz.npy (N, 3), times.npy (T,),
                       freestream.npy (T, 4) [u_inf v_inf w_inf 0], probes.npy (T, P, 4)
Both also write meta.json.

Reads the decomposed case (processor*/) directly when it exists, so reconstructPar is never
needed; cells are concatenated in processor order (identical for every run that copies the
same decomposed mesh). Falls back to a reconstructed case otherwise.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from foam_io import (field_path, match_probes, read_internal_field, read_sampled_set,
                     sampled_time_name, time_dirs)


def probes_at(case, probes, t):
    out = []
    for func, ncomp in (("sampleU", 3), ("sampleP", 1)):
        fdir = case / "postProcessing" / func
        xyz, v = read_sampled_set(fdir, sampled_time_name(fdir, t))
        out.append(match_probes(probes, xyz, v))
    return np.concatenate(out, axis=1).astype(np.float32)


def processor_dirs(case):
    return sorted((p for p in case.glob("processor*") if p.name[9:].isdigit()), key=lambda p: int(p.name[9:]))


def read_cells(dirs_t, name, n=None):
    """Internal field over all processors for one time: dirs_t = [time dir per processor]."""
    parts = [read_internal_field(field_path(d, name), None if n is None else n[k]) for k, d in enumerate(dirs_t)]
    return np.concatenate(parts)


def _fields_job(job):
    dirs_t, n = job
    U = read_cells(dirs_t, "U", n)
    p = read_cells(dirs_t, "p", n)
    return np.column_stack([U, p]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("--kind", choices=["A", "B"], required=True)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    case, out = args.case.resolve(), args.out
    out.mkdir(parents=True, exist_ok=True)
    probes = np.load(args.probes)
    forcing = np.load(case / "constant/forcing.npy")      # t, speed, yaw, ux, uy, uz

    procs = processor_dirs(case)
    base = procs[0] if procs else case
    times = time_dirs(base)
    if args.kind == "A":
        times = times[-1:]
    else:
        # export window = times that have cell centres written from export_from onward
        times = [(t, d) for t, d in times if (d / "U").exists() or (d / "U.gz").exists()]
        t0 = min(t for t, d in times if field_path(d, "C").exists())
        times = [(t, d) for t, d in times if t >= t0 - 1e-9]
    if not times:
        raise SystemExit(f"{case}: no time directories to export")

    def per_proc(d):
        """time dir in processor0 (or the case) -> list of that time dir in every processor"""
        return [p / d.name for p in procs] if procs else [d]

    c_dir = next(d for t, d in times if field_path(d, "C").exists())
    c_parts = [read_internal_field(field_path(d, "C")) for d in per_proc(c_dir)]
    n_proc = [len(c) for c in c_parts]
    xyz = np.concatenate(c_parts)
    n = len(xyz)
    print(f"{case.name}: {n:,} cells from {len(procs) or 1} {'processor dirs' if procs else 'reconstructed case'}, "
          f"{len(times)} times", flush=True)

    def fields(d):
        return _fields_job((per_proc(d), n_proc))

    meta = dict(case=str(case), kind=args.kind, n_cells=int(n), n_probes=int(len(probes)),
                source="processor dirs" if procs else "reconstructed",
                columns=["x", "y", "z", "u", "v", "w", "p"], pressure="kinematic p/rho [m2/s2]")

    if args.kind == "A":
        t, d = times[0]
        arr = np.column_stack([xyz.astype(np.float32), fields(d)])
        np.save(out / f"{out.name}_points.npy", arr)
        pr = probes_at(case, probes, t)
        np.save(out / "probes.npy", pr)
        meta.update(time=t, freestream=[*forcing[0, 3:6].tolist(), 0.0],
                    probe_coverage=float(np.isfinite(pr[:, 0]).mean()))
    else:
        tt = np.array([t for t, _ in times])
        np.save(out / "xyz.npy", xyz.astype(np.float32))
        np.save(out / "times.npy", tt)
        fs = np.column_stack([np.interp(tt, forcing[:, 0], forcing[:, 3 + k]) for k in range(3)]
                             + [np.zeros_like(tt)])
        np.save(out / "freestream.npy", fs.astype(np.float32))
        # one (N, 7) x y z u v w p file per snapshot: same layout as phaseB_transient/*/transient_points
        snap = out / "transient_points"
        snap.mkdir(exist_ok=True)
        P = np.lib.format.open_memmap(out / "probes.npy", "w+", np.float32, (len(tt), len(probes), 4))
        xyz32 = xyz.astype(np.float32)
        workers = max(1, min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4))
        jobs = [(per_proc(d), n_proc) for _, d in times]
        with ProcessPoolExecutor(workers) as ex:              # ASCII parsing is the bottleneck
            for k, ((t, d), f) in enumerate(zip(times, ex.map(_fields_job, jobs))):
                np.save(snap / f"t_{t:.3f}.npy", np.column_stack([xyz32, f]))
                P[k] = probes_at(case, probes, t)
        P.flush()
        meta.update(n_times=len(tt), t_first=float(tt[0]), t_last=float(tt[-1]),
                    probe_coverage=float(np.isfinite(P[0, :, 0]).mean()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
