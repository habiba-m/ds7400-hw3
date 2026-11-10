"""Validate Datasets A and B after the OpenFOAM runs.

    python validate_datasets.py --root /scratch/crv4jj/ellipsoid_datasets

Writes <root>/validation/: report.md, A_summary.csv, B_summary.csv and plots.

Checks
  per case  : finished, checkMesh OK, cell count, solver convergence / completion,
              y+ on the body, inflow at upstream probes matches the prescribed forcing
  Dataset A : drag-coefficient convergence vs mesh size (power-law fit + Celik GCI),
              field convergence at the SHARED probe coordinates vs the finest mesh
  Dataset B : identical snapshot times on every mesh, force histories per condition,
              time-averaged field error at the shared probes vs the finest mesh
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import config as cfg
from foam_io import read_dat

FIELDS = ("u", "v", "w", "p")
A_REF = np.pi * cfg.SEMI_AXES[0] * cfg.SEMI_AXES[2]     # frontal area normal to +y


# --------------------------------------------------------------------------- #
# Log / postProcessing parsing
# --------------------------------------------------------------------------- #
def parse_checkmesh(case):
    log = case / "log.checkMesh"
    if not log.exists():
        return dict(mesh_ok=None, cells=None)
    s = log.read_text(errors="replace")
    m = re.search(r"^\s*cells:\s+(\d+)", s, re.M)
    failed = re.search(r"Failed (\d+) mesh checks", s)
    return dict(mesh_ok=("Mesh OK" in s) and not failed, cells=int(m.group(1)) if m else None)


def parse_solver_log(case, app):
    log = case / f"log.{app}"
    if not log.exists():
        return dict(ran=False)
    s = log.read_text(errors="replace")
    res = {}
    for f in ("Ux", "Uy", "Uz", "p", "k", "omega"):
        r = re.findall(rf"Solving for {f}, Initial residual = ([0-9.eE+-]+)", s)
        if r:
            res[f] = float(r[-1])
    times = re.findall(r"^Time = ([0-9.eE+-]+)", s, re.M)
    co = re.findall(r"Courant Number mean: [0-9.eE+-]+ max: ([0-9.eE+-]+)", s)
    cont = re.findall(r"continuity errors : sum local = ([0-9.eE+-]+)", s)
    return dict(ran=True, ended=bool(re.search(r"^End\s*$", s, re.M)),
                converged="solution converged" in s,
                last_time=float(times[-1]) if times else None,
                final_residuals=res,
                max_courant=max(map(float, co)) if co else None,
                continuity_local=float(cont[-1]) if cont else None)


def read_forces(case):
    """(t, F_total (T,3)) from the forces function object (ESI force.dat or legacy forces.dat)."""
    files = sorted((case / "postProcessing/forces").glob("*/force.dat")) or \
        sorted((case / "postProcessing/forces").glob("*/forces.dat"))
    if not files:
        return None, None
    ts, Fs = [], []
    for f in files:                                   # restarts write new time dirs
        header, a = read_dat(f)
        if a.size == 0:
            continue
        if header and "total_x" in header:
            j = header.index("total_x")
            F = a[:, j:j + 3]
        else:                                         # (pressure)(viscous)[(porous)]
            F = a[:, 1:4] + a[:, 4:7]
        ts.append(a[:, 0]); Fs.append(F)
    t, F = np.concatenate(ts), np.concatenate(Fs)
    order = np.argsort(t, kind="stable")
    return t[order], F[order]


def read_yplus(case):
    files = sorted((case / "postProcessing/yPlus").glob("*/yPlus.dat"))
    if not files:
        return None
    for line in reversed(files[-1].read_text().splitlines()):
        p = line.split()
        if len(p) >= 5 and p[1] == "ellipsoid":
            return dict(min=float(p[2]), max=float(p[3]), avg=float(p[4]))
    return None


# --------------------------------------------------------------------------- #
# Convergence analysis
# --------------------------------------------------------------------------- #
def celik_gci(h, f, Fs=1.25):
    """Celik et al. (2008) GCI for three meshes, h ordered fine -> coarse."""
    (h1, h2, h3), (f1, f2, f3) = h, f
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        return dict(p=np.nan, f_ext=f1, gci_fine=0.0, monotone=True)
    s = np.sign(e32 / e21)
    p = 2.0
    for _ in range(100):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-8:
            break
        p = p_new
    f_ext = (r21**p * f1 - f2) / (r21**p - 1)
    gci = Fs * abs((f1 - f2) / f1) / (r21**p - 1)
    return dict(p=float(p), f_ext=float(f_ext), gci_fine=float(gci), monotone=bool(s > 0))


def powerlaw_fit(h, f):
    """Least-squares f = f0 + C h^p over all meshes (grid search on p)."""
    best = None
    for p in np.linspace(0.5, 4.0, 351):
        X = np.column_stack([np.ones_like(h), h**p])
        coef, *_ = np.linalg.lstsq(X, f, rcond=None)
        r = np.sum((X @ coef - f) ** 2)
        if best is None or r < best[0]:
            best = (r, p, coef)
    r, p, (f0, C) = best
    return dict(p=float(p), f0=float(f0), C=float(C), rms=float(np.sqrt(r / len(h))))


SCALE = np.array([cfg.U_REF] * 3 + [0.5 * cfg.U_REF**2])   # u,v,w / U_REF ; p / (U_REF^2 / 2)


def rel_err(pred, ref):
    """RMS difference per field over probes valid in both arrays (axis -2 = probes),
    normalized by U_REF (velocity) and U_REF^2/2 (pressure). Normalizing by the
    field's own spread would blow up for near-zero components like u and w."""
    ok = np.isfinite(pred).all(-1) & np.isfinite(ref).all(-1)
    sq = np.where(ok[..., None], (pred - ref) ** 2, 0).sum(-2)
    return np.sqrt(sq / np.maximum(ok.sum(-1), 1)[..., None]) / SCALE


def upstream_mask(probes):
    return probes[:, 1] < cfg.DOMAIN[1, 0] + 0.5        # within 0.5 m of the inlet


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    root = args.root
    man = json.loads((root / "manifest.json").read_text())
    by_name = {c["name"]: c for c in man["cases"]}
    probes = np.load(root / "probes/probes.npy")
    groups = dict(np.load(root / "probes/probe_groups.npz"))
    up = upstream_mask(probes)
    out = root / "validation"
    out.mkdir(exist_ok=True)
    report = ["# Dataset validation report", ""]
    plt = None
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    # ------------------------------ per case ------------------------------ #
    rows = []
    for c in man["cases"]:
        case = Path(c["case_dir"])
        exp = Path(c["export_dir"])
        mesh_case = Path(c["mesh_case"]) if c["dataset"] == "B" else case
        r = dict(name=c["name"], dataset=c["dataset"], mesh_index=c["mesh_index"], h=c["h"],
                 condition=c.get("condition", ""), done=(case / "DONE").exists())
        r.update(parse_checkmesh(mesh_case))
        r.update(parse_solver_log(case, "simpleFoam" if c["steady"] else "pimpleFoam"))
        t, F = read_forces(case)
        if F is not None and c["steady"]:
            fdir = np.array(c["freestream"][:3]) / np.linalg.norm(c["freestream"][:3])
            cd = F @ fdir / (0.5 * c["speed"] ** 2 * A_REF)
            tail = cd[-max(len(cd) // 10, 1):]
            r.update(Cd=float(cd[-1]), Cd_tail_rel_std=float(tail.std() / abs(tail.mean())))
        r["yplus"] = read_yplus(case)
        r["exported"] = (exp / "meta.json").exists()
        if r["exported"]:
            meta = json.loads((exp / "meta.json").read_text())
            r["probe_coverage"] = meta.get("probe_coverage")
            pr = np.load(exp / "probes.npy", mmap_mode="r")
            if c["steady"]:
                fs = np.array(c["freestream"][:3])
                r["inflow_err"] = float(np.nanmean(np.linalg.norm(pr[up, :3] - fs, axis=1))
                                        / np.linalg.norm(fs))
            else:
                fs = np.load(exp / "freestream.npy")[:, :3]
                d = np.linalg.norm(np.asarray(pr[:, up, :3]) - fs[:, None], axis=2)
                r["inflow_err"] = float(np.nanmean(d / np.linalg.norm(fs, axis=1)[:, None]))
        rows.append(r)

    def status(r):
        bad = []
        if not r["done"]: bad.append("not done")
        if r.get("mesh_ok") is False: bad.append("checkMesh failed")
        if r.get("ran") and not r.get("ended"): bad.append("solver did not finish")
        if r.get("Cd_tail_rel_std", 0) > 1e-3: bad.append("Cd not iteratively converged")
        if r.get("inflow_err") is not None and r["inflow_err"] > 0.02: bad.append("inflow != forcing")
        if r.get("probe_coverage") is not None and r["probe_coverage"] < 0.99: bad.append("probe coverage < 99%")
        return "; ".join(bad) or "OK"

    for ds in ("A", "B"):
        sel = [r for r in rows if r["dataset"] == ds]
        if not sel:
            continue
        keys = ["name", "condition", "mesh_index", "h", "cells", "mesh_ok", "done", "ended",
                "converged", "last_time", "Cd", "Cd_tail_rel_std", "max_courant",
                "continuity_local", "probe_coverage", "inflow_err", "status"]
        with open(out / f"{ds}_summary.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys + ["yplus_avg", "yplus_max"] + [f"res_{k}" for k in ("Ux", "Uy", "p", "k", "omega")])
            for r in sel:
                r["status"] = status(r)
                yp = r.get("yplus") or {}
                res = r.get("final_residuals") or {}
                w.writerow([r.get(k) for k in keys] + [yp.get("avg"), yp.get("max")]
                           + [res.get(k) for k in ("Ux", "Uy", "p", "k", "omega")])
        n_ok = sum(r["status"] == "OK" for r in sel)
        report += [f"## Dataset {ds}: {n_ok}/{len(sel)} cases pass per-case checks", ""]
        for r in sel:
            if r["status"] != "OK":
                report.append(f"- `{r['name']}`: {r['status']}")
        report.append("")

    # ------------------------------ Dataset A ------------------------------ #
    A = sorted([r for r in rows if r["dataset"] == "A" and r["exported"] and r.get("cells")],
               key=lambda r: r["cells"])
    if len(A) >= 3:
        vol = np.prod(cfg.DOMAIN[:, 1] - cfg.DOMAIN[:, 0])
        N = np.array([r["cells"] for r in A], float)
        h_rep = (vol / N) ** (1 / 3)                  # representative cell size (Celik 2008)
        cd = np.array([r.get("Cd", np.nan) for r in A])
        report += ["## Dataset A: mesh convergence", ""]
        if np.isfinite(cd).all():
            fit = powerlaw_fit(h_rep, cd)
            # GCI on finest, and the meshes closest to r = 1.3 and 1.3^2 coarser
            i1 = len(A) - 1
            i2 = int(np.argmin(abs(h_rep - 1.3 * h_rep[i1])))
            i3 = int(np.argmin(abs(h_rep - 1.3**2 * h_rep[i1])))
            g = celik_gci(h_rep[[i1, i2, i3]], cd[[i1, i2, i3]])
            report += [f"- Cd range over {len(A)} meshes: {cd.min():.5f} .. {cd.max():.5f}",
                       f"- power-law fit Cd = {fit['f0']:.5f} + C h^{fit['p']:.2f} (rms {fit['rms']:.2e})",
                       f"- GCI ({A[i1]['name']}, {A[i2]['name']}, {A[i3]['name']}): "
                       f"p = {g['p']:.2f}, Cd_ext = {g['f_ext']:.5f}, GCI_fine = {g['gci_fine']:.2%}, "
                       f"monotone = {g['monotone']}", ""]
        # field convergence at the shared probes, reference = finest mesh
        ref = np.load(Path(by_name[A[-1]["name"]]["export_dir"]) / "probes.npy")
        errs = {g: [] for g in ("all", "near", "wake", "rest")}
        for r in A:
            exp = Path(by_name[r["name"]]["export_dir"])
            pr = np.load(exp / "probes.npy")
            for gname in errs:
                idx = slice(None) if gname == "all" else groups[gname]
                errs[gname].append(rel_err(pr[idx], ref[idx]))
        errs = {k: np.array(v) for k, v in errs.items()}
        np.savez(out / "A_probe_errors.npz", cells=N, h_rep=h_rep, **errs)
        report += ["RMS difference (/U_REF, /(U_REF²/2)) at the shared probes vs the finest mesh "
                   f"({A[-1]['name']}, {int(N[-1]):,} cells):", "",
                   "| mesh | cells | " + " | ".join(FIELDS) + " | wake u | wake p |",
                   "|---|---|" + "---|" * (len(FIELDS) + 2)]
        for k in sorted(set(np.linspace(0, len(A) - 2, min(8, len(A) - 1)).astype(int))):
            e, ew = errs["all"][k], errs["wake"][k]
            report.append(f"| {A[k]['name']} | {int(N[k]):,} | " + " | ".join(f"{x:.2%}" for x in e)
                          + f" | {ew[0]:.2%} | {ew[3]:.2%} |")
        report.append("")
        if plt:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            if np.isfinite(cd).all():
                ax[0].plot(h_rep, cd, "o", ms=4)
                hh = np.linspace(0, h_rep.max(), 100)
                ax[0].plot(hh, fit["f0"] + fit["C"] * hh ** fit["p"], "-", lw=1)
                ax[0].set_xlabel("representative cell size (V/N)$^{1/3}$ [m]")
                ax[0].set_ylabel("$C_d$"); ax[0].set_title("Drag convergence")
            for j, f in enumerate(FIELDS):
                ax[1].loglog(N[:-1], errs["all"][:-1, j], "o-", ms=3, label=f)
            ax[1].set_xlabel("cells"); ax[1].set_ylabel("RMS diff. vs finest / ref. scale")
            ax[1].legend(); ax[1].set_title("Field convergence")
            fig.tight_layout(); fig.savefig(out / "A_convergence.png", dpi=150); plt.close(fig)

    # ------------------------------ Dataset B ------------------------------ #
    B = [r for r in rows if r["dataset"] == "B" and r["exported"]]
    if B:
        report += ["## Dataset B: transient consistency", ""]
        conds = sorted({r["condition"] for r in B})
        if plt:
            fig, axs = plt.subplots(len(conds), 1, figsize=(9, 2.4 * len(conds)), sharex=True)
        err_tab = {}
        for ci, cond in enumerate(conds):
            sel = sorted([r for r in B if r["condition"] == cond], key=lambda r: r["mesh_index"])
            if not sel:
                continue
            exps = {r["name"]: Path(by_name[r["name"]]["export_dir"]) for r in sel}
            tsets = [np.load(exps[r["name"]] / "times.npy") for r in sel]
            same_t = all(len(t) == len(tsets[-1]) and np.allclose(t, tsets[-1]) for t in tsets)
            if not same_t:
                report.append(f"- **{cond}: snapshot times differ between meshes** "
                              f"({[len(t) for t in tsets]} snapshots)")
            ref = np.load(exps[sel[-1]["name"]] / "probes.npy", mmap_mode="r")
            for r in sel[:-1]:
                pr = np.load(exps[r["name"]] / "probes.npy", mmap_mode="r")
                T = min(len(pr), len(ref))
                e = np.nanmean(rel_err(np.asarray(pr[:T]), np.asarray(ref[:T])), axis=0)
                err_tab[(cond, r["mesh_index"])] = e
            if plt:
                for r in sel:
                    t, F = read_forces(Path(by_name[r["name"]]["case_dir"]))
                    if F is not None:
                        axs[ci].plot(t, F[:, 1], lw=0.8, label=f"m{r['mesh_index']:02d}")
                axs[ci].set_title(cond, fontsize=10); axs[ci].set_ylabel("$F_y$ / ρ [m⁴/s²]")
        if plt:
            axs[0].legend(ncol=5, fontsize=7); axs[-1].set_xlabel("t [s]")
            fig.tight_layout(); fig.savefig(out / "B_forces.png", dpi=150); plt.close(fig)
        if err_tab:
            report += ["Time-averaged RMS difference (/U_REF, /(U_REF²/2)) at the shared probes vs the finest "
                       "mesh of the same condition:", "",
                       "| condition | mesh | " + " | ".join(FIELDS) + " |", "|---|---|" + "---|" * len(FIELDS)]
            for (cond, m), e in sorted(err_tab.items()):
                report.append(f"| {cond} | m{m:02d} | " + " | ".join(f"{x:.2%}" for x in e) + " |")
            report.append("")

    (out / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
