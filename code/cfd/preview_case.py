"""CFD-only z = 0 slice GIFs of one transient case, to check that the flow evolves
before spending time on training.

    python preview_case.py /scratch/crv4jj/ellipsoid_datasets/exports/B_m00_yaw_sine

Works on any folder containing transient_points/t_<time>.npy files with columns
x y z u v w p (cfd/ Dataset B exports and the older phaseB_transient cases).
Writes <case>/preview/<field>.gif, all_fields.gif and prints how much each field
changes over time on the slice.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

FIELDS = ("u", "v", "w", "p")
UNITS = {"u": "m/s", "v": "m/s", "w": "m/s", "p": "m²/s²"}

ap = argparse.ArgumentParser()
ap.add_argument("case", type=Path)
ap.add_argument("--res", type=float, default=0.01, help="pixel size [m]")
ap.add_argument("--window", default="-1,1,-0.8,3", help="xmin,xmax,ymin,ymax [m]")
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--every", type=int, default=1, help="use every n-th snapshot")
ap.add_argument("--p-rel-inlet", action="store_true", help="plot p - p_inlet(t)")
args = ap.parse_args()

files = sorted(((float(f.stem[2:]), f) for f in (args.case / "transient_points").glob("t_*.npy")))
files = files[::args.every]
if not files:
    raise SystemExit(f"no transient_points/t_*.npy in {args.case}")
times = np.array([t for t, _ in files])
xyz = np.load(files[0][1], mmap_mode="r")[:, :3].astype(np.float64)

x0, x1, y0, y1 = map(float, args.window.split(","))
gx, gy = np.arange(x0, x1 + 1e-9, args.res), np.arange(y0, y1 + 1e-9, args.res)
X, Y = np.meshgrid(gx, gy)
g = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])

# body mask + barycentric weights (computed once; mesh is static)
tree = cKDTree(xyz)
d_nn, j_nn = tree.query(g)
own = tree.query(xyz[j_nn], k=2)[0][:, 1]
fluid = (d_nn <= 2.5 * own).reshape(X.shape)
pad = 0.2
sel = np.flatnonzero((np.abs(xyz[:, 2]) < 0.2) & (xyz[:, 0] > x0 - pad) & (xyz[:, 0] < x1 + pad)
                     & (xyz[:, 1] > y0 - pad) & (xyz[:, 1] < y1 + pad))
tri = Delaunay(xyz[sel])
simp = tri.find_simplex(g)
inside = simp >= 0
Tm = tri.transform[simp[inside]]
bary = np.einsum("nij,nj->ni", Tm[:, :3], g[inside] - Tm[:, 3])
W = np.column_stack([bary, 1 - bary.sum(1)])
V = sel[tri.simplices[simp[inside]]]
inlet = np.flatnonzero(xyz[:, 1] < xyz[:, 1].min() + 0.15)

S = np.empty((len(files), *X.shape, 4), np.float32)
fs = np.empty((len(files), 4))
for n, (t, f) in enumerate(files):
    a = np.load(f)
    vals = a[j_nn, 3:7].astype(np.float64)
    vals[inside] = np.einsum("nk,nkf->nf", W, a[V, 3:7])
    fs[n] = a[inlet, 3:7].mean(0)
    if args.p_rel_inlet:
        vals[:, 3] -= fs[n, 3]
    S[n] = vals.reshape(*X.shape, 4)

print(f"{args.case.name}: {len(files)} snapshots, t = {times[0]:.3f} .. {times[-1]:.3f} s, "
      f"{len(xyz):,} cells")
print("change over time on the slice (fluid pixels):")
for j, f in enumerate(FIELDS):
    v = S[:, fluid, j]
    span = np.percentile(v, 99.5) - np.percentile(v, 0.5)
    dt_max = np.abs(v - v[0]).max(0)
    print(f"  {f}: spatial range {span:9.3f} {UNITS[f]:6s} | max change vs first snapshot: "
          f"median pixel {np.median(dt_max):9.3f}, 95th pct {np.percentile(dt_max, 95):9.3f}")
yaw = np.degrees(np.arctan2(fs[:, 0], fs[:, 1]))
spd = np.linalg.norm(fs[:, :3], axis=1)
print(f"inflow: speed {spd.min():.1f}..{spd.max():.1f} m/s, yaw {yaw.min():.1f}..{yaw.max():.1f} deg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

outdir = args.case / "preview"
outdir.mkdir(exist_ok=True)
lims = [np.percentile(S[:, fluid, j], [0.5, 99.5]) for j in range(4)]
pname = "p − p_inlet" if args.p_rel_inlet else "p"


def render(fields, fname, figsize):
    cols = min(len(fields), 2)
    rows = int(np.ceil(len(fields) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    ims = []
    for ax, j in zip(axes.ravel(), fields):
        f = FIELDS[j]
        cmap = plt.get_cmap("viridis" if f == "v" else "RdBu_r").copy()
        cmap.set_bad("0.75")
        im = ax.imshow(np.ma.masked_where(~fluid, S[0, :, :, j]), origin="lower", extent=(x0, x1, y0, y1),
                       cmap=cmap, vmin=lims[j][0], vmax=lims[j][1])
        label = pname if f == "p" else f
        ax.set_title(f"CFD — {label}"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, label=f"{label} [{UNITS[f]}]", shrink=0.85)
        ims.append((im, j))
    sup = fig.suptitle("")

    def update(n):
        sup.set_text(f"{args.case.name} · z = 0 · t = {times[n] * 1000:.0f} ms · "
                     f"inflow {spd[n]:.1f} m/s, yaw {yaw[n]:.1f}°")
        for im, j in ims:
            im.set_data(np.ma.masked_where(~fluid, S[n, :, :, j]))
        return [im for im, _ in ims] + [sup]

    animation.FuncAnimation(fig, update, frames=len(files)).save(
        outdir / fname, writer=animation.PillowWriter(fps=args.fps), dpi=90)
    plt.close(fig)
    print("wrote", outdir / fname)


for j, f in enumerate(FIELDS):
    render([j], f"{f}.gif", (5.5, 7))
render([0, 1, 2, 3], "all_fields.gif", (10, 13))
