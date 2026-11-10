"""z = 0 slice: CFD linearly interpolated from cell centres, prediction decoded at the pixels.
Rows p, v, u, w; columns prediction | CFD; one colour scale per row shared by both columns
and fixed over time. w ~ 0 on the symmetry plane, so it uses the u range (an auto range
would only magnify solver noise)."""
import numpy as np
from scipy.spatial import Delaunay, cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

ROWS = (3, 1, 0, 2)                                       # p, v, u, w
NAMES = ("u", "v", "w", "p")
UNITS = ("m/s", "m/s", "m/s", "m²/s²")


class Slice:
    def __init__(self, xyz, window=(-1.0, 1.0, -0.8, 3.0), res=0.005):
        x0, x1, y0, y1 = window
        X, Y = np.meshgrid(np.arange(x0, x1 + 1e-9, res), np.arange(y0, y1 + 1e-9, res))
        self.extent, self.shape = (x0, x1, y0, y1), X.shape
        self.pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]).astype(np.float32)
        xyz = np.asarray(xyz, np.float64)
        tree = cKDTree(xyz)
        d_nn, self.j_nn = tree.query(self.pts)
        own = tree.query(xyz[self.j_nn], k=2)[0][:, 1]
        self.fluid = (d_nn <= 2.5 * own).reshape(self.shape)          # body pixels -> grey
        pad = 0.2
        sel = np.flatnonzero((np.abs(xyz[:, 2]) < 0.2) & (xyz[:, 0] > x0 - pad) & (xyz[:, 0] < x1 + pad)
                             & (xyz[:, 1] > y0 - pad) & (xyz[:, 1] < y1 + pad))
        tri = Delaunay(xyz[sel])
        simp = tri.find_simplex(self.pts.astype(np.float64))
        self.inside = simp >= 0
        Tm = tri.transform[simp[self.inside]]
        bary = np.einsum("nij,nj->ni", Tm[:, :3], self.pts[self.inside] - Tm[:, 3])
        self.W = np.column_stack([bary, 1 - bary.sum(1)]).astype(np.float32)
        self.V = sel[tri.simplices[simp[self.inside]]]

    def cfd(self, frame):
        vals = frame[self.j_nn, 3:7].copy()
        vals[self.inside] = np.einsum("nk,nkf->nf", self.W, frame[self.V, 3:7])
        return vals.reshape(*self.shape, 4)

    def grid(self, vals):
        return np.asarray(vals).reshape(*self.shape, 4)


def _limits(gt, fluid):
    """gt (..., ny, nx, 4) -> per-field (lo, hi), w symmetric with the u range."""
    lims = [np.percentile(gt[..., j][..., fluid], [0.5, 99.5]) for j in range(4)]
    m = np.abs(lims[0]).max()
    lims[2] = np.array([-m, m])
    return lims


def _figure(sl, lims, first_pred, first_gt, figsize):
    fig, axes = plt.subplots(4, 2, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.3)                                    # before the colour bars
    ims = []
    for r, j in enumerate(ROWS):
        cmap = plt.get_cmap("viridis" if NAMES[j] == "v" else "RdBu_r").copy()
        cmap.set_bad("0.75")
        for col, (src, title) in enumerate(((first_pred, "Prediction"), (first_gt, "CFD"))):
            ax = axes[r, col]
            im = ax.imshow(np.ma.masked_where(~sl.fluid, src[..., j]), origin="lower", extent=sl.extent,
                           cmap=cmap, vmin=lims[j][0], vmax=lims[j][1], interpolation="nearest")
            ax.set_title(f"{title} — {NAMES[j]}", fontsize=11)
            ax.set_ylabel("y [m]")
            if r == 3:
                ax.set_xlabel("x [m]")
            ims.append((im, col, j))
        fig.colorbar(im, ax=axes[r, :].tolist(), label=f"{NAMES[j]} [{UNITS[j]}]", shrink=0.9)
    return fig, ims


def save_png(path, sl, pred, gt, title):
    fig, _ = _figure(sl, _limits(gt, sl.fluid), pred, gt, (9, 15))
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_gif(path, sl, preds, gts, titles, fps=8):
    """preds, gts (n_frames, ny, nx, 4)."""
    fig, ims = _figure(sl, _limits(gts, sl.fluid), preds[0], gts[0], (9, 15))
    sup = fig.suptitle(titles[0], fontsize=11)

    def update(n):
        for im, col, j in ims:
            im.set_data(np.ma.masked_where(~sl.fluid, (preds if col == 0 else gts)[n][..., j]))
        sup.set_text(titles[n])
        return [im for im, _, _ in ims] + [sup]

    animation.FuncAnimation(fig, update, frames=len(preds), blit=False).save(
        path, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
