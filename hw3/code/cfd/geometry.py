"""Ellipsoid STL surfaces and the shared probe set."""
import struct

import numpy as np

import config as cfg


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
         (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
         (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(level):
        cache, new = {}, []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = verts[i] + verts[j]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return np.array(verts), np.array(faces)


def ellipsoid_mesh(axes, level):
    """Triangulated ellipsoid with outward-facing (counter-clockwise) facets."""
    v, f = icosphere(level)
    v = v * np.asarray(axes)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = (n * tri.mean(1)).sum(1) < 0
    f[flip] = f[flip][:, ::-1]
    return v, f


def write_binary_stl(path, verts, faces, name="ellipsoid"):
    tri = verts[faces].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    with open(path, "wb") as fh:
        fh.write(name.encode().ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(faces)))
        rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, tri
        fh.write(rec.tobytes())


def rho(xyz):
    a, b, c = cfg.SEMI_AXES
    return np.sqrt((xyz[:, 0] / a) ** 2 + (xyz[:, 1] / b) ** 2 + (xyz[:, 2] / c) ** 2)


def region_masks(xyz):
    """Same definition as region_masks() in the training script."""
    x, y, z = xyz.T
    r = rho(xyz)
    near = (r >= 1.0) & (r <= 3.0)
    wake = (y >= 0.4) & (y <= 2.5) & (np.abs(x) <= 0.8) & (np.abs(z) <= 0.6) & ~near
    return near, wake, ~(near | wake)


def make_probes():
    """Fixed probe coordinates shared by every mesh and every case.

    Returns (P, 3) float64 and a dict of index groups (near/wake/rest/plane).
    """
    p = cfg.PROBES
    rng = np.random.default_rng(p["seed"])
    lo, hi = cfg.DOMAIN[:, 0], cfg.DOMAIN[:, 1]
    targets = [int(f * p["n_random"]) for f in p["fractions"]]
    targets[-1] = p["n_random"] - sum(targets[:-1])

    buckets = [[], [], []]
    a = np.array(cfg.SEMI_AXES)
    while any(sum(map(len, b)) < t for b, t in zip(buckets, targets)):
        # mix uniform-in-box and near-body samples so every bucket fills quickly
        u = rng.uniform(lo, hi, size=(400_000, 3))
        d = rng.normal(size=(200_000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        nb = d * a * rng.uniform(1.0, 3.0, size=(200_000, 1))
        cand = np.concatenate([u, nb])
        cand = cand[(rho(cand) >= p["rho_min"]) & np.all((cand > lo) & (cand < hi), 1)]
        for k, m in enumerate(region_masks(cand)):
            buckets[k].append(cand[m])
    groups, pts = {}, []
    for k, name in enumerate(("near", "wake", "rest")):
        arr = np.concatenate(buckets[k])[: targets[k]]
        groups[name] = np.arange(len(arr)) + sum(map(len, pts))
        pts.append(arr)

    pl = p["plane_z0"]
    gx = np.arange(pl["x"][0], pl["x"][1] + 1e-9, pl["spacing"])
    gy = np.arange(pl["y"][0], pl["y"][1] + 1e-9, pl["spacing"])
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    plane = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    plane = plane[rho(plane) >= p["rho_min"]]
    groups["plane_z0"] = np.arange(len(plane)) + sum(map(len, pts))
    pts.append(plane)
    return np.concatenate(pts), groups
