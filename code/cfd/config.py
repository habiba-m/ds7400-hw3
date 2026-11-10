"""Single source of truth for Datasets A and B.

Geometry, domain, freestream and region definitions mirror
train_final_mesh_independent_siren.py (A, B, C semi-axes; BOUNDS; U_REF;
flow along +y; "aoa" rotates the freestream in the x-y plane).

Everything marked ASSUMPTION should be checked against the existing
ellipsoid_*_u34 OpenFOAM cases so Datasets A/B stay physically consistent
with the data the original model was trained on.
"""
import numpy as np

# --------------------------------------------------------------------------- #
# Geometry (body fixed at the origin in every case)
# --------------------------------------------------------------------------- #
SEMI_AXES = (0.20, 0.50, 0.15)        # a (x), b (y, streamwise), c (z)  [m]
STL_SUBDIV = 6                        # icosphere level -> 81,920 facets (same STL for all meshes)

# Computational domain. ASSUMPTION: equal to the model's BOUNDS.
DOMAIN = np.array([[-2.0, 2.0],       # x
                   [-3.0, 6.0],       # y  (inlet at y_min, outlet at y_max)
                   [-2.0, 2.0]])      # z
BOUNDS = DOMAIN.copy()                # coordinate normalization used by the model

# --------------------------------------------------------------------------- #
# Flow (incompressible, kinematic pressure p/rho [m^2/s^2])
# --------------------------------------------------------------------------- #
U_REF = 34.3                          # m/s, +y
NU = 1.5e-5                           # m^2/s air. ASSUMPTION
TURB_INTENSITY = 0.01                 # inlet I. ASSUMPTION
VISC_RATIO = 10.0                     # inlet nu_t/nu. ASSUMPTION
TURBULENCE_MODEL = "kOmegaSST"        # ASSUMPTION (RANS / URANS)


def freestream(speed, yaw_deg):
    """(u, v, w) for speed along +y rotated by yaw in the x-y plane.
    yaw=15 -> (8.877, 33.131, 0), matching FREESTREAM['fine15']."""
    t = np.radians(yaw_deg)
    return np.array([speed * np.sin(t), speed * np.cos(t), 0.0])


# --------------------------------------------------------------------------- #
# Mesh-resolution sweep
#
# One scalar per mesh: background cell size h. Refinement *levels* are held
# fixed, so every region scales with h and the meshes form a consistent
# family. Successive h values are not integer ratios -> meshes are not nested
# (cell centres do not coincide), which is what a mesh-independence test needs.
# --------------------------------------------------------------------------- #
REFINEMENT = dict(
    surface_level=(5, 5),             # ellipsoid surface cells = h / 32
    near_level=3,                     # 1 <= rho <= 3 region (model's "near") = h / 8
    wake_level=2,                     # wake box = h / 4
    wake_box=((-1.2, 0.0, -0.6), (1.2, 2.5, 0.6)),  # widened in x to cover +-15 deg yaw
    n_cells_between_levels=3,
    n_layers=3,
    layer_expansion=1.2,
    final_layer_thickness=0.4,        # relative to surface cell size
)

DATASET_A = dict(
    n_meshes=8,                       # enough to span the convergence range (v1 had 50)
    h_coarse=0.20,                    # ~0.35 M cells (pilot B_m00 had 354k)
    h_fine=0.08,                      # ~3-5 M cells (rough estimate)
    speed=U_REF, yaw_deg=0.0,         # identical forcing for every mesh
    steady=True,
    n_iter=3000,
    write_interval=3000,
)

DATASET_B = dict(
    n_meshes=4,
    h_coarse=0.20,                    # ~0.35 M cells
    h_fine=0.13,                      # ~1.3 M cells (transient storage scales with this)
    holdout_mesh=2,                   # never trained on: interpolation test of mesh independence
    t_spinup=0.25,                    # s, constant forcing so the wake washes out of the domain once
    t_end=0.75,
    write_dt=0.005,                   # snapshot spacing [s]
    export_from=0.20,                 # first exported snapshot time [s]
    max_co=2.0,
    table_dt=1e-3,                    # resolution of the inlet time tables
    schedules_per_family=4,
    schedule_seed=20261030,
)


# --------------------------------------------------------------------------- #
# Transient inflow schedules for Dataset B
#
# Six families, each with randomized parameters drawn from a fixed seed. Every schedule
# returns (speed [m/s], yaw [deg]) as a function of tau = t - t_spinup, starts exactly at
# (U_REF, 0) and is C1-smooth (oscillations fade in), so the inlet never injects a
# numerical shock. Limits: yaw within +-15 deg (wake box), speed 25-47 m/s.
# The forcing window is 0.5 s; the inlet-to-body transport delay is ~0.09 s.
# --------------------------------------------------------------------------- #
FORCING_WINDOW = 0.5


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _bump(x):
    """1 - cos pulse on 0 <= x <= 1, zero outside (C1)."""
    return np.where((x > 0) & (x < 1), 0.5 * (1 - np.cos(2 * np.pi * np.clip(x, 0, 1))), 0.0)


def _ramp(r):
    t0, T, ds, dy = r.uniform(0, 0.12), r.uniform(0.08, 0.25), r.choice([-1, 1]) * r.uniform(5, 9), \
        r.choice([0.0, r.uniform(-8, 8)])
    f = lambda tau: (U_REF + ds * _smoothstep((tau - t0) / T), dy * _smoothstep((tau - t0) / T))
    return f, dict(t0=t0, T=T, d_speed=ds, d_yaw=dy)


def _speed_sine(r):
    A, P = r.choice([-1, 1]) * r.uniform(3, 8), r.uniform(0.08, 0.3)
    f = lambda tau: (U_REF + A * _smoothstep(tau / 0.05) * np.sin(2 * np.pi * tau / P), 0.0 * tau)
    return f, dict(amp=A, period=P)


def _yaw_step(r):
    t0, T, y1 = r.uniform(0, 0.15), r.uniform(0.04, 0.15), r.choice([-1, 1]) * r.uniform(5, 15)
    f = lambda tau: (U_REF + 0.0 * tau, y1 * _smoothstep((tau - t0) / T))
    return f, dict(t0=t0, T=T, yaw=y1)


def _yaw_sine(r):
    A, P = r.choice([-1, 1]) * r.uniform(4, 12), r.uniform(0.12, 0.35)
    f = lambda tau: (U_REF + 0.0 * tau, A * _smoothstep(tau / 0.05) * np.sin(2 * np.pi * tau / P))
    return f, dict(amp=A, period=P)


def _gust(r):
    n = int(r.integers(1, 3))
    t0 = np.sort(r.uniform(0, 0.3, n)); T = r.uniform(0.05, 0.15, n)
    A = r.uniform(4, 12, n) * r.choice([-1, 1], n); Ay = r.uniform(-6, 6, n) * r.integers(0, 2, n)
    f = lambda tau: (U_REF + sum(A[i] * _bump((tau - t0[i]) / T[i]) for i in range(n)),
                     sum(Ay[i] * _bump((tau - t0[i]) / T[i]) for i in range(n)))
    return f, dict(t0=t0.tolist(), T=T.tolist(), d_speed=A.tolist(), d_yaw=Ay.tolist())


def _combined(r):
    T, ds = r.uniform(0.1, 0.3), r.choice([-1, 1]) * r.uniform(4, 9)
    A, P = r.uniform(4, 10), r.uniform(0.12, 0.35)
    f = lambda tau: (U_REF + ds * _smoothstep(tau / T),
                     A * _smoothstep(tau / 0.05) * np.sin(2 * np.pi * tau / P))
    return f, dict(T=T, d_speed=ds, yaw_amp=A, yaw_period=P)


FAMILIES = {"ramp": _ramp, "speedsine": _speed_sine, "yawstep": _yaw_step,
            "yawsine": _yaw_sine, "gust": _gust, "combined": _combined}


def make_schedules():
    """{name: dict(fn, family, params, split)} -- deterministic from schedule_seed.
    Per family: draws 0-2 -> train; draw 3 -> val for even family index, test for odd
    (18 train / 3 val / 3 test schedules)."""
    b = DATASET_B
    rng = np.random.default_rng(b["schedule_seed"])
    out = {}
    for fi, (fam, maker) in enumerate(FAMILIES.items()):
        for j in range(b["schedules_per_family"]):
            fn, params = maker(rng)
            split = "train" if j < b["schedules_per_family"] - 1 else ("val" if fi % 2 == 0 else "test")
            out[f"s{len(out):02d}_{fam}"] = dict(fn=fn, family=fam, split=split,
                                               params={k: (np.round(v, 4).tolist() if hasattr(v, "__len__")
                                                           else round(float(v), 4)) for k, v in params.items()})
    return out


# --------------------------------------------------------------------------- #
# Common probe set: identical physical coordinates for every mesh and case
# --------------------------------------------------------------------------- #
PROBES = dict(
    n_random=200_000,
    fractions=(0.30, 0.30, 0.40),     # near / wake / rest, as in balanced_indices()
    rho_min=1.02,                     # keep probes just off the wall
    plane_z0=dict(x=(-1.0, 1.0), y=(-0.8, 3.0), spacing=0.02),  # z=0 plane for plots
    seed=20261026,
)

# --------------------------------------------------------------------------- #
# Cluster
# --------------------------------------------------------------------------- #
SLURM = dict(
    account="sds_baek_energetic",
    partition="standard",
    n_procs=40,
    time_A="12:00:00",
    time_B="24:00:00",
    mem_per_cpu="4G",
    modules=["gcc/14.2.0", "openmpi/5.0.7", "openfoam/v2606"],   # Rivanna, in load order
    setup=['source "$EBROOTOPENFOAM/OpenFOAM-v2606/etc/bashrc"'],  # what the openfoam module asks for    # shell lines run after the module loads (OpenFOAM environment)
    python_module="miniforge",        # FILL IN: any python with numpy + scipy
)


def mesh_sizes(spec):
    """Log-spaced background cell sizes, index 0 = coarsest."""
    return np.geomspace(spec["h_coarse"], spec["h_fine"], spec["n_meshes"])
