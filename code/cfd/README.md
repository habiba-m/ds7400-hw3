# Ellipsoid OpenFOAM datasets (Rivanna)

> **v2:** Dataset A is now 8 meshes. Dataset B is 4 meshes × 24 randomized inflow schedules
> (`config.make_schedules`), with mesh m02 held out. Nothing is reconstructed: the export reads
> `processor*/` directly. Splits are written to `splits.json`. See `../README.md`.
> Some of the text below still describes v1.

**Dataset A:** 8 mesh resolutions, steady `simpleFoam`, identical forcing (34.3 m/s, +y).
**Dataset B:** 4 mesh resolutions × 24 transient schedules, `pimpleFoam`. Each mesh is built
once (kept decomposed) and shared by all its schedules.
Every case is also sampled at **one shared probe set** (~218k fixed physical points), so
mesh independence is measured at identical coordinates.

## Run on Rivanna
1. Copy this `cfd/` folder to Rivanna, e.g. `/scratch/crv4jj/cfd`.
2. Edit `SLURM` in `config.py`: `account`, `partition`, and `modules` (the exact
   OpenFOAM module you used for `ellipsoid_*_u34`, plus a python with numpy and scipy).
3. Generate the cases:
   ```bash
   cd /scratch/crv4jj/cfd && python generate_cases.py --root /scratch/crv4jj/ellipsoid_datasets
   ```
4. Run **one case by hand first** (A_m00, the coarsest) to catch dictionary or version problems:
   ```bash
   cd /scratch/crv4jj/ellipsoid_datasets && sbatch --array=0 run_A.slurm
   ```
   Check `A/A_m00/log.*`, then submit everything:
   ```bash
   /scratch/crv4jj/ellipsoid_datasets/submit_all.sh
   ```
5. Validate once the runs finish:
   ```bash
   python /scratch/crv4jj/cfd/validate_datasets.py --root /scratch/crv4jj/ellipsoid_datasets
   ```
   This writes `validation/report.md`, summary CSVs and plots.

Re-submitting is safe: finished cases (`DONE` file) are skipped.

## Outputs (`<root>/exports/<case>/`)
- **A:** `<case>_points.npy` (N, 7: x y z u v w p, same layout as your existing
  `*_points.npy` files) and `probes.npy` (P, 4).
- **B:** `transient_points/t_<time>.npy` (N, 7 per snapshot, same layout as the older
  `phaseB_transient` data, so `train_transient_prelim.py --root <root>/exports` reads it),
  `xyz.npy` (N, 3), `times.npy`, `freestream.npy`
  (T, 4: the time-varying FREESTREAM for residual targets), `probes.npy` (T, P, 4).

## Assumptions to check against your existing u34 cases
- OpenFOAM **ESI** (openfoam.com, v2112+), not the Foundation version.
- kOmegaSST RANS/URANS, wall functions, ν = 1.5e-5, 1% inlet turbulence.
- Domain = your BOUNDS, x[-2,2] y[-3,6] z[-2,2].
- Far-field sides use `inletOutlet` with p `zeroGradient`, so yawed inflow can enter and leave.
- Mesh family: background cell size h (A: 0.20→0.08 m, B: 0.20→0.11 m),
  with fixed refinement levels (surface h/32, near region h/8, wake h/4, 3 layers).
  Estimated size: ~0.27M–3.4M cells.
- B timing: 0.25 s spin-up at constant flow, then forcing until 0.75 s,
  with a snapshot every 5 ms from 0.20 s (111 snapshots).

## Conditions (B)
`speed_ramp` 34.3→45 m/s · `speed_sine` ±6 m/s @ 0.2 s · `yaw_step` 0→15° ·
`yaw_sine` ±10° @ 0.25 s · `gust` 1−cos, +10 m/s over 0.1 s.
