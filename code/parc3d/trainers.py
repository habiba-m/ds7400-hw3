"""Stage 1: autoencoder (any mesh -> tokens -> field). Stage 2: latent propagator (rollout)."""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .core import (FIELDS, autocast, encode_frame, encode_rows, evaluate_frames, field_loss,
                   lr_lambda, mean_nrmse, nrmse, predict, save_ae)
from .model import AutoEncoder, Propagator


def _ln(z):
    # compare token sets after layer norm: the decoder layer-norms its keys, so raw scale is free
    return F.layer_norm(z.float(), z.shape[-1:])


def _write_log(path, rows):
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)


# =========================================================
# Stage 1: autoencoder
# =========================================================
def train_autoencoder(cfg, ds, train_items, val_items, out_dir, steps=None, eval_every=None,
                      monitor_items=None, model=None):
    a = cfg["ae"]
    steps = steps or a["steps"]
    eval_every = eval_every or a["eval_every"]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = ds.device
    torch.manual_seed(cfg["seed"])
    model = model or AutoEncoder(cfg["model"]).to(dev)
    print(f"autoencoder: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a["warmup"], steps))
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    lo, hi = a["n_context"]
    train_runs = {i for i, _ in train_items}
    pairs = [p for p in ds.mesh_pairs if p[0] in train_runs and p[1] in train_runs]
    monitor_items = monitor_items or train_items[:: max(1, len(train_items) // 4)][:4]

    log, best, t0 = [], float("inf"), time.time()
    for step in range(1, steps + 1):
        n_ctx = int(rng.integers(lo, hi + 1))
        batch = [train_items[j] for j in rng.integers(len(train_items), size=a["batch"])]
        ctx = torch.stack([ds.context(i, k, n_ctx, gen) for i, k in batch])
        q = torch.stack([ds.query(i, k, a["n_query"], gen) for i, k in batch])
        fs = torch.stack([ds.fs[i][k] for i, k in batch])
        c = torch.stack([ds.cond[i][k] for i, k in batch])
        with autocast(dev, a["amp"]):
            z = encode_rows(model, ds.norm, ctx, fs, c)
            rec = field_loss(model, ds.norm, z, q, fs)
            # invariance to how the mesh is sampled: same snapshot, different context size
            i0, k0 = batch[0]
            n2 = int(rng.integers(lo, hi + 1))
            z2 = encode_rows(model, ds.norm, ds.context(i0, k0, n2, gen)[None], fs[:1], c[:1])
            inv = F.mse_loss(_ln(z[:1]), _ln(z2))
            # invariance to the mesh itself: same flow and time, different mesh
            mesh = torch.zeros((), device=dev)
            if pairs and a["lambda_mesh"] > 0:
                pi, pj = pairs[int(rng.integers(len(pairs)))]
                kk = int(rng.integers(ds.runs[pi].T))
                zi = encode_rows(model, ds.norm, ds.context(pi, kk, n_ctx, gen)[None],
                                 ds.fs[pi][kk:kk + 1], ds.cond[pi][kk:kk + 1])
                zj = encode_rows(model, ds.norm, ds.context(pj, kk, n_ctx, gen)[None],
                                 ds.fs[pj][kk:kk + 1], ds.cond[pj][kk:kk + 1])
                mesh = F.mse_loss(_ln(zi), _ln(zj))
            loss = rec + a["lambda_invariance"] * inv + a["lambda_mesh"] * mesh
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if not torch.isfinite(loss):
            raise SystemExit(f"loss is {loss.item()} at step {step}: stopping")

        if step == 1 or step % 100 == 0:
            print(f"step {step:6d} | loss {loss.item():.5f} | rec {rec.item():.5f} | inv {inv.item():.5f} | "
                  f"mesh {mesh.item():.5f} | lr {sched.get_last_lr()[0]:.2e} | "
                  f"{(time.time() - t0) / step:.3f} s/step", flush=True)
        if step % eval_every == 0 or step == steps:
            val = evaluate_frames(model, ds, val_items, hi)
            mon = evaluate_frames(model, ds, monitor_items, hi)
            v, m = mean_nrmse(val), mean_nrmse(mon)
            per = np.mean([r["nrmse"] for r in val], 0)
            row = dict(step=step, loss=loss.item(), rec=rec.item(), inv=inv.item(), mesh=mesh.item(),
                       train_nrmse=m, val_nrmse=v, **{f"val_{f}": float(x) for f, x in zip(FIELDS, per)})
            log.append(row)
            print(f"  [eval] step {step}: NRMSE all cells  train {m:.4f}  val {v:.4f}  "
                  f"(val u {per[0]:.3f} v {per[1]:.3f} w {per[2]:.3f} p {per[3]:.3f})", flush=True)
            save_ae(out_dir / "last.pt", model, ds.norm, cfg, step=step, val_nrmse=v)
            if v < best:
                best = v
                save_ae(out_dir / "best.pt", model, ds.norm, cfg, step=step, val_nrmse=v)
                print(f"  saved best.pt (val NRMSE {v:.4f})", flush=True)
            _write_log(out_dir / "log.csv", log)
    print(f"autoencoder done in {(time.time() - t0) / 60:.1f} min, best val NRMSE {best:.4f}", flush=True)
    return model, log


# =========================================================
# Stage 2: latent propagator
# =========================================================
@torch.no_grad()
def encode_all(model, ds, ids, n_ctx, seed=0):
    """Tokens for every snapshot of each run: {i: (T, L, d)} (fixed context seed)."""
    model.eval()
    return {i: torch.cat([encode_frame(model, ds, i, k, n_ctx, seed).float() for k in range(ds.runs[i].T)])
            for i in ids}


def rollout(prop, z0, cond, n_steps=None):
    """z0 (B, L, d); cond (B, T, 6) -> tokens (B, T', L, d), z[:, 0] = z0."""
    n_steps = cond.shape[1] - 1 if n_steps is None else n_steps
    zs, z = [z0], z0
    for h in range(n_steps):
        z = prop(z, cond[:, h], cond[:, h + 1])
        zs.append(z)
    return torch.stack(zs, 1)


@torch.no_grad()
def evaluate_rollout(ae, prop, ds, Z, ids, per_run):
    """Roll out each run from its first snapshot; error on all cells at evenly spaced frames."""
    prop.eval()
    rows = []
    for i in ids:
        r = ds.runs[i]
        zs = rollout(prop, Z[i][:1], ds.cond[i][None])[0]
        for k in np.unique(np.linspace(0, r.T - 1, min(per_run, r.T)).round().astype(int)):
            gt = r.frame(k)
            pred = predict(ae, ds.norm, zs[k:k + 1], gt[:, :3], ds.fs[i][k])
            rows.append(dict(run=r.name, k=int(k), t=float(r.times[k]), nrmse=nrmse(pred, gt[:, 3:7], ds.norm.u_ref)))
    prop.train()
    return rows


def train_propagator(cfg, ds, ae, train_ids, val_ids, out_dir):
    a = cfg["dyn"]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = ds.device
    for p in ae.parameters():
        p.requires_grad_(False)
    ae.eval()
    n_ctx = cfg["ae"]["n_context"][1]
    t0 = time.time()
    Z = encode_all(ae, ds, sorted(set(train_ids) | set(val_ids)), n_ctx)
    print(f"encoded {sum(len(z) for z in Z.values())} snapshots in {time.time() - t0:.0f} s", flush=True)
    z_std = torch.cat([Z[i] for i in train_ids]).std().item()

    torch.manual_seed(cfg["seed"])
    prop = Propagator(cfg["model"]).to(dev)
    print(f"propagator: {sum(p.numel() for p in prop.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(prop.parameters(), lr=a["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a["warmup"], a["steps"]))
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    T_min = min(ds.runs[i].T for i in train_ids)

    log, best, t0 = [], float("inf"), time.time()
    for step in range(1, a["steps"] + 1):
        H = min(a["horizons"][min((step - 1) // a["stage_steps"], len(a["horizons"]) - 1)], T_min - 1)
        batch = []
        for _ in range(a["batch"]):
            i = int(rng.choice(train_ids))
            batch.append((i, int(rng.integers(0, ds.runs[i].T - H))))
        z0 = torch.stack([Z[i][k] for i, k in batch])
        z0 = z0 + a["noise"] * z_std * torch.randn_like(z0)
        cond = torch.stack([ds.cond[i][k:k + H + 1] for i, k in batch])
        target = torch.stack([Z[i][k + 1:k + H + 1] for i, k in batch])
        with autocast(dev, a["amp"]):
            zs = rollout(prop, z0, cond)[:, 1:]                                  # (B, H, L, d)
            lat = F.mse_loss(zs.float(), target) / z_std ** 2
            fl = torch.zeros((), device=dev)
            for h in range(H):
                q = torch.stack([ds.query(i, k + h + 1, a["n_query"], gen) for i, k in batch])
                fs = torch.stack([ds.fs[i][k + h + 1] for i, k in batch])
                fl = fl + field_loss(ae, ds.norm, zs[:, h], q, fs) / H
            loss = lat + a["lambda_field"] * fl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0)
        opt.step(); sched.step()
        if not torch.isfinite(loss):
            raise SystemExit(f"loss is {loss.item()} at step {step}: stopping")

        if step == 1 or step % 100 == 0:
            print(f"step {step:6d} | H {H:3d} | loss {loss.item():.5f} | latent {lat.item():.5f} | "
                  f"field {fl.item():.5f} | {(time.time() - t0) / step:.3f} s/step", flush=True)
        if step % a["eval_every"] == 0 or step == a["steps"]:
            rows = evaluate_rollout(ae, prop, ds, Z, val_ids, cfg["ae"]["eval_frames_per_run"])
            # checkpoint selection on the fields named in dyn.select_fields (default u v w):
            # one badly-behaved field must not pick a checkpoint that is worse on the others
            v = mean_nrmse(rows, a.get("select_fields", (0, 1, 2)))
            per = np.mean([r["nrmse"] for r in rows], 0)
            log.append(dict(step=step, H=H, loss=loss.item(), latent=lat.item(), field=fl.item(),
                            val_rollout_nrmse=v, **{f"val_{f}": float(x) for f, x in zip(FIELDS, per)}))
            print(f"  [eval] step {step}: full-trajectory rollout NRMSE val (selection fields) {v:.4f} "
                  f"(u {per[0]:.3f} v {per[1]:.3f} w {per[2]:.3f} p {per[3]:.3f})", flush=True)
            ck = dict(prop=prop.state_dict(), model_cfg=cfg["model"], config=cfg, step=step, val_nrmse=v)
            torch.save(ck, out_dir / "last.pt")
            if v < best:
                best = v
                torch.save(ck, out_dir / "best.pt")
                print(f"  saved best.pt (val rollout NRMSE {v:.4f})", flush=True)
            _write_log(out_dir / "log.csv", log)
    print(f"propagator done in {(time.time() - t0) / 60:.1f} min, best val rollout NRMSE {best:.4f}", flush=True)
    return prop, log
