"""Encode / decode / loss / metrics shared by training, sanity gates and evaluation.
Training and plotting go through the same functions, so they cannot disagree."""
import contextlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import Normalizer
from .model import AutoEncoder, Propagator

FIELDS = ("u", "v", "w", "p")


def autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def encode_rows(model, norm, rows, fs, cond):
    """rows (B, n, 7) physical x y z u v w p; fs (B, 3) inflow; cond (B, 6) -> tokens (B, L, d)."""
    return model.encoder(norm.xyz(rows[..., :3]), norm.fields(rows[..., 3:7], fs[:, None]), cond)


def field_loss(model, norm, z, q_rows, fs):
    """MSE in normalized residual units at query rows (B, Q, 7)."""
    pred = model.decoder(z, norm.xyz(q_rows[..., :3]))
    return F.mse_loss(pred.float(), norm.fields(q_rows[..., 3:7], fs[:, None]))


def encode_frame(model, ds, i, k, n_ctx, seed=0):
    gen = torch.Generator(device=ds.device).manual_seed(seed)
    rows = ds.context(i, k, n_ctx, gen)[None]
    return encode_rows(model, ds.norm, rows, ds.fs[i][k:k + 1], ds.cond[i][k:k + 1])


@torch.no_grad()
def predict(model, norm, z, xyz, fs, chunk=65536):
    """Physical u v w p at any points xyz (N, 3) from tokens z (1, L, d); fs (3,) inflow."""
    xyz = torch.as_tensor(np.asarray(xyz, np.float32), device=z.device)
    out = []
    for s in range(0, len(xyz), chunk):
        r = model.decoder(z, norm.xyz(xyz[s:s + chunk])[None])[0]
        out.append(norm.unfields(r.float(), fs).cpu().numpy())
    return np.concatenate(out)


def nrmse(pred, gt, u_ref):
    """Per channel RMSE / std(ground truth); std floored at 1 % of the reference scale so a
    field that is ~0 (w on a symmetry plane) does not explode."""
    m = np.isfinite(gt).all(1)
    d = pred[m] - gt[m]
    floor = 0.01 * np.array([u_ref] * 3 + [0.5 * u_ref ** 2])
    return np.sqrt((d ** 2).mean(0)) / np.maximum(gt[m].std(0), floor)


@torch.no_grad()
def evaluate_frames(model, ds, items, n_ctx, seed=0):
    """Reconstruction error on ALL cells of the native mesh for (run, snapshot) items."""
    was = model.training
    model.eval()
    rows = []
    for i, k in items:
        gt = ds.runs[i].frame(k)
        z = encode_frame(model, ds, i, k, n_ctx, seed)
        pred = predict(model, ds.norm, z, gt[:, :3], ds.fs[i][k])
        rows.append(dict(run=ds.runs[i].name, k=int(k), t=float(ds.runs[i].times[k]),
                         nrmse=nrmse(pred, gt[:, 3:7], ds.norm.u_ref)))
    model.train(was)
    return rows


def eval_items(ds, ids, per_run):
    out = []
    for i in ids:
        T = ds.runs[i].T
        out += [(i, int(k)) for k in np.unique(np.linspace(0, T - 1, min(per_run, T)).round().astype(int))]
    return out


def lr_lambda(warmup, total, floor=0.05):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return f


def save_ae(path, model, norm, cfg, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), norm=norm.state_dict(), model_cfg=cfg["model"],
                    config=cfg, **extra), path)


def load_ae(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = AutoEncoder(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Normalizer.from_state(ck["norm"]), ck


def load_prop(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    prop = Propagator(ck["model_cfg"]).to(device)
    prop.load_state_dict(ck["prop"])
    prop.eval()
    return prop, ck


def mean_nrmse(rows, fields=(0, 1, 2, 3)):
    return float(np.mean([r["nrmse"][list(fields)].mean() for r in rows])) if rows else float("nan")
