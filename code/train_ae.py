#!/usr/bin/env python3
"""Stage 1: train the autoencoder (any mesh -> latent tokens -> field at any point).

    python train_ae.py --config configs/pilot.json
Writes <out>/ae/{best.pt, last.pt, log.csv, config.json}. best.pt = lowest val NRMSE on all
cells of the val runs (never saved from an empty split: the data loader refuses one).
"""
from pathlib import Path

import torch

from parc3d import config
from parc3d.core import eval_items
from parc3d.data import Dataset
from parc3d.trainers import train_autoencoder

args = config.parser(__doc__).parse_args()
cfg = config.load(args.config, args.set)
out = Path(cfg["out"]) / "ae"
config.save(cfg, out / "config.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

ds = Dataset(cfg, device)
for s in ("train", "val", "test"):
    print(f"{s:5s}:", [ds.runs[i].name for i in ds.splits[s]], flush=True)
train_items = [(i, k) for i in ds.splits["train"] for k in range(ds.runs[i].T)]
per = cfg["ae"]["eval_frames_per_run"]
train_autoencoder(cfg, ds, train_items, eval_items(ds, ds.splits["val"], per), out,
                  monitor_items=eval_items(ds, ds.splits["train"], 2))
