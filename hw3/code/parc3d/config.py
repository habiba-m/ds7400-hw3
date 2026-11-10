"""One JSON config per experiment. Overrides: --set ae.steps=100 data.cases='["B_m00_gust"]'."""
import argparse
import copy
import json
from pathlib import Path


def load(path, overrides=()):
    cfg = json.loads(Path(path).read_text())
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw                                   # bare strings
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            if p not in node:
                raise SystemExit(f"--set {key}: no section {p!r} in config")
            node = node[p]
        if leaf not in node:
            raise SystemExit(f"--set {key}: no key {leaf!r} in config (typo?)")
        node[leaf] = val
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return ap


def save(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(copy.deepcopy(cfg), indent=1))
