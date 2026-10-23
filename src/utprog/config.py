"""YAML experiment configs: inheritance, variants and a stable hash.

`sweeps` maps dotted paths to lists of values (e.g. "sensor_common.corr_frac":
[0, 0.02, 0.05]); a value may be a dict to move several settings together (e.g.
POD a50 and a90). Every combination is crossed with every variant, so results
come out as conditions rather than single numbers.

A config may name a parent with `extends: other.yaml`; dictionaries merge
recursively and the child wins. `variants` is not merged: a child that sets it
replaces the parent's. Each variant is the base config deep-merged with that
variant's overrides and runs as its own result directory.

Write floats with a decimal point in YAML (1.0e-11, not 1e-11): PyYAML reads
the latter as a string.
"""
import copy
import itertools
import hashlib
import json
from pathlib import Path

import yaml


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load(path.parent / parent)
        if "variants" in cfg:
            base.pop("variants", None)
        cfg = deep_merge(base, cfg)
    return cfg


def _nest(path, value):
    out = value
    for k in reversed(path.split(".")):
        out = {k: out}
    return out


def _label(path, value):
    if isinstance(value, dict):                  # a dict sweeps several values together
        return ",".join(f"{k}={v}" for k, v in value.items())
    return f"{path.split('.')[-1]}={value}"


def variants(cfg):
    """[(name, resolved config)]: the base merged with each variant (or the base alone, 'base'),
    crossed with every combination of `sweeps`."""
    vs = cfg.get("variants") or {"base": {}}
    sweeps = cfg.get("sweeps") or {}
    base = {k: v for k, v in cfg.items() if k not in ("variants", "sweeps")}
    combos = [()] if not sweeps else list(itertools.product(*[[(k, v) for v in vals] for k, vals in sweeps.items()]))
    out = []
    for name, over in vs.items():
        for combo in combos:
            c = deep_merge(base, over or {})
            for k, v in combo:
                c = deep_merge(c, _nest(k, v))
            vname = name + "".join(f"|{_label(k, v)}" for k, v in combo)
            if vname != "base" or cfg.get("variants"):
                c["variant"] = vname
            out.append((vname, c))
    return out


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
