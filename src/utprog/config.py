"""YAML experiment configs: inheritance, variants and a stable hash.

A config may name a parent with `extends: other.yaml`; dictionaries merge
recursively and the child wins. `variants` is not merged: a child that sets it
replaces the parent's. Each variant is the base config deep-merged with that
variant's overrides and runs as its own result directory.

Write floats with a decimal point in YAML (1.0e-11, not 1e-11): PyYAML reads
the latter as a string.
"""
import copy
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


def variants(cfg):
    """[(name, resolved config)]: the base merged with each variant, or the base alone as 'base'."""
    vs = cfg.get("variants") or {}
    base = {k: v for k, v in cfg.items() if k != "variants"}
    if not vs:
        return [("base", base)]
    return [(name, dict(deep_merge(base, over or {}), variant=name)) for name, over in vs.items()]


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
