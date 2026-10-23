"""Run provenance. A result without a manifest is not used in a report.

The manifest records the resolved config and its hash, the git commit (and
whether the tree was dirty), the pinned NASGRO pipeline commit, package
versions, the root seed, the date and the runtime.
"""
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from . import __version__
from ._external import ROOT, lock
from .config import config_hash

PACKAGES = ("numpy", "scipy", "pandas", "pyyaml", "matplotlib")


def git_info():
    def git(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return dict(commit=commit or "unknown", dirty=bool(status) if status is not None else None)


def versions():
    out = dict(python=platform.python_version(), utprog=__version__)
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def write(run_dir, cfg, started, extra=None):
    lk = lock()
    m = dict(
        config_hash=config_hash(cfg), config=cfg, git=git_info(),
        nasgro_sa=dict(repo=lk["repo"], commit=lk["commit"], verified=True),
        versions=versions(), seed=cfg.get("seed"),
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        runtime_s=round(time.time() - started, 1), **(extra or {}))
    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(m, indent=1, default=_default))
    return path


def require(run_dir):
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no manifest.json; results without one are not used in reports")
    return json.loads(path.read_text())
