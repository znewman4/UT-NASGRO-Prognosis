"""The validated NASGRO pipeline: located, hash-checked and imported read-only.

external/nasgro_sa is a git submodule of znewman4/NASGRO-Sensitivity-Analysis at
the commit pinned in external/nasgro_sa.lock.json. Its package `nasgro_scenarios`
is the rebuilt `nasgro_sa` pipeline (see that repository's README). Every locked
file is checked against its SHA-256 before the package is imported, so an edited
or different copy fails loudly instead of feeding results. Set UTPROG_NASGRO_SA
to use a checkout elsewhere; it is held to the same hashes.

Nothing in this repository writes to external/.
"""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = ROOT / "external" / "nasgro_sa.lock.json"
ENV_VAR = "UTPROG_NASGRO_SA"
MODULES = ("cases", "geometry", "life", "nasgro", "residual_stress")


class IntegrityError(RuntimeError):
    """The external pipeline differs from the pinned, validated version."""


def lock():
    return json.loads(LOCK_FILE.read_text())


def location():
    return Path(os.environ.get(ENV_VAR, ROOT / "external" / "nasgro_sa"))


def file_hash(path):
    """SHA-256 of a file with CRLF normalised to LF (so a Windows checkout still matches)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify(path=None):
    """Check every locked file under `path`; return the lock or raise IntegrityError."""
    path = Path(path) if path is not None else location()
    lk = lock()
    bad = {}
    for rel, digest in lk["sha256"].items():
        f = path / rel
        if not f.is_file():
            bad[rel] = "missing"
        elif file_hash(f) != digest:
            bad[rel] = "hash mismatch"
    if bad:
        raise IntegrityError(
            f"{path} does not match the validated pipeline at commit {lk['commit']}: {bad}. "
            "Run `git submodule update --init` and do not edit external/.")
    return lk


_NS = None


def nasgro():
    """The verified pipeline's modules (cases, geometry, life, nasgro, residual_stress) and its lock."""
    global _NS
    if _NS is None:
        path = location()
        lk = verify(path)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        pkg = lk["package"]
        mods = {m: importlib.import_module(f"{pkg}.{m}") for m in MODULES}
        for m in mods.values():
            if Path(m.__file__).resolve().parent != (path / pkg).resolve():
                raise IntegrityError(f"{m.__name__} was imported from {m.__file__}, not from the verified copy in {path}")
        _NS = SimpleNamespace(path=path, lock=lk, **mods)
    return _NS
