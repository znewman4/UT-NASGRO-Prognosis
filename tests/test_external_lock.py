"""M0: the validated pipeline is pinned and read-only."""
import shutil

import pytest

from utprog import _external


def test_pinned_pipeline_matches_lock():
    lk = _external.verify()
    assert lk["commit"] == "7bf96977d8557badd49c7f08607d648605caaeaf"


def test_imported_modules_are_the_verified_copy(ns):
    for m in _external.MODULES:
        assert str(ns.path) in getattr(ns, m).__file__


def test_edited_copy_is_refused(tmp_path):
    src = _external.location()
    dst = tmp_path / "nasgro_sa"
    shutil.copytree(src / "nasgro_scenarios", dst / "nasgro_scenarios")
    shutil.copytree(src / "results", dst / "results")
    _external.verify(dst)
    f = dst / "nasgro_scenarios" / "nasgro.py"
    f.write_text(f.read_text().replace("R_MIN, R_MAX = -2.0, 0.99", "R_MIN, R_MAX = -2.0, 0.95"))
    with pytest.raises(_external.IntegrityError):
        _external.verify(dst)
