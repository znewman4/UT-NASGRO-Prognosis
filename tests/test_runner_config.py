"""Runner: common random numbers across arms, reproducibility, config handling, manifest."""
import json

import numpy as np

from test_coverage import toy_cfg
from utprog import config, manifest, runner


def small_cfg():
    cfg = toy_cfg(3)
    cfg["policies"]["continuous"] = dict(kind="continuous", dt_frac=0.05)
    cfg["arms"] = dict(list=dict(a=dict(sensor="sd02", policy="periodic", estimator="grid"),
                                 b=dict(sensor="sd02", policy="periodic", estimator="grid"),
                                 c=dict(sensor="sd02", policy="continuous", estimator="grid")))
    cfg["scoring"].update(n_random=2, fixed_fractions=[0.5], sbc_draws=0, n_samples=500)
    return cfg


def test_identical_arms_give_identical_records():
    runner._init(small_cfg())
    _, rec, _ = runner.run_specimen(0)
    a = [r for r in rec if r["arm"] == "a"]
    b = [r for r in rec if r["arm"] == "b"]
    for ra, rb in zip(a, b):
        assert {k: v for k, v in ra.items() if k != "arm"} == {k: v for k, v in rb.items() if k != "arm"}
    c = [r for r in rec if r["arm"] == "c"]
    assert [r["N"] for r in c] == [r["N"] for r in a]          # same forecast times for every arm


def test_specimen_results_do_not_depend_on_specimen_count():
    cfg = small_cfg()
    runner._init(cfg)
    first = runner.run_specimen(2)
    cfg["n_specimens"] = 50
    runner._init(cfg)
    again = runner.run_specimen(2)
    assert first[0] == again[0]
    np.testing.assert_allclose([r["q50"] for r in first[1]], [r["q50"] for r in again[1]])


def test_run_writes_manifest_and_summary(tmp_path):
    out = runner.run(small_cfg(), tmp_path / "run")
    m = manifest.require(out)
    assert m["config_hash"] == config.config_hash(small_cfg())
    assert m["nasgro_sa"]["commit"].startswith("28f217ce")
    assert (out / "summary.csv").is_file()


def test_config_inheritance_and_variants(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nvariants: {v1: {a: {x: 5}}}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na: {y: 3}\nvariants: {w: {}}\n")
    cfg = config.load(tmp_path / "child.yaml")
    assert cfg["a"] == {"x": 1, "y": 3}
    vs = dict(config.variants(cfg))
    assert list(vs) == ["w"] and vs["w"]["variant"] == "w"
    assert config.config_hash({"p": 1, "q": 2}) == config.config_hash(json.loads('{"q": 2, "p": 1}'))
