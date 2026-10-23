"""Sensors give common random numbers across arms; policies give the schedules they claim."""
import numpy as np
import pytest

from utprog import policies
from utprog.estimators.likelihood import NoiseModel
from utprog.rng import stream
from utprog.sensors.depth import DepthSensor, ou_field
from utprog.truth.specimen import Specimen

CTX = dict(N_nominal=1e5, grid=100.0, horizon=1e6)


def spec(life=1e5):
    return Specimen(index=0, case="toy", a0=1e-3, life=life, a_fail=1e-2, truth_theta={},
                    depth_fn=lambda N: 1e-3 * (1.0 + np.asarray(N) / 1e5))


def values(obs):
    return np.array([o.value for o in obs])


def test_same_seed_same_errors_scaled_by_sd():
    s, t = spec(), np.arange(0, 9e4, 1e4)
    o1 = DepthSensor(NoiseModel(sd_abs=1e-4), 100.0, 1e6).observe(s, t, stream(1, 0, "sensor"))
    o2 = DepthSensor(NoiseModel(sd_abs=2e-4), 100.0, 1e6).observe(s, t, stream(1, 0, "sensor"))
    a = s.depth(t)
    np.testing.assert_allclose(values(o2) - a, 2.0 * (values(o1) - a), rtol=1e-9)


def test_periodic_readings_are_a_subset_of_continuous():
    s, nz = spec(), NoiseModel(sd_abs=1e-4, corr_cycles=2000.0, bias_sd=5e-5)
    sensor = DepthSensor(nz, 100.0, 1e6)
    per = policies.Periodic(1e4).times(s, CTX)
    cont = policies.Continuous(1e3).times(s, CTX)
    assert set(per) <= set(cont)
    op = {o.N: o.value for o in sensor.observe(s, per, stream(2, 5, "sensor"))}
    oc = {o.N: o.value for o in sensor.observe(s, cont, stream(2, 5, "sensor"))}
    for N, v in op.items():
        assert oc[N] == v


def test_ou_field_has_unit_variance_and_lag_correlation():
    rng = np.random.default_rng(0)
    e = ou_field(rng.standard_normal(200000), 0.8)
    assert e.var() == pytest.approx(1.0, abs=0.03)
    assert np.corrcoef(e[:-1], e[1:])[0, 1] == pytest.approx(0.8, abs=0.01)


def test_no_detection_below_floor_and_censoring():
    s = Specimen(index=0, case="toy", a0=1e-4, life=1e5, a_fail=1e-2, truth_theta={},
                 depth_fn=lambda N: np.full(np.shape(N), 4e-5))
    nz = NoiseModel(sd_abs=1e-5, a_floor=5e-5)
    obs = DepthSensor(nz, 100.0, 1e6).observe(s, np.arange(0, 1e4, 1e3), stream(3, 0, "sensor"))
    assert all(o.kind == "miss" for o in obs)
    nz = NoiseModel(sd_abs=1e-5, a_size_min=1e-4)
    obs = DepthSensor(nz, 100.0, 1e6).observe(s, np.arange(0, 1e4, 1e3), stream(3, 0, "sensor"))
    assert all(o.kind == "censored" for o in obs)


def test_policy_schedules():
    s = spec(life=95000.0)
    assert policies.NoMeasurement().times(s, CTX).size == 0
    np.testing.assert_array_equal(policies.OneOff(0.0).times(s, CTX), [0.0])
    t = policies.from_config(dict(kind="periodic", T_frac=0.2), CTX).times(s, CTX)
    np.testing.assert_array_equal(t, [0, 2e4, 4e4, 6e4, 8e4])
    c = policies.from_config(dict(kind="continuous", dt_frac=0.01), CTX).times(s, CTX)
    assert c[0] == 0 and np.all(np.diff(c) == 1000.0) and c[-1] < s.life
    assert policies.Periodic(2e4).next_after(4e4, s, CTX) == 6e4
    assert policies.Periodic(2e4).next_after(8e4, s, CTX) == 1e5       # planned even past the failure
    s.events = np.array([25000.0])
    e = policies.EventTriggered(policies.Periodic(2e4), 3, 100.0).times(s, CTX)
    assert {25000.0, 25100.0, 25200.0, 25300.0} <= set(e)
